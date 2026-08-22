"""Outcome-only RLVR through TRL's maintained GRPO trainer.

Ownership boundary (see STANDARD-LLM-STACK-MIGRATION-PLAN.md §8): TRL owns the
RL algorithm, advantage estimation, clipping, distributed optimization, and
checkpointing.  This module owns only the experiment: which worlds are rolled
out, how the byte-action protocol is generated turn by turn, which tokens the
learner actually produced, and what the privileged verifier scores.

The learner-visible signal is deliberately impoverished: a rollout receives one
scalar computed by the same verifier the evaluator uses, after the episode has
terminated.  No teacher target, preferred action, or intermediate label crosses
this boundary, which is what makes the condition comparable to dense teacher
supervision rather than a disguised imitation run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import ACTION, BOS, END_TURN, EOS, OBS, PAD, _family, encode_bytes
from .evaluate import EVALUATION_METRIC_NAMES, _aggregate_rows, _execute_batched, _matched_sets
from .standard_stack import EXPECTED_PARAMETER_COUNT, assert_model_contract, load_tokenizer
from .train import _state_dict_sha256
from .transfer import _MilestoneSaves, _checkpoint_state_roundtrip, _initial_model_artifact


RLVR_CONTRACT = "step1_rlvr_grpo_v1"
ARMS = ("outcome_only_from_initialization", "outcome_only_from_dense")
EPISODE_KEY = re.compile(r"^step1-world/(\d+)$")
# The learner is scored by the verifier only.  These names appear in the TRL
# reward log; only the first carries a nonzero weight by default.
REWARD_FUNCTION_NAMES = ("verified_success", "verified_spend", "verified_protocol_failure", "verified_legal_termination")


def episode_key(world_seed: int) -> str:
    """Name one world instance for the GRPO group sampler.

    The dataset row is an *episode identity*, not learner-visible text: the
    rollout renders the observation itself and returns the prompt token IDs.
    Nothing in this string reaches the model.
    """
    if world_seed < 0:
        raise ValueError(f"world seed must be non-negative, got {world_seed}")
    return f"step1-world/{world_seed}"


def parse_episode_key(key: str) -> int:
    match = EPISODE_KEY.fullmatch(key)
    if not match:
        raise ValueError(f"not a Step 1 episode key: {key!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class RlvrPlan:
    """The exact, frozen interpretation of one outcome-only training budget."""

    root_seed: int
    arm: str
    world_size: int
    per_device_sequences: int
    steps_per_generation: int
    num_generations: int
    generation_batch_size: int
    worlds_per_update: int
    max_steps: int
    rollout_episode_budget: int
    train_seed_base: int
    distinct_train_worlds: int
    max_turns: int
    max_action_tokens: int
    max_completion_tokens: int
    context_length: int
    milestones_updates: tuple[int, ...]
    milestone_evaluation_episodes: int

    def report(self) -> dict:
        return {
            **asdict(self),
            "milestones_updates": list(self.milestones_updates),
            "budget_unit": "rollout episodes (generation_batch_size * optimizer updates)",
            "budget_policy": (
                "matched to the dense arm on world episodes consumed; token, transition, "
                "FLOP, and wall-clock budgets are measured and reported separately"
            ),
        }


def load_rlvr_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["_meta"] = {"hash": hashlib.sha256(canonical).hexdigest(), "path": str(path.resolve())}
    return config


def rlvr_plan(config: dict, world_size: int = 1) -> RlvrPlan:
    """Validate exact GRPO group/accumulation arithmetic before any GPU work."""
    grpo, rollout, world, run = config["grpo"], config["rollout"], config["world"], config["run"]
    if run.get("arm") not in ARMS:
        raise ValueError(f"run.arm must be one of {ARMS}, got {run.get('arm')!r}")
    per_device, steps_per_generation = grpo["per_device_sequences"], grpo["steps_per_generation"]
    num_generations, max_steps = grpo["num_generations"], grpo["max_steps"]
    if min(per_device, steps_per_generation, num_generations, max_steps, world_size) < 1:
        raise ValueError("GRPO batch, group, and budget settings must all be positive")
    generation_batch_size = per_device * world_size * steps_per_generation
    if generation_batch_size % num_generations:
        raise ValueError(
            "per_device_sequences * world_size * steps_per_generation must be divisible by "
            f"num_generations so every group is complete: {generation_batch_size} % {num_generations} != 0"
        )
    worlds_per_update = generation_batch_size // num_generations
    milestones = tuple(grpo["milestones_updates"])
    if not milestones or tuple(sorted(set(milestones))) != milestones or milestones[0] < 1:
        raise ValueError("milestones_updates must be unique, increasing, and positive")
    if milestones[-1] != max_steps:
        raise ValueError(f"the final milestone must be the terminal update {max_steps}, got {milestones[-1]}")
    max_completion = rollout["max_completion_tokens"]
    if max_completion + rollout["max_prompt_tokens"] > world["context_length"]:
        raise ValueError("rollout prompt plus completion budget exceeds the frozen context length")
    if rollout["max_action_tokens"] > max_completion:
        raise ValueError("a single action may not be allowed more tokens than the whole completion")
    return RlvrPlan(
        root_seed=run["root_seed"],
        arm=run["arm"],
        world_size=world_size,
        per_device_sequences=per_device,
        steps_per_generation=steps_per_generation,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        worlds_per_update=worlds_per_update,
        max_steps=max_steps,
        rollout_episode_budget=max_steps * generation_batch_size,
        train_seed_base=run["root_seed"] + rollout["train_seed_offset"],
        distinct_train_worlds=max_steps * worlds_per_update,
        max_turns=rollout["max_turns"],
        max_action_tokens=rollout["max_action_tokens"],
        max_completion_tokens=max_completion,
        context_length=world["context_length"],
        milestones_updates=milestones,
        milestone_evaluation_episodes=config["evaluation"]["milestone_episodes"],
    )


def episode_dataset(plan: RlvrPlan):
    """One row per distinct training world, consumed exactly once.

    The stream is not shuffled: update ``k`` always rolls out the same declared
    world seeds, so the run is replayable from the plan alone.
    """
    from datasets import Dataset

    seeds = [plan.train_seed_base + index for index in range(plan.distinct_train_worlds)]
    return Dataset.from_dict({"prompt": [episode_key(seed) for seed in seeds], "world_seed": seeds})


@dataclass
class RolloutCounters:
    """Budget axes STEP-1 §6 requires alongside optimizer updates."""

    episodes: int = 0
    world_transitions: int = 0
    generated_action_tokens: int = 0
    observation_tokens: int = 0
    prompt_tokens: int = 0
    turns: int = 0
    malformed_episodes: int = 0
    invalid_episodes: int = 0
    exhausted_episodes: int = 0
    terminated_episodes: int = 0
    successful_episodes: int = 0

    def report(self) -> dict:
        return asdict(self)


class WorldRollout:
    """Turn-by-turn byte-protocol rollout used as TRL's ``rollout_func``.

    TRL passes the per-process prompt slice (each world already repeated
    ``num_generations`` times) and expects ``prompt_ids``/``completion_ids``.
    Observations injected between the learner's turns are returned in
    ``env_mask`` as external tokens, so TRL trains only on tokens the model
    actually produced.
    """

    def __init__(self, config: dict, plan: RlvrPlan):
        self.world = config["world"]
        self.rendering = config["world"]["rendering"]
        self.plan = plan
        self.sampling = config["rollout"]
        self.counters = RolloutCounters()

    def _generation_config(self, max_new_tokens: int):
        from transformers import GenerationConfig

        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.sampling["temperature"],
            top_p=self.sampling["top_p"],
            top_k=self.sampling["top_k"],
            eos_token_id=END_TURN,
            pad_token_id=PAD,
            # The frozen model config disables the cache for training; batched
            # rollout generation re-enables it explicitly, exactly as the
            # evaluator does.
            use_cache=True,
        )

    @torch.no_grad()
    def _sample_actions(self, model, contexts: list[list[int]], allowances: list[int], device) -> list[list[int]]:
        """Batched continuation of each live episode's own token history.

        Rows are bucketed by their remaining allowance so an episode close to
        the context limit cannot shorten another episode's action.
        """
        generated: list[list[int]] = [[] for _ in contexts]
        buckets: dict[int, list[int]] = {}
        for index, allowance in enumerate(allowances):
            buckets.setdefault(allowance, []).append(index)
        for allowance, rows in buckets.items():
            config = self._generation_config(allowance)
            for start in range(0, len(rows), self.sampling["generation_rows"]):
                group = rows[start:start + self.sampling["generation_rows"]]
                values = [contexts[index] for index in group]
                width = max(len(value) for value in values)
                input_ids = torch.full((len(group), width), PAD, dtype=torch.long, device=device)
                attention_mask = torch.zeros((len(group), width), dtype=torch.bool, device=device)
                for row, value in enumerate(values):
                    input_ids[row, width - len(value):] = torch.tensor(value, dtype=torch.long, device=device)
                    attention_mask[row, width - len(value):] = True
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=config)
                for index, row in zip(group, output):
                    generated[index] = row[width:].tolist()
        return generated

    @staticmethod
    def _action_text(payload: list[int]) -> str | None:
        if any(token >= 256 for token in payload):
            return None
        try:
            return bytes(payload).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None

    def _generation_model(self, trainer):
        """TRL's own accessor for the underlying model during generation."""
        from trl.models import unwrap_model_for_generation

        return unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator)

    def __call__(self, prompts: list, trainer) -> dict:
        from world_py import Batch

        device = trainer.args.device
        seeds = [parse_episode_key(prompt) for prompt in prompts]
        worlds = [Batch(_family(self.world), seed=seed, n_episodes=1) for seed in seeds]
        prompt_ids = [[BOS, OBS, *encode_bytes(world.observations(self.rendering)[0]), ACTION] for world in worlds]
        for ids in prompt_ids:
            if len(ids) > self.sampling["max_prompt_tokens"]:
                raise AssertionError(f"initial observation exceeds the declared prompt budget: {len(ids)} tokens")
        completion_ids: list[list[int]] = [[] for _ in prompts]
        env_mask: list[list[int]] = [[] for _ in prompts]
        malformed = [0] * len(prompts)
        invalid = [0] * len(prompts)
        exhausted = [False] * len(prompts)
        turns = [0] * len(prompts)
        live = list(range(len(prompts)))
        autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=trainer.args.fp16 and device.type == "cuda")
        with self._generation_model(trainer) as model, autocast:
            for _ in range(self.plan.max_turns):
                if not live:
                    break
                contexts, allowances, active = [], [], []
                for index in live:
                    remaining = self.plan.max_completion_tokens - len(completion_ids[index]) - 1  # reserve EOS
                    allowance = min(self.plan.max_action_tokens, remaining)
                    if allowance < 1:
                        exhausted[index] = True
                        continue
                    contexts.append(prompt_ids[index] + completion_ids[index])
                    allowances.append(allowance)
                    active.append(index)
                if not active:
                    break
                samples = self._sample_actions(model, contexts, allowances, device)
                still_live = []
                for index, sample in zip(active, samples):
                    stop = sample.index(END_TURN) + 1 if END_TURN in sample else len(sample)
                    completion_ids[index].extend(sample[:stop])
                    env_mask[index].extend([1] * stop)
                    self.counters.generated_action_tokens += stop
                    turns[index] += 1
                    text = self._action_text(sample[:stop - 1]) if END_TURN in sample else None
                    if text is None:
                        malformed[index] += 1
                        continue
                    record = worlds[index].step_attempts([text], self.rendering)[0]
                    if record["parsed_action"] is None:
                        malformed[index] += 1
                        continue
                    if not record["accepted"]:
                        invalid[index] += 1
                        continue
                    self.counters.world_transitions += 1
                    if worlds[index].done()[0]:
                        continue
                    observation = [OBS, *encode_bytes(record["observation_after"]), ACTION]
                    if len(completion_ids[index]) + len(observation) + 1 > self.plan.max_completion_tokens:
                        exhausted[index] = True
                        continue
                    completion_ids[index].extend(observation)
                    env_mask[index].extend([0] * len(observation))
                    self.counters.observation_tokens += len(observation)
                    still_live.append(index)
                live = still_live
        for index in live:
            # Still live after the declared turn limit: the episode never reached
            # a verdict and is scored as an unterminated failure.
            exhausted[index] = True

        extras: dict[str, list] = {key: [] for key in (
            "outcome_success", "outcome_spent", "outcome_steps", "outcome_terminated", "outcome_correct",
            "outcome_budget_violation", "outcome_unreachable", "rollout_malformed", "rollout_invalid",
            "rollout_exhausted", "rollout_turns", "rollout_world_seed",
        )}
        for index, world in enumerate(worlds):
            terminated, correct, spent, steps, budget_violation, unreachable = world.privileged_outcomes()[0]
            # Exactly the evaluator's success definition; RLVR must not be
            # scored more leniently than the metric it is compared on.
            success = bool(terminated and correct and not malformed[index] and not invalid[index])
            if not exhausted[index]:
                completion_ids[index].append(EOS)
                # EOS is appended by the harness, never generated by the model,
                # so it is external and carries no gradient.
                env_mask[index].append(0)
            self.counters.episodes += 1
            self.counters.prompt_tokens += len(prompt_ids[index])
            self.counters.turns += turns[index]
            self.counters.malformed_episodes += bool(malformed[index])
            self.counters.invalid_episodes += bool(invalid[index])
            self.counters.exhausted_episodes += bool(exhausted[index])
            self.counters.terminated_episodes += bool(terminated)
            self.counters.successful_episodes += success
            for key, value in (
                ("outcome_success", success), ("outcome_spent", int(spent)), ("outcome_steps", int(steps)),
                ("outcome_terminated", bool(terminated)), ("outcome_correct", bool(correct)),
                ("outcome_budget_violation", bool(budget_violation)), ("outcome_unreachable", bool(unreachable)),
                ("rollout_malformed", malformed[index]), ("rollout_invalid", invalid[index]),
                ("rollout_exhausted", exhausted[index]), ("rollout_turns", turns[index]),
                ("rollout_world_seed", seeds[index]),
            ):
                extras[key].append(value)
        for index, ids in enumerate(completion_ids):
            if len(ids) != len(env_mask[index]):
                raise AssertionError("rollout token and env-mask lengths diverged")
            if len(prompt_ids[index]) + len(ids) > self.plan.context_length:
                raise AssertionError("rollout exceeded the frozen context length")
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            # TRL recomputes the policy log-probabilities for these tokens; the
            # rollout does not supply a sampling distribution of its own.
            "logprobs": None,
            "env_mask": env_mask,
            **extras,
        }


def verified_success(*, outcome_success: list[bool], **_kwargs) -> list[float]:
    """The outcome-only reward: one verified terminal verdict per trajectory."""
    return [1.0 if value else 0.0 for value in outcome_success]


def verified_spend(*, outcome_spent: list[int], **_kwargs) -> list[float]:
    """Verified probe spend, reported (and optionally weighted) as a cost term."""
    return [-float(value) for value in outcome_spent]


def verified_protocol_failure(*, rollout_malformed: list[int], rollout_invalid: list[int], **_kwargs) -> list[float]:
    """Diagnostic rate of trajectories killed by the action protocol itself."""
    return [1.0 if bad or worse else 0.0 for bad, worse in zip(rollout_malformed, rollout_invalid)]


def verified_legal_termination(*, outcome_terminated: list[bool], rollout_malformed: list[int],
                               rollout_invalid: list[int], **_kwargs) -> list[float]:
    """Credit for reaching a verdict at all, right or wrong.

    Still outcome-only: `terminated` is a field of the privileged verifier's
    terminal report, not an intermediate teacher label.  Its purpose is the
    degenerate-group problem — under a purely binary reward, a group in which
    every rollout fails carries no advantage even when some of those rollouts
    played legally and others died on an illegal move.  Weighted well below
    success, it separates those cases without competing with correctness.
    """
    return [
        1.0 if terminated and not bad and not worse else 0.0
        for terminated, bad, worse in zip(outcome_terminated, rollout_malformed, rollout_invalid)
    ]


def _reward_weights(grpo: dict) -> list[float]:
    """Verified success always carries the primary weight; the rest are declared.

    `legal_termination_reward_weight` defaults to zero so every earlier arm
    keeps its exact objective.
    """
    return [1.0, grpo["spend_reward_weight"], 0.0, grpo.get("legal_termination_reward_weight", 0.0)]


def _grpo_config(config: dict, plan: RlvrPlan, workspace: Path):
    from trl import GRPOConfig

    grpo = config["grpo"]
    arguments = GRPOConfig(
        output_dir=str(workspace / "checkpoints"),
        per_device_train_batch_size=plan.per_device_sequences,
        gradient_accumulation_steps=plan.steps_per_generation,
        # TRL rejects setting `generation_batch_size` alongside this; it derives
        # the same value as `per_device * ranks * steps_per_generation`, which is
        # asserted below rather than passed twice.
        steps_per_generation=plan.steps_per_generation,
        num_generations=plan.num_generations,
        max_completion_length=plan.max_completion_tokens,
        max_steps=plan.max_steps,
        learning_rate=grpo["learning_rate"],
        weight_decay=grpo["weight_decay"],
        lr_scheduler_type=grpo["lr_scheduler_type"],
        warmup_ratio=grpo["warmup_fraction"],
        beta=grpo["beta"],
        epsilon=grpo["epsilon"],
        loss_type=grpo["loss_type"],
        scale_rewards=grpo["scale_rewards"],
        num_iterations=grpo["num_iterations"],
        mask_truncated_completions=grpo["mask_truncated_completions"],
        reward_weights=_reward_weights(grpo),
        temperature=config["rollout"]["temperature"],
        top_p=config["rollout"]["top_p"],
        top_k=config["rollout"]["top_k"],
        # The declared world stream is consumed in order so update k is replayable.
        shuffle_dataset=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=None,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        log_completions=False,
        seed=plan.root_seed,
        data_seed=plan.root_seed,
    )
    if arguments.generation_batch_size != plan.generation_batch_size:
        raise AssertionError(
            f"TRL resolved a generation batch of {arguments.generation_batch_size}, "
            f"but the declared plan requires {plan.generation_batch_size}"
        )
    return arguments


def _trainer(model, config: dict, plan: RlvrPlan, workspace: Path, rollout: WorldRollout):
    from trl import GRPOTrainer

    return GRPOTrainer(
        model=model,
        reward_funcs=[verified_success, verified_spend, verified_protocol_failure, verified_legal_termination],
        args=_grpo_config(config, plan, workspace),
        train_dataset=episode_dataset(plan),
        processing_class=load_tokenizer(),
        rollout_func=rollout,
        callbacks=[_MilestoneSaves(plan.milestones_updates)],
    )


def _verify_dense_source(source_run: Path, config: dict) -> tuple[dict, Path]:
    """Accept a warm-start checkpoint only under its exact dense identity."""
    expected = config["source"]
    report = json.loads((source_run / "production" / "training_report.json").read_text())
    experiment = json.loads((source_run / "production" / "model" / "experiment.json").read_text())
    resolved = json.loads((source_run / "resolved_config.json").read_text())
    checks = {
        "dense contract": (report.get("contract"), "step1_dense_training_v1"),
        "source SHA": (report.get("source_git_sha"), expected["git_sha"]),
        "source config hash": (report.get("config_hash"), expected["config_hash"]),
        "source rendering": (resolved.get("world", {}).get("rendering"), config["world"]["rendering"]),
        "source root seed": (resolved.get("run", {}).get("root_seed"), config["run"]["root_seed"]),
        "parameter count": (experiment.get("parameter_count"), EXPECTED_PARAMETER_COUNT),
        "exact source serialization": (report.get("serialization", {}).get("exact_state_dict", {}).get("exact"), True),
        "reported source state SHA": (
            report.get("serialization", {}).get("exact_state_dict", {}).get("actual_state_sha256"),
            expected["model_state_sha256"],
        ),
    }
    mismatches = {name: {"actual": actual, "expected": wanted} for name, (actual, wanted) in checks.items() if actual != wanted}
    if mismatches:
        raise AssertionError(f"dense warm-start source mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return report, source_run / "production" / "model"


def _starting_model(config: dict, plan: RlvrPlan, workspace: Path, source_run: Path | None, state):
    """Materialize the arm's exact starting weights once and reload them everywhere."""
    from transformers import AutoModelForCausalLM

    if plan.arm == "outcome_only_from_dense":
        if source_run is None:
            raise ValueError("the dense warm-start arm requires --source-run")
        source_report, artifact = _verify_dense_source(source_run, config)
        model = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True)
        state_sha = _state_dict_sha256(model.state_dict())
        if state_sha != config["source"]["model_state_sha256"]:
            raise AssertionError("loaded dense warm-start state hash mismatch")
        provenance = {
            "policy": "dense_A_checkpoint",
            "source_run": str(source_run.resolve()),
            "source_git_sha": config["source"]["git_sha"],
            "source_config_hash": config["source"]["config_hash"],
            "prior_dense_cost": source_report["token_accounting"],
        }
    else:
        serialization_path = workspace / "budget-0-serialization.json"
        artifact = workspace / "budget-0-model"
        if state.is_main_process:
            _, artifact, serialization = _initial_model_artifact(workspace, config)
            atomic_json(serialization_path, serialization)
        state.wait_for_everyone()
        model = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True)
        state_sha = _state_dict_sha256(model.state_dict())
        provenance = {
            "policy": "random_from_local_gpt_neox_config",
            "budget_zero_serialization": json.loads(serialization_path.read_text()),
        }
    assert_model_contract(model)
    return model, {**provenance, "root_seed": plan.root_seed, "parameter_count": EXPECTED_PARAMETER_COUNT, "model_state_sha256": state_sha}


def _reduce_counters(counters: RolloutCounters, accelerator) -> dict:
    """Sum every measured budget axis across ranks without a project all-reduce."""
    fields = sorted(counters.report())
    local = torch.tensor([getattr(counters, name) for name in fields], dtype=torch.float64, device=accelerator.device)
    total = accelerator.reduce(local, reduction="sum").detach().cpu().tolist()
    return {name: int(value) for name, value in zip(fields, total)}


def _training_signal(log_history: list[dict]) -> dict:
    """Read the learning signal from the standard Trainer log, not from a probe.

    ``frac_reward_zero_std`` is TRL's own measure of groups whose rewards are
    identical; a run in which it stays at 1.0 produced no advantage at all and
    could not have learned from the verifier regardless of its budget.
    """
    def series(key: str) -> list[float]:
        return [float(entry[key]) for entry in log_history if key in entry]

    rewards = series("rewards/verified_success/mean")
    zero_std = series("frac_reward_zero_std")
    # Present only when a KL anchor is enabled; a paralysed anchor and an
    # ineffective one are told apart by this series plus the protocol-failure
    # trend, not by the reward curve alone.
    kl = series("kl")
    tail = rewards[-max(1, len(rewards) // 10):] if rewards else []
    return {
        "kl_to_reference_mean": sum(kl) / len(kl) if kl else None,
        "kl_to_reference_final": kl[-1] if kl else None,
        "logged_updates": len(rewards),
        "verified_success_reward_first": rewards[0] if rewards else None,
        "verified_success_reward_final_decile_mean": sum(tail) / len(tail) if tail else None,
        "verified_success_reward_max": max(rewards) if rewards else None,
        "frac_reward_zero_std_mean": sum(zero_std) / len(zero_std) if zero_std else None,
        "frac_reward_zero_std_final": zero_std[-1] if zero_std else None,
        "updates_with_any_reward_variance": sum(1 for value in zero_std if value < 1.0),
        "protocol_failure_rate_final": (
            series("rewards/verified_protocol_failure/mean")[-1] if series("rewards/verified_protocol_failure/mean") else None
        ),
        "interpretation": "no group reward variance implies no GRPO gradient signal, whatever the budget",
    }


@torch.no_grad()
def _teacher_forced_nll(model, config: dict, shard_path: Path, device) -> float:
    """Distributional distance from the teacher, comparable across stages.

    This is a diagnostic, never a training signal: it answers whether an
    outcome-only update moved the policy at all, on the same quantity the dense
    arm reports.
    """
    from .data import BinaryShard, collate

    dataset = BinaryShard(shard_path)
    try:
        total, count = 0.0, 0
        for start in range(0, len(dataset), 4):
            batch = collate([dataset[index] for index in range(start, min(start + 4, len(dataset)))],
                            config["world"]["context_length"])
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            labels = int((batch["labels"][:, 1:] != -100).sum())
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("teacher-forced diagnostic loss is absent or non-finite")
            total += float(loss) * labels
            count += labels
    finally:
        dataset.close()
    if count <= 0:
        raise AssertionError("teacher-forced diagnostic received no supervised action tokens")
    return total / count


def _diagnostic_shard(config: dict, run_dir: Path) -> Path:
    """One deterministic teacher shard on the evaluator's own validation seeds."""
    from .data import generate_rust_shard

    params, seed, count, rendering, _comparison = _matched_sets(config)["validation"]
    binary, _manifest, _replay = generate_rust_shard(
        params, seed, count, config["world"]["context_length"], rendering,
        run_dir / "datasets", "validation",
    )
    return binary


@torch.no_grad()
def _evaluate_sets(model, sets: dict, device) -> dict:
    result = {}
    for name, (params, seed, count, rendering, comparison) in sets.items():
        metrics = _aggregate_rows(_execute_batched(model, params, seed, count, rendering, device))
        for metric, value in metrics.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise FloatingPointError(f"non-finite RLVR metric {name}.{metric}: {value!r}")
        result[name] = {"comparison": comparison, "metrics": metrics}
    return result


def _milestone_sets(config: dict, plan: RlvrPlan, final: bool) -> dict:
    """Matched evaluation sets; the full dense matrix is reserved for the endpoint."""
    sets = _matched_sets(config)
    if final:
        return sets
    params, seed, _count, rendering, comparison = sets["validation"]
    return {"validation": (params, seed, plan.milestone_evaluation_episodes, rendering, comparison)}


def assert_rlvr_report_contract(report: dict, config: dict, plan: RlvrPlan) -> None:
    """Exact operational contract. It sets no scientific threshold."""
    if report.get("contract") != RLVR_CONTRACT:
        raise AssertionError("RLVR contract mismatch")
    if report.get("experiment_config_sha256") != config["_meta"]["hash"]:
        raise AssertionError("RLVR configuration hash mismatch")
    if report.get("plan") != plan.report():
        raise AssertionError("RLVR plan mismatch")
    algorithm = report.get("algorithm", {})
    if algorithm.get("trainer") != "trl.GRPOTrainer" or algorithm.get("signal") != "outcome_only_privileged_verifier":
        raise AssertionError("RLVR algorithm identity mismatch")
    if tuple(algorithm.get("reward_functions", ())) != REWARD_FUNCTION_NAMES:
        raise AssertionError("RLVR reward function set mismatch")
    weights = algorithm.get("reward_weights", [None])
    if weights[0] != 1.0 or any(weight >= 1.0 for weight in weights[1:]):
        raise AssertionError("verified success must carry the strictly largest reward weight")
    if report.get("initialization", {}).get("root_seed") != plan.root_seed:
        raise AssertionError("RLVR initialization seed mismatch")
    budget = report.get("budget_accounting", {})
    if budget.get("optimizer_updates") != plan.max_steps:
        raise AssertionError("RLVR did not complete its declared update budget")
    if budget.get("rollout_episodes") != plan.rollout_episode_budget:
        raise AssertionError(
            f"rolled-out episodes {budget.get('rollout_episodes')} != declared budget {plan.rollout_episode_budget}"
        )
    for axis in ("world_transitions", "generated_action_tokens", "wall_clock_seconds"):
        value = budget.get(axis)
        if not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
            raise AssertionError(f"budget axis {axis} was not measured")
    milestones = report.get("milestones", [])
    if [point.get("budget_updates") for point in milestones] != list(plan.milestones_updates):
        raise AssertionError("RLVR milestone grid mismatch")
    for point in milestones:
        serialization = point.get("serialization", {})
        if serialization.get("exact") is not True:
            raise AssertionError(f"milestone {point.get('budget_updates')} artifact is not exact")
        final = point["budget_updates"] == plan.max_steps
        expected_sets = {"validation", "structural", "rendering_b", "reversible_control"} if final else {"validation"}
        if set(point.get("evaluation", {})) != expected_sets:
            raise AssertionError(f"milestone {point['budget_updates']} evaluation set mismatch")
        for name, item in point["evaluation"].items():
            if set(item.get("metrics", {})) != set(EVALUATION_METRIC_NAMES):
                raise AssertionError(f"milestone {point['budget_updates']} {name} metric fields mismatch")
        nll = point.get("teacher_forced_action_nll")
        if not isinstance(nll, (int, float)) or isinstance(nll, bool) or not math.isfinite(nll):
            raise AssertionError(f"milestone {point['budget_updates']} has no finite teacher-forced diagnostic")
    if report.get("training_signal", {}).get("logged_updates", 0) < 1:
        raise AssertionError("RLVR training log carries no reward history")


def load_config(path: Path) -> dict:
    """Accept the runner's resolved JSON or a checked-in TOML unchanged."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return load_rlvr_config(path)


def resolve_source_run(config: dict, plan: RlvrPlan, source_run: Path | None) -> Path | None:
    """Locate an attached upstream Kaggle output by its exact identity.

    Cloud-to-cloud dependencies arrive under `/kaggle/input`; a directory name
    is never sufficient evidence, so discovery goes through the same identity
    check the transfer stage uses.
    """
    if plan.arm != "outcome_only_from_dense":
        return None
    if source_run is not None:
        return source_run
    from .transfer import locate_dense_source

    return locate_dense_source([Path("/kaggle/input")], config["source"])


def run(config_path: Path, output_dir: Path, source_run: Path | None = None) -> Path:
    from accelerate import PartialState

    import trl

    config = load_config(config_path)
    state = PartialState()
    plan = rlvr_plan(config, world_size=state.num_processes)
    if state.num_processes != 2:
        raise AssertionError(f"Step 1 RLVR requires exactly two Trainer/Accelerate ranks, got {state.num_processes}")
    source_run = resolve_source_run(config, plan, source_run)
    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "analysis" / "rlvr_plan.json", plan.report())
    state.wait_for_everyone()

    workspace = output_dir / plan.arm
    model, initialization = _starting_model(config, plan, workspace, source_run, state)
    rollout = WorldRollout(config, plan)
    trainer = _trainer(model, config, plan, workspace, rollout)
    started = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - started
    if trainer.state.global_step != plan.max_steps:
        raise AssertionError(f"RLVR stopped at update {trainer.state.global_step}, expected {plan.max_steps}")
    budget = _reduce_counters(rollout.counters, trainer.accelerator)
    finished_ranks = trainer.accelerator.gather_for_metrics(
        torch.tensor([trainer.args.process_index], device=trainer.args.device)
    ).detach().cpu().tolist()
    if sorted(finished_ranks) != list(range(plan.world_size)):
        raise AssertionError(f"not every rank completed RLVR training: {finished_ranks}")
    log_history = list(trainer.state.log_history)
    trainer.accelerator.wait_for_everyone()
    device = trainer.args.device
    model.to("cpu")
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report_path = output_dir / "rlvr_report.json"
    if not state.is_main_process:
        # Training is complete; rank zero alone runs the long closed-loop
        # evaluations, with no pending collective that could time out.
        return report_path

    milestones = []
    diagnostic_shard = _diagnostic_shard(config, output_dir)
    for step in plan.milestones_updates:
        checkpoint = workspace / "checkpoints" / f"checkpoint-{step}"
        point_model, serialization = _checkpoint_state_roundtrip(checkpoint)
        point_model.to(device).eval()
        evaluation = _evaluate_sets(point_model, _milestone_sets(config, plan, step == plan.max_steps), device)
        milestones.append({
            "budget_updates": step,
            "rollout_episodes": step * plan.generation_batch_size,
            "artifact": str(checkpoint.resolve()),
            "serialization": serialization,
            "evaluation": evaluation,
            "teacher_forced_action_nll": _teacher_forced_nll(point_model, config, diagnostic_shard, device),
        })
        point_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grpo = config["grpo"]
    report = {
        "contract": RLVR_CONTRACT,
        "experiment_config_sha256": config["_meta"]["hash"],
        "algorithm": {
            "library": "trl",
            "library_version": trl.__version__,
            "trainer": "trl.GRPOTrainer",
            "paper": "https://huggingface.co/papers/2402.03300",
            "signal": "outcome_only_privileged_verifier",
            "reward_functions": list(REWARD_FUNCTION_NAMES),
            "reward_weights": _reward_weights(grpo),
            "reward_definition": {
                "verified_success": "1.0 iff the verifier reports a terminated, correct episode with no malformed or invalid action",
                "verified_spend": "negated verified probe spend",
                "verified_protocol_failure": "diagnostic only; weight 0",
                "verified_legal_termination": "1.0 iff the verifier reports a terminated episode with no protocol failure, right or wrong",
            },
            "loss_type": grpo["loss_type"],
            "scale_rewards": grpo["scale_rewards"],
            "beta_kl": grpo["beta"],
            "epsilon": grpo["epsilon"],
            "num_generations": plan.num_generations,
            "num_iterations": grpo["num_iterations"],
            "privileged_intermediate_labels": False,
            "logged_reward_field": (
                "TRL's `reward` log entry is the unweighted sum over reward functions; the trained "
                "objective is the weighted sum, so read `rewards/verified_success/mean`"
            ),
        },
        "plan": plan.report(),
        "initialization": initialization,
        "world": {**config["world"], "train_seed_base": plan.train_seed_base, "distinct_train_worlds": plan.distinct_train_worlds},
        "budget_accounting": {
            **budget,
            "optimizer_updates": plan.max_steps,
            "rollout_episodes": budget["episodes"],
            "wall_clock_seconds": elapsed,
            "ranks_finished": sorted(finished_ranks),
            "note": "episodes and transitions are measured; no single axis makes RLVR and dense supervision equivalent",
        },
        "training_signal": _training_signal(log_history),
        "milestones": milestones,
        "scientific_acceptance_policy": "measurement only; no capability threshold gates this run",
    }
    assert_rlvr_report_contract(report, config, plan)
    atomic_json(report_path, report)
    atomic_json(output_dir / "logs" / "training_log_history.json", log_history)
    atomic_json(output_dir / "evaluation" / "metrics.json", {
        "contract": RLVR_CONTRACT,
        "milestones": [
            {"budget_updates": point["budget_updates"], "rollout_episodes": point["rollout_episodes"],
             "evaluation": point["evaluation"]}
            for point in milestones
        ],
        "training_signal": report["training_signal"],
    })
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    args = parser.parse_args()
    print(run(args.resolved_config, args.run_dir, args.source_run))


if __name__ == "__main__":
    main()
