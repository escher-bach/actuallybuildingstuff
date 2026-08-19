"""Learner-conditioned dense supervision: STEP-1 §6's third condition.

The three conditions in STEP-1 §6 differ only in which states carry supervision
and what the signal is.  Dense teaching labels states the *teacher* visits.
Outcome-only RLVR labels nothing and scores the end.  This stage labels the
states the *learner* visits, with the same privileged teacher and the same
cross-entropy objective — DAgger's data collection, not a new learning rule.
TRL is deliberately absent: substituting a policy gradient here would change the
gradient paradigm and stop answering the question.

THEORY-PHASE.md §8 registers the predictions this run tests, P1 and the on-policy
half of P3, before it was implemented.

Two things this stage does *not* claim, stated here so the report cannot drift:

- There is no matched teacher-conditioned continuation arm.  Both arms are
  scored against their own starting policy under an identical instrument
  (`budget-0`), which matches the *measurement* but not the *training*.  Any
  movement is therefore consistent with "learner-state supervision helped" and
  with "512 more supervised updates of any kind helped".  Separating them is one
  config change and one run.
- The extended metrics live in this stage's report only.  `evaluate.py`'s frozen
  metric set and every existing contract are untouched, so every number here is
  comparable with every number already retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import ACTION, BOS, OBS, SequenceDataset, _family, collate, encode_bytes
from .evaluate import EVALUATION_METRIC_NAMES, _aggregate_rows, _execute_batched, _matched_sets
from .learner import MALFORMED_ATTEMPT, CollectionSettings, assert_collection_contract, collect_tranche
from .standard_stack import EXPECTED_PARAMETER_COUNT, assert_model_contract, load_tokenizer
from .train import _state_dict_sha256


LEARNER_CONDITIONED_CONTRACT = "step1_learner_conditioned_dagger_v1"
ARMS = ("learner_conditioned_from_initialization", "learner_conditioned_from_dense")
EXTENDED_METRIC_NAMES = (
    "success_conditioned_on_legal_play", "protocol_failure_rate", "commitment_rate",
    "premature_commitment_rate", "mean_probes", "probe_histogram",
    "mean_live_hypotheses_at_commitment",
)


@dataclass(frozen=True)
class DaggerPlan:
    """The exact, frozen interpretation of one learner-conditioned budget."""

    root_seed: int
    arm: str
    world_size: int
    per_device_sequences: int
    gradient_accumulation_steps: int
    sequences_per_update: int
    rounds: int
    episodes_per_round: int
    collection_episode_budget: int
    updates_per_round: int
    max_steps: int
    train_seed_base: int
    context_length: int
    milestones_updates: tuple[int, ...]
    milestone_evaluation_episodes: int
    recovery_probe_episodes: int
    recovery_probe_seed: int

    def report(self) -> dict:
        return {
            **asdict(self),
            "milestones_updates": list(self.milestones_updates),
            "budget_unit": "optimizer updates over on-policy episodes collected in rounds",
            "budget_policy": (
                "collection episodes and optimizer updates are declared and matched across arms; "
                "supervised tokens are data dependent and reported, never used to change an update"
            ),
            "aggregation": (
                "DAgger: each round trains on every example collected so far, so the buffer grows "
                "while the per-round update count stays fixed"
            ),
        }


def dagger_plan(config: dict, world_size: int = 1) -> DaggerPlan:
    """Validate the round/update arithmetic before any GPU work happens."""
    run, collection, training, world = config["run"], config["collection"], config["training"], config["world"]
    if run.get("arm") not in ARMS:
        raise ValueError(f"run.arm must be one of {ARMS}, got {run.get('arm')!r}")
    per_device = training["per_device_sequences"]
    accumulation = training["gradient_accumulation_steps"]
    rounds, updates_per_round = collection["rounds"], training["updates_per_round"]
    episodes_per_round = collection["episodes_per_round"]
    if min(per_device, accumulation, rounds, updates_per_round, episodes_per_round, world_size) < 1:
        raise ValueError("round, batch, and budget settings must all be positive")
    if episodes_per_round % world_size:
        raise ValueError(
            "episodes_per_round must divide across ranks so every rank collects an equal slice: "
            f"{episodes_per_round} % {world_size} != 0"
        )
    milestones = tuple(updates_per_round * index for index in range(1, rounds + 1))
    return DaggerPlan(
        root_seed=run["root_seed"],
        arm=run["arm"],
        world_size=world_size,
        per_device_sequences=per_device,
        gradient_accumulation_steps=accumulation,
        sequences_per_update=per_device * world_size * accumulation,
        rounds=rounds,
        episodes_per_round=episodes_per_round,
        collection_episode_budget=rounds * episodes_per_round,
        updates_per_round=updates_per_round,
        max_steps=rounds * updates_per_round,
        train_seed_base=run["root_seed"] + collection["train_seed_offset"],
        context_length=world["context_length"],
        milestones_updates=milestones,
        milestone_evaluation_episodes=config["evaluation"]["milestone_episodes"],
        recovery_probe_episodes=collection["recovery_probe_episodes"],
        # The recovery probe is an evaluation: it uses the held-out validation
        # seeds and nothing collected from it is ever trained on.
        recovery_probe_seed=run["root_seed"] + 1_000_000,
    )


def round_seeds(plan: DaggerPlan, index: int) -> list[int]:
    """The declared world stream for round `index`, consumed exactly once."""
    start = plan.train_seed_base + index * plan.episodes_per_round
    return [start + offset for offset in range(plan.episodes_per_round)]


def _aggregate_extended(rows: list[dict]) -> dict:
    """The measurements THEORY-PHASE §7 asks for and success rate hides.

    A raw success rate mixes a decision failure with an interface failure. Every
    episode leaves the evaluator at its first protocol failure, so malformed and
    invalid are disjoint per episode and legal play is exactly the remainder.
    """
    if not rows:
        raise AssertionError("extended aggregation received no rows")
    total = len(rows)
    legal = [row for row in rows if not row["malformed"] and not row["invalid"]]
    committed = [row for row in rows if row["committed"]]
    premature = [row for row in committed if not row["licensed_at_commitment"]]
    return {
        # Success among episodes that never played illegally: decision quality
        # with the interface term removed.
        "success_conditioned_on_legal_play": (
            sum(row["success"] for row in legal) / len(legal) if legal else None
        ),
        "protocol_failure_rate": (total - len(legal)) / total,
        "commitment_rate": len(committed) / total,
        # The P1 quantity: committing while the evidence still admits more than
        # one hypothesis. Undefined when nothing committed.
        "premature_commitment_rate": len(premature) / len(committed) if committed else None,
        "mean_probes": sum(row["probes"] for row in rows) / total,
        "probe_histogram": {str(key): value for key, value in
                            sorted(Counter(row["probes"] for row in rows).items())},
        "mean_live_hypotheses_at_commitment": (
            sum(row["live_hypotheses_at_commitment"] for row in committed) / len(committed)
            if committed else None
        ),
    }


@torch.no_grad()
def _execute_retry_tolerant(model, params: dict, seed: int, count: int, rendering: str,
                            device, settings: CollectionSettings, max_retries: int) -> list[dict]:
    """Score a policy under the recovery the world and the teacher already allow.

    The frozen evaluator drops an episode at its first malformed or invalid
    action.  That is a convention of the evaluator, not a rule of the world:
    `step_attempts` leaves the state unchanged and the episode `Running`, and the
    teacher supplies a target from that unchanged state.  Learner-conditioned
    training is built on exactly that, so scoring it with a rule that grants no
    second attempt charges it full price for behaviour the regime was taught.

    This grants up to `max_retries` consecutive failures per state and is
    otherwise the frozen evaluator: same worlds, same greedy decoding, same
    success definition, same privileged instrumentation.  A failed attempt stays
    in the context, which is how the arm was trained.

    It is reported alongside the frozen number, never instead of it.
    """
    from world_py import Batch

    from .learner import _decode_actions

    n_probe = params["n_probe"]
    worlds = [Batch(_family(params), seed=seed + index, n_episodes=1) for index in range(count)]
    histories: list[list[int]] = [[BOS] for _ in worlds]
    probes = [0] * count
    retries = [0] * count
    recovered = [0] * count
    consecutive = [0] * count
    commitment: list[dict | None] = [None] * count
    exhausted = [False] * count
    live = list(range(count))
    limit = params["step_slack"] + params["n_probe"] + params["n_hyp"] + 4
    steps = [0] * count
    for _turn in range(settings.max_turns):
        if not live:
            break
        contexts, active = [], []
        for index in live:
            observation = [OBS, *encode_bytes(worlds[index].observations(rendering)[0]), ACTION]
            context = histories[index] + observation
            if len(context) + settings.max_action_tokens + 2 > settings.context_length:
                exhausted[index] = True
                continue
            contexts.append(context)
            active.append(index)
        if not active:
            break
        attempts = _decode_actions(model, contexts, settings, device)
        still_live = []
        for index, context, (tokens, text) in zip(active, contexts, attempts):
            licensed = worlds[index].privileged_teacher_targets()[0]["licenses_commitment"]
            live_hypotheses = worlds[index].privileged_consistent_counts()[0]
            histories[index] = context + tokens
            record = worlds[index].step_attempts([text if text is not None else MALFORMED_ATTEMPT], rendering)[0]
            accepted = text is not None and record["parsed_action"] is not None and record["accepted"]
            if not accepted:
                retries[index] += 1
                consecutive[index] += 1
                if consecutive[index] > max_retries:
                    continue
                still_live.append(index)
                continue
            if consecutive[index]:
                recovered[index] += 1
            consecutive[index] = 0
            if record["parsed_action"] < n_probe:
                probes[index] += 1
            else:
                commitment[index] = {"licensed": bool(licensed), "live_hypotheses": int(live_hypotheses)}
            steps[index] += 1
            if worlds[index].done()[0] or steps[index] >= limit:
                continue
            still_live.append(index)
        live = still_live
    rows = []
    for index, world in enumerate(worlds):
        terminated, correct, spent, *_ = world.privileged_outcomes()[0]
        rows.append({
            # Success is the verifier's, exactly as the frozen evaluator defines
            # it, minus the "no failed attempt ever occurred" clause the world
            # itself does not impose.
            "success": bool(terminated and correct),
            "spent": int(spent), "steps": steps[index], "probes": probes[index],
            "retries": retries[index], "recovered": recovered[index],
            "exhausted": exhausted[index],
            "committed": commitment[index] is not None,
            "licensed_at_commitment": commitment[index]["licensed"] if commitment[index] else None,
            "live_hypotheses_at_commitment": commitment[index]["live_hypotheses"] if commitment[index] else None,
        })
    return rows


def aggregate_retry_tolerant(rows: list[dict]) -> dict:
    """The same quantities as the frozen read, with recovery permitted."""
    if not rows:
        raise AssertionError("retry-tolerant evaluation received no rows")
    total = len(rows)
    committed = [row for row in rows if row["committed"]]
    premature = [row for row in committed if not row["licensed_at_commitment"]]
    return {
        "success_rate": sum(row["success"] for row in rows) / total,
        "commitment_rate": len(committed) / total,
        "premature_commitment_rate": len(premature) / len(committed) if committed else None,
        "mean_probes": sum(row["probes"] for row in rows) / total,
        "mean_live_hypotheses_at_commitment": (
            sum(row["live_hypotheses_at_commitment"] for row in committed) / len(committed)
            if committed else None
        ),
        "mean_retries": sum(row["retries"] for row in rows) / total,
        "episodes_needing_a_retry": sum(row["retries"] > 0 for row in rows) / total,
        "mean_recoveries": sum(row["recovered"] for row in rows) / total,
        "exhausted_rate": sum(row["exhausted"] for row in rows) / total,
    }


@torch.no_grad()
def _evaluate_sets(model, sets: dict, device) -> dict:
    """The frozen matched-set evaluation, plus this stage's own extended read."""
    result = {}
    for name, (params, seed, count, rendering, comparison) in sets.items():
        rows = _execute_batched(model, params, seed, count, rendering, device, instrument=True)
        metrics = _aggregate_rows(rows)
        for metric, value in metrics.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise FloatingPointError(f"non-finite metric {name}.{metric}: {value!r}")
        result[name] = {"comparison": comparison, "metrics": metrics, "extended": _aggregate_extended(rows)}
    return result


def _milestone_sets(config: dict, plan: DaggerPlan, final: bool) -> dict:
    """The full matched matrix at the endpoint; validation only in between."""
    sets = _matched_sets(config)
    if final:
        return sets
    params, seed, _count, rendering, comparison = sets["validation"]
    return {"validation": (params, seed, plan.milestone_evaluation_episodes, rendering, comparison)}


@torch.no_grad()
def recovery_probe(model, config: dict, plan: DaggerPlan, settings: CollectionSettings, device) -> dict:
    """P3 on a fixed held-out block: does a failed action stay failed?

    The frozen evaluator cannot answer this. It ends an episode at its first
    protocol failure, so persistence is structurally unobservable to it. The
    collector can observe it, because it is built to continue from the unchanged
    state a failed attempt leaves behind.

    Nothing collected here is trained on, and the seeds are the evaluator's own
    held-out block, so the measurement is matched across milestones and arms.
    """
    seeds = [plan.recovery_probe_seed + offset for offset in range(plan.recovery_probe_episodes)]
    _examples, _traces, counters = collect_tranche(
        model, config["world"], seeds, config["world"]["rendering"], settings, device,
    )
    failures = counters.malformed_attempts + counters.invalid_attempts
    attempts = failures + counters.accepted_attempts
    return {
        "episodes": len(seeds),
        "attempts": attempts,
        "failed_attempts": failures,
        "attempt_failure_rate": failures / attempts if attempts else None,
        "failures_recovered": counters.failures_recovered,
        "episodes_abandoned_repeated_failure": counters.episodes_abandoned_repeated_failure,
        # Of the episodes that failed at least once, the share that went on to
        # play a legal action from the unchanged state.
        "recovery_rate": (
            counters.failures_recovered / (counters.failures_recovered + counters.failures_unrecovered)
            if (counters.failures_recovered + counters.failures_unrecovered) else None
        ),
        "note": "measurement only; these episodes are never trained on",
    }


def _training_arguments(config: dict, plan: DaggerPlan, workspace: Path, seed: int):
    from transformers import TrainingArguments

    training = config["training"]
    return TrainingArguments(
        output_dir=str(workspace / "trainer"),
        per_device_train_batch_size=plan.per_device_sequences,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        # Constant with warmup, not cosine: a fresh Trainer per round would
        # restart a cosine schedule four times and the sawtooth would be
        # indistinguishable from the effect under test.
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_ratio=training["warmup_fraction"],
        max_steps=plan.updates_per_round,
        fp16=torch.cuda.is_available(),
        bf16=False,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        average_tokens_across_devices=True,
    )


def _train_round(model, config: dict, plan: DaggerPlan, workspace: Path, buffer: list, seed: int):
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=_training_arguments(config, plan, workspace, seed),
        train_dataset=SequenceDataset(buffer),
        data_collator=lambda sequences: collate(sequences, plan.context_length),
        processing_class=load_tokenizer(),
    )
    trainer.train()
    if trainer.state.global_step != plan.updates_per_round:
        raise AssertionError(
            f"round ended at update {trainer.state.global_step}, expected {plan.updates_per_round}"
        )
    return trainer


def _verify_source(source_run: Path, config: dict) -> tuple[dict, Path]:
    """Accept a warm-start checkpoint only under its exact dense identity."""
    from .rlvr import _verify_dense_source

    return _verify_dense_source(source_run, config)


def _starting_model(config: dict, plan: DaggerPlan, workspace: Path, source_run: Path | None, state):
    """Materialize the arm's exact starting weights and record what they are."""
    from transformers import AutoModelForCausalLM

    from .transfer import _initial_model_artifact

    if plan.arm == "learner_conditioned_from_dense":
        if source_run is None:
            raise ValueError("the dense warm-start arm requires an attached dense source run")
        source_report, artifact = _verify_source(source_run, config)
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
            _model, artifact, serialization = _initial_model_artifact(workspace, config)
            atomic_json(serialization_path, serialization)
        state.wait_for_everyone()
        model = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True)
        state_sha = _state_dict_sha256(model.state_dict())
        provenance = {
            "policy": "random_from_local_gpt_neox_config",
            "budget_zero_serialization": json.loads(serialization_path.read_text()),
        }
    assert_model_contract(model)
    return model, {**provenance, "root_seed": plan.root_seed,
                   "parameter_count": EXPECTED_PARAMETER_COUNT, "model_state_sha256": state_sha}


def _save_milestone(trainer_or_model, workspace: Path, updates: int, main_process: bool) -> Path:
    """One standard HF artifact per declared budget point."""
    artifact = workspace / f"budget-{updates}" / "model"
    if hasattr(trainer_or_model, "save_model"):
        trainer_or_model.save_model(str(artifact))
    elif main_process:
        trainer_or_model.save_pretrained(artifact, safe_serialization=True)
    if main_process:
        load_tokenizer().save_pretrained(artifact)
    return artifact


def _sum_counters(counters, accelerator) -> dict:
    """Sum every collection axis across ranks without a project all-reduce."""
    fields = sorted(counters.report())
    local = torch.tensor([getattr(counters, name) for name in fields],
                         dtype=torch.float64, device=accelerator.device)
    total = accelerator.reduce(local, reduction="sum").detach().cpu().tolist()
    return {name: int(value) for name, value in zip(fields, total)}


def assert_report_contract(report: dict, config: dict, plan: DaggerPlan) -> None:
    """Exact operational contract. It sets no scientific threshold."""
    if report.get("contract") != LEARNER_CONDITIONED_CONTRACT:
        raise AssertionError("learner-conditioned contract mismatch")
    if report.get("experiment_config_sha256") != config["_meta"]["hash"]:
        raise AssertionError("learner-conditioned configuration hash mismatch")
    if report.get("plan") != plan.report():
        raise AssertionError("learner-conditioned plan mismatch")
    algorithm = report.get("algorithm", {})
    if algorithm.get("objective") != "supervised_cross_entropy_on_teacher_corrections":
        raise AssertionError("learner-conditioned objective identity mismatch")
    if algorithm.get("policy_gradient") is not False:
        raise AssertionError("this stage must not report itself as policy-gradient learning")
    if algorithm.get("state_distribution") != "learner_conditioned":
        raise AssertionError("state distribution identity mismatch")
    budget = report.get("budget_accounting", {})
    if budget.get("optimizer_updates") != plan.max_steps:
        raise AssertionError("the declared update budget was not completed")
    if budget.get("collection_episodes") != plan.collection_episode_budget:
        raise AssertionError(
            f"collected {budget.get('collection_episodes')} episodes against a declared "
            f"{plan.collection_episode_budget}"
        )
    for axis in ("world_transitions", "generated_action_tokens", "supervised_correction_tokens",
                 "wall_clock_seconds"):
        value = budget.get(axis)
        if not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
            raise AssertionError(f"budget axis {axis} was not measured")
    if "states_refused_unlicensed_commit" not in budget:
        raise AssertionError(
            "the count of refused truth-derived commit targets is missing; without it the report "
            "cannot show the guard ran"
        )
    milestones = report.get("milestones", [])
    expected = [0, *plan.milestones_updates]
    if [point.get("budget_updates") for point in milestones] != expected:
        raise AssertionError(f"milestone grid mismatch: expected {expected}")
    for point in milestones:
        if point.get("serialization", {}).get("exact") is not True:
            raise AssertionError(f"milestone {point.get('budget_updates')} artifact is not exact")
        final = point["budget_updates"] == plan.max_steps
        wanted = {"validation", "structural", "rendering_b", "reversible_control"} if final else {"validation"}
        if set(point.get("evaluation", {})) != wanted:
            raise AssertionError(f"milestone {point['budget_updates']} evaluation set mismatch")
        for name, item in point["evaluation"].items():
            if set(item.get("metrics", {})) != set(EVALUATION_METRIC_NAMES):
                raise AssertionError(f"milestone {point['budget_updates']} {name} metric fields mismatch")
            if set(item.get("extended", {})) != set(EXTENDED_METRIC_NAMES):
                raise AssertionError(f"milestone {point['budget_updates']} {name} extended fields mismatch")
        if point.get("recovery_probe", {}).get("episodes") != plan.recovery_probe_episodes:
            raise AssertionError(f"milestone {point['budget_updates']} recovery probe did not run")
    if len(report.get("rounds", [])) != plan.rounds:
        raise AssertionError("one collection record per round is required")


def load_config(path: Path) -> dict:
    """Accept the runner's resolved JSON or a checked-in TOML unchanged."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["_meta"] = {"hash": hashlib.sha256(canonical).hexdigest(), "path": str(path.resolve())}
    return config


def resolve_source_run(config: dict, plan: DaggerPlan, source_run: Path | None) -> Path | None:
    """Locate an attached upstream Kaggle output by its exact identity."""
    if plan.arm != "learner_conditioned_from_dense":
        return None
    if source_run is not None:
        return source_run
    from .transfer import locate_dense_source

    return locate_dense_source([Path("/kaggle/input")], config["source"])


def run(config_path: Path, output_dir: Path, source_run: Path | None = None) -> Path:
    from accelerate import PartialState

    config = load_config(config_path)
    state = PartialState()
    plan = dagger_plan(config, world_size=state.num_processes)
    if state.num_processes != 2:
        raise AssertionError(f"this stage requires exactly two ranks, got {state.num_processes}")
    settings = CollectionSettings.from_config(config)
    source_run = resolve_source_run(config, plan, source_run)
    workspace = output_dir / plan.arm
    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "analysis" / "dagger_plan.json", plan.report())
    state.wait_for_everyone()

    model, initialization = _starting_model(config, plan, workspace, source_run, state)
    model.to(state.device)
    started = time.monotonic()

    # Budget zero: the starting policy, saved and scored under the same
    # instrument as every later point. It is the only baseline this run carries.
    if state.is_main_process:
        _save_milestone(model, workspace, 0, True)
    state.wait_for_everyone()

    buffer: list = []
    rounds: list[dict] = []
    trainer = None
    for index in range(plan.rounds):
        seeds = round_seeds(plan, index)
        model.eval()
        examples, _traces, counters = collect_tranche(
            model, config["world"], seeds[state.process_index::plan.world_size],
            config["world"]["rendering"], settings, state.device,
        )
        assert_collection_contract(examples, counters)
        from accelerate.utils import gather_object

        # Concatenated in rank order, so every rank builds the identical buffer
        # without needing an ordering key.
        buffer = buffer + list(gather_object(examples))
        model.train()
        trainer = _train_round(model, config, plan, workspace, buffer, plan.root_seed + index)
        completed = plan.updates_per_round * (index + 1)
        _save_milestone(trainer, workspace, completed, state.is_main_process)
        rounds.append({
            "round": index,
            "budget_updates": completed,
            "world_seed_range": [seeds[0], seeds[-1]],
            "collection": _sum_counters(counters, trainer.accelerator),
            "buffer_examples": len(buffer),
        })
        model = trainer.model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.monotonic() - started

    report_path = output_dir / "learner_conditioned_report.json"
    if not state.is_main_process:
        # Training is finished; rank zero alone runs the long evaluations, with
        # no pending collective that could time out.
        return report_path

    from .rlvr import _diagnostic_shard, _teacher_forced_nll
    from .transfer import _checkpoint_state_roundtrip

    device = state.device
    diagnostic_shard = _diagnostic_shard(config, output_dir)
    milestones = []
    for updates in (0, *plan.milestones_updates):
        artifact = workspace / f"budget-{updates}" / "model"
        point_model, serialization = _checkpoint_state_roundtrip(artifact)
        point_model.to(device).eval()
        milestones.append({
            "budget_updates": updates,
            "collection_episodes": (updates // plan.updates_per_round) * plan.episodes_per_round,
            "artifact": str(artifact.resolve()),
            "serialization": serialization,
            "evaluation": _evaluate_sets(point_model, _milestone_sets(config, plan, updates == plan.max_steps), device),
            "teacher_forced_action_nll": _teacher_forced_nll(point_model, config, diagnostic_shard, device),
            "recovery_probe": recovery_probe(point_model, config, plan, settings, device),
        })
        point_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collection_totals = {
        key: sum(record["collection"][key] for record in rounds)
        for key in rounds[0]["collection"]
    }
    report = {
        "contract": LEARNER_CONDITIONED_CONTRACT,
        "experiment_config_sha256": config["_meta"]["hash"],
        "algorithm": {
            "family": "DAgger",
            "paper": "https://proceedings.mlr.press/v15/ross11a/ross11a.pdf",
            "objective": "supervised_cross_entropy_on_teacher_corrections",
            "policy_gradient": False,
            "state_distribution": "learner_conditioned",
            "expert": "privileged_dense_teacher",
            "collection_decoding": "greedy, identical to the frozen evaluator",
            "teacher_in_rollout_context": False,
            "unlicensed_commit_targets": "refused; a commit the evidence does not license is truth-derived",
            "trainer": "transformers.Trainer",
        },
        "plan": plan.report(),
        "initialization": initialization,
        "world": {**config["world"], "train_seed_base": plan.train_seed_base},
        "rounds": rounds,
        "budget_accounting": {
            **collection_totals,
            "optimizer_updates": plan.max_steps,
            "collection_episodes": collection_totals["episodes"],
            "wall_clock_seconds": elapsed,
            "note": (
                "episodes and updates are matched by declaration; supervised tokens are data "
                "dependent and are reported, never used to change an update"
            ),
        },
        "milestones": milestones,
        "control_status": (
            "no matched teacher-conditioned continuation arm was run; budget-0 matches the "
            "instrument but not the training, so a change here is not attributed to the state "
            "distribution alone"
        ),
        "scientific_acceptance_policy": "measurement only; no capability threshold gates this run",
    }
    assert_report_contract(report, config, plan)
    atomic_json(report_path, report)
    atomic_json(output_dir / "analysis" / "result-report.json", report)
    atomic_json(output_dir / "evaluation" / "metrics.json", {
        "contract": LEARNER_CONDITIONED_CONTRACT,
        "milestones": [{"budget_updates": point["budget_updates"], "evaluation": point["evaluation"],
                        "recovery_probe": point["recovery_probe"]} for point in milestones],
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
