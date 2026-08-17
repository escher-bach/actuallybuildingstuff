"""World-executed, parser-mediated evaluation; teacher-forced NLL is secondary."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import ACTION, BOS, BinaryShard, END_TURN, OBS, collate, encode_bytes


# This keeps generated KV caches comfortably bounded on a 16-GB T4 while
# reducing autoregressive generation calls by up to 32x versus one episode per
# call.  Worlds remain independently stepped through the public Batch API so a
# malformed/invalid action terminates only its own evaluation episode.
GENERATION_BATCH_SIZE = 32
EVALUATION_SET_NAMES = ("validation", "structural", "rendering_b", "reversible_control")
EVALUATION_METRIC_NAMES = (
    "success_rate", "failure_rate", "malformed_action_rate", "invalid_action_rate",
    "mean_spent", "mean_success_excess_cost", "mean_steps",
)


def _family(params: dict):
    from world_py import FamilyParams
    return FamilyParams(n_hyp=params["n_hyp"], n_probe=params["n_probe"], n_evidence=params["n_evidence"], cost_lo=params["cost_lo"], cost_hi=params["cost_hi"], budget_slack=params["budget_slack"], min_depth=params["min_depth"], step_slack=params["step_slack"], variant=params["variant"])


def _decode_generated_action(generated: list[int]) -> str | None:
    if END_TURN not in generated:
        return None
    payload = generated[:generated.index(END_TURN)]
    if any(token >= 256 for token in payload):
        return None
    try:
        return bytes(payload).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


@torch.no_grad()
def _decode_actions_batched(model, prefixes: list[list[int]], device: torch.device, batch_size: int = GENERATION_BATCH_SIZE, temperature: float = 0.0) -> list[str | None]:
    """Use standard batched ``generate`` while retaining row-local decoding.

    ``temperature = 0.0`` is the frozen greedy rule every retained evaluation
    uses.  A positive temperature samples instead, which exists only for the
    decoding diagnostic: it must never silently change a reported metric.
    """
    if batch_size < 1:
        raise ValueError("generation batch_size must be positive")
    decoded: list[str | None] = [None] * len(prefixes)
    max_positions = model.config.max_position_embeddings
    # Rows that have no context room are malformed independently.  Bucket the
    # remaining rows by their own generation allowance: a near-context row
    # must not reduce the continuation budget of an otherwise healthy row.
    buckets: dict[int, list[tuple[int, list[int]]]] = {}
    for index, prefix in enumerate(prefixes):
        if len(prefix) < max_positions:
            allowance = min(96, max_positions - len(prefix))
            buckets.setdefault(allowance, []).append((index, prefix))
    for limit, eligible in buckets.items():
        for start in range(0, len(eligible), batch_size):
            group = eligible[start:start + batch_size]
            indices = [index for index, _ in group]
            values = [prefix for _, prefix in group]
            width = max(map(len, values))
            input_ids = torch.full((len(values), width), model.config.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(values), width), dtype=torch.bool, device=device)
            for row, prefix in enumerate(values):
                input_ids[row, width - len(prefix):] = torch.tensor(prefix, dtype=torch.long, device=device)
                attention_mask[row, width - len(prefix):] = True
            sampling = {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature > 0 else {"do_sample": False}
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=limit,
                eos_token_id=END_TURN,
                pad_token_id=model.config.pad_token_id,
                use_cache=True,
                **sampling,
            )
            for index, row in zip(indices, generated):
                decoded[index] = _decode_generated_action(row[width:].tolist())
    return decoded


@torch.no_grad()
def _teacher_cost(params: dict, seed: int) -> int:
    from world_py import Batch
    batch = Batch(_family(params), seed=seed, n_episodes=1)
    while not batch.done()[0]:
        action = min(batch.privileged_teacher_targets()[0]["preferred_actions"])
        batch.step([action])
    return int(batch.privileged_outcomes()[0][2])


def _aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        raise AssertionError("evaluation set is empty")
    successes = [row for row in rows if row["success"]]
    return {
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "failure_rate": sum(not row["success"] for row in rows) / len(rows),
        "malformed_action_rate": sum(row["malformed"] > 0 for row in rows) / len(rows),
        "invalid_action_rate": sum(row["invalid"] > 0 for row in rows) / len(rows),
        "mean_spent": sum(row["spent"] for row in rows) / len(rows),
        # Failed/malformed traces can have arbitrary partial spend.  They are
        # intentionally excluded from this teacher comparison.
        "mean_success_excess_cost": (
            sum(row["spent"] - row["teacher_spent"] for row in successes) / len(successes)
            if successes else None
        ),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
    }


def _attempt_status(record: dict) -> str:
    """Classify the public ``Batch.step_attempts`` record without reparsing it."""
    if record["parsed_action"] is None:
        return "malformed"
    return "accepted" if record["accepted"] else "invalid"


def _matched_sets(config: dict) -> dict[str, tuple[dict, int, int, str, dict]]:
    """Declare which comparisons are matched rather than silently changing seeds."""
    world, seed = config["world"], config["run"]["root_seed"]
    validation_seed = seed + 1_000_000
    return {
        "validation": (world, validation_seed, world["validation_episodes"], world["rendering"], {
            "label": "held_out_in_distribution", "seed_policy": "held_out_seed",
        }),
        "structural": ({**world, "n_hyp": world["n_hyp"] + 1}, seed + 2_000_000, world["structural_episodes"], world["rendering"], {
            "label": "structural_generalization", "seed_policy": "held_out_structural_seed_not_a_paired_contrast",
        }),
        "rendering_b": ({**world, "rendering": "b"}, validation_seed, world["transfer_episodes"], "b", {
            "label": "zero_shot_rendering_transfer", "seed_policy": "matched_to_validation_seed; no_rendering_b_calibration",
        }),
        "reversible_control": ({**world, "variant": "reversible"}, validation_seed, world["validation_episodes"], world["rendering"], {
            "label": "matched_seed_variant_control", "seed_policy": "matched_to_validation_seed",
        }),
    }


@torch.no_grad()
def _execute_batched(model, params: dict, seed: int, count: int, rendering: str, device: torch.device, temperature: float = 0.0, instrument: bool = False) -> list[dict]:
    """Execute independent worlds with batched generation and row-local stops.

    ``instrument`` adds privileged *row* fields — how many probes the episode
    bought, and what the evidence licensed at the moment it committed.  It adds
    no metric and changes no existing one: `_aggregate_rows` ignores the extra
    keys, so every retained comparison is unaffected.  It is off by default
    because it costs two extra world calls per step, and only the
    learner-conditioned stage reads it.
    """
    from world_py import Batch

    worlds = [Batch(_family(params), seed=seed + index, n_episodes=1) for index in range(count)]
    prefixes = [[BOS] for _ in worlds]
    malformed, invalid, steps = [0] * count, [0] * count, [0] * count
    probes = [0] * count
    commitment: list[dict | None] = [None] * count
    active = set(range(count))
    limit = params["step_slack"] + params["n_probe"] + params["n_hyp"] + 4
    while active:
        live = sorted(active)
        decode_prefixes = []
        for index in live:
            prefix = prefixes[index]
            prefix += [OBS] + encode_bytes(worlds[index].observations(rendering)[0]) + [ACTION]
            decode_prefixes.append(prefix)
        texts = _decode_actions_batched(model, decode_prefixes, device, temperature=temperature)
        for index, text in zip(live, texts):
            if text is None:
                malformed[index] += 1
                active.remove(index)
                continue
            # Read the privileged state *before* the action is applied: what
            # the evidence licensed at the moment of choosing is the quantity,
            # not what it licenses afterwards.
            licensed = worlds[index].privileged_teacher_targets()[0]["licenses_commitment"] if instrument else None
            live_hypotheses = worlds[index].privileged_consistent_counts()[0] if instrument else None
            record = worlds[index].step_attempts([text], rendering)[0]
            status = _attempt_status(record)
            if status == "malformed":
                malformed[index] += 1
                active.remove(index)
                continue
            if status == "invalid":
                invalid[index] += 1
                active.remove(index)
                continue
            if instrument:
                if record["parsed_action"] < params["n_probe"]:
                    probes[index] += 1
                else:
                    commitment[index] = {"licensed": bool(licensed), "live_hypotheses": int(live_hypotheses)}
            prefixes[index] += encode_bytes(text) + [END_TURN]
            steps[index] += 1
            if worlds[index].done()[0] or steps[index] >= limit:
                active.remove(index)
    rows = []
    for index, world in enumerate(worlds):
        terminated, correct, spent, *_ = world.privileged_outcomes()[0]
        row = {
            "success": bool(terminated and correct and not malformed[index] and not invalid[index]),
            "spent": int(spent),
            "malformed": malformed[index],
            "invalid": invalid[index],
            "steps": steps[index],
            "teacher_spent": _teacher_cost(params, seed + index),
        }
        if instrument:
            row.update({
                "probes": probes[index],
                "committed": commitment[index] is not None,
                "licensed_at_commitment": commitment[index]["licensed"] if commitment[index] else None,
                "live_hypotheses_at_commitment": commitment[index]["live_hypotheses"] if commitment[index] else None,
            })
        rows.append(row)
    return rows


def assert_evaluation_contract(metrics: dict) -> None:
    """Require complete, finite scientific measurements without target thresholds."""
    if set(metrics.get("sets", {})) != set(EVALUATION_SET_NAMES):
        raise AssertionError(f"evaluation set mismatch: {sorted(metrics.get('sets', {}))}")
    for name, item in metrics["sets"].items():
        values = item.get("metrics")
        if set(values or {}) != set(EVALUATION_METRIC_NAMES):
            raise AssertionError(f"evaluation metric fields mismatch for {name}: {sorted(values or {})}")
        for key, value in values.items():
            if key == "mean_success_excess_cost" and value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise FloatingPointError(f"non-finite evaluation metric for {name}.{key}: {value!r}")
        if not item.get("comparison", {}).get("label") or not item["comparison"].get("seed_policy"):
            raise AssertionError(f"evaluation comparison provenance missing for {name}")
    if not isinstance(metrics.get("teacher_forced_action_nll"), (int, float)) or not math.isfinite(metrics["teacher_forced_action_nll"]):
        raise FloatingPointError("teacher_forced_action_nll must be finite")


@torch.no_grad()
def evaluate(resolved_config: Path, run_dir: Path) -> dict:
    config = json.loads(resolved_config.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM
    artifact = run_dir / "production" / "model"
    model = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True).to(device).eval()
    output = {
        "checkpoint": str(artifact),
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "sets": {},
    }
    for name, (params, seed, count, rendering, comparison) in _matched_sets(config).items():
        rows = _execute_batched(model, params, seed, count, rendering, device)
        output["sets"][name] = {"comparison": comparison, "metrics": _aggregate_rows(rows)}
    # Retain teacher-forced NLL as a diagnostic, never as the success metric.
    dataset = BinaryShard(run_dir / "datasets" / "validation.bin")
    try:
        total, count = 0.0, 0
        for start in range(0, len(dataset), 4):
            batch = collate([dataset[i] for i in range(start, min(start + 4, len(dataset)))], config["world"]["context_length"])
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            labels = int((batch["labels"][:, 1:] != -100).sum())
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("teacher-forced evaluation loss is absent or non-finite")
            total += float(loss) * labels
            count += labels
    finally:
        dataset.close()
    if count <= 0:
        raise AssertionError("teacher-forced evaluation received no supervised action tokens")
    output["teacher_forced_action_nll"] = total / count
    assert_evaluation_contract(output)
    atomic_json(run_dir / "evaluation" / "metrics.json", output)
    return output
