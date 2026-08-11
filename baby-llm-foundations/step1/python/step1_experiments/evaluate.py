"""World-executed, parser-mediated evaluation; teacher-forced NLL is secondary."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import ACTION, BOS, BinaryShard, END_TURN, OBS, collate, encode_bytes


def _family(params: dict):
    from world_py import FamilyParams
    return FamilyParams(n_hyp=params["n_hyp"], n_probe=params["n_probe"], n_evidence=params["n_evidence"], cost_lo=params["cost_lo"], cost_hi=params["cost_hi"], budget_slack=params["budget_slack"], min_depth=params["min_depth"], step_slack=params["step_slack"], variant=params["variant"])


@torch.no_grad()
def _decode_action(model, prefix: list[int], device: torch.device, limit: int = 96) -> tuple[str | None, float]:
    if len(prefix) >= model.config.max_position_embeddings:
        return None, 0.0
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids=ids,
        max_new_tokens=min(limit, model.config.max_position_embeddings - len(prefix)),
        do_sample=False,
        eos_token_id=END_TURN,
        pad_token_id=model.config.pad_token_id,
        use_cache=True,
    )[0, len(prefix):].tolist()
    if END_TURN not in generated:
        return None, 0.0
    payload = generated[:generated.index(END_TURN)]
    if any(token >= 256 for token in payload):
        return None, 0.0
    try:
        return bytes(payload).decode("utf-8", errors="strict"), 0.0
    except UnicodeDecodeError:
        return None, 0.0


@torch.no_grad()
def _teacher_cost(params: dict, seed: int) -> int:
    from world_py import Batch
    batch = Batch(_family(params), seed=seed, n_episodes=1)
    while not batch.done()[0]:
        action = min(batch.privileged_teacher_targets()[0]["preferred_actions"])
        batch.step([action])
    return int(batch.privileged_outcomes()[0][2])


@torch.no_grad()
def _execute(model, params: dict, seed: int, rendering: str, device: torch.device) -> dict:
    from world_py import Batch, parse_action
    batch, prefix = Batch(_family(params), seed=seed, n_episodes=1), [BOS]
    malformed, confidences, steps = 0, [], 0
    while not batch.done()[0] and steps < params["step_slack"] + params["n_probe"] + params["n_hyp"] + 4:
        prefix += [OBS] + encode_bytes(batch.observations(rendering)[0]) + [ACTION]
        text, confidence = _decode_action(model, prefix, device)
        if text is None:
            malformed += 1; break
        try: action = parse_action(text, params["n_probe"], params["n_hyp"], rendering)
        except ValueError:
            malformed += 1; break
        prefix += encode_bytes(text) + [END_TURN]; confidences.append(confidence); batch.step([action]); steps += 1
    terminated, correct, spent, *_ = batch.privileged_outcomes()[0]
    return {"success": bool(terminated and correct), "spent": int(spent), "malformed": malformed, "steps": steps, "confidence": sum(confidences) / len(confidences) if confidences else 0.0, "teacher_spent": _teacher_cost(params, seed)}


@torch.no_grad()
def evaluate(resolved_config: Path, run_dir: Path) -> dict:
    config = json.loads(resolved_config.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM
    artifact = run_dir / "model"
    model = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True).to(device).eval()
    output = {"checkpoint": str(artifact), "sets": {}}
    sets = {"validation": (config["world"], config["run"]["root_seed"] + 1_000_000, config["world"]["validation_episodes"], config["world"]["rendering"]),
            "structural": ({**config["world"], "n_hyp": config["world"]["n_hyp"] + 1}, config["run"]["root_seed"] + 2_000_000, config["world"]["structural_episodes"], config["world"]["rendering"]),
            "rendering_b": ({**config["world"], "rendering": "b"}, config["run"]["root_seed"] + 3_000_000, config["world"]["transfer_episodes"], "b"),
            "reversible_control": ({**config["world"], "variant": "reversible"}, config["run"]["root_seed"] + 4_000_000, config["world"]["validation_episodes"], config["world"]["rendering"])}
    for name, (params, seed, count, rendering) in sets.items():
        rows = [_execute(model, params, seed + index, rendering, device) for index in range(count)]
        successes = sum(row["success"] for row in rows)
        output["sets"][name] = {"success_rate": successes / count, "malformed_action_rate": sum(row["malformed"] > 0 for row in rows) / count,
                                "mean_probe_cost": sum(row["spent"] for row in rows) / count, "regret_to_teacher": sum(row["spent"] - row["teacher_spent"] for row in rows) / count,
                                "mean_action_confidence": sum(row["confidence"] for row in rows) / count, "mean_steps": sum(row["steps"] for row in rows) / count}
    # Retain teacher-forced NLL as a diagnostic, never as the success metric.
    dataset = BinaryShard(run_dir / "datasets" / "validation.bin"); total, count = 0.0, 0
    for start in range(0, len(dataset), 4):
        batch = collate([dataset[i] for i in range(start, min(start + 4, len(dataset)))], config["world"]["context_length"])
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model(**batch).loss
        labels = int((batch["labels"][:, 1:] != -100).sum())
        total += float(loss) * labels; count += labels
    output["teacher_forced_action_nll"] = total / count
    atomic_json(run_dir / "evaluation" / "metrics.json", output); return output
