"""World-executed, parser-mediated evaluation; teacher-forced NLL is secondary."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import ACTION, BOS, END_TURN, OBS, Sequence, collate, encode_bytes, load_generated_dataset
from .model import Step1Transformer, masked_next_token_loss


def _family(params: dict):
    from world_py import FamilyParams
    return FamilyParams(n_hyp=params["n_hyp"], n_probe=params["n_probe"], n_evidence=params["n_evidence"], cost_lo=params["cost_lo"], cost_hi=params["cost_hi"], budget_slack=params["budget_slack"], min_depth=params["min_depth"], step_slack=params["step_slack"], variant=params["variant"])


@torch.no_grad()
def _decode_action(model: Step1Transformer, prefix: list[int], device: torch.device, limit: int = 96) -> tuple[str | None, float]:
    generated: list[int] = []; confidences: list[float] = []
    for _ in range(limit):
        if len(prefix) + len(generated) >= model.context_length: return None, 0.0
        ids = torch.tensor([prefix + generated], dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probabilities = model(ids)[:, -1].float().softmax(-1)
        value, token = probabilities.max(-1); token = int(token.item()); confidences.append(float(value.item()))
        if token == END_TURN:
            try: return bytes(generated).decode("utf-8", errors="strict"), sum(confidences) / len(confidences)
            except UnicodeDecodeError: return None, 0.0
        if token < 256: generated.append(token)
        else: return None, 0.0
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
def _execute(model: Step1Transformer, params: dict, seed: int, rendering: str, device: torch.device) -> dict:
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
    config = json.loads(resolved_config.read_text()); checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    if checkpoint["config_hash"] != config["_meta"]["hash"]: raise RuntimeError("checkpoint configuration hash mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Step1Transformer(**{k: config["model"][k] for k in ("layers", "width", "heads", "mlp_width", "rope_base")}).to(device)
    model.context_length = config["world"]["context_length"]; model.load_state_dict(checkpoint["model"]); model.eval()
    output = {"checkpoint": "latest.pt", "checkpoint_tokens": checkpoint["tokens"], "sets": {}}
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
    dataset = load_generated_dataset(run_dir / "datasets" / "validation.pt"); total, count = 0.0, 0
    for start in range(0, len(dataset), 4):
        batch = collate([dataset[i] for i in range(start, min(start + 4, len(dataset)))], config["world"]["context_length"]); ids, mask = batch["tokens"].to(device), batch["loss_mask"].to(device)
        loss, labels = masked_next_token_loss(model(ids), ids, mask); total += float(loss); count += int(labels)
    output["teacher_forced_action_nll"] = total / count
    atomic_json(run_dir / "evaluation" / "metrics.json", output); return output
