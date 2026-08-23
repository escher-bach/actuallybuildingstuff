"""Batched conversion from the Rust public-token boundary to torch tensors."""

from __future__ import annotations

from typing import Any

import torch

import step2_world_py


MODEL_FIELDS = (
    "role_ids",
    "key_ids",
    "position_ids",
    "payloads",
    "attention_mask",
    "action_targets",
    "action_target_mask",
    "outcome_targets",
    "outcome_target_mask",
)


def world_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "d_min": int(config["d_min"]),
        "d_max": int(config["d_max"]),
        "gain_min": float(config["gain_min"]),
        "gain_max": float(config["gain_max"]),
        "action_limit": float(config["action_limit"]),
        "calibration_pulse": float(config["calibration_pulse"]),
        "max_control_steps": int(config["max_control_steps"]),
    }


def tensorize(raw: dict[str, Any], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    integer_fields = {"role_ids", "key_ids", "position_ids", "attention_mask"}
    tensors: dict[str, torch.Tensor] = {}
    for field in MODEL_FIELDS:
        dtype = torch.long if field in integer_fields else torch.float32
        tensors[field] = torch.tensor(raw[field], dtype=dtype, device=device)
    return tensors


def generate_torch_batch(
    *,
    seed: int,
    start_index: int,
    batch_size: int,
    max_tokens: int,
    world: dict[str, Any],
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw = step2_world_py.generate_training_batch(
        seed=seed,
        start_index=start_index,
        batch_size=batch_size,
        max_tokens=max_tokens,
        **world_kwargs(world),
    )
    metadata = {key: value for key, value in raw.items() if key not in MODEL_FIELDS}
    return tensorize(raw, device), metadata


def tensorize_rollout(raw: dict[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    return tensorize(raw, device)
