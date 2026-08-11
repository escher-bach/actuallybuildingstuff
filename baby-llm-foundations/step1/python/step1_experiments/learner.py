"""Narrow learner-conditioned collection adapter; optimization stays in Trainer."""
from __future__ import annotations

import torch

from .data import ACTION, BOS, END_TURN, EOS, OBS, Sequence, encode_bytes


@torch.no_grad()
def collect_single_action(model, batch, params: dict, rendering: str) -> tuple[list[Sequence], list[dict]]:
    """Collect one policy action per live world episode and teacher corrections.

    The Rust boundary owns parsing and independent state transitions.  This
    adapter only renders the model context and turns its returned correction
    spans into ordinary `labels == -100`-compatible sequences.
    """
    device = next(model.parameters()).device
    attempts: list[str] = []
    for index in batch.live_episode_indices():
        prefix = [BOS, OBS, *encode_bytes(batch.observations(rendering)[index]), ACTION]
        generated = model.generate(
            input_ids=torch.tensor([prefix], device=device),
            max_new_tokens=96,
            do_sample=True,
            eos_token_id=END_TURN,
            pad_token_id=model.config.pad_token_id,
        )[0, len(prefix):].tolist()
        payload = generated[:generated.index(END_TURN)] if END_TURN in generated else generated
        try:
            attempts.append(bytes(token for token in payload if token < 256).decode("utf-8", errors="strict"))
        except UnicodeDecodeError:
            attempts.append("<malformed-utf8>")

    records = batch.step_attempts(attempts, rendering)
    sequences: list[Sequence] = []
    for record in records:
        ids = [BOS, OBS, *encode_bytes(record["observation_before"]), ACTION, *encode_bytes(record["learner_text"]), END_TURN,
               OBS, *encode_bytes(record["observation_after"]), ACTION]
        mask = [0] * len(ids)
        corrections = sorted(record["preferred_corrections"])
        if record["terminal_outcome"] is None and corrections:
            from world_py import render_action
            correction = render_action(corrections[0], params["n_probe"], params["n_hyp"], rendering)
            correction_ids = [*encode_bytes(correction), END_TURN]
            ids.extend(correction_ids)
            mask.extend([1] * len(correction_ids))
        ids.append(EOS); mask.append(0)
        sequences.append(Sequence(ids, mask, [0] * len(ids)))
    return sequences, records
