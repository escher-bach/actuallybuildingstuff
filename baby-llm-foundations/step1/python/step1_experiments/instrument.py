"""Fixed-model preflight: variable-gap retrieval and tiny-world overfit."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .artifacts import atomic_json
from .data import BinaryShard, collate
from .model import Step1Transformer, masked_next_token_loss
from .train import _optimizer


def retrieval_batch(seed: int, batch_size: int = 16, length: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Each answer requires content lookup over a randomized source-query gap."""
    rng = random.Random(seed); rows, masks = [], []
    for _ in range(batch_size):
        row, mask = [256] + [rng.randrange(1, 220) for _ in range(length - 1)], [0] * length
        keys = rng.sample(range(1, 96), 4); values = rng.sample(range(128, 220), 4)
        sources = rng.sample(range(2, 72, 3), 4)
        for position, key, value in zip(sources, keys, values): row[position:position + 2] = [key, value]
        # Four-token slots prevent query triples from overwriting one another.
        # The marker is before (not at) each query, so the supervised target is
        # always the independently sampled value rather than the marker token.
        for position, (key, value) in enumerate(zip(keys * 2, values * 2), start=96):
            query = 96 + (position - 96) * 4
            row[query:query + 3] = [255, key, value]; mask[query + 2] = 1
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.uint8)


def run(resolved_config: Path, run_dir: Path) -> None:
    config = json.loads(resolved_config.read_text()); local, rank = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"])
    torch.cuda.set_device(local); device = torch.device("cuda", local); dist.init_process_group("nccl")
    torch.manual_seed(config["run"]["root_seed"] + rank)
    model = Step1Transformer(**{k: config["model"][k] for k in ("layers", "width", "heads", "mlp_width", "rope_base")}).to(device)
    ddp = DDP(model, device_ids=[local]); optimizer = _optimizer(ddp.module, config["training"]); scaler = torch.amp.GradScaler("cuda")
    tiny_shard = BinaryShard(run_dir / "datasets" / "train.bin")
    tiny = [tiny_shard[index] for index in range(min(32, len(tiny_shard)))]
    input_tokens, turn = 0, 0
    # This is capped by input tokens, not labels. Half the updates are world traces.
    while input_tokens < config["training"]["token_budget"]:
        optimizer.zero_grad(set_to_none=True)
        if turn % 2 == 0:
            tokens, mask = retrieval_batch(config["run"]["root_seed"] + rank * 1_000_000 + turn)
            local_input_count = tokens.numel(); tokens, mask = tokens.to(device), mask.to(device)
        else:
            batch = collate([tiny[(turn * 4 + i) % len(tiny)] for i in range(4)], config["world"]["context_length"])
            local_input_count = int(batch["attention_mask"].sum()); tokens, mask = batch["tokens"].to(device), batch["loss_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16): logits = ddp(tokens); total, labels = masked_next_token_loss(logits, tokens, mask)
        scaler.scale(total).backward(); scaler.unscale_(optimizer)
        global_labels = labels.detach().to(torch.float64); dist.all_reduce(global_labels)
        for parameter in ddp.parameters():
            if parameter.grad is not None: parameter.grad.mul_(dist.get_world_size() / global_labels.item())
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        local_inputs = torch.tensor(local_input_count, device=device, dtype=torch.int64); dist.all_reduce(local_inputs); input_tokens += int(local_inputs)
        turn += 1
    # Held-out randomized bindings: a fixed positional rule cannot pass this.
    tokens, mask = retrieval_batch(config["run"]["root_seed"] + 9_000_000 + rank, 128)
    tokens, mask = tokens.to(device), mask.to(device)
    with torch.no_grad(): predicted = ddp.module(tokens).argmax(-1)[:, :-1]; target, active = tokens[:, 1:], mask[:, 1:].bool(); correct = ((predicted == target) & active).sum().to(torch.float64); total_targets = active.sum().to(torch.float64)
    dist.all_reduce(correct); dist.all_reduce(total_targets)
    world_sum, world_count = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(tiny), 4):
            batch = collate(tiny[start:start + 4], config["world"]["context_length"]); ids, loss = batch["tokens"].to(device), batch["loss_mask"].to(device)
            value, count = masked_next_token_loss(ddp.module(ids), ids, loss); world_sum += value; world_count += count
    dist.all_reduce(world_sum); dist.all_reduce(world_count)
    result = {"input_tokens": input_tokens, "variable_gap_associative_retrieval_accuracy": (correct / total_targets).item(), "tiny_world_action_nll": (world_sum / world_count).item()}
    if rank == 0:
        thresholds = config["instrument"]
        atomic_json(run_dir / "benchmarks" / "instrument.json", {**result, "thresholds": thresholds})
        if result["variable_gap_associative_retrieval_accuracy"] < thresholds["associative_retrieval_accuracy"]: raise RuntimeError("instrument variable-gap associative retrieval failed")
        if result["tiny_world_action_nll"] > thresholds["tiny_world_loss"]: raise RuntimeError("instrument tiny-world overfit failed")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--resolved-config", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); run(args.resolved_config, args.run_dir)
