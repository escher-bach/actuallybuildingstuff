"""DDP training child.  Parent runner owns phase transitions and packaging."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .artifacts import atomic_json
from .data import BinaryShard, DistributedSequenceSampler, make_dataloader
from .model import Step1Transformer, masked_next_token_loss


def _rank() -> tuple[int, int, int]:
    local, rank, size = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return local, rank, size


def _atomic_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    with temporary.open("rb") as handle: os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(model: torch.nn.Module, loss: torch.Tensor) -> None:
    if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all(): raise FloatingPointError(f"non-finite gradient: {name}")


def _optimizer(model: torch.nn.Module, config: dict):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "embedding" in name or "norm" in name else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=config["learning_rate"], betas=(0.9, .95), eps=1e-8)


def _lr(config: dict, tokens: int) -> float:
    budget, warm = config["token_budget"], int(config["token_budget"] * config["warmup_fraction"])
    if tokens < warm: return config["learning_rate"] * tokens / max(warm, 1)
    progress = min(1.0, (tokens - warm) / max(budget - warm, 1))
    return config["min_learning_rate"] + .5 * (config["learning_rate"] - config["min_learning_rate"]) * (1 + math.cos(math.pi * progress))


def run(resolved_config: Path, run_dir: Path, preflight_only: bool = False, resume: bool = False) -> None:
    config = json.loads(resolved_config.read_text()); local, rank, world_size = _rank(); device = torch.device("cuda", local)
    random.seed(config["run"]["root_seed"] + rank); torch.manual_seed(config["run"]["root_seed"] + rank); torch.cuda.manual_seed_all(config["run"]["root_seed"] + rank)
    dataset = BinaryShard(run_dir / "datasets" / "train.bin")
    sampler = DistributedSequenceSampler(dataset, config["run"]["root_seed"], rank, world_size)
    model = Step1Transformer(**{k: config["model"][k] for k in ("layers", "width", "heads", "mlp_width", "rope_base")}).to(device)
    model.context_length = config["world"]["context_length"]
    ddp = DDP(model, device_ids=[local], output_device=local)
    optimizer, scaler = _optimizer(ddp.module, config["training"]), torch.amp.GradScaler("cuda")
    micro = config["training"]["microbatch_sequences"]
    if len(sampler) == 0: raise RuntimeError("DDP rank received no sequences")
    global_worker_budget = max(1, min(8, (os.cpu_count() or 2) // 2))
    workers = max(1, global_worker_budget // world_size)
    loader = make_dataloader(dataset, config["world"]["context_length"], micro, workers=workers, sampler=sampler)
    iterator = iter(loader)
    def next_batch(batch_size: int) -> dict[str, torch.Tensor]:
        nonlocal loader, iterator
        if loader.batch_size != batch_size:
            loader = make_dataloader(dataset, config["world"]["context_length"], batch_size, workers=workers, sampler=sampler)
            iterator = iter(loader)
        try: return next(iterator)
        except StopIteration:
            sampler.set_epoch(sampler.epoch + 1)
            iterator = iter(loader)
            return next(iterator)
    def one_step(microbatch: int, commit: bool) -> int:
        batch = {k: v.to(device, non_blocking=True) for k, v in next_batch(microbatch).items()}
        if batch["tokens"].dtype != torch.long or not all(v.shape == batch["tokens"].shape for v in batch.values()): raise AssertionError("invalid batch tensor contract")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ddp(batch["tokens"]); total, count = masked_next_token_loss(logits, batch["tokens"], batch["loss_mask"])
        scaler.scale(total).backward()
        global_count = count.detach().to(torch.float64); dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        scaler.unscale_(optimizer)
        # DDP averaged local sums; convert it to globally normalized token loss.
        for p in ddp.parameters():
            if p.grad is not None: p.grad.mul_(world_size / global_count.item())
        _finite(ddp.module, total)
        if commit:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        input_count = batch["attention_mask"].sum().detach().to(torch.int64)
        dist.all_reduce(input_count, op=dist.ReduceOp.SUM)
        return int(input_count.item())
    # Bounded OOM-only adjustment before real training. The probe includes the
    # full optimizer/checkpoint path, then restores every mutable state so no
    # uncounted 6e-4 update leaks into the official warm-up or token budget.
    baseline = {"model": copy.deepcopy(ddp.module.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "scaler": copy.deepcopy(scaler.state_dict()), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
    while True:
        try:
            optimizer.zero_grad(set_to_none=True); one_step(micro, True); torch.cuda.synchronize(device)
            ddp.module.load_state_dict(baseline["model"]); optimizer.load_state_dict(baseline["optimizer"]); scaler.load_state_dict(baseline["scaler"])
            torch.set_rng_state(baseline["rng"]); torch.cuda.set_rng_state_all(baseline["cuda_rng"]); optimizer.zero_grad(set_to_none=True)
            break
        except torch.OutOfMemoryError:
            ddp.module.load_state_dict(baseline["model"]); optimizer.load_state_dict(baseline["optimizer"]); scaler.load_state_dict(baseline["scaler"])
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); micro //= 2
            if micro < 1: raise RuntimeError("preflight exhausted configured microbatch retries")
    if rank == 0:
        config["training"]["resolved_microbatch_sequences"] = micro; atomic_json(resolved_config, config)
        _atomic_checkpoint(run_dir / "checkpoints" / "preflight.pt", {"model": ddp.module.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config_hash": config["_meta"]["hash"], "tokens": 0})
    dist.barrier()
    if preflight_only: dist.destroy_process_group(); return
    tokens, step, index, start = 0, 0, 0, time.monotonic()
    latest = run_dir / "checkpoints" / "latest.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("config_hash") != config["_meta"]["hash"]:
            raise RuntimeError("resume checkpoint configuration hash mismatch")
        ddp.module.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        tokens, step, index = state["tokens"], state["step"], state["sampler_offset"]
        torch.set_rng_state(state["rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"]); sampler.set_epoch(state["sampler_epoch"])
    metric_path = run_dir / "logs" / "metrics.jsonl"
    last_checkpoint_at, previous_tokens = time.monotonic(), tokens
    while tokens < config["training"]["token_budget"]:
        optimizer.zero_grad(set_to_none=True); update_tokens = update_action_tokens = 0
        while update_tokens < config["training"]["global_tokens_per_update"]:
            batch = {k: v.to(device, non_blocking=True) for k, v in next_batch(micro).items()}; index += int(batch["tokens"].shape[0])
            with torch.autocast("cuda", dtype=torch.float16): logits = ddp(batch["tokens"]); total, count = masked_next_token_loss(logits, batch["tokens"], batch["loss_mask"])
            scaler.scale(total).backward()
            action_count = count.detach().to(torch.float64); dist.all_reduce(action_count, op=dist.ReduceOp.SUM); update_action_tokens += int(action_count.item())
            input_count = batch["attention_mask"].sum().detach().to(torch.int64)
            dist.all_reduce(input_count, op=dist.ReduceOp.SUM); update_tokens += int(input_count.item())
        scaler.unscale_(optimizer)
        for p in ddp.parameters():
            if p.grad is not None: p.grad.mul_(world_size / update_action_tokens)
        _finite(ddp.module, total); torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        for group in optimizer.param_groups: group["lr"] = _lr(config["training"], tokens)
        scaler.step(optimizer); scaler.update(); tokens += update_tokens; step += 1
        if rank == 0:
            with metric_path.open("a") as handle: handle.write(json.dumps({"step": step, "tokens": tokens, "loss_sum": float(total.detach()), "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - start}) + "\n")
            periodic_due = time.monotonic() - last_checkpoint_at >= config["training"]["checkpoint_minutes"] * 60
            target_due = any(previous_tokens < target <= tokens for target in config["training"]["save_tokens"])
            if target_due or periodic_due or tokens >= config["training"]["token_budget"]:
                _atomic_checkpoint(run_dir / "checkpoints" / "latest.pt", {"model": ddp.module.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "tokens": tokens, "step": step, "config_hash": config["_meta"]["hash"], "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "sampler_epoch": sampler.epoch, "sampler_offset": index})
                _atomic_checkpoint(run_dir / "checkpoints" / f"recovery-{step:08d}.pt", {"model": ddp.module.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "tokens": tokens, "step": step, "config_hash": config["_meta"]["hash"], "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "sampler_epoch": sampler.epoch, "sampler_offset": index})
                last_checkpoint_at = time.monotonic()
        previous_tokens = tokens
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--resolved-config", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); run(args.resolved_config, args.run_dir, args.preflight_only, args.resume)
