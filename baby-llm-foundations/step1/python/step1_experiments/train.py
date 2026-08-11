"""DDP training child.  Parent runner owns phase transitions and packaging."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
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
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(model: torch.nn.Module, loss: torch.Tensor) -> None:
    if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all(): raise FloatingPointError(f"non-finite gradient: {name}")


def ddp_global_mean_loss(local_sum: torch.Tensor, global_count: torch.Tensor | int, world_size: int) -> torch.Tensor:
    """Scale a local loss sum so DDP's gradient average is the global token mean.

    This normalization must happen *before* GradScaler multiplies the loss.
    Scaling an unnormalized token sum can overflow FP16 gradients even when the
    true mean loss and unscaled gradients are finite.
    """
    count = torch.as_tensor(global_count, device=local_sum.device, dtype=local_sum.dtype)
    if count.numel() != 1 or not torch.isfinite(count) or count.item() <= 0:
        raise FloatingPointError(f"invalid global supervised-token count: {count}")
    return local_sum * (float(world_size) / count)


def _all_reduce_count(local_count: torch.Tensor, device: torch.device) -> torch.Tensor:
    count = local_count.detach().to(device=device, dtype=torch.float64)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return count


def _backward_global_mean(
    scaler: torch.amp.GradScaler,
    local_sum: torch.Tensor,
    global_count: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    finite = torch.isfinite(local_sum.detach()).to(device=local_sum.device, dtype=torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() != 1:
        raise FloatingPointError("non-finite loss sum on at least one DDP rank")
    normalized = ddp_global_mean_loss(local_sum, global_count, world_size)
    scaler.scale(normalized).backward()
    return normalized.detach()


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
    loader = None
    iterator = None
    sampler_offset = 0
    def reset_data(batch_size: int, epoch: int = 0, skip_sequences: int = 0) -> None:
        nonlocal loader, iterator, sampler_offset
        sampler.set_epoch(epoch); sampler_offset = 0
        loader = make_dataloader(dataset, config["world"]["context_length"], batch_size, workers=workers, sampler=sampler)
        iterator = iter(loader)
        while sampler_offset < skip_sequences:
            batch = next_batch(batch_size)
            if sampler_offset > skip_sequences:
                raise RuntimeError("checkpoint sampler offset is not aligned to the resolved microbatch")
    def next_batch(batch_size: int) -> dict[str, torch.Tensor]:
        nonlocal loader, iterator, sampler_offset
        if loader is None or iterator is None or loader.batch_size != batch_size:
            reset_data(batch_size, sampler.epoch)
        try: batch = next(iterator)
        except StopIteration:
            sampler.set_epoch(sampler.epoch + 1)
            iterator = iter(loader)
            sampler_offset = 0
            batch = next(iterator)
        sampler_offset += int(batch["tokens"].shape[0])
        return batch
    reset_data(micro)
    def one_step(cpu_batch: dict[str, torch.Tensor], commit: bool) -> int:
        batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
        if batch["tokens"].dtype != torch.long or not all(v.shape == batch["tokens"].shape for v in batch.values()): raise AssertionError("invalid batch tensor contract")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ddp(batch["tokens"]); total, count = masked_next_token_loss(logits, batch["tokens"], batch["loss_mask"])
        global_count = _all_reduce_count(count, device)
        normalized = _backward_global_mean(scaler, total, global_count, world_size)
        scaler.unscale_(optimizer)
        _finite(ddp.module, normalized)
        if commit:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        input_count = batch["attention_mask"].sum().detach().to(torch.int64)
        dist.all_reduce(input_count, op=dist.ReduceOp.SUM)
        return int(input_count.item())
    # Bounded OOM-only adjustment before real training. The probe includes the
    # full optimizer/checkpoint path, then restores every mutable state so no
    # uncounted 6e-4 update leaks into the official warm-up or token budget.
    baseline = {"model": copy.deepcopy(ddp.module.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "scaler": copy.deepcopy(scaler.state_dict()), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}
    def restore_baseline() -> None:
        ddp.module.load_state_dict(baseline["model"]); optimizer.load_state_dict(baseline["optimizer"]); scaler.load_state_dict(baseline["scaler"])
        torch.set_rng_state(baseline["rng"]); torch.cuda.set_rng_state_all(baseline["cuda_rng"]); random.setstate(baseline["python_rng"])
        optimizer.zero_grad(set_to_none=True)
    retries = 0
    probe_batch = None
    while True:
        try:
            probe_batch = next_batch(micro)
            optimizer.zero_grad(set_to_none=True); one_step(probe_batch, True); torch.cuda.synchronize(device)
            restore_baseline()
            break
        except torch.OutOfMemoryError:
            restore_baseline(); torch.cuda.empty_cache(); retries += 1
            if retries > config["training"]["max_preflight_retries"] or micro <= 1:
                raise RuntimeError("preflight exhausted configured microbatch retries")
            micro //= 2; reset_data(micro)
    if rank == 0:
        config["training"]["resolved_microbatch_sequences"] = micro; atomic_json(resolved_config, config)
        _atomic_checkpoint(run_dir / "checkpoints" / "preflight.pt", {"model": ddp.module.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config_hash": config["_meta"]["hash"], "tokens": 0})
    dist.barrier()
    # Every retained run proves that the checkpoint can be loaded and used for
    # a finite inference pass before training time is spent.
    preflight_state = torch.load(run_dir / "checkpoints" / "preflight.pt", map_location=device, weights_only=False)
    if preflight_state.get("config_hash") != config["_meta"]["hash"]:
        raise RuntimeError("preflight checkpoint configuration hash mismatch")
    ddp.module.load_state_dict(preflight_state["model"]); optimizer.load_state_dict(preflight_state["optimizer"]); scaler.load_state_dict(preflight_state["scaler"])
    assert probe_batch is not None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        check_logits = ddp.module(probe_batch["tokens"].to(device, non_blocking=True))
    if not torch.isfinite(check_logits).all(): raise FloatingPointError("non-finite logits after preflight checkpoint reload")
    dist.barrier()
    if preflight_only: dist.destroy_process_group(); return
    # The probe is not part of the scientific data order.
    del preflight_state, check_logits, baseline, probe_batch
    torch.cuda.empty_cache()
    reset_data(micro)
    tokens, step, start = 0, 0, time.monotonic()
    latest = run_dir / "checkpoints" / "latest.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("config_hash") != config["_meta"]["hash"]:
            raise RuntimeError("resume checkpoint configuration hash mismatch")
        ddp.module.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        tokens, step = state["tokens"], state["step"]
        torch.set_rng_state(state["rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"]); random.setstate(state["python_rng"])
        reset_data(micro, state["sampler_epoch"], state["sampler_offset"])
    metric_path = run_dir / "logs" / "metrics.jsonl"
    last_checkpoint_at, previous_tokens = time.monotonic(), tokens
    while tokens < config["training"]["token_budget"]:
        optimizer.zero_grad(set_to_none=True); update_tokens = 0; cpu_batches = []; local_action_tokens = 0
        while update_tokens < config["training"]["global_tokens_per_update"]:
            cpu_batch = next_batch(micro); cpu_batches.append(cpu_batch)
            local_action_tokens += int(cpu_batch["loss_mask"][:, 1:].sum().item())
            input_count = cpu_batch["attention_mask"].sum().to(device=device, dtype=torch.int64)
            dist.all_reduce(input_count, op=dist.ReduceOp.SUM); update_tokens += int(input_count.item())
        global_action_tokens = torch.tensor(local_action_tokens, device=device, dtype=torch.float64)
        dist.all_reduce(global_action_tokens, op=dist.ReduceOp.SUM)
        if global_action_tokens.item() <= 0: raise AssertionError("optimizer update has no supervised action tokens")
        global_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        observed_local_action_tokens = 0
        for batch_index, cpu_batch in enumerate(cpu_batches):
            batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
            sync = nullcontext() if batch_index == len(cpu_batches) - 1 else ddp.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = ddp(batch["tokens"]); total, count = masked_next_token_loss(logits, batch["tokens"], batch["loss_mask"])
                _backward_global_mean(scaler, total, global_action_tokens, world_size)
            global_loss_sum += total.detach().to(torch.float64)
            observed_local_action_tokens += int(count.item())
        if observed_local_action_tokens != local_action_tokens: raise AssertionError("supervised-token pre-count changed before backward")
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0, error_if_nonfinite=True)
        for group in optimizer.param_groups: group["lr"] = _lr(config["training"], tokens)
        scaler.step(optimizer); scaler.update(); tokens += update_tokens; step += 1
        if rank == 0:
            elapsed = time.monotonic() - start
            with metric_path.open("a") as handle: handle.write(json.dumps({"step": step, "tokens": tokens, "action_nll": float((global_loss_sum / global_action_tokens).item()), "action_tokens": int(global_action_tokens.item()), "input_tokens_per_second": tokens / max(elapsed, 1e-9), "grad_norm": float(grad_norm), "grad_scale": scaler.get_scale(), "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed}) + "\n")
            periodic_due = time.monotonic() - last_checkpoint_at >= config["training"]["checkpoint_minutes"] * 60
            target_due = any(previous_tokens < target <= tokens for target in config["training"]["save_tokens"])
            if target_due or periodic_due or tokens >= config["training"]["token_budget"]:
                checkpoint = {"model": ddp.module.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "tokens": tokens, "step": step, "config_hash": config["_meta"]["hash"], "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(), "sampler_epoch": sampler.epoch, "sampler_offset": sampler_offset, "wall_seconds": elapsed}
                _atomic_checkpoint(run_dir / "checkpoints" / "latest.pt", checkpoint)
                _atomic_checkpoint(run_dir / "checkpoints" / f"recovery-{step:08d}.pt", checkpoint)
                last_checkpoint_at = time.monotonic()
        previous_tokens = tokens
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--resolved-config", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try: run(args.resolved_config, args.run_dir, args.preflight_only, args.resume)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
