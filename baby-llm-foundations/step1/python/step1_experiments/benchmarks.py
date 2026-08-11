"""Measured CPU/data throughput gates; never substitutes local numbers for Kaggle."""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable

from .artifacts import atomic_json
from .data import BOS, Sequence, collate, generate_world_sequences, make_dataloader


def _measure(work: Callable[[], int], minimum_seconds: int = 30, minimum_iterations: int = 1000) -> dict:
    work()  # warm-up, deliberately excluded
    samples, total, start = [], 0, time.perf_counter()
    while len(samples) < minimum_iterations or time.perf_counter() - start < minimum_seconds:
        before = time.perf_counter(); amount = work(); elapsed = time.perf_counter() - before
        if elapsed > 0: samples.append(amount / elapsed)
        total += amount
    samples.sort()
    return {"seconds": time.perf_counter() - start, "work": total, "median_per_second": statistics.median(samples),
            "iqr_per_second": statistics.quantiles(samples, n=4)[2] - statistics.quantiles(samples, n=4)[0] if len(samples) >= 4 else 0}


def cpu_benchmark(config: dict, directory: Path) -> dict:
    world = config["world"]
    # Raw text is materialized before timing. Both paths use the byte tokenizer, sequence policy, and collator.
    source, _ = generate_world_sequences(world, config["run"]["root_seed"], 256, world["context_length"], world["rendering"])
    raw_path = directory / "materialized_raw_text.bin"
    transcripts = [bytes([v for v in s.tokens if v < 256]) for s in source]
    raw_path.write_bytes(b"\0".join(transcripts))
    cursor = 0
    def raw() -> int:
        nonlocal cursor
        # Timed baseline exactly follows read materialized text -> byte tokenize
        # -> pack -> deliver batch. It deliberately does no world generation.
        chunks = raw_path.read_bytes().split(b"\0")
        chosen = chunks[cursor % len(chunks): cursor % len(chunks) + 8] or chunks[:8]; cursor += 8
        packed = [Sequence([BOS, *text[: world["context_length"] - 1]], [0] * (min(len(text), world["context_length"] - 1) + 1), [0] * (min(len(text), world["context_length"] - 1) + 1)) for text in chosen]
        collate(packed, world["context_length"])
        return sum(len(text) for text in chosen)
    generated = 0
    def world_work() -> int:
        nonlocal generated
        sequences, _ = generate_world_sequences(world, config["run"]["root_seed"] + generated, 8, world["context_length"], world["rendering"]); generated += 8
        collate(sequences, world["context_length"])
        return sum(len(s.tokens) for s in sequences)
    result = {"raw_text": _measure(raw), "world": _measure(world_work), "policy": "30 seconds or 1000 batches after warm-up"}
    result["world_to_raw_ratio"] = result["world"]["median_per_second"] / result["raw_text"]["median_per_second"]
    result["passes_80_percent_gate"] = result["world_to_raw_ratio"] >= 0.8
    atomic_json(directory / "cpu.json", result)
    return result


def dataloader_benchmark(dataset, context: int, directory: Path) -> dict:
    loader = make_dataloader(dataset, context, batch_size=min(8, len(dataset)))
    iterator = iter(loader)
    def consume() -> int:
        nonlocal iterator
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        return int(batch["attention_mask"].sum())
    result = _measure(consume)
    result["workers"] = loader.num_workers
    result["pinned_memory"] = loader.pin_memory
    atomic_json(directory / "dataloader.json", result)
    return result
