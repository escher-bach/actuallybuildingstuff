"""Byte tokenizer, deterministic world trajectories, and binary shard reader."""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import random
import struct
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .artifacts import atomic_json, sha256_file

PAD, BOS, EOS, OBS, ACTION, END_TURN = range(256, 262)
VOCAB_SIZE = 262
TOKEN_TABLE = {**{str(i): i for i in range(256)}, "PAD": PAD, "BOS": BOS, "EOS": EOS, "OBS": OBS, "ACTION": ACTION, "END_TURN": END_TURN}
TOKENIZER_HASH = hashlib.sha256(json.dumps(TOKEN_TABLE, sort_keys=True).encode()).hexdigest()


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="strict"))


def decode_bytes(tokens: list[int]) -> str:
    if any(t < 0 or t >= 256 for t in tokens):
        raise ValueError("transport tokens cannot be UTF-8 decoded")
    return bytes(tokens).decode("utf-8", errors="strict")


@dataclass(frozen=True)
class Sequence:
    tokens: list[int]
    loss: list[int]
    channels: list[int]


class BinaryShard(Dataset[Sequence]):
    """Zero-copy-ish sequential decoder for the Rust `BLMSHRD1` format."""
    def __init__(self, path: Path):
        self.path, self.handle = path, path.open("rb")
        self.map = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        if self.map[:8] != b"BLMSHRD1":
            raise ValueError(f"not a Step 1 shard: {path}")
        version, count = struct.unpack_from("<IQ", self.map, 8)
        if version != 1:
            raise ValueError(f"unsupported shard version {version}")
        self.offsets: list[tuple[int, int]] = []
        offset = 20
        for _ in range(count):
            length, examples = struct.unpack_from("<QQ", self.map, offset)
            payload = offset + 16
            end = payload + length * 4 + length * 2 + examples * 24
            if end > len(self.map):
                raise ValueError("truncated shard")
            self.offsets.append((payload, length))
            offset = end
        if offset != len(self.map):
            raise ValueError("unexpected trailing shard data")

    def __len__(self) -> int: return len(self.offsets)
    def __getitem__(self, index: int) -> Sequence:
        offset, length = self.offsets[index]
        tokens = list(struct.unpack_from(f"<{length}I", self.map, offset))
        loss_start = offset + 4 * length
        loss = list(self.map[loss_start:loss_start + length])
        channels = list(self.map[loss_start + length:loss_start + 2 * length])
        return Sequence(tokens, loss, channels)


class SequenceDataset(Dataset[Sequence]):
    def __init__(self, sequences: list[Sequence]): self.sequences = sequences
    def __len__(self) -> int: return len(self.sequences)
    def __getitem__(self, index: int) -> Sequence: return self.sequences[index]


def collate(sequences: list[Sequence], context: int) -> dict[str, torch.Tensor]:
    if not sequences: raise AssertionError("empty batch")
    length = max(len(s.tokens) for s in sequences)
    if length > context: raise AssertionError(f"sequence {length} exceeds context {context}")
    tokens = torch.full((len(sequences), length), PAD, dtype=torch.long)
    loss, channels, attention = torch.zeros_like(tokens, dtype=torch.uint8), torch.zeros_like(tokens, dtype=torch.uint8), torch.zeros_like(tokens, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        n = len(sequence.tokens)
        if not (n == len(sequence.loss) == len(sequence.channels)): raise AssertionError("sequence fields misaligned")
        tokens[row, :n] = torch.tensor(sequence.tokens, dtype=torch.long)
        loss[row, :n], channels[row, :n], attention[row, :n] = torch.tensor(sequence.loss, dtype=torch.uint8), torch.tensor(sequence.channels, dtype=torch.uint8), True
    if not ((tokens >= 0).all() and (tokens < VOCAB_SIZE).all()): raise AssertionError("token ID out of vocabulary")
    if (loss[tokens == PAD] != 0).any(): raise AssertionError("padding has nonzero loss")
    return {"tokens": tokens, "loss_mask": loss, "target_channels": channels, "attention_mask": attention}


class DistributedSequenceSampler(Sampler[int]):
    def __init__(self, dataset: Dataset[Sequence], seed: int, rank: int, world_size: int, drop_last: bool = True):
        self.dataset, self.seed, self.rank, self.world_size, self.drop_last, self.epoch = dataset, seed, rank, world_size, drop_last, 0
    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __iter__(self) -> Iterator[int]:
        values = list(range(len(self.dataset)))
        random.Random((self.seed << 32) ^ self.epoch).shuffle(values)
        usable = len(values) - (len(values) % self.world_size) if self.drop_last else len(values)
        return iter(values[:usable][self.rank:usable:self.world_size])
    def __len__(self) -> int:
        return len(self.dataset) // self.world_size if self.drop_last else (len(self.dataset) + self.world_size - 1) // self.world_size


def _family(params: dict):
    from world_py import FamilyParams
    return FamilyParams(n_hyp=params["n_hyp"], n_probe=params["n_probe"], n_evidence=params["n_evidence"], cost_lo=params["cost_lo"], cost_hi=params["cost_hi"], budget_slack=params["budget_slack"], min_depth=params["min_depth"], step_slack=params["step_slack"], variant=params["variant"])


def generate_world_sequences(params: dict, seed: int, episodes: int, context: int, rendering: str) -> tuple[list[Sequence], dict]:
    """One Rust batch call per transition; Python only assembles offline buffers."""
    from world_py import Batch
    batch = Batch(_family(params), seed=seed, n_episodes=episodes)
    buffers = [[BOS] for _ in range(episodes)]
    masks = [[0] for _ in range(episodes)]
    channels = [[0] for _ in range(episodes)]
    while not all(batch.done()):
        observations, targets, live = batch.observations(rendering), batch.privileged_teacher_targets(), batch.live_episode_indices()
        actions: list[int] = []
        for i in live:
            action = min(targets[i]["preferred_actions"])
            # Cross the public renderer boundary: the compact integer is only
            # for Batch.step, never a model target or evaluation shortcut.
            from world_py import render_action
            text = render_action(action, params["n_probe"], params["n_hyp"], rendering)
            prefix, action_tokens = [OBS] + encode_bytes(observations[i]) + [ACTION], encode_bytes(text) + [END_TURN]
            buffers[i].extend(prefix + action_tokens); masks[i].extend([0] * len(prefix) + [1] * len(action_tokens)); channels[i].extend([0] * len(prefix) + [1] * len(action_tokens))
            actions.append(action)
        batch.step(actions)
    output = []
    for ids, mask, channel in zip(buffers, masks, channels):
        ids.append(EOS); mask.append(0); channel.append(0)
        if len(ids) > context: raise RuntimeError(f"overlong trajectory ({len(ids)} > {context}); no truncation is permitted")
        output.append(Sequence(ids, mask, channel))
    manifest = {"world_family_version": "world-0.1.0", "root_seed": seed, "episode_count": episodes, "rendering": rendering, "tokenizer": TOKEN_TABLE, "tokenizer_hash": TOKENIZER_HASH, "token_count": sum(map(lambda s: len(s.tokens), output))}
    manifest["content_hash"] = hashlib.sha256(json.dumps([s.tokens for s in output]).encode()).hexdigest()
    return output, manifest


def write_generated_dataset(root: Path, name: str, sequences: list[Sequence], manifest: dict) -> Path:
    # A compact torch serialization is used only for generated Python data; Rust shards retain their own binary format.
    path = root / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([s.__dict__ for s in sequences], path)
    manifest = {**manifest, "file": path.name, "file_sha256": sha256_file(path)}
    atomic_json(root / "manifests" / f"{name}.json", manifest)
    return path


def load_generated_dataset(path: Path) -> SequenceDataset:
    return SequenceDataset([Sequence(**item) for item in torch.load(path, map_location="cpu", weights_only=True)])


def make_dataloader(dataset: Dataset[Sequence], context: int, batch_size: int, workers: int | None = None, sampler: Sampler[int] | None = None) -> DataLoader:
    """One global worker budget; callers divide it across DDP ranks."""
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=workers,
                      collate_fn=partial(collate, context=context), pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=2 if workers else None)


def generate_rust_shard(params: dict, seed: int, episodes: int, context: int, rendering: str, directory: Path, stem: str) -> tuple[Path, Path, Path]:
    """Delegate generation/packing to Rust; Python only reads the stable binary."""
    from world_py import generate_teacher_shard
    paths = generate_teacher_shard(_family(params), seed, episodes, rendering, context, str(directory), stem)
    return tuple(Path(path) for path in paths)
