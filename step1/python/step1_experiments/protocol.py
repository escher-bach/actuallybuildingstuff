"""Dependency-free causal-LM transport protocol shared by data and audits."""
from __future__ import annotations


PAD, BOS, EOS, OBS, ACTION, END_TURN = range(256, 262)
VOCAB_SIZE = 262


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="strict"))


def decode_bytes(tokens: list[int]) -> str:
    if any(token < 0 or token >= 256 for token in tokens):
        raise ValueError("transport tokens cannot be UTF-8 decoded")
    return bytes(tokens).decode("utf-8", errors="strict")
