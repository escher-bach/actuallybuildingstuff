"""Build the checked-in 262-token Hugging Face fast-tokenizer artifact.

The ByteLevel implementation comes from ``tokenizers``. This builder only
freezes the project's experimental byte-ID and protocol-marker contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from .standard_stack import TOKENIZER_PATH


SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|obs|>", "<|action|>", "<|end_turn|>")
SPECIAL_TOKEN_IDS = {token: 256 + index for index, token in enumerate(SPECIAL_TOKENS)}


def _byte_to_unicode() -> dict[int, str]:
    """Return Tokenizers' standard GPT-2 ByteLevel alphabet by byte value."""
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    codepoints = visible[:]
    next_codepoint = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + next_codepoint)
            next_codepoint += 1
    return dict(zip(visible, map(chr, codepoints), strict=True))


def create_tokenizer() -> Tokenizer:
    byte_vocabulary = {symbol: byte for byte, symbol in _byte_to_unicode().items()}
    tokenizer = Tokenizer(models.BPE(vocab={**byte_vocabulary, **SPECIAL_TOKEN_IDS}, merges=[], fuse_unk=False))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
    return tokenizer


def save_tokenizer(output: Path = TOKENIZER_PATH) -> None:
    output.mkdir(parents=True, exist_ok=True)
    create_tokenizer().save(str(output / "tokenizer.json"))
    (output / "special_tokens_map.json").write_text(json.dumps({
        "pad_token": SPECIAL_TOKENS[0], "bos_token": SPECIAL_TOKENS[1], "eos_token": SPECIAL_TOKENS[2],
        "additional_special_tokens": list(SPECIAL_TOKENS[3:]),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 2048,
        "clean_up_tokenization_spaces": False, "add_bos_token": False, "add_eos_token": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=TOKENIZER_PATH)
    args = parser.parse_args()
    save_tokenizer(args.output)


if __name__ == "__main__":
    main()
