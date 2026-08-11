"""Standard Hugging Face artifacts for the Step 1 model stack.

This module deliberately owns only artifact locations and the small adapter
needed to instantiate the maintained Transformers implementation. It must not
grow model layers, a loss function, or training plumbing.
"""
from __future__ import annotations

from pathlib import Path


_STEP1_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = _STEP1_ROOT / "artifacts" / "pythia70m-byte-gpt-neox" / "config.json"
TOKENIZER_PATH = _STEP1_ROOT / "artifacts" / "byte-utf8-tokenizer"
EXPECTED_PARAMETER_COUNT = 19_183_616


def load_model_config():
    """Load the checked-in frozen GPT-NeoX config without fetching weights."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(MODEL_CONFIG_PATH, local_files_only=True)


def create_model():
    """Create random Pythia-70M-profile weights from the local config."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_config(load_model_config())


def load_tokenizer():
    """Load the checked-in fast-tokenizer artifact without network access."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)


def assert_model_contract(model) -> None:
    """Check the frozen architecture's public experiment contract."""
    expected = {
        "model_type": "gpt_neox", "vocab_size": 262, "hidden_size": 512,
        "intermediate_size": 2048, "num_hidden_layers": 6,
        "num_attention_heads": 8, "max_position_embeddings": 2048,
        "pad_token_id": 256, "bos_token_id": 257, "eos_token_id": 258,
        "tie_word_embeddings": False, "use_cache": False,
    }
    mismatches = {name: (getattr(model.config, name, None), value) for name, value in expected.items() if getattr(model.config, name, None) != value}
    if mismatches:
        raise AssertionError(f"frozen GPT-NeoX config mismatch: {mismatches}")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(f"GPT-NeoX parameter count {count:,} != {EXPECTED_PARAMETER_COUNT:,}")
