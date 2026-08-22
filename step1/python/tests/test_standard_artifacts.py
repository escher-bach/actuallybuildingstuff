from __future__ import annotations

import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from step1_experiments.build_tokenizer import SPECIAL_TOKENS, _byte_to_unicode
from step1_experiments.standard_stack import (
    EXPECTED_PARAMETER_COUNT,
    MODEL_CONFIG_PATH,
    TOKENIZER_PATH,
)


class StandardArtifactContracts(unittest.TestCase):
    def test_frozen_gpt_neox_config(self) -> None:
        config = AutoConfig.from_pretrained(MODEL_CONFIG_PATH, local_files_only=True)
        self.assertEqual(config.model_type, "gpt_neox")
        self.assertEqual(config.vocab_size, 262)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.intermediate_size, 2048)
        self.assertEqual(config.num_hidden_layers, 6)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.max_position_embeddings, 2048)
        self.assertEqual(config.pad_token_id, 256)
        self.assertEqual(config.bos_token_id, 257)
        self.assertEqual(config.eos_token_id, 258)
        self.assertFalse(config.tie_word_embeddings)
        self.assertFalse(config.use_cache)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.25)
        self.assertEqual(config.rope_parameters["rope_theta"], 10_000.0)

    def test_fast_tokenizer_has_fixed_byte_and_special_ids(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        self.assertEqual(tokenizer.vocab_size, 262)
        for byte, token in _byte_to_unicode().items():
            self.assertEqual(tokenizer.backend_tokenizer.token_to_id(token), byte)
        self.assertEqual(
            [tokenizer.encode(token, add_special_tokens=False) for token in SPECIAL_TOKENS],
            [[256], [257], [258], [259], [260], [261]],
        )
        for text in ("ASCII\x00", "héllo", "λ → observation", "inspect(3)", "commit(2)"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(ids, list(text.encode("utf-8")))
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)

    @unittest.skipUnless(importlib.util.find_spec("world_py"), "requires the built Rust Python extension")
    def test_rust_shard_identity_is_bound_to_tokenizer_artifact(self) -> None:
        from world_py import tokenizer_identity

        artifact_hash = hashlib.sha256((TOKENIZER_PATH / "tokenizer.json").read_bytes()).hexdigest()
        self.assertEqual(
            tokenizer_identity(),
            ("byte-utf8-fast", "v2", artifact_hash),
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires the Kaggle PyTorch runtime")
    def test_random_model_has_frozen_parameter_count(self) -> None:
        from step1_experiments.standard_stack import assert_model_contract, create_model

        model = create_model()
        assert_model_contract(model)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), EXPECTED_PARAMETER_COUNT)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires the Kaggle PyTorch runtime")
    def test_save_pretrained_round_trip_has_exact_model_state(self) -> None:
        """Serialization is proved by state equality, not output tolerance."""
        from transformers import AutoModelForCausalLM
        from step1_experiments.standard_stack import create_model
        from step1_experiments.train import assert_exact_state_dict_roundtrip

        model = create_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            reloaded = AutoModelForCausalLM.from_pretrained(Path(directory), local_files_only=True).eval()
        report = assert_exact_state_dict_roundtrip(model, reloaded)
        self.assertTrue(report["exact"])
        self.assertEqual(report["expected_key_count"], report["actual_key_count"])


if __name__ == "__main__":
    unittest.main()
