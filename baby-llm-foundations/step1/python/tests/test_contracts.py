from __future__ import annotations

import struct
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from step1_experiments.data import END_TURN, BinaryShard, Sequence, collate
from step1_experiments.runner import _run_checked
from step1_experiments.train import training_plan


class DataContracts(unittest.TestCase):
    def test_global_input_token_plan_is_invariant_to_world_size(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "kaggle" / "t4x2_preflight.toml"
        config = tomllib.loads(config_path.read_text())
        one_rank = training_plan(config, world_size=1)
        two_ranks = training_plan(config, world_size=2)
        self.assertEqual(one_rank.gradient_accumulation_steps, 4)
        self.assertEqual(two_ranks.gradient_accumulation_steps, 2)
        self.assertEqual(one_rank.nominal_global_input_tokens_per_update, 32768)
        self.assertEqual(two_ranks.nominal_global_input_tokens_per_update, 32768)
        self.assertEqual(two_ranks.max_steps, 64)
        self.assertEqual(two_ranks.checkpoint_interval_nominal_global_input_tokens, 1_048_576)

    def test_token_plan_rejects_partial_global_updates(self) -> None:
        config = {
            "run": {"mode": "dense"},
            "world": {"context_length": 2048},
            "training": {
                "microbatch_sequences": 4,
                "global_tokens_per_update": 32768,
                "token_budget": 2_000_000,
                "checkpoint_interval_updates": 32,
                "checkpoint_total_limit": 2,
            },
        }
        with self.assertRaisesRegex(ValueError, "token_budget must be divisible"):
            training_plan(config, world_size=2)

    def test_collator_uses_standard_minus_100_labels(self) -> None:
        batch = collate([Sequence([1, 2, END_TURN], [0, 1, 1], [0, 1, 1])], context=8)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, END_TURN]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, END_TURN]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True]])

    def test_binary_shard_reader_obeys_layout(self) -> None:
        tokens, loss, channels = [257, 65, 261, 258], [0, 1, 1, 0], [0, 1, 1, 0]
        payload = bytearray(b"BLMSHRD1")
        payload += struct.pack("<IQ", 1, 1) + struct.pack("<QQ", len(tokens), 1)
        payload += struct.pack(f"<{len(tokens)}I", *tokens) + bytes(loss) + bytes(channels)
        payload += struct.pack("<QQQ", 0, 0, len(tokens))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(payload)
            shard = BinaryShard(path)
            self.assertEqual(shard[0], Sequence(tokens, loss, channels))
            shard.close()

    def test_subprocess_failure_surfaces_child_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with self.assertRaisesRegex(RuntimeError, "UNIQUE_CHILD_FAILURE"):
                _run_checked([sys.executable, "-c", "print('UNIQUE_CHILD_FAILURE'); raise SystemExit(7)"], log, Path(directory), 30)


if __name__ == "__main__":
    unittest.main()
