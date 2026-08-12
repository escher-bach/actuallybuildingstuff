from __future__ import annotations

import struct
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import torch

from step1_experiments.data import END_TURN, BinaryShard, Sequence, collate
from step1_experiments.runner import _run_checked
from step1_experiments.train import (
    _per_rank_numerical_diagnostic,
    _tensor_difference_report,
    assert_exact_state_dict_roundtrip,
    exact_state_dict_report,
    training_plan,
)


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

    def test_serialization_gate_requires_exact_state_dict_values(self) -> None:
        torch.manual_seed(7)
        expected = torch.nn.Linear(3, 2)
        actual = torch.nn.Linear(3, 2)
        actual.load_state_dict(expected.state_dict())
        exact = assert_exact_state_dict_roundtrip(expected, actual)
        self.assertTrue(exact["exact"])
        self.assertEqual(exact["expected_state_sha256"], exact["actual_state_sha256"])
        with torch.no_grad():
            actual.bias.add_(0.25)
        report = exact_state_dict_report(expected, actual)
        self.assertFalse(report["exact"])
        self.assertNotEqual(report["expected_state_sha256"], report["actual_state_sha256"])
        self.assertEqual(report["unequal_keys"], ["bias"])
        self.assertGreater(report["max_parameter_abs_difference_for_unequal_floats"], 0.24)
        with self.assertRaisesRegex(AssertionError, "save_pretrained reload changed model state"):
            assert_exact_state_dict_roundtrip(expected, actual)

    def test_numerical_differences_are_reported_without_a_tolerance_gate(self) -> None:
        report = _tensor_difference_report(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 1.0]))
        self.assertFalse(report["exactly_equal"])
        self.assertEqual(report["max_abs_difference"], 1.0)
        self.assertEqual(report["mean_abs_difference"], 0.75)

    def test_rank_safe_numerical_diagnostic_records_every_rank(self) -> None:
        class Accelerator:
            num_processes = 2

            @staticmethod
            def gather_for_metrics(_values):
                return torch.tensor([0.0, 0.0, 1.0, 0.125, 0.03125, 0.0], dtype=torch.float64)

        class Trainer:
            accelerator = Accelerator()

            class args:
                device = torch.device("cpu")

        report = _per_rank_numerical_diagnostic(
            Trainer(), {"max_abs_difference": 0.0, "mean_abs_difference": 0.0, "exactly_equal": True}
        )
        self.assertEqual(report, [
            {"rank": 0, "max_abs_difference": 0.0, "mean_abs_difference": 0.0, "exactly_equal": True},
            {"rank": 1, "max_abs_difference": 0.125, "mean_abs_difference": 0.03125, "exactly_equal": False},
        ])

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
