from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from step1_experiments.data import (
    END_TURN,
    VOCAB_SIZE,
    BinaryShard,
    DistributedSequenceSampler,
    Sequence,
    SequenceDataset,
    collate,
)
from step1_experiments.instrument import retrieval_batch
from step1_experiments.model import Step1Transformer, masked_next_token_loss
from step1_experiments.runner import _run_checked
from step1_experiments.train import _atomic_checkpoint, _gradients_are_finite, _lr, ddp_global_mean_loss


class OptimizationContracts(unittest.TestCase):
    def test_gradient_finiteness_is_checked_after_unscaling(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        self.assertTrue(_gradients_are_finite(model))
        model.weight.grad[0, 0] = torch.inf
        self.assertFalse(_gradients_are_finite(model))

    def test_ddp_loss_is_normalized_before_backward(self) -> None:
        local_sum = torch.tensor(2321.0, requires_grad=True)
        normalized = ddp_global_mean_loss(local_sum, global_count=817, world_size=2)
        self.assertAlmostEqual(normalized.item(), 2321.0 * 2 / 817, places=5)
        normalized.backward()
        self.assertAlmostEqual(local_sum.grad.item(), 2 / 817, places=7)

    def test_ddp_average_of_scaled_local_sums_equals_global_mean(self) -> None:
        # Rank-local sums have gradients 2 and 6 and counts 2 and 3.
        # DDP averages the two scaled gradients after backward.
        rank0 = torch.tensor(1.0, requires_grad=True)
        rank1 = torch.tensor(1.0, requires_grad=True)
        ddp_global_mean_loss(2 * rank0, global_count=5, world_size=2).backward()
        ddp_global_mean_loss(6 * rank1, global_count=5, world_size=2).backward()
        ddp_average = (rank0.grad + rank1.grad) / 2
        self.assertAlmostEqual(ddp_average.item(), 8 / 5, places=6)

    def test_lr_schedule_starts_at_zero_and_ends_at_minimum(self) -> None:
        config = {"token_budget": 1000, "warmup_fraction": 0.02, "learning_rate": 6e-4, "min_learning_rate": 6e-5}
        self.assertEqual(_lr(config, 0), 0.0)
        self.assertAlmostEqual(_lr(config, 20), 6e-4)
        self.assertAlmostEqual(_lr(config, 1000), 6e-5)


class DataContracts(unittest.TestCase):
    def test_retrieval_targets_are_nonoverlapping_bound_values(self) -> None:
        tokens, mask = retrieval_batch(7, batch_size=8)
        for row, active_mask in zip(tokens.tolist(), mask.tolist()):
            targets = [index for index, active in enumerate(active_mask) if active]
            self.assertEqual(len(targets), 8)
            for target_index in targets:
                self.assertEqual(row[target_index - 2], 255)
                key, value = row[target_index - 1], row[target_index]
                self.assertTrue(any(row[source] == key and row[source + 1] == value for source in range(2, 73)))

    def test_collator_and_shifted_loss_contract(self) -> None:
        sequence = Sequence([1, 2, END_TURN], [0, 1, 1], [0, 1, 1])
        batch = collate([sequence], context=8)
        logits = torch.zeros(1, 3, VOCAB_SIZE, requires_grad=True)
        total, count = masked_next_token_loss(logits, batch["tokens"], batch["loss_mask"])
        self.assertEqual(count.item(), 2)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_distributed_sampler_partitions_without_overlap(self) -> None:
        dataset = SequenceDataset([Sequence([i], [0], [0]) for i in range(11)])
        rank0 = set(DistributedSequenceSampler(dataset, 3, 0, 2))
        rank1 = set(DistributedSequenceSampler(dataset, 3, 1, 2))
        self.assertFalse(rank0 & rank1)
        self.assertEqual(len(rank0 | rank1), 10)

    def test_binary_shard_reader_obeys_layout(self) -> None:
        tokens, loss, channels = [257, 65, 261, 258], [0, 1, 1, 0], [0, 1, 1, 0]
        payload = bytearray(b"BLMSHRD1")
        payload += struct.pack("<IQ", 1, 1)
        payload += struct.pack("<QQ", len(tokens), 1)
        payload += struct.pack(f"<{len(tokens)}I", *tokens)
        payload += bytes(loss) + bytes(channels)
        payload += struct.pack("<QQQ", 0, 0, len(tokens))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"; path.write_bytes(payload)
            shard = BinaryShard(path)
            self.assertEqual(shard[0], Sequence(tokens, loss, channels))
            shard.close()


class ModelAndArtifactContracts(unittest.TestCase):
    def test_fixed_model_parameter_count(self) -> None:
        model = Step1Transformer()
        self.assertEqual(model.parameter_counts()["total"], 21_345_408)

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            state = {"config_hash": "abc", "tensor": torch.arange(5)}
            _atomic_checkpoint(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["config_hash"], "abc")
            self.assertTrue(torch.equal(loaded["tensor"], state["tensor"]))

    def test_subprocess_failure_surfaces_child_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with self.assertRaisesRegex(RuntimeError, "UNIQUE_CHILD_FAILURE"):
                _run_checked([sys.executable, "-c", "print('UNIQUE_CHILD_FAILURE'); raise SystemExit(7)"], log, Path(directory), 30)


if __name__ == "__main__":
    unittest.main()
