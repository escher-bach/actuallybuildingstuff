from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from step1_experiments.evaluate import (
    END_TURN,
    _aggregate_rows,
    _attempt_status,
    _decode_actions_batched,
    _execute_batched,
    _matched_sets,
    assert_evaluation_contract,
)


class EvaluationContracts(unittest.TestCase):
    def test_aggregation_excludes_failed_rows_from_excess_cost(self) -> None:
        metrics = _aggregate_rows([
            {"success": True, "spent": 5, "teacher_spent": 3, "malformed": 0, "invalid": 0, "steps": 2},
            {"success": False, "spent": 99, "teacher_spent": 2, "malformed": 1, "invalid": 0, "steps": 0},
        ])
        self.assertEqual(metrics["mean_success_excess_cost"], 2.0)
        self.assertEqual(metrics["mean_spent"], 52.0)
        self.assertEqual(metrics["failure_rate"], 0.5)
        self.assertEqual(metrics["malformed_action_rate"], 0.5)

    def test_aggregation_reports_null_success_cost_when_no_episode_succeeds(self) -> None:
        metrics = _aggregate_rows([
            {"success": False, "spent": 0, "teacher_spent": 3, "malformed": 1, "invalid": 0, "steps": 0},
        ])
        self.assertIsNone(metrics["mean_success_excess_cost"])

    def test_batched_generation_preserves_row_order_and_reduces_calls(self) -> None:
        class Model:
            class config:
                max_position_embeddings = 32
                pad_token_id = 256

            def __init__(self):
                self.batch_sizes: list[int] = []

            def generate(self, input_ids, **_kwargs):
                self.batch_sizes.append(input_ids.shape[0])
                continuation = torch.tensor([[105, 110, END_TURN]], dtype=torch.long, device=input_ids.device)
                return torch.cat([input_ids, continuation.repeat(input_ids.shape[0], 1)], dim=1)

        model = Model()
        result = _decode_actions_batched(model, [[257, 65], [257, 66, 67], [257, 68]], torch.device("cpu"), batch_size=2)
        self.assertEqual(result, ["in", "in", "in"])
        self.assertEqual(model.batch_sizes, [2, 1])

    def test_mixed_context_lengths_preserve_row_local_generation_allowance(self) -> None:
        class Model:
            class config:
                max_position_embeddings = 100
                pad_token_id = 256

            def __init__(self):
                self.calls: list[tuple[int, int]] = []

            def generate(self, input_ids, **kwargs):
                self.calls.append((input_ids.shape[0], kwargs["max_new_tokens"]))
                continuation = torch.tensor([[105, 110, END_TURN]], dtype=torch.long, device=input_ids.device)
                return torch.cat([input_ids, continuation.repeat(input_ids.shape[0], 1)], dim=1)

        model = Model()
        result = _decode_actions_batched(model, [
            [257, 65],
            [257] * 100,
            [257] * 99,
            [257, 69],
        ], torch.device("cpu"), batch_size=4)
        self.assertEqual(result, ["in", None, "in", "in"])
        self.assertEqual(model.calls, [(2, 96), (1, 1)])

    def test_attempt_status_uses_only_public_batch_attempt_record(self) -> None:
        self.assertEqual(_attempt_status({"parsed_action": None, "accepted": False}), "malformed")
        self.assertEqual(_attempt_status({"parsed_action": 7, "accepted": False}), "invalid")
        self.assertEqual(_attempt_status({"parsed_action": 7, "accepted": True}), "accepted")

    def test_executor_steps_through_public_attempt_api_not_parser_or_raw_step(self) -> None:
        class Batch:
            instances: list["Batch"] = []

            def __init__(self, *_args, **_kwargs):
                self.attempts: list[str] = []
                self.complete = False
                Batch.instances.append(self)

            def observations(self, _rendering):
                return ["observation"]

            def step_attempts(self, attempts, _rendering):
                self.attempts.extend(attempts)
                self.complete = True
                return [{"parsed_action": 1, "accepted": True}]

            def step(self, _actions):
                raise AssertionError("evaluation must not call raw Batch.step")

            def done(self):
                return [self.complete]

            def privileged_outcomes(self):
                return [(True, True, 9)]

        class Model:
            class config:
                max_position_embeddings = 64
                pad_token_id = 256

            def generate(self, input_ids, **_kwargs):
                continuation = torch.tensor([[105, 110, END_TURN]], dtype=torch.long, device=input_ids.device)
                return torch.cat([input_ids, continuation], dim=1)

        world_py = SimpleNamespace(Batch=Batch, FamilyParams=lambda **kwargs: kwargs)
        params = {
            "n_hyp": 1, "n_probe": 1, "n_evidence": 1, "cost_lo": 1,
            "cost_hi": 1, "budget_slack": 1, "min_depth": 1,
            "step_slack": 1, "variant": "default",
        }
        with mock.patch.dict("sys.modules", {"world_py": world_py}), mock.patch(
            "step1_experiments.evaluate._teacher_cost", return_value=4,
        ):
            rows = _execute_batched(Model(), params, seed=3, count=1, rendering="a", device=torch.device("cpu"))
        self.assertEqual(Batch.instances[0].attempts, ["in"])
        self.assertEqual(rows, [{
            "success": True, "spent": 9, "malformed": 0, "invalid": 0,
            "steps": 1, "teacher_spent": 4,
        }])

    def test_matched_sets_and_finite_contract_have_no_capability_threshold(self) -> None:
        config = {
            "run": {"root_seed": 7},
            "world": {"rendering": "a", "validation_episodes": 2, "structural_episodes": 2, "transfer_episodes": 2, "n_hyp": 6},
        }
        sets = _matched_sets(config)
        self.assertEqual(sets["rendering_b"][1], sets["validation"][1])
        self.assertEqual(sets["rendering_b"][4]["label"], "zero_shot_rendering_transfer")
        values = {
            "success_rate": 0.0, "failure_rate": 1.0, "malformed_action_rate": 1.0,
            "invalid_action_rate": 0.0, "mean_spent": 0.0,
            "mean_success_excess_cost": None, "mean_steps": 0.0,
        }
        report = {
            "teacher_forced_action_nll": 3.0,
            "sets": {name: {"comparison": comparison, "metrics": values} for name, (*_, comparison) in sets.items()},
        }
        assert_evaluation_contract(report)


if __name__ == "__main__":
    unittest.main()
