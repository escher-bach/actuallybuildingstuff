from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from step2_experiments.data import generate_torch_batch
from step2_experiments.model import (
    Step2Config,
    Step2ForTrajectoryPrediction,
    assert_selected_parameter_report,
    assert_selected_profile,
    parameter_report,
)
from step2_experiments.train import (
    build_optimizer,
    closed_loop_eval,
    model_checksum,
    run_overfit_gate,
    scheduler_last_epoch,
)


ROOT = Path(__file__).resolve().parents[3]
SELECTED_CONFIG = ROOT / "step2" / "artifacts" / "icrt-derived-small" / "model_config.json"
WORLD = {
    "d_min": 1,
    "d_max": 4,
    "gain_min": 0.75,
    "gain_max": 1.25,
    "action_limit": 0.20,
    "calibration_pulse": 0.10,
    "max_control_steps": 4,
}


def tiny_config() -> Step2Config:
    return Step2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        attention_heads=4,
        max_position_embeddings=192,
        num_roles=10,
        payload_dim=8,
        action_horizon=16,
        vision_resampler_tokens=8,
    )


class ModelIntegrationTests(unittest.TestCase):
    def test_selected_exact_profile_runs_real_world_backward(self) -> None:
        torch.manual_seed(1)
        config = Step2Config.from_project_json(SELECTED_CONFIG)
        assert_selected_profile(config)
        model = Step2ForTrajectoryPrediction(config)
        report = parameter_report(model)
        assert_selected_parameter_report(report)
        batch, _ = generate_torch_batch(
            seed=101,
            start_index=3,
            batch_size=1,
            max_tokens=192,
            world=WORLD,
        )
        output = model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.action_head.weight.grad)
        self.assertTrue(torch.isfinite(model.action_head.weight.grad).all())

    def test_fixed_real_batch_overfits_with_tiny_core(self) -> None:
        torch.manual_seed(2)
        model = Step2ForTrajectoryPrediction(tiny_config())
        batch, _ = generate_torch_batch(
            seed=202,
            start_index=0,
            batch_size=2,
            max_tokens=192,
            world=WORLD,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
        model.eval()
        with torch.no_grad():
            initial = model(**batch).loss.item()
        model.train()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = model(**batch).loss.item()
        self.assertLess(final, 0.75 * initial, (initial, final))

    def test_standard_pretrained_round_trip_preserves_outputs(self) -> None:
        torch.manual_seed(3)
        model = Step2ForTrajectoryPrediction(tiny_config()).eval()
        batch, _ = generate_torch_batch(
            seed=303,
            start_index=1,
            batch_size=1,
            max_tokens=192,
            world=WORLD,
        )
        with torch.no_grad():
            before = model(**batch).action_predictions
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = Step2ForTrajectoryPrediction.from_pretrained(directory).eval()
            with torch.no_grad():
                after = restored(**batch).action_predictions
        self.assertTrue(torch.equal(before, after))

    def test_random_visual_path_has_expected_shape_and_gradient(self) -> None:
        torch.manual_seed(4)
        model = Step2ForTrajectoryPrediction(tiny_config())
        images = torch.randn(2, 3, 32, 32)
        visual = model.encode_images(images)
        self.assertEqual(tuple(visual.shape), (2, 8, 64))
        visual.square().mean().backward()
        self.assertIsNotNone(model.visual_patch_stem.weight.grad)
        self.assertTrue(torch.isfinite(model.visual_patch_stem.weight.grad).all())

    def test_accelerate_gate_runs_on_real_rust_batch(self) -> None:
        accelerator = Accelerator(cpu=True)
        config = {
            "run": {
                "seed": 5,
                "overfit_per_device_batch_size": 1,
                "learning_rate": 2.0e-3,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "overfit_updates": 12,
                "overfit_warmup_updates": 2,
                "overfit_required_fraction": 0.90,
            },
            "model": {"sequence_length": 192},
            "world": {**WORLD, "overfit_seed": 505},
        }
        result, model = run_overfit_gate(accelerator, tiny_config(), config)
        self.assertTrue(result["passed"])
        del model
        accelerator.free_memory()

    def test_model_and_oracle_closed_loop_paths_share_rust_rollout(self) -> None:
        accelerator = Accelerator(cpu=True)
        model = Step2ForTrajectoryPrediction(tiny_config()).to(accelerator.device)
        config = {
            "model": {"sequence_length": 192},
            "world": {**WORLD, "d_min": 1, "d_max": 4},
        }
        model_result = closed_loop_eval(
            accelerator,
            model,
            config,
            seed=606,
            start_index=0,
            episodes_per_rank=4,
        )
        oracle_result = closed_loop_eval(
            accelerator,
            model,
            config,
            seed=606,
            start_index=0,
            episodes_per_rank=4,
            use_oracle=True,
        )
        self.assertEqual(model_result["episodes"], 4)
        self.assertEqual(oracle_result["success_rate"], 1.0)
        accelerator.free_memory()

    def test_accelerate_state_restore_is_exact(self) -> None:
        accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
        model = Step2ForTrajectoryPrediction(tiny_config())
        optimizer = build_optimizer(model, 1.0e-3, 0.0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 1, 4)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        batch, _ = generate_torch_batch(
            seed=707,
            start_index=0,
            batch_size=1,
            max_tokens=192,
            world=WORLD,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        before = model_checksum(accelerator.unwrap_model(model))
        with tempfile.TemporaryDirectory() as directory:
            accelerator.save_state(directory)
            with torch.no_grad():
                next(accelerator.unwrap_model(model).parameters()).add_(1.0)
            accelerator.load_state(directory)
        after = model_checksum(accelerator.unwrap_model(model))
        self.assertTrue(torch.equal(before, after))
        accelerator.free_memory()

    def test_manual_scheduler_advances_once_per_global_update(self) -> None:
        accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
        model = Step2ForTrajectoryPrediction(tiny_config())
        optimizer = build_optimizer(model, 1.0e-3, 0.0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 1, 4)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        observed = []
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            next(model.parameters()).sum().backward()
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
            observed.append(scheduler_last_epoch(scheduler))
        self.assertEqual(observed, [1, 2, 3, 4])
        accelerator.free_memory()


if __name__ == "__main__":
    unittest.main()
