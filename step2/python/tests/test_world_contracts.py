from __future__ import annotations

import unittest

import step2_world_py

from step2_experiments.data import MODEL_FIELDS, tensorize


WORLD = {
    "d_min": 1,
    "d_max": 4,
    "gain_min": 0.75,
    "gain_max": 1.25,
    "action_limit": 0.20,
    "calibration_pulse": 0.10,
    "max_control_steps": 4,
}


class WorldContractTests(unittest.TestCase):
    def test_public_oracle_and_support_validation(self) -> None:
        result = step2_world_py.validate_generated_worlds(
            seed=123,
            start_index=0,
            count=256,
            **WORLD,
        )
        self.assertEqual(result["dimension_counts"][1:5], [64, 64, 64, 64])
        self.assertLess(result["max_oracle_error"], 1.0e-5)
        self.assertGreater(result["action_targets"], 0)
        self.assertGreater(result["outcome_targets"], result["action_targets"])
        self.assertLessEqual(result["max_length"], 192)

    def test_batched_public_tensor_contract(self) -> None:
        raw = step2_world_py.generate_training_batch(
            seed=17,
            start_index=0,
            batch_size=8,
            max_tokens=192,
            **WORLD,
        )
        tensors = tensorize(raw)
        self.assertEqual(set(tensors), set(MODEL_FIELDS))
        self.assertEqual(tuple(tensors["role_ids"].shape), (8, 192))
        self.assertEqual(tuple(tensors["payloads"].shape), (8, 192, 8))
        self.assertEqual(tuple(tensors["action_targets"].shape), (8, 192, 16))
        self.assertTrue((tensors["action_target_mask"].sum(dim=-1) <= 1).all())
        self.assertNotIn("indices", tensors)
        self.assertNotIn("dimensions", tensors)

    def test_truncation_is_a_loud_error(self) -> None:
        with self.assertRaises(ValueError):
            step2_world_py.generate_training_batch(
                seed=17,
                start_index=3,
                batch_size=1,
                max_tokens=16,
                **WORLD,
            )

    def test_privileged_oracle_solves_but_zero_policy_does_not(self) -> None:
        oracle = step2_world_py.RolloutBatch(
            seed=41,
            start_index=0,
            batch_size=16,
            max_tokens=192,
            **WORLD,
        )
        while not oracle.all_done():
            oracle.step(oracle.privileged_oracle_actions())
        oracle_summary = oracle.summary()
        self.assertTrue(all(oracle_summary["success"]))

        zero = step2_world_py.RolloutBatch(
            seed=41,
            start_index=0,
            batch_size=16,
            max_tokens=192,
            **WORLD,
        )
        while not zero.all_done():
            batch = zero.learner_batch()
            actions = [([] if done else [0.0] * dimension) for done, dimension in zip(batch["done"], batch["dimensions"])]
            zero.step(actions)
        self.assertFalse(any(zero.summary()["success"]))


if __name__ == "__main__":
    unittest.main()
