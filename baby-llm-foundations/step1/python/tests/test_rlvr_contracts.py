from __future__ import annotations

import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from step1_experiments.data import ACTION, BOS, END_TURN, EOS, OBS, PAD, encode_bytes
from step1_experiments.evaluate import EVALUATION_METRIC_NAMES
from step1_experiments.rlvr import (
    REWARD_FUNCTION_NAMES,
    RLVR_CONTRACT,
    WorldRollout,
    assert_rlvr_report_contract,
    episode_key,
    load_rlvr_config,
    parse_episode_key,
    rlvr_plan,
    verified_protocol_failure,
    verified_spend,
    verified_success,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "seed0": PROJECT_ROOT / "step1/configs/kaggle/t4x2_rlvr_seed0.toml",
    "smoke": PROJECT_ROOT / "step1/configs/kaggle/t4x2_rlvr_smoke.toml",
    "warmstart": PROJECT_ROOT / "step1/configs/kaggle/t4x2_rlvr_warmstart_seed0.toml",
    "klanchor": PROJECT_ROOT / "step1/configs/kaggle/t4x2_rlvr_klanchor_seed0.toml",
    "bigstep": PROJECT_ROOT / "step1/configs/kaggle/t4x2_rlvr_bigstep_seed0.toml",
}
LAUNCHER = PROJECT_ROOT / "step1/kaggle/step1_t4x2_launcher.ipynb"
DENSE_CONFIG = PROJECT_ROOT / "step1/configs/kaggle/t4x2_dense_seed0.toml"
HAS_WORLD = importlib.util.find_spec("world_py") is not None


class _ScriptedPolicy(torch.nn.Module):
    """Emits one declared byte action per generation call, per live row."""

    def __init__(self, script: list[str | None]):
        super().__init__()
        self.script = script
        self.config = SimpleNamespace(pad_token_id=PAD, max_position_embeddings=2048)

    def generate(self, input_ids=None, attention_mask=None, generation_config=None, **_kwargs):
        payloads = []
        for row in range(input_ids.shape[0]):
            text = self.script[row]
            ids = [65] * generation_config.max_new_tokens if text is None else encode_bytes(text) + [END_TURN]
            payloads.append(ids[: generation_config.max_new_tokens])
        width = max(len(payload) for payload in payloads)
        prompt_width = input_ids.shape[1]
        output = torch.full((input_ids.shape[0], prompt_width + width), PAD, dtype=torch.long)
        output[:, :prompt_width] = input_ids
        for row, payload in enumerate(payloads):
            output[row, prompt_width:prompt_width + len(payload)] = torch.tensor(payload, dtype=torch.long)
        return output


class _LocalRollout(WorldRollout):
    def _generation_model(self, trainer):
        return contextlib.nullcontext(trainer.model_wrapped)


def _stub_trainer(model) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(device=torch.device("cpu"), fp16=False),
        model_wrapped=model,
        accelerator=None,
    )


def _metrics() -> dict:
    return {name: (None if name == "mean_success_excess_cost" else 0.0) for name in EVALUATION_METRIC_NAMES}


def _report(config: dict, plan) -> dict:
    final = plan.milestones_updates[-1]
    sets = ("validation", "structural", "rendering_b", "reversible_control")
    return {
        "contract": RLVR_CONTRACT,
        "experiment_config_sha256": config["_meta"]["hash"],
        "algorithm": {
            "trainer": "trl.GRPOTrainer",
            "signal": "outcome_only_privileged_verifier",
            "reward_functions": list(REWARD_FUNCTION_NAMES),
            "reward_weights": [1.0, 0.0, 0.0],
        },
        "plan": plan.report(),
        "initialization": {"root_seed": plan.root_seed},
        "budget_accounting": {
            "optimizer_updates": plan.max_steps,
            "rollout_episodes": plan.rollout_episode_budget,
            "world_transitions": 17,
            "generated_action_tokens": 23,
            "wall_clock_seconds": 1.5,
        },
        "training_signal": {"logged_updates": plan.max_steps},
        "milestones": [
            {
                "budget_updates": step,
                "serialization": {"exact": True},
                "teacher_forced_action_nll": 0.0853,
                "evaluation": {
                    name: {"comparison": {}, "metrics": _metrics()}
                    for name in (sets if step == final else ("validation",))
                },
            }
            for step in plan.milestones_updates
        ],
    }


class RlvrPlanContracts(unittest.TestCase):
    def test_seed0_budget_matches_the_dense_sequence_budget(self) -> None:
        config = load_rlvr_config(CONFIGS["seed0"])
        plan = rlvr_plan(config, world_size=2)
        self.assertEqual(plan.generation_batch_size, 64)
        self.assertEqual(plan.worlds_per_update, 8)
        self.assertEqual(plan.max_steps, 763)
        # The dense seed-0 arm consumed 100,007,936 nominal input tokens in
        # 2,048-token sequences; RLVR rolls out exactly that many episodes.
        self.assertEqual(plan.rollout_episode_budget, 100_007_936 // 2048)
        self.assertEqual(plan.rollout_episode_budget, 48_832)
        self.assertEqual(plan.distinct_train_worlds, 6_104)
        self.assertEqual(plan.milestones_updates[-1], plan.max_steps)

    def test_every_config_declares_a_complete_and_consistent_plan(self) -> None:
        for name, path in CONFIGS.items():
            with self.subTest(config=name):
                config = load_rlvr_config(path)
                plan = rlvr_plan(config, world_size=2)
                self.assertEqual(plan.rollout_episode_budget, plan.max_steps * plan.generation_batch_size)
                self.assertEqual(plan.generation_batch_size % plan.num_generations, 0)
                self.assertEqual(plan.milestones_updates[-1], plan.max_steps)
                self.assertLessEqual(
                    config["rollout"]["max_completion_tokens"] + config["rollout"]["max_prompt_tokens"],
                    config["world"]["context_length"],
                )

    def test_training_worlds_use_a_seed_band_disjoint_from_every_evaluation_set(self) -> None:
        config = load_rlvr_config(CONFIGS["seed0"])
        plan = rlvr_plan(config, world_size=2)
        used = set()
        for offset, count in ((1_000_000, 1024), (2_000_000, 1024), (3_000_000, 1024)):
            used.update(range(plan.root_seed + offset, plan.root_seed + offset + count))
        training = set(range(plan.train_seed_base, plan.train_seed_base + plan.distinct_train_worlds))
        self.assertEqual(training & used, set())

    def test_evaluation_worlds_are_the_dense_seed0_worlds(self) -> None:
        dense = load_rlvr_config(DENSE_CONFIG)
        rlvr = load_rlvr_config(CONFIGS["seed0"])
        self.assertEqual(dense["run"]["root_seed"], rlvr["run"]["root_seed"])
        for field in ("n_hyp", "n_probe", "n_evidence", "cost_lo", "cost_hi", "budget_slack",
                      "min_depth", "step_slack", "variant", "rendering", "context_length",
                      "validation_episodes", "structural_episodes", "transfer_episodes"):
            self.assertEqual(dense["world"][field], rlvr["world"][field], field)

    def test_plan_rejects_incoherent_group_budget_and_milestones(self) -> None:
        base = load_rlvr_config(CONFIGS["seed0"])
        for mutate in (
            lambda config: config["grpo"].update(num_generations=7),
            lambda config: config["grpo"].update(milestones_updates=[95, 191]),
            lambda config: config["grpo"].update(milestones_updates=[191, 95, 763]),
            lambda config: config["rollout"].update(max_completion_tokens=2048),
            lambda config: config["run"].update(arm="whatever_gets_a_result"),
        ):
            config = json.loads(json.dumps(base))
            mutate(config)
            with self.assertRaises(ValueError):
                rlvr_plan(config, world_size=2)

    def test_episode_keys_round_trip_and_reject_foreign_prompts(self) -> None:
        self.assertEqual(parse_episode_key(episode_key(24_260_811)), 24_260_811)
        for bad in ("", "step1-world/", "step1-world/-1", "BUDGET 7", "step1-world/1/2"):
            with self.assertRaises(ValueError):
                parse_episode_key(bad)


class RlvrRewardContracts(unittest.TestCase):
    def test_success_reward_is_the_evaluator_success_rule(self) -> None:
        rollout = {
            "outcome_success": [True, False, False, False],
            "outcome_spent": [4, 2, 0, 7],
            "rollout_malformed": [0, 0, 1, 0],
            "rollout_invalid": [0, 1, 0, 0],
        }
        self.assertEqual(verified_success(**rollout), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(verified_spend(**rollout), [-4.0, -2.0, -0.0, -7.0])
        self.assertEqual(verified_protocol_failure(**rollout), [0.0, 1.0, 1.0, 0.0])

    def test_reward_function_names_are_the_logged_contract(self) -> None:
        self.assertEqual(
            REWARD_FUNCTION_NAMES,
            (verified_success.__name__, verified_spend.__name__, verified_protocol_failure.__name__),
        )


@unittest.skipUnless(HAS_WORLD, "world_py is built by the runner before tests execute")
class RlvrRolloutContracts(unittest.TestCase):
    def _rollout(self, script: list[str | None], overrides: dict | None = None):
        config = load_rlvr_config(CONFIGS["smoke"])
        for section, values in (overrides or {}).items():
            config[section].update(values)
        plan = rlvr_plan(config, world_size=2)
        rollout = _LocalRollout(config, plan)
        prompts = [episode_key(plan.train_seed_base + index) for index in range(len(script))]
        return rollout, rollout(prompts, _stub_trainer(_ScriptedPolicy(script))), plan

    def test_observation_frames_never_carry_gradient(self) -> None:
        rollout, output, plan = self._rollout(["inspect(probe_1)"] * 3)
        for ids, mask in zip(output["completion_ids"], output["env_mask"]):
            self.assertEqual(len(ids), len(mask))
            for start, token in enumerate(ids):
                if token == OBS and mask[start] == 0:
                    end = next(index for index in range(start, len(ids)) if ids[index] == ACTION)
                    self.assertTrue(all(value == 0 for value in mask[start:end + 1]))
            self.assertEqual(mask[-1], 0)
            self.assertEqual(ids[-1], EOS)
        for ids in output["prompt_ids"]:
            self.assertEqual((ids[0], ids[-1]), (BOS, ACTION))
            self.assertEqual(ids[1], OBS)
        self.assertEqual(rollout.counters.episodes, 3)
        self.assertEqual(rollout.counters.world_transitions, 3)

    def test_malformed_and_unterminated_actions_end_only_their_own_episode(self) -> None:
        # Row 0 commits and terminates; row 1 emits unparseable bytes; row 2
        # never emits END_TURN within its allowance.
        rollout, output, _plan = self._rollout(["commit(cause_1)", "not-an-action", None])
        self.assertEqual(output["rollout_malformed"], [0, 1, 1])
        self.assertEqual(output["rollout_invalid"], [0, 0, 0])
        self.assertEqual(output["rollout_turns"], [1, 1, 1])
        self.assertTrue(output["outcome_terminated"][0])
        self.assertFalse(any(output["outcome_terminated"][1:]))
        # Only the accepted commitment reached the world.
        self.assertEqual(rollout.counters.world_transitions, 1)

    def test_repeated_probe_is_an_invalid_action_and_not_a_transition(self) -> None:
        # The scripted policy repeats its action every turn, so the second turn
        # inspects an already-inspected probe.
        rollout, output, _plan = self._rollout(["inspect(probe_1)"])
        self.assertEqual(output["rollout_turns"], [2])
        self.assertEqual(output["rollout_invalid"], [1])
        self.assertEqual(output["rollout_malformed"], [0])
        self.assertEqual(rollout.counters.world_transitions, 1)
        self.assertFalse(output["outcome_success"][0])

    def test_completion_budget_is_never_exceeded_and_marks_unfinished_episodes(self) -> None:
        _rollout, output, plan = self._rollout(
            ["inspect(probe_1)"] * 2, {"rollout": {"max_completion_tokens": 300}}
        )
        for ids, exhausted, prompt in zip(output["completion_ids"], output["rollout_exhausted"], output["prompt_ids"]):
            self.assertLessEqual(len(ids), 300)
            self.assertLessEqual(len(prompt) + len(ids), plan.context_length)
            # Only an episode that reached a verdict is closed with EOS.
            self.assertEqual(ids[-1] == EOS, not exhausted)


class RlvrReportContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_rlvr_config(CONFIGS["seed0"])
        self.plan = rlvr_plan(self.config, world_size=2)

    def test_well_formed_report_passes(self) -> None:
        assert_rlvr_report_contract(_report(self.config, self.plan), self.config, self.plan)

    def test_report_rejects_short_budgets_missing_evaluation_and_inexact_artifacts(self) -> None:
        for mutate in (
            lambda report: report["budget_accounting"].update(rollout_episodes=17),
            lambda report: report["budget_accounting"].update(optimizer_updates=1),
            lambda report: report["budget_accounting"].pop("world_transitions"),
            lambda report: report["milestones"][-1]["evaluation"].pop("rendering_b"),
            lambda report: report["milestones"][0]["serialization"].update(exact=False),
            lambda report: report["milestones"][1].pop("teacher_forced_action_nll"),
            lambda report: report["milestones"].pop(),
            lambda report: report["algorithm"].update(signal="teacher_preferred_actions"),
            lambda report: report["algorithm"].update(reward_weights=[0.0, 1.0, 0.0]),
            lambda report: report.update(contract="something_else"),
        ):
            report = _report(self.config, self.plan)
            mutate(report)
            with self.assertRaises(AssertionError):
                assert_rlvr_report_contract(report, self.config, self.plan)


class LauncherContracts(unittest.TestCase):
    """The launcher is a launcher: no experiment logic, no published debris."""

    def setUp(self) -> None:
        document = json.loads(LAUNCHER.read_text(encoding="utf-8"))
        self.cells = document["cells"]
        self.text = "\n".join("".join(cell.get("source", [])) for cell in self.cells)

    def test_shape_is_one_markdown_and_three_code_cells(self) -> None:
        self.assertEqual([cell["cell_type"] for cell in self.cells], ["markdown", "code", "code", "code"])

    def test_exactly_one_commit_and_config_placeholder(self) -> None:
        self.assertEqual(self.text.count("__FINAL_COMMIT_SHA__"), 1)
        self.assertEqual(self.text.count("__CONFIG_REL__"), 1)

    def test_ephemeral_source_and_output_only_working_tree(self) -> None:
        self.assertIn('RUNTIME = Path("/tmp/step1-runtime")', self.text)
        self.assertIn("SOURCE = RUNTIME /", self.text)
        self.assertIn("CARGO_HOME", self.text)
        # Nothing may clone or build beneath the published output tree.
        self.assertNotIn('SOURCE = WORKING', self.text)
        self.assertIn('OUTPUT = WORKING / "step1-results"', self.text)

    def test_invokes_only_the_repository_runner(self) -> None:
        self.assertIn("step1_experiments.runner", self.text)
        for forbidden in ("torchrun", "locate_dense_source", "maturin", "assert report", "success_rate"):
            self.assertNotIn(forbidden, self.text)


if __name__ == "__main__":
    unittest.main()
