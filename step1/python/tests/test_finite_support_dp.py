from __future__ import annotations

import unittest
from dataclasses import dataclass

from step1_experiments.finite_support_dp import (
    ExactFiniteSupportDP,
    Transition,
    transductive_empirical_optimum,
)
from step1_experiments.protocol import ACTION, BOS, OBS


@dataclass(frozen=True)
class ToyParticle:
    identity: int
    prefix: tuple[int, ...]


class ExactFiniteSupportDPContracts(unittest.TestCase):
    def test_probe_can_split_an_ambiguous_exact_prefix(self) -> None:
        root = (BOS, OBS, ord("r"), ACTION)
        left = (BOS, OBS, ord("r"), ACTION, ord("p"), OBS, ord("0"), ACTION)
        right = (BOS, OBS, ord("r"), ACTION, ord("p"), OBS, ord("1"), ACTION)
        particles = (ToyParticle(0, root), ToyParticle(1, root))

        def legal(prefix: tuple[int, ...]) -> tuple[str, ...]:
            return ("commit_0", "commit_1", "probe") if prefix == root else ("commit_0", "commit_1")

        def transition(particle: ToyParticle, action: str) -> Transition[ToyParticle]:
            if action == "probe":
                return Transition(successor=ToyParticle(particle.identity, left if particle.identity == 0 else right))
            chosen = int(action[-1])
            return Transition(terminal_correct=chosen == particle.identity)

        solver = ExactFiniteSupportDP[ToyParticle](
            prefix_of=lambda particle: particle.prefix,
            legal_actions=legal,
            transition=transition,
            action_priority=lambda action: (action == "probe", action),
        )
        successes, forest = solver.solve(particles)
        self.assertEqual(successes, 2)
        self.assertEqual(len(forest), 1)
        self.assertEqual(forest[0].action, "probe")
        self.assertEqual(len(forest[0].children), 2)
        self.assertEqual(sum(child.optimal_successes for child in forest[0].children), 2)

    def test_unique_prefixes_permit_transductive_memorization(self) -> None:
        particles = tuple(
            ToyParticle(index, (BOS, OBS, ord(str(index)), ACTION)) for index in range(3)
        )

        solver = ExactFiniteSupportDP[ToyParticle](
            prefix_of=lambda particle: particle.prefix,
            legal_actions=lambda _prefix: ("commit_0", "commit_1", "commit_2"),
            transition=lambda particle, action: Transition(
                terminal_correct=int(action[-1]) == particle.identity
            ),
            action_priority=lambda action: action,
        )
        successes, forest = solver.solve(particles)
        self.assertEqual(successes, 3)
        self.assertEqual(len(forest), 3)
        self.assertEqual({node.action for node in forest}, {"commit_0", "commit_1", "commit_2"})

    def test_transition_requires_exactly_one_outcome_kind(self) -> None:
        with self.assertRaises(ValueError):
            Transition[ToyParticle]()
        with self.assertRaises(ValueError):
            Transition(successor=ToyParticle(0, (BOS,)), terminal_correct=True)

    def test_real_backend_is_explicitly_transductive(self) -> None:
        params = {
            "n_hyp": 6,
            "n_probe": 5,
            "n_evidence": 2,
            "cost_lo": 1,
            "cost_hi": 3,
            "budget_slack": 1,
            "min_depth": 2,
            "step_slack": 2,
            "variant": "irreversible",
            "rendering": "a",
        }
        result = transductive_empirical_optimum(params, seed=91_000_000, episodes=8)
        self.assertEqual(result["contract"], "step1_transductive_empirical_optimum_v1")
        self.assertIn("not a population", result["scientific_status"])
        self.assertEqual(result["support"]["episodes"], 8)
        self.assertEqual(result["optimum"]["successes"], 8)
        self.assertEqual(result["policy_boundary"]["key"], "exact causal-LM token prefix")


if __name__ == "__main__":
    unittest.main()
