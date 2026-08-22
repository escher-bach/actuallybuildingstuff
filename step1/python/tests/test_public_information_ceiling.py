from __future__ import annotations

import inspect
import unittest

from step1_experiments.public_information_ceiling import (
    PublicEvidencePolicy,
    append_observation,
    available_actions,
    current_observation,
    idealized_success_after_probes,
    observed_evidence,
    semantic_public_upper_bound,
)
from step1_experiments.protocol import ACTION, BOS, END_TURN, OBS, encode_bytes


def observation(history: tuple[str, ...] = ()) -> str:
    lines = [f"SEEN probe_{index + 1} => mark_{value}" for index, value in enumerate(history)]
    lines.extend([
        "BUDGET 5",
        (
            "AVAILABLE inspect(probe_3), inspect(probe_1), inspect(probe_2), "
            "commit(cause_2), commit(cause_4), commit(cause_3), "
            "commit(cause_6), commit(cause_1), commit(cause_5)"
        ),
        "STATUS running",
    ])
    return "\n".join(lines)


class PublicInformationCeilingContracts(unittest.TestCase):
    def test_policy_accepts_only_the_exact_serialized_prefix(self) -> None:
        signature = inspect.signature(PublicEvidencePolicy.choose)
        self.assertEqual(list(signature.parameters), ["self", "prefix"])
        self.assertEqual(list(PublicEvidencePolicy.__dataclass_fields__), ["max_probes", "inspect_selection"])

    def test_current_observation_is_decoded_from_transport_framing(self) -> None:
        first = append_observation((BOS,), observation())
        prefix = first + tuple(encode_bytes("inspect(probe_3)")) + (END_TURN,)
        prefix = append_observation(prefix, observation(("amber",)))
        self.assertEqual(current_observation(prefix), observation(("amber",)))
        self.assertEqual(prefix[-1], ACTION)
        self.assertIn(OBS, prefix)

    def test_parser_reads_only_public_history_and_available_actions(self) -> None:
        value = observation(("amber", "blue"))
        self.assertEqual(observed_evidence(value), ("amber", "blue"))
        self.assertEqual(available_actions(value)[0], "inspect(probe_3)")

    def test_policy_reproduces_the_deployed_public_bayes_rule(self) -> None:
        policy = PublicEvidencePolicy(max_probes=4)
        self.assertEqual(policy.choose(append_observation((BOS,), observation())), "inspect(probe_3)")
        self.assertEqual(
            policy.choose(append_observation((BOS,), observation(("amber", "blue")))),
            "commit(cause_3)",
        )
        four_amber = observation(("amber", "amber", "amber", "amber"))
        self.assertEqual(policy.choose(append_observation((BOS,), four_amber)), "commit(cause_1)")
        four_blue = observation(("blue", "blue", "blue", "blue"))
        self.assertEqual(policy.choose(append_observation((BOS,), four_blue)), "commit(cause_2)")

    def test_closed_form_matches_the_40_to_45_percent_band(self) -> None:
        self.assertAlmostEqual(idealized_success_after_probes(0), 1 / 6)
        self.assertAlmostEqual(idealized_success_after_probes(2), 5 / 12)
        self.assertAlmostEqual(idealized_success_after_probes(3), 11 / 24)
        self.assertAlmostEqual(idealized_success_after_probes(4), 23 / 48)
        self.assertEqual(semantic_public_upper_bound(), 0.5)


if __name__ == "__main__":
    unittest.main()
