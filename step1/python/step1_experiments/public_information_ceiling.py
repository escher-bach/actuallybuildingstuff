"""Audit the policy ceiling available through the learner's serialized bytes.

The policy in this module receives exactly the causal-LM prefix assembled by
the evaluator: transport tokens plus rendered observation and action bytes. It
does not receive a ``world_py.Batch``, an ``Instance``, an evidence table,
probe costs, the latent truth, teacher targets, or a consistent-hypothesis set.
The driver uses the public parser to pass the returned action string back to
the world and consults the privileged verifier only after an episode ends.

This deliberately characterizes the deployed generator rather than the world
that was intended. With ``n_evidence = 2``, the sampler's anti-identity rewrite
forces hypothesis zero to return evidence one on every probe and hypothesis
one to return evidence zero on every probe. The remaining four hypotheses are
exchangeable. Consequently the public evidence channel has a semantic ceiling
of 1/2: identify either special hypothesis, or guess one of the four ordinary
hypotheses. Four inspections are the largest uniformly safe cap because every
accepted instance has ``min_depth >= 2`` and therefore ``step_limit >= 5``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .protocol import ACTION, BOS, END_TURN, OBS, encode_bytes
from .world_config import family_from_config as _family


Prefix = tuple[int, ...]
INSPECT_RE = re.compile(r"^inspect\(probe_(\d+)\)$")
COMMIT_RE = re.compile(r"^commit\(cause_(\d+)\)$")
SEEN_RE = re.compile(r"^SEEN probe_\d+ => mark_([a-z0-9_]+)$")


def append_observation(prefix: Prefix, observation: str) -> Prefix:
    """Append the exact evaluator-side observation framing."""
    return prefix + (OBS, *encode_bytes(observation), ACTION)


def append_action(prefix: Prefix, action: str) -> Prefix:
    """Append the exact evaluator-side action framing."""
    return prefix + (*encode_bytes(action), END_TURN)


def current_observation(prefix: Sequence[int]) -> str:
    """Decode only the current rendered observation from a model prefix."""
    try:
        start = len(prefix) - 1 - list(reversed(prefix)).index(OBS)
        end = start + 1 + list(prefix[start + 1 :]).index(ACTION)
    except ValueError as error:
        raise ValueError("prefix does not contain a framed current observation") from error
    payload = prefix[start + 1 : end]
    if any(token < 0 or token >= 256 for token in payload):
        raise ValueError("current observation contains a transport token")
    return bytes(payload).decode("utf-8", errors="strict")


def available_actions(observation: str) -> tuple[str, ...]:
    """Read Rendering A's public action list without consulting the world."""
    line = next((line for line in observation.splitlines() if line.startswith("AVAILABLE ")), None)
    if line is None:
        raise ValueError("Rendering A observation has no AVAILABLE line")
    payload = line.removeprefix("AVAILABLE ")
    return tuple(part.strip() for part in payload.split(", ") if part.strip())


def observed_evidence(observation: str) -> tuple[str, ...]:
    """Return the public evidence words already present in the transcript."""
    values: list[str] = []
    for line in observation.splitlines():
        match = SEEN_RE.match(line)
        if match:
            values.append(match.group(1))
    return tuple(values)


def _numbered(action: str, pattern: re.Pattern[str]) -> int:
    match = pattern.match(action)
    if not match:
        raise ValueError(f"action does not match the expected public grammar: {action!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class PublicEvidencePolicy:
    """Bayes rule induced by the sampler's learner-visible label asymmetry.

    ``choose`` intentionally has one input: the exact token prefix. The class
    stores no world object and no world parameters. On mixed evidence,
    hypotheses one and two (zero- and one-indexed internally) are impossible,
    while causes 3--6 remain exchangeable, so any one of them is Bayes-equivalent.
    On a constant history, the always-amber or always-blue cause is preferred.
    """

    max_probes: int = 4
    inspect_selection: str = "first_rendered"

    def choose(self, prefix: Prefix) -> str:
        observation = current_observation(prefix)
        actions = available_actions(observation)
        inspections = tuple(action for action in actions if INSPECT_RE.match(action))
        commitments = tuple(action for action in actions if COMMIT_RE.match(action))
        evidence = observed_evidence(observation)

        if not commitments:
            raise ValueError("running observation exposes no commitment action")

        mixed = len(set(evidence)) > 1
        if not mixed and len(evidence) < self.max_probes and inspections:
            return self._select_inspection(inspections)

        if mixed:
            return self._commit_number(commitments, 3)
        if evidence and all(value == "amber" for value in evidence):
            return self._commit_number(commitments, 1)
        if evidence and all(value == "blue" for value in evidence):
            return self._commit_number(commitments, 2)
        if evidence:
            raise ValueError(f"unexpected binary-evidence history: {evidence!r}")
        return self._commit_number(commitments, 1)

    def _select_inspection(self, inspections: tuple[str, ...]) -> str:
        if self.inspect_selection == "first_rendered":
            return inspections[0]
        if self.inspect_selection == "last_rendered":
            return inspections[-1]
        if self.inspect_selection == "lowest_id":
            return min(inspections, key=lambda action: _numbered(action, INSPECT_RE))
        if self.inspect_selection == "highest_id":
            return max(inspections, key=lambda action: _numbered(action, INSPECT_RE))
        raise ValueError(f"unknown inspection selection: {self.inspect_selection!r}")

    @staticmethod
    def _commit_number(commitments: tuple[str, ...], number: int) -> str:
        for action in commitments:
            if _numbered(action, COMMIT_RE) == number:
                return action
        raise ValueError(f"cause_{number} is absent from the public action list")


def idealized_success_after_probes(probes: int) -> float:
    """Closed form for the deployed 2-special/4-exchangeable generator.

    The expression assumes independent fair evidence for the four ordinary
    hypotheses. Structural rejection perturbs it slightly, so retained claims
    use executed-world measurements rather than substituting this formula.
    """
    if probes < 0:
        raise ValueError("probes cannot be negative")
    if probes == 0:
        return 1.0 / 6.0
    return 0.5 - (2.0 ** (1 - probes)) / 6.0


def semantic_public_upper_bound() -> float:
    """Maximum success once two special and four exchangeable labels remain."""
    return 0.5


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return centre - margin, centre + margin


def _terminal_metrics(outcomes: Iterable[tuple]) -> dict:
    rows = list(outcomes)
    successes = sum(bool(row[0] and row[1]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_wilson_95": [low, high],
        "mean_spent": statistics.fmean(float(row[2]) for row in rows),
        "mean_steps": statistics.fmean(float(row[3]) for row in rows),
    }


def _drive_batch(batch, params: dict, policy: PublicEvidencePolicy) -> dict:
    """Execute a batch while keeping the policy on the serialized boundary."""
    from world_py import parse_action

    prefixes: list[Prefix] = [(BOS,) for _ in range(batch.n_episodes)]
    turns = 0
    while True:
        live = batch.live_episode_indices()
        if not live:
            break
        observations = batch.observations("a")
        encoded: list[int] = []
        rendered: list[str] = []
        for index in live:
            prefixes[index] = append_observation(prefixes[index], observations[index])
            action = policy.choose(prefixes[index])
            if action not in available_actions(observations[index]):
                raise AssertionError(f"observation-only policy selected an unavailable action: {action!r}")
            rendered.append(action)
            encoded.append(parse_action(action, params["n_probe"], params["n_hyp"], "a"))
        batch.step(encoded)
        for index, action in zip(live, rendered):
            prefixes[index] = append_action(prefixes[index], action)
        turns += 1
        if turns > params["n_probe"] + params["n_hyp"] + params["step_slack"] + 4:
            raise AssertionError("public policy exceeded the declared rollout safety bound")
    return _terminal_metrics(batch.privileged_outcomes())


def evaluate_seed_range(params: dict, seed: int, episodes: int, policy: PublicEvidencePolicy) -> dict:
    """Match the frozen evaluator: one independently seeded Batch per episode."""
    from world_py import Batch

    rows: list[tuple] = []
    for index in range(episodes):
        batch = Batch(_family(params), seed=seed + index, n_episodes=1)
        _drive_batch(batch, params, policy)
        rows.extend(batch.privileged_outcomes())
    return _terminal_metrics(rows)


def audit(params: dict, validation_seed: int, validation_episodes: int, population_seed: int, population_episodes: int) -> dict:
    caps: dict[str, dict] = {}
    for cap in range(5):
        policy = PublicEvidencePolicy(max_probes=cap)
        item = evaluate_seed_range(params, validation_seed, validation_episodes, policy)
        item["idealized_iid_success"] = idealized_success_after_probes(cap)
        caps[str(cap)] = item

    selected = PublicEvidencePolicy(max_probes=4)
    # Match evaluate._execute_batched exactly.  A single multi-episode Batch
    # walks accepted candidate indices under one root seed and is a different
    # sampling scheme; it is useful for quick smoke tests but not a retained
    # population estimate of evaluator behavior.
    population = evaluate_seed_range(params, population_seed, population_episodes, selected)
    return {
        "contract": "step1_public_information_policy_audit_v1",
        "policy_boundary": {
            "input": "exact causal-LM token prefix: BOS/OBS/ACTION/END_TURN framing plus UTF-8 bytes",
            "action_source": "rendered AVAILABLE line only",
            "forbidden_policy_inputs": [
                "Instance", "evidence table", "probe_cost", "truth", "consistent()",
                "teacher targets", "valid_actions()", "replay key",
            ],
            "privileged_use": "terminal verifier outcome only, after policy execution, for scoring",
        },
        "generator_diagnosis": {
            "n_evidence": params["n_evidence"],
            "special_public_semantics": {
                "cause_1": "always mark_amber",
                "cause_2": "always mark_blue",
                "cause_3_through_cause_6": "exchangeable random binary columns",
            },
            "semantic_public_upper_bound": semantic_public_upper_bound(),
            "upper_bound_scope": (
                "causal evidence channel under exchangeability; the raw serializer also contains a "
                "pseudorandom action-order fingerprint and therefore has no meaningful seed-free "
                "transductive ceiling"
            ),
        },
        "frozen_validation": {
            "seed_policy": "Batch(seed=validation_seed + episode_index, n_episodes=1)",
            "seed": validation_seed,
            "probe_cap_sweep": caps,
        },
        "population_replication": {
            "seed_policy": "Batch(seed=population_seed + episode_index, n_episodes=1)",
            "seed": population_seed,
            "policy": {"max_probes": 4, "inspect_selection": "first_rendered", "stop_on_mixed": True},
            **population,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--population-seed", type=int, default=31_260_811)
    parser.add_argument("--population-episodes", type=int, default=100_000)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    params = config["world"]
    if params["rendering"] != "a" or params["n_hyp"] != 6 or params["n_evidence"] != 2:
        raise ValueError("the v1 audit is frozen to Rendering A with six hypotheses and binary evidence")
    validation_seed = args.validation_seed or config["run"]["root_seed"] + 1_000_000
    validation_episodes = args.validation_episodes or params["validation_episodes"]
    result = audit(params, validation_seed, validation_episodes, args.population_seed, args.population_episodes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
