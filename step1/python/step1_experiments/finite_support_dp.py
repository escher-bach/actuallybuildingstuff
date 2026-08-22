"""Exact observation-only planning on a declared finite support.

This module intentionally computes a *transductive empirical optimum*, not a
population or scientific capability ceiling.  It is useful precisely because
it demonstrates how a high-cardinality serialized fingerprint can make a
same-support observation-only policy nearly perfect.

The planner's state key is the exact causal-LM token prefix.  Its black-box
backend recreates evaluator-matched worlds, replays rendered public actions,
and reads the verifier only after termination.  It never calls ``Instance``,
``consistent()``, ``valid_actions()``, teacher targets, or ``replay_key()``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, Iterable, Sequence, TypeVar

from .public_information_ceiling import (
    COMMIT_RE,
    INSPECT_RE,
    Prefix,
    append_action,
    append_observation,
    available_actions,
    current_observation,
)
from .protocol import BOS
from .world_config import family_from_config as _family


ParticleT = TypeVar("ParticleT")


@dataclass(frozen=True)
class Transition(Generic[ParticleT]):
    """One deterministic particle transition.

    Exactly one of ``successor`` and ``terminal_correct`` is populated.
    """

    successor: ParticleT | None = None
    terminal_correct: bool | None = None

    def __post_init__(self) -> None:
        if (self.successor is None) == (self.terminal_correct is None):
            raise ValueError("transition must be either terminal or nonterminal")


@dataclass(frozen=True)
class PolicyNode:
    """The optimal decision for one exact-prefix posterior bucket."""

    prefix: Prefix
    support_size: int
    optimal_successes: int
    action: str
    children: tuple["PolicyNode", ...] = ()


@dataclass
class SolverStats:
    decision_nodes_evaluated: int = 0
    action_evaluations: int = 0
    particle_transitions: int = 0
    upper_bound_shortcuts: int = 0
    max_evaluated_depth: int = 0


def _default_action_priority(action: str) -> tuple[int, int | str]:
    """Try terminal commitments first, then probes, for cheap singleton solves."""
    commit = COMMIT_RE.match(action)
    if commit:
        return 0, int(commit.group(1))
    inspect = INSPECT_RE.match(action)
    if inspect:
        return 1, int(inspect.group(1))
    return 2, action


class ExactFiniteSupportDP(Generic[ParticleT]):
    """Backward induction over exact-prefix equivalence classes.

    Particles have equal prior weight.  The transition callback is allowed to
    operate a black-box environment, but the policy key and legal-action set
    are functions only of the exact serialized prefix.
    """

    def __init__(
        self,
        *,
        prefix_of: Callable[[ParticleT], Prefix],
        legal_actions: Callable[[Prefix], Sequence[str]],
        transition: Callable[[ParticleT, str], Transition[ParticleT]],
        action_priority: Callable[[str], object] = _default_action_priority,
    ) -> None:
        self._prefix_of = prefix_of
        self._legal_actions = legal_actions
        self._transition = transition
        self._action_priority = action_priority
        self.stats = SolverStats()

    def solve(self, particles: Iterable[ParticleT]) -> tuple[int, tuple[PolicyNode, ...]]:
        """Return exact optimal successes and one policy forest.

        A forest is needed because a finite prior can expose several distinct
        reset prefixes.  Every retained node is keyed by the full causal-LM
        prefix, so flattening the forest gives a pure prefix-to-action lookup.
        """
        roots = self._partition(tuple(particles))
        if not roots:
            raise ValueError("finite support is empty")
        forest = tuple(self._solve_bucket(bucket, depth=0) for bucket in roots.values())
        return sum(node.optimal_successes for node in forest), forest

    def _solve_bucket(self, bucket: tuple[ParticleT, ...], depth: int) -> PolicyNode:
        prefixes = {self._prefix_of(particle) for particle in bucket}
        if len(prefixes) != 1:
            raise AssertionError("posterior bucket contains more than one exact prefix")
        prefix = next(iter(prefixes))
        actions = tuple(sorted(self._legal_actions(prefix), key=self._action_priority))
        if not actions:
            raise ValueError("nonterminal exact prefix exposes no legal action")

        self.stats.decision_nodes_evaluated += 1
        self.stats.max_evaluated_depth = max(self.stats.max_evaluated_depth, depth)
        best_successes = -1
        best_action = ""
        best_children: tuple[PolicyNode, ...] = ()

        for action in actions:
            self.stats.action_evaluations += 1
            terminal_successes = 0
            successors: list[ParticleT] = []
            for particle in bucket:
                outcome = self._transition(particle, action)
                self.stats.particle_transitions += 1
                if outcome.successor is not None:
                    successors.append(outcome.successor)
                else:
                    terminal_successes += int(bool(outcome.terminal_correct))

            child_nodes = tuple(
                self._solve_bucket(child_bucket, depth + 1)
                for child_bucket in self._partition(tuple(successors)).values()
            )
            successes = terminal_successes + sum(child.optimal_successes for child in child_nodes)
            if successes > best_successes:
                best_successes = successes
                best_action = action
                best_children = child_nodes
            if best_successes == len(bucket):
                self.stats.upper_bound_shortcuts += 1
                break

        return PolicyNode(
            prefix=prefix,
            support_size=len(bucket),
            optimal_successes=best_successes,
            action=best_action,
            children=best_children,
        )

    def _partition(self, particles: tuple[ParticleT, ...]) -> dict[Prefix, tuple[ParticleT, ...]]:
        groups: dict[Prefix, list[ParticleT]] = defaultdict(list)
        for particle in particles:
            groups[self._prefix_of(particle)].append(particle)
        return {prefix: tuple(group) for prefix, group in groups.items()}


@dataclass(frozen=True)
class EvaluatorParticle:
    """A black-box world identity plus public rendered action history."""

    episode_index: int
    actions: tuple[str, ...]
    prefix: Prefix


@dataclass
class BackendStats:
    worlds_constructed: int = 0
    historical_actions_replayed: int = 0
    terminal_verifier_reads: int = 0
    transition_cache_hits: int = 0


@dataclass
class EvaluatorMatchedBackend:
    """Recreate/replay adapter matching ``evaluate._execute_batched`` seeds."""

    params: dict
    seed: int
    rendering: str = "a"
    stats: BackendStats = field(default_factory=BackendStats)
    _cache: dict[tuple[int, tuple[str, ...], str], Transition[EvaluatorParticle]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.rendering != "a":
            raise ValueError("finite-support v1 currently parses Rendering A only")

    def initial_particles(self, episodes: int) -> tuple[EvaluatorParticle, ...]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        return tuple(self._initial_particle(index) for index in range(episodes))

    def legal_actions(self, prefix: Prefix) -> tuple[str, ...]:
        """Derive the action set exclusively from the exact serialized prefix."""
        return available_actions(current_observation(prefix))

    def transition(self, particle: EvaluatorParticle, action: str) -> Transition[EvaluatorParticle]:
        key = particle.episode_index, particle.actions, action
        cached = self._cache.get(key)
        if cached is not None:
            self.stats.transition_cache_hits += 1
            return cached

        batch, reconstructed = self._recreate_and_replay(particle)
        if reconstructed != particle.prefix:
            raise AssertionError("replayed causal prefix differs from the particle key")
        if action not in self.legal_actions(particle.prefix):
            raise ValueError(f"action {action!r} is absent from the serialized AVAILABLE line")

        from world_py import parse_action

        encoded = parse_action(
            action,
            self.params["n_probe"],
            self.params["n_hyp"],
            self.rendering,
        )
        batch.step([encoded])
        after_action = append_action(particle.prefix, action)
        if batch.done()[0]:
            outcome = batch.privileged_outcomes()[0]
            self.stats.terminal_verifier_reads += 1
            if not outcome[0]:
                raise AssertionError("done world returned a nonterminal verifier outcome")
            result: Transition[EvaluatorParticle] = Transition(terminal_correct=bool(outcome[1]))
        else:
            next_observation = batch.observations(self.rendering)[0]
            successor = EvaluatorParticle(
                episode_index=particle.episode_index,
                actions=particle.actions + (action,),
                prefix=append_observation(after_action, next_observation),
            )
            result = Transition(successor=successor)
        self._cache[key] = result
        return result

    def _initial_particle(self, episode_index: int) -> EvaluatorParticle:
        batch = self._new_world(episode_index)
        observation = batch.observations(self.rendering)[0]
        return EvaluatorParticle(
            episode_index=episode_index,
            actions=(),
            prefix=append_observation((BOS,), observation),
        )

    def _new_world(self, episode_index: int):
        from world_py import Batch

        self.stats.worlds_constructed += 1
        return Batch(_family(self.params), seed=self.seed + episode_index, n_episodes=1)

    def _recreate_and_replay(self, particle: EvaluatorParticle):
        from world_py import parse_action

        batch = self._new_world(particle.episode_index)
        prefix: Prefix = (BOS,)
        for action in particle.actions:
            observation = batch.observations(self.rendering)[0]
            prefix = append_observation(prefix, observation)
            if action not in available_actions(observation):
                raise AssertionError("historical action is absent from replayed AVAILABLE line")
            encoded = parse_action(
                action,
                self.params["n_probe"],
                self.params["n_hyp"],
                self.rendering,
            )
            batch.step([encoded])
            self.stats.historical_actions_replayed += 1
            prefix = append_action(prefix, action)
            if batch.done()[0]:
                raise AssertionError("particle history continued beyond termination")
        observation = batch.observations(self.rendering)[0]
        return batch, append_observation(prefix, observation)


def _walk_forest(forest: Sequence[PolicyNode]) -> tuple[PolicyNode, ...]:
    output: list[PolicyNode] = []
    stack = list(reversed(forest))
    while stack:
        node = stack.pop()
        output.append(node)
        stack.extend(reversed(node.children))
    return tuple(output)


def _policy_hash(nodes: Sequence[PolicyNode]) -> str:
    digest = hashlib.sha256()
    for node in sorted(nodes, key=lambda item: item.prefix):
        digest.update(struct.pack("<Q", len(node.prefix)))
        for token in node.prefix:
            digest.update(struct.pack("<I", token))
        action = node.action.encode("utf-8", errors="strict")
        digest.update(struct.pack("<Q", len(action)))
        digest.update(action)
    return digest.hexdigest()


def transductive_empirical_optimum(params: dict, seed: int, episodes: int) -> dict:
    """Compute the exact same-support optimum and its fingerprint diagnostics."""
    backend = EvaluatorMatchedBackend(params=params, seed=seed)
    particles = backend.initial_particles(episodes)
    root_counts = Counter(particle.prefix for particle in particles)
    solver = ExactFiniteSupportDP[
        EvaluatorParticle
    ](
        prefix_of=lambda particle: particle.prefix,
        legal_actions=backend.legal_actions,
        transition=backend.transition,
    )
    optimal_successes, forest = solver.solve(particles)
    retained = _walk_forest(forest)
    selected_action_nodes = Counter(
        "commit" if COMMIT_RE.match(node.action) else "inspect" if INSPECT_RE.match(node.action) else "other"
        for node in retained
    )
    root_action_worlds = Counter()
    for node in forest:
        kind = "commit" if COMMIT_RE.match(node.action) else "inspect" if INSPECT_RE.match(node.action) else "other"
        root_action_worlds[kind] += node.support_size

    return {
        "contract": "step1_transductive_empirical_optimum_v1",
        "scientific_status": (
            "same-support fingerprint memorization diagnostic; not a population public-information ceiling"
        ),
        "policy_boundary": {
            "key": "exact causal-LM token prefix",
            "legal_actions": "parsed only from the serialized AVAILABLE line",
            "environment_access": "recreate and replay public rendered actions",
            "reward_access": "terminal verifier correctness only after termination",
            "forbidden": [
                "Instance",
                "evidence table",
                "probe costs",
                "truth",
                "consistent()",
                "valid_actions()",
                "teacher targets",
                "replay_key()",
            ],
        },
        "support": {
            "seed_policy": "Batch(seed=seed + episode_index, n_episodes=1)",
            "seed": seed,
            "episodes": episodes,
            "distinct_reset_prefixes": len(root_counts),
            "singleton_reset_prefixes": sum(count == 1 for count in root_counts.values()),
            "episodes_in_colliding_reset_prefixes": sum(
                count for count in root_counts.values() if count > 1
            ),
            "max_reset_bucket": max(root_counts.values()),
        },
        "optimum": {
            "successes": optimal_successes,
            "success_rate": optimal_successes / episodes,
            "policy_sha256": _policy_hash(retained),
        },
        "selected_action_tree": {
            "root_nodes": len(forest),
            "retained_decision_nodes": len(retained),
            "retained_edges": sum(len(node.children) for node in retained),
            "max_depth": max(_selected_depth(node) for node in forest),
            "selected_node_kinds": dict(sorted(selected_action_nodes.items())),
            "root_worlds_by_selected_action_kind": dict(sorted(root_action_worlds.items())),
        },
        "search": vars(solver.stats),
        "backend": vars(backend.stats),
    }


def _selected_depth(node: PolicyNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_selected_depth(child) for child in node.children)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    params = config["world"]
    if params["rendering"] != "a":
        raise ValueError("finite-support v1 is frozen to Rendering A")
    seed = args.seed or config["run"]["root_seed"] + 1_000_000
    episodes = args.episodes or params["validation_episodes"]
    result = transductive_empirical_optimum(params, seed, episodes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
