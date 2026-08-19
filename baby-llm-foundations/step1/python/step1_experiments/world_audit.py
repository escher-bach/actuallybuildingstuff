"""CPU replay audit of the world and its teacher, with no model in the loop.

THEORY-PHASE.md §9 asks for measurements that sharpen the learner-conditioned
experiment before it is run.  Two of them need only the world:

1. how many hypotheses the *opening* observation leaves live, which decides
   whether the ⌈log2 n_hyp⌉ probe bound in §2 applies to every episode or only
   to some, and therefore whether the dense arm's ~1.94 probes is under-probing
   at all; and
2. the teacher's own probe count and spend on the same episodes, which STEP 1
   reports for the model and never for the demonstrator it was cloned from.

Both are arithmetic over `world_py`.  This module deliberately imports neither
torch nor anything from `.data`, so the audit runs on a laptop.

The remaining two items in §9 need a checkpoint and are measured inside the
learner-conditioned run, where the checkpoint already lives.
"""
from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path


AUDIT_CONTRACT = "step1_world_audit_v1"


def _family(params: dict):
    from world_py import FamilyParams

    return FamilyParams(
        n_hyp=params["n_hyp"], n_probe=params["n_probe"], n_evidence=params["n_evidence"],
        cost_lo=params["cost_lo"], cost_hi=params["cost_hi"], budget_slack=params["budget_slack"],
        min_depth=params["min_depth"], step_slack=params["step_slack"], variant=params["variant"],
    )


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("cannot average an empty audit series")
    return sum(values) / len(values)


def identification_ceiling(n_hyp: int, probes: int) -> float:
    """Best success attainable after `probes` binary observations.

    An information bound, not a claim about any policy: no probe selection
    extracts more than one bit per probe, so after k probes at most min(2^k,
    n_hyp) of the n_hyp hypotheses can be told apart.
    """
    if probes < 0:
        raise ValueError("probe count cannot be negative")
    return min(2 ** probes, n_hyp) / n_hyp


def opening_liveness(params: dict, seed: int, episodes: int) -> dict:
    """How much the episode's first observation has already narrowed the field."""
    from world_py import Batch

    batch = Batch(_family(params), seed=seed, n_episodes=episodes)
    counts = list(batch.privileged_consistent_counts())
    return {
        "episodes": episodes,
        "mean_live_hypotheses": _mean(counts),
        "min_live_hypotheses": min(counts),
        "max_live_hypotheses": max(counts),
        "histogram": _histogram(counts),
        "equals_n_hyp_in_every_episode": all(count == params["n_hyp"] for count in counts),
    }


def teacher_play(params: dict, seed: int, episodes: int) -> dict:
    """Replay the privileged teacher and record what the demonstrator costs.

    The teacher is also the instrument check for the premature-commitment
    metric used on the learner: a teacher that ever committed with more than
    one hypothesis live would mean the metric fires on optimal play, and the
    measurement would be reading something other than prematurity.
    """
    from world_py import Batch

    n_probe = params["n_probe"]
    batch = Batch(_family(params), seed=seed, n_episodes=episodes)
    probes = [0] * episodes
    live_at_commitment: list[int] = []
    while not all(batch.done()):
        targets = batch.privileged_teacher_targets()
        consistent = batch.privileged_consistent_counts()
        live = batch.live_episode_indices()
        actions = []
        for index in live:
            action = min(targets[index]["preferred_actions"])
            if action < n_probe:
                probes[index] += 1
            else:
                live_at_commitment.append(consistent[index])
            actions.append(action)
        batch.step(actions)
    outcomes = batch.privileged_outcomes()
    spent = [int(row[2]) for row in outcomes]
    steps = [int(row[3]) for row in outcomes]
    correct = [bool(row[0] and row[1]) for row in outcomes]
    return {
        "episodes": episodes,
        "success_rate": sum(correct) / episodes,
        "mean_probes": _mean(probes),
        "probe_histogram": _histogram(probes),
        "mean_spent": _mean(spent),
        "mean_steps": _mean(steps),
        "committed_episodes": len(live_at_commitment),
        "max_live_hypotheses_at_commitment": max(live_at_commitment) if live_at_commitment else None,
        "never_commits_prematurely": all(count == 1 for count in live_at_commitment),
    }


def truth_blind_identification(params: dict, seed: int, episodes: int) -> dict:
    """What identification costs a learner that cannot read the answer.

    `teach` selects probes using `inst.truth`: it needs only the true
    hypothesis to land in a cell of its own, not all `n_hyp` to be separated.
    That is why it identifies inside two binary probes, which cannot separate
    six hypotheses.  A learner has no such shortcut available to it.

    This replays a fixed truth-blind order (ascending probe id, every probe it
    can afford) until the consistent set is a singleton or the budget is gone.
    It is one policy, not the optimum, so `mean_probes` is an upper bound on
    what an optimal truth-blind learner needs.  `identified_rate` is the
    load-bearing figure: an episode this policy cannot identify after buying
    every probe it can afford is one where the evidence never singles out the
    answer along that spending path at all.
    """
    from world_py import Batch

    n_probe = params["n_probe"]
    probes_used, identified = [], []
    for index in range(episodes):
        world = Batch(_family(params), seed=seed + index, n_episodes=1)
        used = 0
        while True:
            if world.privileged_teacher_targets()[0]["licenses_commitment"]:
                identified.append(True)
                break
            affordable = [a for a in world.valid_actions()[0] if a < n_probe]
            if not affordable:
                # Every probe is either bought or unaffordable: no further
                # evidence is reachable in this episode.
                identified.append(False)
                break
            world.step([min(affordable)])
            used += 1
        probes_used.append(used)
    return {
        "episodes": episodes,
        "policy": "ascending probe id, truth-blind, until licensed or no affordable probe remains",
        "identified_rate": sum(identified) / episodes,
        "mean_probes": _mean(probes_used),
        "mean_probes_when_identified": (
            _mean([used for used, ok in zip(probes_used, identified) if ok])
            if any(identified) else None
        ),
        "probe_histogram": _histogram(probes_used),
        "bound_direction": "upper bound on probes; one fixed policy, not the truth-blind optimum",
    }


def truth_blind_ceiling(params: dict, seed: int, episodes: int) -> dict:
    """The exact success ceiling for any learner that cannot read the answer.

    For each episode this searches every subset of probes the budget allows,
    and scores the subset by `1 / |consistent set|` -- the chance of guessing
    right among the hypotheses the evidence still permits.  Taking the maximum
    over subsets hands the learner the best probe set *in hindsight*, which no
    adaptive policy can beat, so the mean is a hard upper bound on truth-blind
    success.

    The teacher scores 1.0 by picking its probes with `inst.truth` in hand.
    The gap between that 1.0 and this bound is not something a learner can
    close by imitating better; it is information the demonstrator had and the
    learner does not.
    """
    from world_py import Batch

    n_probe = params["n_probe"]
    best_scores, identifiable = [], []
    for index in range(episodes):
        best = 0.0
        for mask in range(1 << n_probe):
            subset = [q for q in range(n_probe) if mask & (1 << q)]
            world = Batch(_family(params), seed=seed + index, n_episodes=1)
            affordable = True
            for probe in subset:
                try:
                    world.step([probe])
                except ValueError:
                    # The world rejects an unaffordable probe without mutating
                    # state, so this subset simply does not fit the budget.
                    affordable = False
                    break
            if not affordable:
                continue
            best = max(best, 1.0 / world.privileged_consistent_counts()[0])
        best_scores.append(best)
        identifiable.append(best == 1.0)
    return {
        "episodes": episodes,
        "search": "exhaustive over probe subsets within budget; scored in hindsight",
        "identifiable_rate": sum(identifiable) / episodes,
        "success_ceiling": _mean(best_scores),
        "bound_direction": "upper bound: no adaptive truth-blind policy can exceed a hindsight-chosen subset",
    }


def truth_blind_optimum(params: dict, seed: int, episodes: int) -> dict:
    """What a learner that cannot read the answer can actually achieve.

    The teacher identifies in ~2 probes by choosing them with `inst.truth` in
    hand: it needs only the true hypothesis to end up alone in a cell. It is
    tempting to conclude from that the demonstration cannot be followed. That
    conclusion does not survive measurement, and a single fixed probe order is
    not evidence for it -- a bad fixed order wastes the budget on probes that
    separate nothing.

    This is the Bayes-optimal truth-blind policy by backward induction: at each
    state, commit now for `1 / |consistent|`, or buy an affordable unprobed
    probe and average over the evidence it might return. It answers two things
    the teacher's own numbers cannot: how well a learner *could* do here, and
    how many probes doing that well actually costs.
    """
    from world_py import truth_blind_optimal_success

    values = truth_blind_optimal_success(_family(params), seed, episodes)
    success = [value for value, _probes in values]
    probes = [count for _value, count in values]
    return {
        "episodes": episodes,
        "success_ceiling": _mean(success),
        "certain_identification_rate": sum(1 for value in success if value > 0.999) / episodes,
        "mean_probes": _mean(probes),
        "interpretation": (
            "the task is very nearly solvable without truth; the teacher's lower probe count is a "
            "privilege of knowing the answer, so imitating its cost profile caps the learner below "
            "this ceiling"
        ),
    }


def teacher_target_leakage(params: dict, seed: int, episodes: int, max_turns: int = 8) -> dict:
    """Does querying the teacher off its own trajectory leak the answer?

    The dense teacher is truth-privileged.  On its own trajectory that is
    harmless: it only ever commits from a state where the history already
    pins the answer down, so a commit target is always justified by evidence
    the learner can see.

    Off that trajectory the guarantee is gone.  At a state where the budget
    has been spent without identifying anything, `teach` can still propose
    `Commit(truth)` -- a target no amount of visible evidence supports.
    Supervising it would teach the model to commit on evidence that does not
    determine the answer, which is the behaviour learner conditioning is
    supposed to reduce.

    This walks a deliberately weak truth-blind policy into exactly those
    states and counts them.  It exists because the hazard is *created* by
    learner conditioning and is invisible to every teacher-conditioned check.
    """
    from world_py import Batch

    n_probe = params["n_probe"]
    states = unlicensed = commit_at_unlicensed = commit_only = 0
    for index in range(episodes):
        world = Batch(_family(params), seed=seed + index, n_episodes=1)
        for _ in range(max_turns):
            targets = world.privileged_teacher_targets()[0]
            preferred = sorted(targets["preferred_actions"])
            if not preferred:
                break
            states += 1
            if not targets["licenses_commitment"]:
                unlicensed += 1
                commits = [action for action in preferred if action >= n_probe]
                if commits:
                    commit_at_unlicensed += 1
                    commit_only += len(commits) == len(preferred)
            affordable = [action for action in world.valid_actions()[0] if action < n_probe]
            if not affordable:
                break
            world.step([min(affordable)])
            if world.done()[0]:
                break
    return {
        "episodes": episodes,
        "probe_policy": "ascending probe id, truth-blind; a stand-in for a learner spending badly",
        "teacher_queried_states": states,
        "unlicensed_states": unlicensed,
        "commit_proposed_at_unlicensed_state": commit_at_unlicensed,
        "commit_was_the_only_preferred_action": commit_only,
        "leak_rate_among_unlicensed": commit_at_unlicensed / unlicensed if unlicensed else 0.0,
        "consequence": (
            "a commit target at an unlicensed state is derived from inst.truth and cannot be "
            "justified by visible evidence; STEP-1 §5.2 forbids supervising it"
        ),
    }


def assert_audit_contract(report: dict, params: dict) -> None:
    """Fail on a world that would make the §2 reading unsound or the metric wrong."""
    if report.get("contract") != AUDIT_CONTRACT:
        raise AssertionError("world audit contract mismatch")
    opening = report["opening_liveness"]
    if opening["max_live_hypotheses"] > params["n_hyp"]:
        raise AssertionError("more hypotheses were live than the family declares")
    teacher = report["teacher_play"]
    if teacher["success_rate"] != 1.0:
        raise AssertionError(
            f"the privileged teacher must solve every episode, scored {teacher['success_rate']}"
        )
    if not teacher["never_commits_prematurely"]:
        raise AssertionError(
            "the teacher committed with more than one hypothesis live; the premature-commitment "
            "metric would fire on optimal play and is not measuring prematurity"
        )
    if teacher["mean_probes"] <= 0:
        raise AssertionError("a teacher that never probes cannot calibrate a probe-count metric")
    ceiling = report["truth_blind_ceiling"]
    if not 0.0 <= ceiling["success_ceiling"] <= 1.0:
        raise AssertionError(f"truth-blind ceiling is not a probability: {ceiling['success_ceiling']}")
    if ceiling["success_ceiling"] < teacher["success_rate"] and ceiling["identifiable_rate"] == 1.0:
        raise AssertionError("every episode is identifiable yet the ceiling is below the teacher's score")
    optimum = report["truth_blind_optimum"]
    if not 0.0 <= optimum["success_ceiling"] <= 1.0:
        raise AssertionError(f"truth-blind optimum is not a probability: {optimum['success_ceiling']}")
    if optimum["success_ceiling"] < ceiling["identifiable_rate"] - 0.5:
        raise AssertionError(
            "every episode is identifiable in hindsight yet the adaptive optimum is far below it; "
            "the backward induction is not searching the same actions"
        )
    leakage = report["teacher_target_leakage"]
    if leakage["unlicensed_states"] <= 0:
        raise AssertionError(
            "the leakage probe never reached an unlicensed state, so it checked nothing; "
            "its stand-in policy is too strong to stand in for a learner"
        )


def audit(config: dict, ceiling_episodes: int = 256) -> dict:
    """Audit the validation split the evaluator itself scores.

    The subset search is quadratic in nothing but exhaustive over `2 **
    n_probe` worlds per episode, so it runs on its own smaller sample; every
    other measurement uses the full split.
    """
    world = config["world"]
    seed = config["run"]["root_seed"] + 1_000_000
    episodes = world["validation_episodes"]
    opening = opening_liveness(world, seed, episodes)
    teacher = teacher_play(world, seed, episodes)
    blind = truth_blind_identification(world, seed, episodes)
    optimum = truth_blind_optimum(world, seed, episodes)
    ceiling = truth_blind_ceiling(world, seed, min(ceiling_episodes, episodes))
    leakage = teacher_target_leakage(world, seed, min(ceiling_episodes, episodes))
    report = {
        "contract": AUDIT_CONTRACT,
        "question": (
            "does the opening observation already narrow the field, and what does the "
            "demonstrator itself spend? (THEORY-PHASE.md §9 items 1 and 2)"
        ),
        "world": dict(world),
        "evaluation_set": {
            "seed": seed, "episodes": episodes, "rendering": world["rendering"],
            "note": "the validation seeds evaluate._matched_sets scores every arm on",
        },
        "opening_liveness": opening,
        "teacher_play": teacher,
        "truth_blind_identification": blind,
        "truth_blind_optimum": optimum,
        "truth_blind_ceiling": ceiling,
        "teacher_target_leakage": leakage,
        "identification_ceiling": {
            str(probes): identification_ceiling(world["n_hyp"], probes)
            for probes in range(0, world["n_probe"] + 1)
        },
        "scientific_acceptance_policy": "measurement only; it trains nothing and gates no run",
    }
    assert_audit_contract(report, world)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    report = audit(config)
    document = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
    print(document)


if __name__ == "__main__":
    main()
