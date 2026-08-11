from world_py import Batch, FamilyParams


params = FamilyParams(
    n_hyp=6,
    n_probe=5,
    n_evidence=2,
    cost_lo=1,
    cost_hi=3,
    budget_slack=1,
    min_depth=2,
    step_slack=2,
    variant="reversible",
)
batch = Batch(params, seed=20260810, n_episodes=36)

for rendering in ("a", "b"):
    print(f"{rendering}: {batch.observations(rendering)[0]}")

while not all(batch.done()):
    targets = batch.privileged_teacher_targets()
    live = batch.live_episode_indices()
    assert live == [i for i, is_done in enumerate(batch.done()) if not is_done]
    # `actions[j]` is for original episode `live[j]`, never episode `j`.
    actions = [next(iter(targets[i]["preferred_actions"])) for i in live]
    batch.step(actions)

outcomes = batch.privileged_outcomes()
assert len(outcomes) == 36
assert all(terminated and correct for terminated, correct, *_ in outcomes)
print(f"completed {len(outcomes)} episodes; all correct")
print(f"replay key: {batch.replay_key(0, 'b')}")

# Exercise staggered completion and the compact live-action mapping. Episode 0
# commits immediately; the remaining episodes take their teacher action.
staggered = Batch(
    FamilyParams(
        n_hyp=6,
        n_probe=5,
        n_evidence=2,
        cost_lo=1,
        cost_hi=3,
        budget_slack=1,
        min_depth=2,
        step_slack=2,
        variant="irreversible",
    ),
    seed=20260811,
    n_episodes=4,
)
staggered_targets = staggered.privileged_teacher_targets()
staggered.step(
    [params.n_probe]
    + [next(iter(staggered_targets[i]["preferred_actions"])) for i in (1, 2, 3)]
)
assert staggered.done()[0] and staggered.live_episode_indices() == [1, 2, 3]
next_targets = staggered.privileged_teacher_targets()
staggered.step(
    [next(iter(next_targets[i]["preferred_actions"])) for i in staggered.live_episode_indices()]
)
assert staggered.done()[0]
print("staggered completion compact mapping: episode 0 done; live indices [1, 2, 3]")
