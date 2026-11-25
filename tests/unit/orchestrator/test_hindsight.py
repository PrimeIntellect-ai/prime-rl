from dataclasses import replace

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.hindsight import make_hindsight_batch, make_hindsight_target, reconstruct_rollout


def test_relabel_original_immutable(hindsight_episode):
    ep, world = hindsight_episode()
    original = ep.to_json()
    t = make_hindsight_target(ep, world)
    assert (t.requested, t.achieved, t.original_reward) == ((0,), (1,), 0.0)
    assert (t.prefixes, t.actions, t.action_ids) == (((99, 1), (99, 1, 10)), (10, 11), (10, 11))
    assert ep.to_json() == original
    ep.traces[0].nodes[1].sampler_support = None
    ep.traces[0].nodes[1].logprobs = []
    ep.traces[0].nodes[1].weight_versions = None
    assert make_hindsight_target(ep, world) == t


def test_eligibility_skips_goal_rendering(hindsight_episode):
    eps, worlds = zip(
        *(hindsight_episode(req, prior) for req, prior in ((0, {(0,): 1.0}), (0, None), (1, {(1,): 1.0}), (1, None))),
        strict=True,
    )
    requested = []

    def record(w):
        def render(g, a):
            requested.append(g)
            return w.prefixes(g, a)

        return replace(w, prefixes=render)

    metrics = {}
    ts = make_hindsight_batch(eps, [record(w) for w in worlds], metrics=metrics)
    assert len(ts) == len(eps) and ts[0] is None and [t.original_reward for t in ts[1:]] == [0.0, 1.0, 1.0]
    assert requested == [(0,), (0,), (1,), (1,), (1,)]
    assert metrics == dict(
        original_episodes=4, unsupported_omitted=1, singleton_hits=1, supported_distinct_goal_misses=1
    )
    with pytest.raises(ValueError):
        make_hindsight_batch(eps, worlds[:-1])


@pytest.mark.parametrize(
    "issue",
    ["support", "incomplete", "unverified", "bonus", "prefix", "branch", "calls", "horizon", "goal-support", "illegal"],
)
def test_reconstruction_rejects_invalid_evidence(hindsight_episode, issue):
    ep, w = hindsight_episode()
    t = ep.traces[0]
    if issue == "support":
        t.nodes[1].sampler_support.ids[0] = 42
    if issue == "incomplete":
        t.is_completed = False
    if issue == "unverified":
        t.rewards["reward"].score = 1
    if issue == "bonus":
        t.rewards["bonus"] = vf.Reward(score=0.25)
    if issue == "prefix":
        w = replace(w, prefixes=lambda g, a: [(9,), (9, 10)])
    if issue == "branch":
        t.nodes[1].parent = None
    if issue == "calls":
        t.calls[0].node = 0
    if issue == "horizon":
        w = replace(w, horizon=1)
    if issue == "goal-support":
        w = replace(w, supported_goals={(1,)})
    if issue == "illegal":
        t.nodes[1].token_ids[1] = 999
    with pytest.raises(ValueError):
        reconstruct_rollout(ep, w)
