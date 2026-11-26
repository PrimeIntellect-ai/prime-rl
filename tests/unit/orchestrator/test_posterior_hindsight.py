import math
from dataclasses import replace
from random import Random

import pytest
import torch

from prime_rl.orchestrator.posterior_hindsight import centered_update_loss, make_posterior_targets
from prime_rl.trainer.rl.posterior_hindsight import centered_pair_loss, weighted_hindsight_loss


@pytest.mark.parametrize(
    ("score_dtype", "baseline_dtype", "q", "b"),
    [
        (torch.bfloat16, torch.float32, 0.999, 0.25),
        (torch.bfloat16, torch.float32, 1 - 1e-9, 0.25),
        (torch.float16, torch.float32, 0.9999, 0.25),
        (torch.float32, torch.float32, 0.999, 0.999),
        (torch.float64, torch.float32, 1 - 1e-10, 0.25),
        (torch.float32, torch.float64, 1 - 1e-10, 1 - 1e-9),
    ],
)
def test_centered_gradient_through_both_score_roles(hindsight_episode, score_dtype, baseline_dtype, q, b):
    prior = {(0,): 1 - q, (1,): q}
    ep, world = hindsight_episode(request=0, prior=prior)
    factual_logs = list(map(float, ep.traces[0].nodes[-1].logprobs))

    def frozen(snapshot, rows):
        assert snapshot == "pinned" and [r.prefixes[0] for r in rows] == [(99, 1)]
        return [factual_logs]  # Equal conditional likelihoods give posterior equal to the task prior.

    target = make_posterior_targets([ep], [world], [prior], "pinned", frozen, Random(42))
    scores = torch.tensor([-1.0, -2.0], dtype=score_dtype, requires_grad=True)
    baseline = torch.tensor(b, dtype=baseline_dtype, requires_grad=True)

    def fresh(rows):
        assert [r.prefixes[0] for r in rows] == [(99, 1), (99, 0)]
        return scores

    loss = centered_update_loss(target, "pinned", fresh, baseline)
    grad, grad_baseline = torch.autograd.grad(loss, (scores, baseline), allow_unused=True)
    dtype = torch.promote_types(torch.float32, torch.promote_types(score_dtype, baseline_dtype))
    assert loss.dtype == dtype and grad_baseline is None
    w = torch.tensor(q, dtype=dtype)
    b = baseline.detach().to(dtype)
    foil = torch.tensor(prior[(0,)], dtype=dtype)
    expected = torch.stack((-w * (1 - b), foil * b)).to(score_dtype)
    assert expected[1] > 0 and grad[1] > 0
    torch.testing.assert_close(grad, expected, rtol=1e-5, atol=0)
    # A manually supplied Python posterior still carries its own precision before transfer.
    external = [replace(target[0], posterior=q, foil_posterior=None)]
    fallback = centered_update_loss(external, "pinned", fresh, baseline)
    torch.testing.assert_close(torch.autograd.grad(fallback, scores)[0], expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("requested", [0, 1])
def test_extreme_likelihood_retains_both_roles(hindsight_episode, requested):
    prior = {(0,): 0.5, (1,): 0.5}
    ep, world = hindsight_episode(request=requested, prior=prior)
    logs = list(map(float, ep.traces[0].nodes[-1].logprobs))
    # For equal priors this makes the alternative exactly exp(-45) times as likely as the factual path.
    alternative = [logs[0], logs[1] - 45.0]
    targets = make_posterior_targets([ep], [world], [prior], "pinned", lambda *_: [alternative], Random(42))
    ratio = math.exp(-45)
    tail = ratio / (1 + ratio)
    if requested == 1:
        assert targets[0].posterior == 1.0 and targets[0].foil_posterior == pytest.approx(tail, abs=0)
        expected = [-0.75, 0.25 * tail]
    else:
        assert targets[0].posterior == pytest.approx(tail, abs=0) and targets[0].foil_posterior == 1.0
        expected = [-0.75 * tail, 0.25]
    scores = torch.tensor([-1.0, -2.0], dtype=torch.bfloat16, requires_grad=True)
    loss = centered_update_loss(targets, "pinned", lambda _: scores, torch.tensor(0.25))
    assert loss.dtype == torch.float32
    (gradient,) = torch.autograd.grad(loss, scores)
    assert gradient.count_nonzero().item() == 2
    torch.testing.assert_close(gradient, torch.tensor(expected, dtype=scores.dtype), rtol=0, atol=0)


def test_centered_and_weighted_batch_special_cases(hindsight_episode):
    prior = {(0,): 0.25, (1,): 0.75, (2,): 0.0}
    cases = [(0, {(0,): 1.0}), (1, {(1,): 1.0}), (0, prior), (1, prior)]
    eps, worlds = zip(*(hindsight_episode(request=g, prior=p) for g, p in cases), strict=True)
    source = [ep.to_json() for ep in eps]
    alternate_logs = [math.log(0.8), math.log(0.7)]
    factual = math.exp(math.fsum(map(float, eps[0].traces[0].nodes[-1].logprobs)))
    alternative = 0.8 * 0.7

    def frozen(snapshot, rows):
        assert snapshot == "pinned"
        assert [r.prefixes[0] for r in rows] == [(99, 1), (99, 0)]
        return [alternate_logs, alternate_logs]

    targets = make_posterior_targets(eps, worlds, [p for _, p in cases], "pinned", frozen, Random(23))
    assert source == [ep.to_json() for ep in eps]
    assert [t.original_reward for t in targets] == [0, 1, 0, 1]
    assert targets[0].achieved is None and targets[1].foil is None
    weights = [
        0.0,
        1.0,
        0.75 * alternative / (0.75 * alternative + 0.25 * factual),
        0.75 * factual / (0.75 * factual + 0.25 * alternative),
    ]
    assert [t.posterior for t in targets] == pytest.approx(weights)
    paths = torch.tensor([-0.4, -1.3], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.25, dtype=torch.float64)

    def fresh(rows):
        assert [r.prefixes[0][-1] for r in rows] == [0, 1, 1, 0, 1, 0]
        return torch.stack([paths[r.prefixes[0][-1]] for r in rows])

    loss = centered_update_loss(targets, "pinned", fresh, b)
    achieved = sum(weights)
    foil = len(targets) - achieved
    expected = torch.stack([foil * b, -achieved * (1 - b)]) / len(targets)
    torch.testing.assert_close(torch.autograd.grad(loss, paths)[0], expected)

    with pytest.raises(ValueError, match="collecting policy"):
        centered_update_loss(targets, "stale", fresh, b)
    w = torch.tensor(weights[1:], dtype=torch.float64, requires_grad=True)
    zero_baseline = weighted_hindsight_loss(paths[1].expand(3), w, len(targets))
    gp, gw = torch.autograd.grad(zero_baseline, (paths, w), allow_unused=True)
    assert gw is None
    torch.testing.assert_close(gp, torch.tensor([0.0, -achieved / len(targets)], dtype=torch.float64))
    empty = weighted_hindsight_loss(paths[:0], w[:0], 4)
    assert empty.item() == 0 and not torch.autograd.grad(empty, paths)[0].any()


@pytest.mark.parametrize("missing", ["support", "version", "short-span", "invalid-logp"])
def test_requires_factual_sampler_evidence(hindsight_episode, missing):
    ep, world = hindsight_episode()
    node = ep.traces[0].nodes[-1]
    if missing == "support":
        node.sampler_support = None
    elif missing == "version":
        node.weight_versions[0].version = "stale"
    elif missing == "short-span":
        node.weight_versions[0].end = 1
    else:
        node.logprobs[0] = float("nan")
    with pytest.raises(ValueError):
        make_posterior_targets(
            [ep], [world], [{(0,): 0.25, (1,): 0.75}], "pinned", lambda *_: pytest.fail("Unexpected score"), Random(1)
        )


def test_learner_preserves_external_fp64_coefficients():
    y = torch.tensor([-1.0], dtype=torch.bfloat16, requires_grad=True)
    j = y.detach().clone().requires_grad_()
    posterior = torch.tensor([1 - 1e-10], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.25)
    loss = centered_pair_loss(y, j, posterior, b)
    assert loss.dtype == torch.float64
    gj, gw = torch.autograd.grad(loss, (j, posterior), allow_unused=True)
    assert gw is None and gj.item() > 0
    torch.testing.assert_close(gj, ((1 - posterior.detach()) * b).to(j), atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="sum to one"):
        centered_pair_loss(y, j, posterior, b, foil_posterior=torch.tensor([0.5]))


@pytest.mark.parametrize("prior", [{(0,): 1.0}, {(0,): 0.2, (1,): 0.3}, {(0,): 0.7, (1,): 0.2, (2,): 0.1}])
def test_prior_must_match_ordinary_goal_support(hindsight_episode, prior):
    ep, world = hindsight_episode()
    with pytest.raises(ValueError, match="Goal prior"):
        make_posterior_targets([ep], [world], [prior], "pinned", lambda *_: pytest.fail("Unexpected score"), Random(1))
