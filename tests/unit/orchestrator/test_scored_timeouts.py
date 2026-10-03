"""Opt-in ``train_timeouts = "score"``: healthy agent-deadline exhaustions are scored task outcomes.

Uses real timed-out / completed traces from the super35-pi-rl-v5 run (Nemotron-3.5-Super, Pi harness) when available.
"""
import copy
import glob
import json
import os

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.dispatcher import is_scored_budget_failure
from prime_rl.orchestrator.train_sink import _prune_zero_advantages
from prime_rl.orchestrator.trajectories import trace_to_samples

V5 = "/home/huggingface/radhika/rl-super/outputs/super35-pi-rl-v5/monitors/file/traces/stream"


def _real_episodes(stop: str, n: int) -> list[vf.Episode]:
    out = []
    for f in sorted(glob.glob(V5 + "/*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d["run"]["work"]["type"] == "train" and d["traces"][0]["stop_condition"] == stop:
                out.append(vf.Episode.model_validate(d))
                if len(out) >= n:
                    return out
    return out


def _as_pre_dispatch(ep: vf.Episode) -> vf.Episode:
    """Undo the dispatcher's drop-mode mutation stored in the archive (Timeout error, ok=False)."""
    ep = copy.deepcopy(ep)
    for tr in ep.traces:
        tr.errors = [e for e in tr.errors if e.type != "Timeout"]
        tr.ok = True
    ep.ok = True
    return ep


def _set_reward(ep: vf.Episode, value: float) -> vf.Episode:
    ep = copy.deepcopy(ep)
    for tr in ep.traces:
        for k, r in tr.rewards.items():
            if r is not None:
                tr.rewards[k] = r.model_copy(update={"score": value})
    return ep


needs_v5 = pytest.mark.skipif(not os.path.isdir(V5), reason="v5 traces not available")


@needs_v5
def test_classifier_on_real_timeout():
    ep = _as_pre_dispatch(_real_episodes("agent_timeout", 1)[0])
    tr = ep.traces[0]
    assert tr.is_timeout and tr.stop_condition == "agent_timeout" and not tr.has_error and tr.num_turns > 0
    assert is_scored_budget_failure(tr)
    # recorded error -> not a scored budget failure (infra / harness problems stay excluded)
    bad = copy.deepcopy(tr)
    bad.errors.append(vf.Error(type="SandboxError", message="x"))
    bad.ok = False
    assert not is_scored_budget_failure(bad)
    # other deadline stages are not agent budgets
    fin = copy.deepcopy(tr)
    fin.stop_condition = "finalize_timeout"
    assert not is_scored_budget_failure(fin)
    # ungraded -> excluded
    ung = copy.deepcopy(tr)
    ung.rewards = {k: None for k in ung.rewards}
    assert not is_scored_budget_failure(ung)
    # empty trajectory -> excluded
    emp = copy.deepcopy(tr)
    emp.nodes = []
    emp.calls = []
    assert not is_scored_budget_failure(emp)


@needs_v5
def test_grpo_advantages_and_pruning_with_scored_timeouts():
    algo = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    done = _real_episodes("agent_completed", 5)
    tos = [_as_pre_dispatch(e) for e in _real_episodes("agent_timeout", 3)]
    group = [_set_reward(e, 1.0) for e in done] + [_set_reward(e, 0.0) for e in tos]  # [1,1,1,1,1,0,0,0]
    survivors = [t for _, t in iter_trainable_traces(group)]
    assert len(survivors) == 8, "scored timeouts must survive the trainable-trace filter"
    import asyncio
    asyncio.run(algo.score_group(group))
    advs = []
    for _, t in iter_trainable_traces(group):
        samples = trace_to_samples(t)
        assert samples, "every survivor yields samples"
        for s in samples:
            # tokens / logprobs / mask / advantages aligned
            assert len(s.token_ids) == len(s.mask)
            if s.logprobs is not None:
                assert len(s.logprobs) == len(s.token_ids)
            if s.advantages is not None:
                assert len(s.advantages) == len(s.token_ids)
            vals = {round(a, 6) for a, m in zip(s.advantages, s.mask) if m}
            assert len(vals) == 1
            advs.append(vals.pop())
            assert _prune_zero_advantages(copy.deepcopy(s)), "non-zero-advantage samples are kept"
    assert set(advs) == {0.375, -0.625}


@needs_v5
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_uniform_groups_have_zero_advantage(value):
    algo = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    group = [_set_reward(_as_pre_dispatch(e), value) for e in _real_episodes("agent_timeout", 4)]
    import asyncio
    asyncio.run(algo.score_group(group))
    for _, t in iter_trainable_traces(group):
        for s in trace_to_samples(t):
            assert all(abs(a) < 1e-9 for a, m in zip(s.advantages, s.mask) if m)
            assert not _prune_zero_advantages(copy.deepcopy(s)), "zero-advantage samples are pruned"


@needs_v5
def test_drop_mode_excludes_timeouts():
    """Archived v5 traces carry the drop-mode Timeout error: they must be excluded from training."""
    eps = _real_episodes("agent_timeout", 3)
    assert eps and all(t.has_error for e in eps for t in e.traces)
    assert not list(iter_trainable_traces(eps))


@needs_v5
def test_packing_retains_scored_timeout_samples():
    """Scored-timeout samples go through the real trainer batch packer unchanged (no truncation, mask/advantage kept)."""
    import asyncio
    from prime_rl.trainer.batch import build_bin_cost, prepare_batch
    algo = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    done = _real_episodes("agent_completed", 2)
    tos = [_as_pre_dispatch(e) for e in _real_episodes("agent_timeout", 2)]
    group = [_set_reward(e, 1.0) for e in done] + [_set_reward(e, 0.0) for e in tos]
    asyncio.run(algo.score_group(group))
    samples, timeout_tokens = [], 0
    for ep, t in iter_trainable_traces(group):
        ss = trace_to_samples(t, env_name="test")
        for smp in ss:
            smp.temperatures = [1.0] * len(smp.token_ids)  # as TrainSink does before packing
        ss = [smp for smp in ss if _prune_zero_advantages(smp)]
        if t.is_timeout:
            timeout_tokens += sum(sum(1 for m in s.mask if m) for s in ss)
        samples += ss
    assert timeout_tokens > 0
    mbs = prepare_batch(rollouts=samples, seq_len=131072, num_train_workers=1, bin_cost=build_bin_cost(None), pad_to_multiple_of=1)
    packed_trainable = sum(int(sum(1 for m in getattr(mb, "loss_mask", getattr(mb, "mask", [])) if m)) for worker in mbs for mb in worker)
    total_trainable = sum(sum(1 for m in s.mask if m) for s in samples)
    assert packed_trainable == total_trainable, (packed_trainable, total_trainable)
