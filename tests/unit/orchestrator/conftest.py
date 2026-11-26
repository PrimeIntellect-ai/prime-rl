import math

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.types import SamplerSupport, WeightVersionSpan

from prime_rl.orchestrator.hindsight import FixedHorizonWorld


def make_hindsight_episode(request=0, prior=None):
    requested = (request,)
    goals = {(0,): 0.25, (1,): 0.75} if prior is None else prior
    w = FixedHorizonWorld(
        "explicit",
        (0,),
        requested,
        frozenset(g for g, prob in goals.items() if prob > 0),
        (10, 11),
        2,
        "completed",
        lambda s, t: ((s[0] + t - 10) % 2,),
        lambda g, a: [(99, g[0], *a[:i]) for i in range(len(a))],
    )
    # Actual action sequence (10,11) reaches goal (1,).
    task = vf.TraceTask(type="Parity", data=vf.TaskData(idx=0, prompt="reference"))
    root = vf.MessageNode(message=vf.UserMessage(content=""), token_ids=[99], mask=[False])
    answer = vf.MessageNode(
        message=vf.AssistantMessage(content="ab"),
        token_ids=[request, 10, 11],
        mask=[False, True, True],
        sampled=True,
        parent=0,
        logprobs=[math.log(0.4), math.log(0.3)],
        sampler_support=SamplerSupport(np.array([10, 11, 10, 11]), np.array([2, 2]), np.ones(2, bool)),
        weight_versions=[WeightVersionSpan(start=0, end=2, version="pinned")],
    )
    trace = vf.Trace[vf.TaskData](
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[root, answer],
        calls=[vf.ModelCall(node=1, model="policy")],
        rewards={"reward": vf.Reward(score=float(request == 1))},
        ok=True,
        is_completed=True,
        stop_condition="completed",
    )
    return vf.Episode(task=task, traces=[trace], env=vf.EnvInfo(id="explicit"), ok=True), w


@pytest.fixture
def hindsight_episode():
    return make_hindsight_episode
