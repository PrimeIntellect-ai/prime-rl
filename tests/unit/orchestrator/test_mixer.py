from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import PrefixSourceConfig
from prime_rl.orchestrator.train_source import TrainSource, inflight_caps, mixer_weight


def test_dispatch_corrects_for_acceptance() -> None:
    # 1:1 prompt share; math groups are half accepted, swe groups all accepted
    assert mixer_weight(16.0, 0.0, 0.5) == 2 * mixer_weight(16.0, 0.0, 1.0)


def test_deficit_term_vanishes_once_batch_target_is_queued() -> None:
    assert mixer_weight(8.0, 0.0, 0.5) == 16.0
    assert mixer_weight(8.0, 4.0, 0.5) == 12.0
    assert mixer_weight(8.0, 20.0, 0.5) == 8.0
    # an env that accepts nothing is dispatched like one that accepts everything
    assert mixer_weight(8.0, 0.0, 0.0) == mixer_weight(8.0, 0.0, 1.0) == 8.0


def test_inflight_caps_split_slots_by_littles_law_with_staleness_clip() -> None:
    caps = inflight_caps({"fast": 100.0, "slow": 100.0}, {"fast": 10.0, "slow": 30.0}, 400, None)
    assert caps == {"fast": 125.0, "slow": 375.0}
    bounds = {"fast": 1, "slow": 1}
    assert inflight_caps({"fast": 100.0, "slow": 10.0}, {"fast": 1.0, "slow": 1000.0}, 400, bounds)["slow"] == 20.0


def prefix_episode(task: vf.Task, *, reward: float, step: int, calls: int = 3) -> vf.Episode:
    """A fresh episode whose trace commits ``calls`` model calls, a tool result between each."""
    nodes = [vf.MessageNode(message=vf.UserMessage(content="q"), token_ids=[1], mask=[False])]
    for i in range(calls):
        if i:
            tool = vf.ToolMessage(content="out", tool_call_id=str(i))
            nodes.append(vf.MessageNode(message=tool, token_ids=[2], mask=[False], parent=len(nodes) - 1))
        reply = vf.AssistantMessage(content="a")
        nodes.append(vf.MessageNode(message=reply, token_ids=[10 + i], mask=[True], parent=len(nodes) - 1))
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=[vf.ModelCall(node=index) for index, node in enumerate(nodes) if any(node.mask)],
        rewards={"reward": vf.Reward(score=reward)},
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(name="swe"),
        task=trace.task,
        group=vf.GroupInfo(id="g"),
        traces=[trace],
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=step)),
    )


def test_prefix_source_dispatches_only_with_eligible_episodes() -> None:
    task = vf.Task(vf.TaskData(idx=0, prompt="q"))
    env_config = SimpleNamespace(ratio=1.0, group_size=2, max_off_policy_steps=8, curriculum=None)
    env = SimpleNamespace(name="swe", tasks=iter([task]), num_tasks=1, config=env_config)
    prefix = PrefixSourceConfig(name="swe-prefix", env="swe", rollouts="failed", max_age=2)
    source = TrainSource([env], batch_size=4, prefixes=[prefix])

    # Cold start: the empty buffer gets weight 0.
    fresh = source.next_task(step=1, capacity=8, inflight={})
    assert fresh.env_name == "swe" and fresh.prefix is None
    assert source.weights(1)["swe-prefix"] == 0.0

    # A finished fresh group feeds the buffer, all-fail or not; passed episodes are filtered out.
    group = [prefix_episode(task, reward=0.0, step=1), prefix_episode(task, reward=1.0, step=1)]
    source.on_group(fresh.group_id, "swe", group)
    assert len(source.buffers["swe-prefix"].entries) == 1
    requests = (source.next_task(step=2, capacity=8, inflight={}) for _ in range(100))
    request = next(r for r in requests if r.env_name == "swe-prefix")
    assert len(request.prefix.calls) in (1, 2) and request.task is task
    assert request.prefix.source["reward"] == 0.0
    assert not source.buffers["swe-prefix"].entries  # uses = 1

    # Prefix groups skip the curriculum.
    continuation = prefix_episode(task, reward=1.0, step=2)
    continuation.env.name = "swe-prefix"
    assert source.on_result([continuation]) is True

    # Episodes older than max_age expire.
    source.on_group(source.next_task(step=2, capacity=8, inflight={}).group_id, "swe", group)
    assert source.weights(5)["swe-prefix"] == 0.0
