import io

import pytest
import torch
import verifiers.v1 as vf

from prime_rl.configs.algorithm import NGUAlgoConfig
from prime_rl.orchestrator.algo.ngu import anchored_advantages
from prime_rl.orchestrator.ngu import NGUController


def episodes(task, rewards, version=0):
    result = []
    for reward in rewards:
        trace = vf.Trace(
            task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            nodes=[
                vf.MessageNode(
                    message=vf.AssistantMessage(content="x"), token_ids=[1], mask=[True], sampled=True, logprobs=[-0.1]
                )
            ],
            rewards={"solved": vf.Reward(score=reward)},
            ok=True,
        )
        result.append(
            vf.Episode(
                ok=True,
                task=trace.task,
                traces=[trace],
                env=vf.EnvInfo(id="test", name="test"),
                run=vf.TrainRunInfo(
                    id="run", work=vf.TrainWorkInfo(step=version + 1, policy=vf.PolicySpan(start=version, end=version))
                ),
            )
        )
    return [vf.WireEpisode.model_validate(e.model_dump(context={"float_decimals": None})) for e in result]


@pytest.mark.parametrize("rewards", [[0, 0], [1, 1], [0, 1], [0, 0, 0, 1]])
def test_ngu_fresh_advantages_match_grpo(rewards):
    result = anchored_advantages(rewards, len(rewards), sum(rewards))
    assert result == pytest.approx([r - sum(rewards) / len(rewards) for r in rewards])


def test_ngu_anchor_uses_expired_rewards_and_balances_negatives():
    assert anchored_advantages([0, 0, 0, 1], 100, 1) == pytest.approx([-0.33, -0.33, -0.33, 0.99])
    assert anchored_advantages([1, 1], 100, 2) == [0, 0]
    with pytest.raises(ValueError, match="binary"):
        anchored_advantages([0.5, 1], 2, 1)
    with pytest.raises(ValueError, match="exceed"):
        anchored_advantages([1, 1], 3, 1)


def test_ngu_retry_new_group_same_task_and_expiry():
    task = vf.Task(vf.TaskData(idx=0, prompt="x"))
    controller = NGUController(NGUAlgoConfig(seed=1), "test")
    first = controller.start(task, 1)
    assert controller.finish(first.group_id, episodes(task, [0] * 4), complete=True, min_version=0) is None
    retry = controller.next_retry(5)
    assert retry.task is task and retry.group_id != first.group_id and retry.step == 5
    cohort = controller.finish(retry.group_id, episodes(task, [0, 0, 0, 1], 4), complete=True, min_version=1)
    assert (cohort.attempts, cohort.successes, len(cohort.episodes)) == (8, 1, 4)
    assert anchored_advantages([e.traces[0].reward for e in cohort.episodes], 8, 1) == pytest.approx(
        [-0.875 / 3] * 3 + [0.875]
    )
    assert not controller.visits and not controller.retries
    assert controller.counters["expired_payloads"] == 4


def test_ngu_visits_independent_errors_not_failures_and_give_up():
    task = vf.Task(vf.TaskData(idx=0, prompt="x"))
    controller = NGUController(NGUAlgoConfig(continuation_probability=0), "test")
    first, second = controller.start(task, 1), controller.start(task, 1)
    assert controller.finish(first.group_id, episodes(task, [0, 0]), complete=True, min_version=0) is None
    cohort = controller.finish(second.group_id, episodes(task, [0, 1]), complete=True, min_version=0)
    assert cohort.attempts == 2
    failed = controller.start(task, 1)
    assert controller.finish(failed.group_id, episodes(task, [0]), complete=False, min_version=0) is None
    assert controller.counters["valid_attempts"] == 4
    assert controller.counters["incomplete_visits"] == 1
    assert not controller.retries


def test_ngu_checkpoint_preserves_counts_payloads_and_rng():
    task = vf.Task(vf.TaskData(idx=0, prompt="x"))
    config = NGUAlgoConfig(seed=1)
    controller = NGUController(config, "test")
    request = controller.start(task, 1)
    controller.finish(request.group_id, episodes(task, [0] * 4), complete=True, min_version=0)
    assert len(next(iter(controller.visits.values())).cohort.episodes) == 4
    buffer = io.BytesIO()
    torch.save(controller.state_dict(), buffer)
    buffer.seek(0)
    restored = NGUController(config, "test")
    restored.load_state_dict(torch.load(buffer, weights_only=False))
    assert restored.rng.getstate() == controller.rng.getstate()
    retry = restored.next_retry(3)
    cohort = restored.finish(retry.group_id, episodes(task, [0, 1], 2), complete=True, min_version=0)
    assert (cohort.attempts, cohort.successes, len(cohort.episodes)) == (6, 1, 6)
    assert len({e.id for e in cohort.episodes}) == 6


@pytest.mark.parametrize(
    "config",
    [
        {"continuation_probability": 1},
        {"continuation_probability": -0.1},
        {"sampling": {"source": {"name": "frozen", "base_url": "http://localhost"}}},
    ],
)
def test_ngu_invalid_configuration(config):
    with pytest.raises(ValueError):
        NGUAlgoConfig(**config)


def test_ngu_sink_checkpoint_preserves_wire_payloads_and_aliases():
    import numpy as np

    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator.metrics import TrainEpisodes
    from prime_rl.orchestrator.train_sink import TrainSink
    from prime_rl.orchestrator.types import Progress

    def sink():
        return TrainSink(
            OrchestratorConfig(),
            tokenizer=None,
            train_envs=None,
            progress=Progress(),
            batch_size=4,
            token_batch_size=None,
        )

    episode = episodes(vf.Task(vf.TaskData(idx=0, prompt="x")), [0])[0]
    trace = episode.traces[0]
    node = trace.nodes[0]
    node.logprobs = [-0.123456789012345]
    node.advantages = [-0.876543210987654]
    node.routed_experts = np.array([[[1, 2]]], dtype=np.uint8)
    original = sink()
    original.episode_by_trace[trace.id] = episode
    original.pending_episodes = TrainEpisodes([episode], sampled_trace_ids={trace.id})
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = sink()
    restored.load_state_dict(torch.load(buffer, weights_only=False))
    recovered = restored.episode_by_trace[trace.id]
    assert recovered is restored.pending_episodes.episodes[0]
    restored_node = recovered.traces[0].nodes[0]
    assert restored_node.logprobs == node.logprobs
    assert restored_node.advantages == node.advantages
    np.testing.assert_array_equal(restored_node.routed_experts, node.routed_experts)


def test_ngu_group_can_span_fixed_size_batches():
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator.metrics import TrainEpisodes
    from prime_rl.orchestrator.train_sink import TrainSink
    from prime_rl.orchestrator.types import Progress
    from prime_rl.transports.batch import TrainingSample

    group = episodes(vf.Task(vf.TaskData(idx=0, prompt="x")), [0, 0, 0, 0, 1])
    advantages = anchored_advantages([0, 0, 0, 0, 1], 5, 1)
    sink = TrainSink(
        OrchestratorConfig(),
        tokenizer=None,
        train_envs=None,
        progress=Progress(),
        batch_size=4,
        token_batch_size=None,
    )
    for episode, advantage in zip(group, advantages, strict=True):
        trace = episode.traces[0]
        sink.episode_by_trace[trace.id] = episode
        sink.pending_batch[trace.id] = [
            TrainingSample(
                token_ids=[1],
                mask=[True],
                logprobs=[-0.1],
                temperatures=[1.0],
                advantages=[advantage],
                env_name="test",
                trace_id=trace.id,
            )
        ]
    sink.pending_episodes = TrainEpisodes(group, sampled_trace_ids=set(sink.pending_batch))
    first = sink.process_batch()
    assert len(first.samples) == 4
    assert len(sink.pending_batch) == 1
    second = sink.process_batch()
    assert [s.advantages[0] for s in first.samples + second.samples] == advantages
    assert not sink.pending_batch
