import asyncio
from collections import Counter

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.orchestrator.replay import read_records, replay_batches, synthetic_advantage
from prime_rl.transports.batch.types import MICRO_BATCH_DTYPES

SEQ_LEN = 64
PAD_TO = 8
NUM_RANKS = 2
TEMPERATURE = 0.7


def recorded_trace(prompt_len: int, completion_lens: list[int], *, start_id: int) -> vf.Trace:
    """A trace as the file monitor records it (unscored, no advantages), with one completion per
    branch forking off a shared prompt."""
    prompt = MessageNode(
        message=UserMessage(content="prompt"),
        token_ids=list(range(start_id, start_id + prompt_len)),
        mask=[False] * prompt_len,
    )
    nodes = [prompt]
    next_id = start_id + prompt_len
    for completion_len in completion_lens:
        nodes.append(
            MessageNode(
                parent=0,
                message=AssistantMessage(content="answer"),
                sampled=True,
                token_ids=list(range(next_id, next_id + completion_len)),
                mask=[True] * completion_len,
                logprobs=[-0.5] * completion_len,
            )
        )
        next_id += completion_len
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig(sampling=vf.SamplingConfig(temperature=TEMPERATURE))),
        nodes=nodes,
        ok=True,
        is_completed=True,
    )


def recorded_episode(trace: vf.Trace, *, ok: bool = True, work: str = "train") -> vf.Episode:
    work_info = vf.TrainWorkInfo(step=1) if work == "train" else vf.EvalWorkInfo(step=1)
    return vf.Episode(
        env=vf.EnvInfo(id="test", name="test-env"),
        task=trace.task,
        traces=[trace],
        ok=ok,
        run=vf.TrainRunInfo(id="run", work=work_info),
    )


def test_replay_batches_pack_recorded_train_rollouts_like_the_orchestrator(tmp_path):
    shapes = [(10, [20]), (5, [30, 12]), (40, [24]), (3, [7]), (16, [48]), (8, [9, 9, 9]), (12, [4])]
    traces = [
        recorded_trace(prompt_len, completions, start_id=1000 * i) for i, (prompt_len, completions) in enumerate(shapes)
    ]
    episodes = [recorded_episode(trace) for trace in traces]
    episodes.insert(1, recorded_episode(recorded_trace(6, [6], start_id=50_000), work="eval"))
    episodes.insert(4, recorded_episode(recorded_trace(6, [6], start_id=60_000), ok=False))
    monitor = FileMonitor(FileMonitorConfig(compress=True))
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_episodes(episodes, step=1, kind="train", subset="all"))
    asyncio.run(monitor.finalize())
    chunks = sorted(get_trace_stream(tmp_path).glob("*.jsonl*"))
    assert [chunk.suffix for chunk in chunks] == [".zst"]
    unpinned_ids = {traces[1].id, traces[4].id}

    def unpin_temperatures(records):
        for record in records:
            for trace in record["traces"]:
                if trace["id"] in unpinned_ids:
                    trace["agent"]["config"]["sampling"]["temperature"] = None
            yield record

    steps = list(
        replay_batches(
            unpin_temperatures(read_records(chunks)),
            rollouts_per_step=3,
            seq_len=SEQ_LEN,
            num_train_workers=NUM_RANKS,
            bin_cost=sum,
            pad_to_multiple_of=PAD_TO,
        )
    )

    # Seven train rollouts make two full steps; the partial third and the eval and failed episodes are dropped.
    assert len(steps) == 2
    for step, grid in enumerate(steps):
        step_traces = {trace.id: trace for trace in traces[3 * step : 3 * step + 3]}
        assert len(grid) == NUM_RANKS
        assert len({len(rank) for rank in grid}) == 1
        rank_loads = [sum(len(mb.input_ids) for mb in rank) for rank in grid]
        largest_row = max(len(mb.input_ids) for rank in grid for mb in rank)
        assert max(rank_loads) - min(rank_loads) <= largest_row

        sequences = Counter()
        for micro_batch in (mb for rank in grid for mb in rank):
            num_tokens = len(micro_batch.input_ids)
            assert num_tokens <= SEQ_LEN and num_tokens % PAD_TO == 0
            assert sum(micro_batch.seq_lens) == num_tokens
            for name, dtype in MICRO_BATCH_DTYPES.items():
                stream = getattr(micro_batch, name)
                assert stream is None or (stream.dtype == dtype and len(stream) == num_tokens), name
            assert micro_batch.payloads is None
            if micro_batch.trace_ids is None:
                assert not micro_batch.loss_mask.any()
                continue
            offset = 0
            for trace_id, branch_index, length in zip(
                micro_batch.trace_ids, micro_batch.branch_indices, micro_batch.sequence_lengths
            ):
                is_last = offset + length == num_tokens
                real = length - (micro_batch.padding if is_last else 0)
                branch = step_traces[trace_id].branches[branch_index]
                tokens = slice(offset, offset + real)
                np.testing.assert_array_equal(micro_batch.input_ids[tokens], branch.token_ids)
                np.testing.assert_array_equal(micro_batch.loss_mask[tokens], branch.sampled_mask)
                np.testing.assert_array_equal(micro_batch.inference_logprobs[tokens], branch.logprobs)
                expected_temperature = 1.0 if trace_id in unpinned_ids else TEMPERATURE
                np.testing.assert_array_equal(micro_batch.temperatures[tokens], np.float32(expected_temperature))
                advantage = np.float32(synthetic_advantage(trace_id))
                expected_advantages = np.where(branch.sampled_mask, advantage, np.float32(0))
                np.testing.assert_array_equal(micro_batch.advantages[tokens], expected_advantages)
                assert not micro_batch.loss_mask[offset + real : offset + length].any()
                sequences[trace_id, branch_index] += 1
                offset += length
        expected = Counter((trace.id, branch.index) for trace in step_traces.values() for branch in trace.branches)
        assert sequences == expected
