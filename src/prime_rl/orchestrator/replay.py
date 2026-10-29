"""Rebuild the trainer's packed micro batches from a run's recorded episodes.

The file monitor records every episode as it arrives, before the orchestrator scores it. This
module runs the recorded train episodes through the orchestrator's own sample building
(``trace_to_samples``) and packing (``prepare_batch``), so the trainer can replay real batches
(document lengths, packing, token ids) without inference or an orchestrator.

Token ids, loss masks, inference logprobs, branch identity and env names come from the records,
and temperatures from each trace's recorded sampling config (1.0 if it has none), which can
differ from the env-level temperature the live orchestrator uses when a trainable seat overrides
temperature. Records are written before scoring, so they carry no advantages: every trace gets one
scalar advantage drawn from a standard normal seeded by its trace id, spread over its sampled
tokens like the orchestrator's rollout scalar. The arrays env servers kept for the trainer
(routing, sampling masks, top logprobs) are not recorded, so replayed batches carry none.
Replayed samples always use the RL loss, so algorithms whose ``action_loss_type`` is ``ce`` or
``ref_kl`` are not supported.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from zlib import crc32

import numpy as np
import orjson
import pyzstd
import verifiers.v1 as vf

from prime_rl.monitors.file.traces.index import episode_kind
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.utils import episode_env_name
from prime_rl.transports.batch.types import MicroBatch, TrainingSample


def read_records(paths: Iterable[Path]) -> Iterator[dict]:
    """Episode records of trace stream chunks (plain or zstd JSONL), one line at a time."""
    for path in paths:
        with pyzstd.open(path, "rb") if path.suffix == ".zst" else open(path, "rb") as file:
            for line in file:
                yield orjson.loads(line)


def synthetic_advantage(trace_id: str) -> float:
    return float(np.random.default_rng(crc32(trace_id.encode())).standard_normal())


def record_samples(record: dict) -> list[list[TrainingSample]]:
    """The training samples of a recorded train episode, one list per trainable trace."""
    if episode_kind(record) != "train" or not record["ok"]:
        return []
    temperatures = {}
    for trace in record["traces"]:
        sampling = trace["agent"]["config"].get("sampling") or {}
        temperatures[trace["id"]] = sampling.get("temperature", 1.0)
        # Only the token graph is needed, and the agent config can name a renderer file that
        # resolves only on the machine that ran the episode, failing WireEpisode validation elsewhere.
        trace["agent"]["config"] = {}
    episode = vf.WireEpisode.model_validate(record)
    samples_by_trace = []
    for _, trace in iter_trainable_traces([episode]):
        samples = trace_to_samples(trace, env_name=episode_env_name(episode))
        for sample in samples:
            sample.temperatures = np.full(len(sample.token_ids), temperatures[trace.id], dtype=np.float32)
            sample.payload = None
            if sample.advantages is None:
                sample.advantages = np.where(sample.mask, synthetic_advantage(trace.id), 0.0).astype(np.float32)
        if samples:
            samples_by_trace.append(samples)
    return samples_by_trace


def replay_batches(
    records: Iterable[dict],
    rollouts_per_step: int,
    seq_len: int,
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
    pad_to_multiple_of: int = 1,
) -> Iterator[list[list[MicroBatch]]]:
    """Pack each run of ``rollouts_per_step`` trainable traces into one step's micro batches per
    data rank, in record order. A trailing partial step is dropped."""
    samples: list[TrainingSample] = []
    num_rollouts = 0
    for record in records:
        for trace_samples in record_samples(record):
            samples.extend(trace_samples)
            num_rollouts += 1
            if num_rollouts == rollouts_per_step:
                yield prepare_batch(samples, seq_len, num_train_workers, bin_cost, pad_to_multiple_of)
                samples, num_rollouts = [], 0
