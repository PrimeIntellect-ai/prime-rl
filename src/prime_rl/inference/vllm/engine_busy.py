"""Engine step counters: how long each engine core spent executing model steps, and what those steps held.

Throughput over busy time is what an engine sustains busy all the time, which estimates its capacity while admission
holds it below its knee (the balance advisor's trainer-bound case). Decode engines step whenever they hold a sequence,
so their spare capacity is in batch size, not time: the step contents (sequences decoding, the context they read,
prefill tokens) let the advisor fit how step time grows with the batch and extrapolate to a full one.

vLLM exports no such metric, and its scheduler stats cannot carry extra fields across the engine-core process boundary,
so every engine core writes its running totals to a small file and the API server serves those files at
``/prime_rl/engine_busy``.
"""

import json
import os
import time
from pathlib import Path

BUSY_DIR_ENV = "PRIME_RL_ENGINE_BUSY_DIR"
WRITE_INTERVAL_S = 1.0

COUNTERS = ("busy_s", "steps", "decode_seqs", "kv_read_tokens", "prefill_tokens")
"""Running totals per engine: seconds executing steps, steps executed, and summed over the scheduled steps, the
sequences decoding, the context those sequences read and the prefill tokens."""


def busy_dir() -> Path:
    """The directory of this server's engines. The first process to ask (the server, before it spawns its engine cores
    and API server processes) fixes it in the environment, so every child inherits the same one."""
    path = os.environ.setdefault(BUSY_DIR_ENV, f"/dev/shm/prime_rl_engine_busy_{os.getpid()}")
    return Path(path)


def write_busy(directory: Path, engine_index: int, counters: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".engine_{engine_index}.{os.getpid()}"
    tmp.write_text(json.dumps(counters))
    os.replace(tmp, directory / f"engine_{engine_index}")


def read_busy(directory: Path) -> dict[str, dict[str, float]]:
    """Counters per engine index (as a string, like vLLM's ``engine`` metrics label)."""
    busy: dict[str, dict[str, float]] = {}
    if not directory.is_dir():
        return busy
    for path in directory.glob("engine_*"):
        try:
            busy[path.name.removeprefix("engine_")] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return busy


def step_contents(scheduler_output) -> tuple[int, int, int]:
    """Sequences decoding, the context they read from KV, and prefill tokens in one scheduled step. A scheduled
    request decodes when it runs one token (plus its speculative ones) on top of its computed context; everything
    else is prefill. A new request can decode on its first step: on a P/D decode engine, or after a full prefix-cache
    hit."""
    decode_seqs = kv_read = prefill = 0
    scheduled = scheduler_output.num_scheduled_tokens
    cached = scheduler_output.scheduled_cached_reqs
    requests = [(r.req_id, r.num_computed_tokens) for r in scheduler_output.scheduled_new_reqs]
    requests += zip(cached.req_ids, cached.num_computed_tokens)
    for req_id, computed in requests:
        tokens = scheduled.get(req_id, 0)
        if tokens <= 1 + len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())):
            decode_seqs += 1
            kv_read += computed
        else:
            prefill += tokens
    return decode_seqs, kv_read, prefill


def monkey_patch_engine_busy_time():
    """Count the wall time of every engine step that executed the model (the engine core runs steps only while it has
    requests, and a step that executed nothing, waiting on a KV transfer, yields instead of computing), and what each
    scheduled step held.

    Data-parallel MoE engines step in lockstep (``DPEngineCoreProc``): a rank without ready requests runs a dummy
    forward so the expert collectives stay in step with its peers. Its GPU is as occupied as theirs, so dummy steps
    count as busy steps holding nothing, and every rank of the group reports the group's busy time. The engines say
    they are ``lockstep`` so the advisor fits the group as one engine."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc

    if getattr(EngineCoreProc._process_engine_step, "_prime_rl_busy", False):
        return
    directory = busy_dir()
    original_step = EngineCoreProc._process_engine_step
    original_schedule = Scheduler.schedule
    original_dummy = EngineCore.execute_dummy_batch

    def schedule(self, *args, **kwargs):
        output = original_schedule(self, *args, **kwargs)
        decode_seqs, kv_read, prefill = step_contents(output)
        counters = self.__dict__.setdefault("_prime_rl_counters", dict.fromkeys(COUNTERS, 0.0))
        counters["decode_seqs"] += decode_seqs
        counters["kv_read_tokens"] += kv_read
        counters["prefill_tokens"] += prefill
        return output

    def _process_engine_step(self) -> bool:
        start = time.perf_counter()
        executed = original_step(self)
        now = time.perf_counter()
        counters = self.scheduler.__dict__.setdefault("_prime_rl_counters", dict.fromkeys(COUNTERS, 0.0))
        if executed:
            counters["busy_s"] += now - start
            counters["steps"] += 1
        if now - getattr(self, "_prime_rl_busy_written", 0.0) >= WRITE_INTERVAL_S:
            self._prime_rl_busy_written = now
            try:
                write_busy(
                    directory,
                    self.engine_index,
                    {
                        **counters,
                        "max_num_seqs": self.scheduler.max_num_running_reqs,
                        "lockstep": isinstance(self, DPEngineCoreProc),
                    },
                )
            except OSError:
                pass
        return executed

    def execute_dummy_batch(self):
        start = time.perf_counter()
        result = original_dummy(self)
        if isinstance(self, DPEngineCoreProc):
            counters = self.scheduler.__dict__.setdefault("_prime_rl_counters", dict.fromkeys(COUNTERS, 0.0))
            counters["busy_s"] += time.perf_counter() - start
            counters["steps"] += 1
        return result

    _process_engine_step._prime_rl_busy = True
    EngineCoreProc._process_engine_step = _process_engine_step
    EngineCore.execute_dummy_batch = execute_dummy_batch
    Scheduler.schedule = schedule
