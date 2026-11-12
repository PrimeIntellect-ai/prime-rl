"""Engine busy time: the seconds each engine core spent executing model steps.

Throughput over busy time is what an engine sustains busy all the time, which the balance advisor reads as its
capacity. vLLM exports no such metric, and its scheduler stats cannot carry extra fields across the engine-core
process boundary, so every engine core writes its running total to a small file and the API server serves those files
at ``/prime_rl/engine_busy``.
"""

import json
import os
import time
from pathlib import Path

BUSY_DIR_ENV = "PRIME_RL_ENGINE_BUSY_DIR"
WRITE_INTERVAL_S = 1.0


def busy_dir() -> Path:
    """The directory of this server's engines. The first process to ask (the server, before it spawns its engine cores
    and API server processes) fixes it in the environment, so every child inherits the same one."""
    return Path(os.environ.setdefault(BUSY_DIR_ENV, f"/dev/shm/prime_rl_engine_busy_{os.getpid()}"))


def read_busy(directory: Path) -> dict[str, float]:
    """Busy seconds per engine index (as a string, like vLLM's ``engine`` metrics label)."""
    busy: dict[str, float] = {}
    for path in directory.glob("engine_*") if directory.is_dir() else ():
        try:
            busy[path.name.removeprefix("engine_")] = float(path.read_text())
        except (OSError, ValueError):
            continue
    return busy


def monkey_patch_engine_busy_time():
    """Count the wall time of every engine step that executed the model (the engine core runs steps only while it has
    requests, and a step that executed nothing, waiting on a KV transfer, yields instead of computing).

    Data-parallel MoE engines step in lockstep (``DPEngineCoreProc``): a rank without ready requests runs a dummy
    forward so the expert collectives stay in step with its peers. Its GPU is as occupied as theirs, so dummy steps
    count as busy."""
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc

    if getattr(EngineCoreProc._process_engine_step, "_prime_rl_busy", False):
        return
    directory = busy_dir()
    original_step = EngineCoreProc._process_engine_step
    original_dummy = EngineCore.execute_dummy_batch

    def _process_engine_step(self) -> bool:
        start = time.perf_counter()
        executed = original_step(self)
        now = time.perf_counter()
        self._prime_rl_busy_s = getattr(self, "_prime_rl_busy_s", 0.0) + (now - start if executed else 0.0)
        if now - getattr(self, "_prime_rl_busy_written", 0.0) >= WRITE_INTERVAL_S:
            self._prime_rl_busy_written = now
            try:  # a full /dev/shm loses the advice, not the engine
                directory.mkdir(parents=True, exist_ok=True)
                tmp = directory / f".engine_{self.engine_index}.{os.getpid()}"
                tmp.write_text(json.dumps(self._prime_rl_busy_s))
                os.replace(tmp, directory / f"engine_{self.engine_index}")
            except OSError:
                pass
        return executed

    def execute_dummy_batch(self):
        start = time.perf_counter()
        result = original_dummy(self)
        if isinstance(self, DPEngineCoreProc):
            self._prime_rl_busy_s = getattr(self, "_prime_rl_busy_s", 0.0) + time.perf_counter() - start
        return result

    _process_engine_step._prime_rl_busy = True
    EngineCoreProc._process_engine_step = _process_engine_step
    EngineCore.execute_dummy_batch = execute_dummy_batch
