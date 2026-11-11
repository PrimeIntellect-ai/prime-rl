"""Engine busy time: seconds each engine core spent executing model steps.

Throughput over busy time is what an engine sustains once it is busy all the time, so it estimates an engine's capacity
while admission holds it below its knee (the balance advisor's trainer-bound case). vLLM exports no such metric, and its
scheduler stats cannot carry extra fields across the engine-core process boundary, so every engine core writes its
running total to a small file and the API server serves those files at ``/prime_rl/engine_busy``.
"""

import os
import time
from pathlib import Path

BUSY_DIR_ENV = "PRIME_RL_ENGINE_BUSY_DIR"
WRITE_INTERVAL_S = 1.0


def busy_dir() -> Path:
    """The directory of this server's engines. The first process to ask (the server, before it spawns its engine cores
    and API server processes) fixes it in the environment, so every child inherits the same one."""
    path = os.environ.setdefault(BUSY_DIR_ENV, f"/dev/shm/prime_rl_engine_busy_{os.getpid()}")
    return Path(path)


def write_busy(directory: Path, engine_index: int, seconds: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".engine_{engine_index}.{os.getpid()}"
    tmp.write_text(f"{seconds:.6f}")
    os.replace(tmp, directory / f"engine_{engine_index}")


def read_busy(directory: Path) -> dict[str, float]:
    """Busy seconds per engine index (as a string, like vLLM's ``engine`` metrics label)."""
    busy: dict[str, float] = {}
    if not directory.is_dir():
        return busy
    for path in directory.glob("engine_*"):
        try:
            busy[path.name.removeprefix("engine_")] = float(path.read_text())
        except (OSError, ValueError):
            continue
    return busy


def monkey_patch_engine_busy_time():
    """Count the wall time of every engine step that executed the model: the engine core runs steps only while it has
    requests, and a step that executed nothing (waiting on a KV transfer) yields instead of computing."""
    from vllm.v1.engine.core import EngineCoreProc

    if getattr(EngineCoreProc._process_engine_step, "_prime_rl_busy", False):
        return
    directory = busy_dir()
    original = EngineCoreProc._process_engine_step

    def _process_engine_step(self) -> bool:
        start = time.perf_counter()
        executed = original(self)
        now = time.perf_counter()
        if executed:
            self._prime_rl_busy_s = getattr(self, "_prime_rl_busy_s", 0.0) + (now - start)
        if now - getattr(self, "_prime_rl_busy_written", 0.0) >= WRITE_INTERVAL_S:
            self._prime_rl_busy_written = now
            try:
                write_busy(directory, self.engine_index, getattr(self, "_prime_rl_busy_s", 0.0))
            except OSError:
                pass
        return executed

    _process_engine_step._prime_rl_busy = True
    EngineCoreProc._process_engine_step = _process_engine_step
