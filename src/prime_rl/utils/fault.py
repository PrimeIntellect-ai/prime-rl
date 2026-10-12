"""Opt-in fault injection for fault-tolerance tests, configured by ``PRIME_FAULT``.

``PRIME_FAULT`` is a comma-separated list of ``<fault>@<key>`` entries, for example
``PRIME_FAULT="trainer_crash@5,ckpt_write_fail@10"``. A step fault fires when its call
site reaches that step; an env fault applies to every request to the env of that name.
The variable is read once at import; unset, every hook is a dict lookup.

Step faults are one-shot when ``PRIME_FAULT_DIR`` is set: the first process to reach
one creates a marker file there, so exactly one rank or engine fires it, and a resumed
or requeued run does not fire it again. The directory must be visible to every node.
"""

import asyncio
import errno
import os
import sys
import time


class InjectedFault(RuntimeError):
    pass


def _crash(name: str) -> None:
    raise InjectedFault(f"PRIME_FAULT {name}")


def _exit(name: str) -> None:
    os._exit(1)


def _hang(name: str) -> None:
    while True:
        time.sleep(3600)


def _disk_full(name: str) -> None:
    raise OSError(errno.ENOSPC, f"PRIME_FAULT {name}")


STEP_FAULTS = {
    "engine_crash": _exit,
    "broadcast_hang": _hang,
    "trainer_crash": _crash,
    "orchestrator_crash": _crash,
    "ckpt_write_fail": _disk_full,
}
ENV_FAULTS = ("env_error", "env_hang")


def parse(spec: str) -> dict[str, str]:
    """``"trainer_crash@5,env_error@math"`` -> ``{"trainer_crash": "5", "env_error": "math"}``."""
    faults = {}
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        name, _, key = entry.partition("@")
        if name not in (*STEP_FAULTS, *ENV_FAULTS) or not key:
            raise ValueError(
                f"Invalid PRIME_FAULT entry {entry!r}, expected <fault>@<key> with a fault in {[*STEP_FAULTS, *ENV_FAULTS]}"
            )
        faults[name] = key
    return faults


FAULTS = parse(os.environ.get("PRIME_FAULT", ""))
FAULT_DIR = os.environ.get("PRIME_FAULT_DIR")


def _claim(marker: str) -> bool:
    if FAULT_DIR is None:
        return True
    os.makedirs(FAULT_DIR, exist_ok=True)
    try:
        os.close(os.open(os.path.join(FAULT_DIR, marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def inject(name: str, step: int | str) -> None:
    """Fire step fault ``name`` if ``PRIME_FAULT`` asks for it at ``step``."""
    if FAULTS.get(name) == str(step) and _claim(f"{name}@{step}"):
        print(f"PRIME_FAULT: injecting {name}@{step} (pid {os.getpid()})", file=sys.stderr, flush=True)
        STEP_FAULTS[name](name)


async def inject_env(env_name: str) -> None:
    """Stall forever (``env_hang``) or fail (``env_error``) a request to ``env_name``."""
    if FAULTS.get("env_hang") == env_name:
        await asyncio.Event().wait()
    if FAULTS.get("env_error") == env_name:
        await asyncio.sleep(1)
        raise InjectedFault(f"PRIME_FAULT env_error@{env_name}")
