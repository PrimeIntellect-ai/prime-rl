"""``python -m prime_rl.entrypoints.supervise --name NAME --restarts N [--group] COMMAND...``: run each shell
command and restart it in place when it fails, up to N times. With ``--group``, one failed command restarts
all of them (the ranks of an expert-parallel inference replica only work together). A restarted command sees
``PRL_RESTART_COUNT``. Exits with the command's code once it fails after its last restart, and with 0 once
every command exited cleanly."""

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time

from prime_rl.utils.logger import get_logger, setup_logger

RESTART_DELAY_S = 10
STOP_TIMEOUT_S = 60


def start(command: str, restart_count: int) -> subprocess.Popen:
    # Its own process group, so a restart also reaches the processes it spawned after it died
    return subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", command],
        env={**os.environ, "PRL_RESTART_COUNT": str(restart_count)},
        start_new_session=True,
    )


def alive(process: subprocess.Popen) -> bool:
    """Whether any process of the command's process group is left."""
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop(processes: list[subprocess.Popen]) -> None:
    """Kill the commands' process groups and wait until all of them are gone (and their GPU memory freed)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, sig)
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while any(alive(process) for process in processes) and time.monotonic() < deadline:
            time.sleep(1)
    for process in processes:
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--group", action="store_true")
    parser.add_argument("commands", nargs="+")
    args = parser.parse_args()
    setup_logger("info")
    logger = get_logger()

    commands: list[str] = args.commands
    restart_counts = [0] * len(commands)
    processes = [start(command, 0) for command in commands]

    def terminate(signum, frame) -> None:
        stop(processes)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    finished: set[int] = set()
    while len(finished) < len(commands):
        time.sleep(1)
        for index, process in enumerate(processes):
            code = process.poll()
            if index in finished or code is None:
                continue
            if code == 0:
                finished.add(index)
                continue
            targets = range(len(commands)) if args.group else [index]
            if restart_counts[index] >= args.restarts:
                logger.error(f"{args.name}: command {index} exited with code {code} after {args.restarts} restarts")
                stop(processes)
                sys.exit(code)
            for target in targets:
                restart_counts[target] += 1
            logger.warning(
                f"{args.name}: command {index} exited with code {code}; restarting "
                f"{'all commands' if args.group else 'it'} ({restart_counts[index]}/{args.restarts})"
            )
            stop([processes[target] for target in targets])
            time.sleep(RESTART_DELAY_S)
            for target in targets:
                processes[target] = start(commands[target], restart_counts[target])


if __name__ == "__main__":
    main()
