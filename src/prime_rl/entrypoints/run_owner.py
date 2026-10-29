import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.monitors.prime import launch_run


def main() -> None:
    """``python -m prime_rl.entrypoints.run_owner {rl,sft} <config> <command...>``: run a
    multi-node SLURM job's step as the owner of its Prime run (``launch_run``). The step's
    clean exit finishes the run; its non-zero exit, or SLURM's SIGTERM on cancel or time
    limit, fails it."""
    kind, path, *command = sys.argv[1:]
    config = (RLConfig if kind == "rl" else SFTConfig).model_validate_json(Path(path).read_text())
    prime = config.orchestrator.monitors.prime if isinstance(config, RLConfig) else config.monitors.prime
    step = subprocess.Popen(command)

    def forward(signum: int, frame: Any) -> None:
        step.send_signal(signum)

    def terminate(signum: int, frame: Any) -> None:
        # srun relays the signal to the step's tasks, which tear down on their own
        # (e.g. finish an in-flight checkpoint); a later SIGTERM only reaches srun.
        forward(signum, frame)
        signal.signal(signal.SIGTERM, forward)
        raise SystemExit(128 + signum) from RuntimeError("SLURM terminated the job")

    signal.signal(signal.SIGTERM, terminate)
    try:
        with launch_run(config, prime):
            code = step.wait()
            if code != 0:
                raise SystemExit(code) from RuntimeError(f"The SLURM job step exited with code {code}")
    finally:
        # The batch script must outlive the step: SLURM ends the job's steps with it.
        step.wait()


if __name__ == "__main__":
    main()
