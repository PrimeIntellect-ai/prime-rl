import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.monitors.prime import launch_run


def main() -> None:
    """``python -m prime_rl.entrypoints.run_owner {rl,sft} <config> <command...>``: run a multi-node
    SLURM job's command as the owner of its Prime run (``launch_run``). The command's
    non-zero exit, or SLURM's SIGTERM on cancel or time limit, fails the run."""
    kind, path, *command = sys.argv[1:]
    config_cls = OrchestratorConfig if kind == "rl" else SFTConfig
    config = config_cls.model_validate_json(Path(path).read_text())

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum) from RuntimeError("SLURM terminated the job")

    signal.signal(signal.SIGTERM, terminate)
    with launch_run(config, config.monitors.prime):
        code = subprocess.run(command).returncode
        if code != 0:
            raise SystemExit(code) from RuntimeError(f"The SLURM job step exited with code {code}")


if __name__ == "__main__":
    main()
