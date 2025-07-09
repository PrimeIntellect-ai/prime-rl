import os
import subprocess
import sys
from itertools import chain
from subprocess import Popen
from typing import List

from zeroband.inference.config import InferenceConfig
from zeroband.orchestrator.config import OrchestratorConfig
from zeroband.trainer.config import TrainerConfig
from zeroband.utils.pydantic_config import BaseSettings, parse_argv


class RLConfig(BaseSettings):
    """Configures an RL training run."""

    trainer: TrainerConfig
    inference: InferenceConfig
    orchestrator: OrchestratorConfig


def cleanup(processes: List[Popen]):
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def to_cli(prefix, d):
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from to_cli(path, v)
        else:
            if isinstance(v, bool):
                if v:
                    yield (f"--{path}",)
            else:
                yield f"--{path}", str(v)


def rl(config: RLConfig):
    processes: List[subprocess.Popen] = []
    try:
        print("Starting processes...")
        # Start inference process
        inference_args = list(chain.from_iterable(to_cli("", config.inference.model_dump())))
        inference_cmd = ["uv", "run", "inference", *inference_args]
        inference_process = subprocess.Popen(
            inference_cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(inference_process)

        # Start orchestrator process
        orchestrator_args = list(chain.from_iterable(to_cli("", config.orchestrator.model_dump())))
        orchestrator_cmd = ["uv", "run", "orchestrator", *orchestrator_args]
        orchestrator_process = subprocess.Popen(
            orchestrator_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(orchestrator_process)

        # Start training process
        train_args = list(chain.from_iterable(to_cli("", config.trainer.model_dump())))
        training_cmd = ["uv", "run", "trainer", *train_args]
        training_process = subprocess.Popen(
            training_cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(training_process)

        # Wait for all processes to complete
        print("Waiting for all processes to complete...")
        orchestrator_process.wait()
        training_process.wait()
        print("Done!")
        cleanup(processes)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, terminating all processes...")
        cleanup(processes)
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        cleanup(processes)
        raise


def main():
    rl(parse_argv(RLConfig))


if __name__ == "__main__":
    main()
