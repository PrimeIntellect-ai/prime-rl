from itertools import chain
import os
from typing import TypeAlias

import torch
from torch.distributed.tensor import DTensor

from zeroband.inference.config import InferenceConfig
from zeroband.training.model import Model
from zeroband.training.orchestrator.orchestrator import run_orchestrator
from zeroband.training.orchestrator.config import OrchestratorConfig
from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger
import multiprocessing as mp


class FakeTokenizer(object):
    def __init__(self):
        self.vocab_size = 1000
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return self.vocab_size


def get_real_tensor(tensor: torch.Tensor | DTensor):
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


OffloadedTensor: TypeAlias = list[tuple[torch.Tensor, int]]


def offload_model_to_cpu(model: Model) -> OffloadedTensor:
    """
    Retun a list of cpu tensor representing the model weight.
    Also reduce to 0 the gpu memory usage.
    """
    tensors_offloaded = []
    for param in chain(model.parameters(), model.buffers()):
        data = get_real_tensor(param.data)
        cpu_data = data.to("cpu", non_blocking=True)
        storage_size = data.untyped_storage().size()
        data.untyped_storage().resize_(1)
        tensors_offloaded.append((cpu_data, storage_size))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return tensors_offloaded


def copy_model_to_cpu(model: Model) -> OffloadedTensor:
    """
    Retun a list of cpu tensor representing the model weight.
    Keep gpu memory intact.
    """

    tensors_offloaded = []
    for param in chain(model.parameters(), model.buffers()):
        data = get_real_tensor(param.data)
        cpu_data = data.to("cpu")
        storage_size = data.untyped_storage().size()
        tensors_offloaded.append((cpu_data, storage_size))

    return tensors_offloaded


def wake_up_model_from_cpu(model: Model, tensors: OffloadedTensor):
    for param, (cpu_data, storage_size) in zip(chain(model.parameters(), model.buffers()), tensors):
        data = get_real_tensor(param.data)
        data.untyped_storage().resize_(storage_size)
        data.copy_(cpu_data, non_blocking=True)
    torch.cuda.synchronize()


def setup_orchestrator_sidecar(config: OrchestratorConfig) -> mp.Process:
    config.num_train_workers = get_world().world_size

    get_logger().info("Starting orchestrator in a separate process")

    # Create a queue for orchestrator to signal when setup is complete
    ctx = mp.get_context("spawn")
    setup_queue = ctx.Queue()
    orchestrator = ctx.Process(target=run_orchestrator, args=(config, setup_queue), daemon=True)
    orchestrator.start()

    # Wait for orchestrator to signal that setup is complete
    get_logger().info("Waiting for orchestrator to complete setup...")
    signal = setup_queue.get()
    if signal == "ready":
        get_logger().success("Orchestrator setup complete, continuing with training")
    else:
        raise RuntimeError(f"Unexpected signal from orchestrator: {signal}")
    return orchestrator  # type: ignore[return-value]


class ServerWithEnv:
    """
    Wrap a function to set environment variables and redirect stdout and stderr to devnull.
    """

    def __init__(self, config, env_vars, fn):
        self.config = config
        self.env_vars = env_vars
        self.fn = fn

    def __call__(self, *args, **kwargs):
        os.environ.update(self.env_vars)
        # Redirect stdout and stderr to devnull at file descriptor level
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull_fd, 1)  # Redirect stdout
            os.dup2(devnull_fd, 2)  # Redirect stderr
            self.fn(self.config, vllm_args=[])
        finally:
            os.dup2(old_stdout_fd, 1)  # Restore stdout
            os.dup2(old_stderr_fd, 2)  # Restore stderr
            os.close(devnull_fd)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def setup_inference_sidecar(config: InferenceConfig) -> mp.Process:
    from zeroband.inference.vllm.server import server

    num_gpus_inference = config.num_gpus()
    num_gpus_available = torch.cuda.device_count()
    num_gpus_training = get_world().world_size

    if num_gpus_inference + num_gpus_training > num_gpus_available:
        raise ValueError(
            f"Not enough GPUs available. {num_gpus_inference} inference GPUs + {num_gpus_training} training GPUs > {num_gpus_available} available GPUs"
        )
    env_vars = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(num_gpus_training + i) for i in range(num_gpus_inference)),
        "VLLM_LOGGING_LEVEL": "ERROR",
    }
    get_logger().info(f"Env vars: {env_vars}")

    get_logger().info("Starting inference in a separate process")

    # Create a queue for orchestrator to signal when setup is complete
    ctx = mp.get_context("spawn")
    inference = ctx.Process(target=ServerWithEnv(config, env_vars, server), daemon=False)
    inference.start()

    return inference  # type: ignore[return-value]


def terminate_sidecar(sidecar: mp.Process, name: str):
    if sidecar.is_alive():
        get_logger().info(f"Terminating {name} process")
        sidecar.terminate()
        sidecar.join(timeout=5)
        if sidecar.is_alive():
            get_logger().warning(f"{name} process did not terminate gracefully, forcing kill")
            sidecar.kill()
            sidecar.join()
