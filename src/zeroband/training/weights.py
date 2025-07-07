import threading
import time
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from zeroband.training.model import Model
from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger


def save_weight_checkpoint(
    model: Model,
    tokenizer: AutoTokenizer,
    path: Path,
    dtype: torch.dtype = torch.bfloat16,
    async_save: bool = False,
) -> Path:
    """
    Save a HF-compatible weight-only checkpoint to the specified path. Saves the
    model weights as `model.pt`, the model config, generation arguments and tokenizer
    using HF's `save_pretrained` method.
    """
    # Get logger and world info
    logger = get_logger()
    is_master = get_world().rank == 0

    # Create checkpoint directory if it doesn't exist
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    # Gather distributed weights for weight checkpoint
    start_time = time.time()
    logger.debug("Gathering sharded weights")
    cpu_state = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value = value.to(dtype)
            # only gather after the downcast to dtype as it will be faster
            value = value.full_tensor()

        if is_master:
            key = get_fqns(model, key)
            assert len(key) == 1
            key = next(iter(key))
            # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
            cpu_state[key] = value.to("cpu", non_blocking=False)

    torch.distributed.barrier()
    logger.debug(f"Gathered sharded weights in {time.time() - start_time:.2f} seconds")

    model_path = path / "model.pt"

    def _save_weight_checkpoint():
        if is_master:
            start_time = time.time()
            logger.debug(f"Saving weights to {path}")

            # Save model weights to temporary file to avoid race condition
            tmp_model_path = path / "model.pt.tmp"
            torch.save(cpu_state, tmp_model_path)

            # Rename temporary file to indicate checkpoint is complete
            tmp_model_path.rename(model_path)

            # Save model config, generation arguments and tokenizer
            model.config.save_pretrained(path)
            model.generation_config.save_pretrained(path)
            tokenizer.save_pretrained(path)

            logger.debug(f"Saved weights to {path} in {time.time() - start_time:.2f} seconds")

    if async_save:
        thread = threading.Thread(target=_save_weight_checkpoint)
        thread.start()
    else:
        _save_weight_checkpoint()

    return model_path
