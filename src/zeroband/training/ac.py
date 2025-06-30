import time

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from zeroband.training.config import ActivationCheckpointConfig
from zeroband.utils.logger import get_logger
from zeroband.utils.models import Model


def setup_ac(model: Model, ac_config: ActivationCheckpointConfig):
    logger = get_logger()
    logger.info(f"Applying activation checkpointing to every {ac_config.interval} layers")
    layers_ckpt = 0
    start_time = time.time()
    for layer_id, transformer_block in model.model.layers.named_children():
        if layer_id % ac_config.interval == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
            model.model.layers.register_module(layer_id, transformer_block)
            layers_ckpt += 1
    logger.info(
        f"Applied activation checkpointing to {layers_ckpt}/{len(model.model.layers)} layers in {time.time() - start_time:.2f} seconds"
    )
