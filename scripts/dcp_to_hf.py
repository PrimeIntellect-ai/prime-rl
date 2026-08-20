"""Convert a DCP trainer checkpoint into HF-format weights offline.

HF weight checkpoints are written online only at eval steps; every other export
goes through this script. It mirrors the trainer's save path exactly: build the
model, DCP-load the checkpoint's model state, gather rank-parallel, convert to
HF format, and write sharded safetensors plus config/tokenizer assets.

Usage (from the prime-rl repo; more ranks = faster gathers and writes):
    uv run torchrun --nproc-per-node 8 scripts/dcp_to_hf.py \
        --model.name <hf-id-or-path> \
        --ckpt-dir <run>/checkpoints/step_N/trainer \
        --output-dir <out>

``--model.*`` must match the run that wrote the checkpoint (impl, attn, ...) so
the state-dict FQNs and the prime->HF conversion line up.
"""

from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict_parallel,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger


class DcpToHfConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    ckpt_dir: Path
    """The DCP checkpoint to convert (``<run>/checkpoints/step_N/trainer``)."""

    output_dir: Path
    """Where to write the HF-format weights."""


def main(config: DcpToHfConfig) -> None:
    logger = setup_logger("info")
    world = get_world()
    setup_torch_distributed()

    parallel_dims = get_parallel_dims(config.model)
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=True)

    logger.info(f"Loading DCP checkpoint from {config.ckpt_dir}")
    dcp_load(state_dict={"app": AppState(model, [], None, None)}, checkpoint_id=config.ckpt_dir)

    logger.info("Gathering and converting weights")
    state_dict = gather_weights_parallel(model, dtype=torch.bfloat16)
    if getattr(model.config, "tie_word_embeddings", False):
        for key in getattr(model, "_tied_weights_keys", []):
            state_dict.pop(key, None)
    state_dict = convert_state_dict_to_hf(model, state_dict)

    logger.info(f"Writing HF weights to {config.output_dir}")
    save_state_dict_parallel(state_dict, config.output_dir)
    if world.is_master:
        from prime_rl.trainer.model import setup_tokenizer

        model.config.save_pretrained(config.output_dir)
        if model.generation_config:
            model.generation_config.save_pretrained(config.output_dir)
        tokenizer_config = config.tokenizer
        if tokenizer_config.name is None:
            tokenizer_config = tokenizer_config.model_copy(update={"name": config.model.name})
        setup_tokenizer(tokenizer_config).save_pretrained(config.output_dir)
        logger.info(f"Done: {config.output_dir}")


if __name__ == "__main__":
    main(cli(DcpToHfConfig))
