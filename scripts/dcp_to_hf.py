"""Convert a DCP trainer checkpoint into HF-format weights offline.

The trainer saves only DCP checkpoints; every HF export goes through this
script. It mirrors the trainer's broadcast save path: build the model, DCP-load
the checkpoint's model state, gather rank-parallel, convert to HF format, and
write sharded safetensors plus config/tokenizer assets. For LoRA checkpoints
(``--model.lora`` matching the run) it writes the adapter (peft layout) instead
of merged weights.

Usage (from the prime-rl repo; more ranks = faster gathers and writes):
    uv run torchrun --nproc-per-node 8 scripts/dcp_to_hf.py \
        --model.name <hf-id-or-path> \
        --ckpt-dir <run>/checkpoints/step_N/trainer \
        --output-dir <out>

``--model.*`` must match the run that wrote the checkpoint (impl, attn, lora,
...) so the state-dict FQNs and the prime->HF conversion line up.
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.lora import get_lora_state, has_lora_layers, save_lora_config
from prime_rl.trainer.model import setup_model, setup_processor, setup_tokenizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict,
    save_state_dict_parallel,
)


class DcpToHfConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    ckpt_dir: Path
    """The DCP checkpoint to convert (``<run>/checkpoints/step_N/trainer``)."""

    output_dir: Path
    """Where to write the HF-format weights."""


def save_model_assets(model, config: DcpToHfConfig) -> None:
    """Save model config, generation config, processor and tokenizer next to the weights."""
    model.config.save_pretrained(config.output_dir)
    if model.generation_config:
        # training sets use_cache=False which can conflict with cache_implementation —
        # save with use_cache=True without mutating the model's config
        gen_config = deepcopy(model.generation_config)
        gen_config.use_cache = True
        gen_config.save_pretrained(config.output_dir)
    # Processor first: it saves its own (unmodified) tokenizer, which the configured
    # tokenizer (pad token, custom chat template) must override.
    processor = setup_processor(config.model)
    if processor is not None:
        processor.save_pretrained(config.output_dir)
    tokenizer_config = config.tokenizer
    if tokenizer_config.name is None:
        tokenizer_config = tokenizer_config.model_copy(update={"name": config.model.name})
    setup_tokenizer(tokenizer_config).save_pretrained(config.output_dir)


def main(config: DcpToHfConfig) -> None:
    logger = setup_logger("info")
    world = get_world()
    setup_torch_distributed()

    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=True)

    logger.info(f"Loading DCP checkpoint from {config.ckpt_dir}")
    dcp_load(state_dict={"app": AppState(model, [], None, None)}, checkpoint_id=config.ckpt_dir)

    if has_lora_layers(model):
        logger.info("Gathering LoRA adapter")
        # All ranks must participate in the DTensor gathers, only master saves
        lora_state_dict = {
            f"base_model.model.{key}": (value.full_tensor() if isinstance(value, DTensor) else value).to("cpu")
            for key, value in get_lora_state().adapter_state_dict().items()
        }
        if world.is_master:
            logger.info(f"Writing adapter to {config.output_dir}")
            save_state_dict(lora_state_dict, config.output_dir, save_sharded=False, adapter=True)
            if config.model.lora:
                save_lora_config(
                    model,
                    config.output_dir,
                    rank=config.model.lora.rank,
                    alpha=config.model.lora.alpha,
                    dropout=config.model.lora.dropout,
                )
            save_model_assets(model, config)
            logger.info(f"Done: {config.output_dir}")
        return

    logger.info("Gathering and converting weights")
    state_dict = gather_weights_parallel(model, dtype=torch.bfloat16)
    if getattr(model.config, "tie_word_embeddings", False):
        for key in getattr(model, "_tied_weights_keys", []):
            state_dict.pop(key, None)
    state_dict = convert_state_dict_to_hf(model, state_dict)

    logger.info(f"Writing HF weights to {config.output_dir}")
    save_state_dict_parallel(state_dict, config.output_dir)
    if world.is_master:
        save_model_assets(model, config)
        logger.info(f"Done: {config.output_dir}")


if __name__ == "__main__":
    main(cli(DcpToHfConfig))
