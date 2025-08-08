from typing import TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torchtitan.models.deepseek_v3.model.moe import DeepSeekV3ModelArgs, MoE
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)

from prime_rl.trainer.config import ModelConfig
from prime_rl.trainer.world import get_world

# TODO: Change all to nn.Module
Model: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM | nn.Module


def convert_tt_moe(model: nn.Module) -> None:
    for layer_id, transformer_block in model.model.layers.named_children():
        # Skip non MoE layers
        if not hasattr(transformer_block.mlp, "gate"):
            print(f"Skipping non MoE layer: {layer_id}")
            continue

        # Map HF MoE args to TT MoE args
        model_args = DeepSeekV3ModelArgs()
        hf_config = transformer_block.mlp.config
        model_args.n_routed_experts = transformer_block.mlp.gate.n_routed_experts
        model_args.dim = hf_config.hidden_size
        model_args.moe_inter_dim = hf_config.moe_intermediate_size
        model_args.n_activated_experts = transformer_block.mlp.gate.top_k
        model_args.route_scale = transformer_block.mlp.gate.routed_scaling_factor
        model_args.use_grouped_mm = False
        model_args.score_func = transformer_block.mlp.gate.scoring_func
        model_args.n_shared_experts = hf_config.n_shared_experts
        model_args.load_balance_coeff = None

        new_mlp = MoE(model_args)
        # Router
        new_mlp.router.gate.weight.data.copy_(transformer_block.mlp.gate.weight.data)
        # Shared experts
        new_mlp.shared_expert.w1.data[0].copy_(transformer_block.mlp.shared_experts.gate_proj.weight.data.T)
        new_mlp.shared_expert.w2.data[0].copy_(transformer_block.mlp.shared_experts.down_proj.weight.data.T)
        new_mlp.shared_expert.w3.data[0].copy_(transformer_block.mlp.shared_experts.up_proj.weight.data.T)
        # Routed experts
        for i in range(model_args.n_routed_experts):
            new_mlp.experts.w1.data[i].copy_(transformer_block.mlp.experts[i].gate_proj.weight.data.T)
            new_mlp.experts.w2.data[i].copy_(transformer_block.mlp.experts[i].down_proj.weight.data.T)
            new_mlp.experts.w3.data[i].copy_(transformer_block.mlp.experts[i].up_proj.weight.data.T)
        transformer_block.mlp = new_mlp


def get_model(config: ModelConfig) -> Model:
    config_model = AutoConfig.from_pretrained(
        config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
    )

    # Support expert parallelism in MoE models
    if hasattr(config_model, "ep_size"):
        if config.ep_mode == "world":
            config_model.ep_size = get_world().world_size
        elif config.ep_mode == "local":
            config_model.ep_size = get_world().local_world_size
        elif isinstance(config.ep_mode, int):
            config_model.ep_size = config.ep_mode
        else:
            raise ValueError(f"Invalid EP mode: {config.ep_mode} ({type(config.ep_mode)})")

    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.name,
        config=config_model,
        trust_remote_code=config.trust_remote_code,
    )

    convert_tt_moe(model)
    return model


def get_tokenizer(config: ModelConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: Model, config: ModelConfig):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if config.reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def setup_ac(model: Model, config: ModelConfig) -> None:
    if not config.ac:
        return
    for layer_id, transformer_block in model.model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers.register_module(layer_id, transformer_block)


def setup_model(config: ModelConfig) -> Model:
    dist.init_process_group()
    model = get_model(config)
    setup_fsdp(model, config)
    setup_ac(model, config)
    if config.compile:
        model = torch.compile(model)
    # TODO: This should be type-hinted as FSDP version of the model
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: Model, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits.float()
