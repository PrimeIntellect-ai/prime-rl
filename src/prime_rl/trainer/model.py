import torch
import torch.distributed as dist
import torch.nn as nn
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.moe import MoE, MoEArgs
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from prime_rl.trainer.config import ModelConfig
from prime_rl.trainer.world import get_world


def to_tt_moe_args(config: PretrainedConfig, use_grouped_mm: bool = True) -> MoEArgs:
    """Map HF config to TT MoE args"""
    if isinstance(config, Qwen3MoeConfig):
        return MoEArgs(
            num_experts=config.num_experts,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=config.norm_topk_prob,
            route_scale=1.0,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=use_grouped_mm,
            load_balance_coeff=None,
        )
    elif isinstance(config, DeepseekV3Config):
        return MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=use_grouped_mm,
            load_balance_coeff=None,
        )
    else:
        raise ValueError(f"Unsupported config: {config}")


def convert_tt_moe(model: nn.Module, config: PretrainedConfig) -> None:
    moe_args = to_tt_moe_args(config)
    for layer_id, transformer_block in model.model.layers.named_children():
        # Skip non MoE layers
        if not hasattr(transformer_block.mlp, "gate"):
            print(f"Skipping non MoE layer: {layer_id}")
            continue

        new_mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        # Router
        new_mlp.router.gate.weight.data.copy_(transformer_block.mlp.gate.weight.data)
        # Shared experts
        new_mlp.shared_expert.w1.data[0].copy_(transformer_block.mlp.shared_experts.gate_proj.weight.data)
        new_mlp.shared_expert.w2.data[0].copy_(transformer_block.mlp.shared_experts.down_proj.weight.data)
        new_mlp.shared_expert.w3.data[0].copy_(transformer_block.mlp.shared_experts.up_proj.weight.data)
        # Routed experts
        for i in range(moe_args.num_experts):
            new_mlp.experts.w1.data[i].copy_(transformer_block.mlp.experts[i].gate_proj.weight.data)
            new_mlp.experts.w2.data[i].copy_(transformer_block.mlp.experts[i].down_proj.weight.data)
            new_mlp.experts.w3.data[i].copy_(transformer_block.mlp.experts[i].up_proj.weight.data)
        transformer_block.mlp = new_mlp


def get_model(config: ModelConfig) -> nn.Module:
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

    convert_tt_moe(model, config_model)
    return model


def get_tokenizer(config: ModelConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: nn.Module, config: ModelConfig, world_mesh: DeviceMesh):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    hsdp_mesh = world_mesh["ep", "fsdp"]
    ep_mesh = world_mesh["ep"]
    fsdp_mesh = world_mesh["fsdp"]

    for layer_id, transformer_block in enumerate(model.model.layers):
        if config.reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        if hasattr(transformer_block.mlp, "experts"):
            parallelize_module(
                module=transformer_block.mlp.experts,
                device_mesh=ep_mesh,
                parallelize_plan=ExpertParallel(),
            )
            fully_shard(
                transformer_block,
                mesh=fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=layer_reshard_after_forward,
            )
        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )

    fully_shard(model, mesh=hsdp_mesh, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def setup_ac(model: nn.Module, config: ModelConfig) -> None:
    if not config.ac:
        return
    for layer_id, transformer_block in model.model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers.register_module(layer_id, transformer_block)


def setup_model(config: ModelConfig) -> nn.Module:
    dist.init_process_group()
    assert dist.get_world_size() % config.ep_mode == 0, "World size must be divisible by EP mode"
    fsdp_dim = dist.get_world_size() // config.ep_mode
    world_mesh = dist.init_device_mesh("cuda", (config.ep_mode, fsdp_dim), mesh_dim_names=("ep", "fsdp"))
    model = get_model(config)
    setup_fsdp(model, config, world_mesh)
    setup_ac(model, config)
    if config.compile:
        model = torch.compile(model)
    # TODO: This should be type-hinted as FSDP version of the model
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits.float()
