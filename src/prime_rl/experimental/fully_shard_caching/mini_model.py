"""Depth-truncated GLM-4.5-Air, randomly initialized, with expert parallelism and FSDP."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import (
    ActivationCheckpointConfig,
    BF16MoEComputeConfig,
    DeepGemmFP8MoEComputeConfig,
    ModelConfig,
    MoERuntimeConfig,
)
from prime_rl.experimental.fully_shard_caching.ops import (
    Fp8GroupedExpertCompute,
    RowScaledGroupedExpertCompute,
)
from prime_rl.experimental.fully_shard_caching.prepared_tensor import install_prepared_weights
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM
from prime_rl.trainer.models.layers.activations import ActivationDispatch
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE
from prime_rl.trainer.moe_runtime import configure_moe_runtime
from prime_rl.trainer.parallel_dims import ParallelDims

GLM_4_5_AIR_CONFIG = dict(
    attention_bias=True,
    head_dim=128,
    hidden_act="silu",
    hidden_size=4096,
    partial_rotary_factor=0.5,
    intermediate_size=10944,
    max_position_embeddings=131072,
    moe_intermediate_size=1408,
    norm_topk_prob=True,
    num_attention_heads=96,
    n_routed_experts=128,
    n_shared_experts=1,
    routed_scaling_factor=1.0,
    num_experts_per_tok=8,
    first_k_dense_replace=1,
    num_key_value_heads=8,
    rms_norm_eps=1e-05,
    rope_theta=1000000,
    tie_word_embeddings=False,
    use_qk_norm=False,
    vocab_size=151552,
    pad_token_id=151329,
)

WRAP_MODES = ("none", "fp8", "toy")

CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"

DEFAULT_NUM_EXPERTS = GLM_4_5_AIR_CONFIG["n_routed_experts"]


class PreparedGroupedExperts(GroupedExperts):
    """Grouped experts that hand their raw parameters to a stateless op.

    The stock module transposes and casts each weight before the grouped-gemm seam, which hides the
    subclass an op must detect, so preparation needs its own module.
    """

    compute = None

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        output = self.compute(
            x.bfloat16(),
            to_local(self.gate_proj),
            to_local(self.up_proj),
            to_local(self.down_proj),
            offsets,
            num_tokens_per_expert,
        )
        return output.type_as(x)


@dataclass(frozen=True)
class MiniModelSpec:
    num_hidden_layers: int
    wrap: str
    num_experts: int
    seed: int
    attn_implementation: str = "flash_attention_3"
    lm_head_chunk_size: int = 8192


def build_op(wrap: str, activation):
    if wrap == "fp8":
        return Fp8GroupedExpertCompute(activation=activation)
    if wrap == "toy":
        return RowScaledGroupedExpertCompute(activation=activation)
    raise ValueError(f"Unknown wrap mode {wrap!r}, expected one of {WRAP_MODES}.")


def build_model_config(wrap: str) -> ModelConfig:
    """Pick the MoE compute whose kernels the wrap mode's op reproduces."""
    compute = BF16MoEComputeConfig() if wrap == "toy" else DeepGemmFP8MoEComputeConfig()
    return ModelConfig(moe=MoERuntimeConfig(compute=compute))


def replace_experts(model: nn.Module, spec: MiniModelSpec) -> list[GroupedExperts]:
    config = model.config
    op = build_op(spec.wrap, ActivationDispatch[config.hidden_act]) if spec.wrap != "none" else None
    experts_modules = []
    for moe in (module for module in model.modules() if isinstance(module, MoE)):
        experts_cls = GroupedExperts if op is None else PreparedGroupedExperts
        experts = experts_cls(
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
            num_experts=moe.experts.num_experts,
            expert_type="gated",
            activation=config.hidden_act,
        )
        if op is not None:
            experts.compute = op
        moe.experts = experts
        experts_modules.append(experts)
    return experts_modules


def seed_name(name: str) -> str:
    """Parameter name with the activation-checkpoint wrapper stripped, so init is the same either way."""
    return name.replace(f".{CHECKPOINT_WRAPPED_MODULE}", "")


def initialize(model: nn.Module, seed: int, device: torch.device) -> None:
    model.to_empty(device=device)
    for _, buffer in model.named_buffers():
        buffer.zero_()
    model.init_buffers_post_meta()

    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if parameter.dim() == 1:
                parameter.fill_(1.0 if name.endswith("weight") else 0.0)
                continue
            stream = seed * 1_000_003 + zlib.crc32(seed_name(name).encode())
            generator = torch.Generator(device=parameter.device).manual_seed(stream % (2**63))
            parameter.normal_(0.0, 0.02, generator=generator)


def install_expert_preparation(experts_modules: list[GroupedExperts]) -> None:
    for experts in experts_modules:
        names = [name for name, _ in experts.named_parameters(recurse=False)]
        install_prepared_weights(experts, {name: experts.compute.prepare for name in names})


def apply_activation_checkpointing(model: nn.Module) -> None:
    wrap_block = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="full"))
    layers = model.model.layers
    for name, block in list(layers.named_children()):
        layers.register_module(name, wrap_block(block))


def expert_modules(model: nn.Module) -> list[GroupedExperts]:
    return [module for module in model.modules() if isinstance(module, GroupedExperts)]


def apply_fsdp(model: nn.Module, parallel_dims: ParallelDims, expert_reshard_after_forward: bool) -> None:
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    hsdp_mesh = parallel_dims.get_mesh("hsdp")
    dp_mod_ep_mesh = parallel_dims.get_mesh("dp_shard_mod_ep") if parallel_dims.ep_enabled else hsdp_mesh

    def config(reshard: bool) -> dict:
        return dict(mp_policy=mp_policy, offload_policy=OffloadPolicy(), reshard_after_forward=reshard)

    for block in model.model.layers:
        mlp = getattr(block, CHECKPOINT_WRAPPED_MODULE, block).mlp
        if isinstance(mlp, MoE):
            fully_shard(mlp.experts, mesh=dp_mod_ep_mesh, **config(expert_reshard_after_forward))
            mlp.experts.set_gradient_divide_factor(parallel_dims.fsdp_gradient_divide_factor)
        fully_shard(block, mesh=hsdp_mesh, **config(True))

    fully_shard(model.model.embed_tokens, mesh=hsdp_mesh, **config(True))
    fully_shard([model.lm_head, model.model.norm], mesh=hsdp_mesh, **config(False))
    fully_shard(model, mesh=hsdp_mesh, **config(True))


def build_mini_model(
    spec: MiniModelSpec,
    parallel_dims: ParallelDims,
    *,
    expert_reshard_after_forward: bool,
    activation_checkpointing: bool,
    install_prepared: bool,
    device: torch.device,
) -> nn.Module:
    config = Glm4MoeConfig(**GLM_4_5_AIR_CONFIG, num_hidden_layers=spec.num_hidden_layers)
    config.n_routed_experts = spec.num_experts
    config._attn_implementation = spec.attn_implementation
    with torch.device("meta"):
        model = Glm4MoeForCausalLM(config)
        experts_modules = replace_experts(model, spec)
        inject_prime_lm_head(model, chunk_size=spec.lm_head_chunk_size)

    if spec.wrap != "none" and install_prepared:
        install_expert_preparation(experts_modules)
    configure_moe_runtime(model, build_model_config(spec.wrap), parallel_dims)
    if activation_checkpointing:
        apply_activation_checkpointing(model)
    apply_fsdp(model, parallel_dims, expert_reshard_after_forward)
    initialize(model, spec.seed, device)
    return model
