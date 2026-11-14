import importlib.util

import torch
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor.parallel import parallelize_module

from prime_rl.configs.trainer import (
    BF16MoEComputeConfig,
    DeepEPMoEDispatchConfig,
    DeepGemmFP8MoEComputeConfig,
    MegaMoeMoEDispatchConfig,
    ModelConfig,
    MoEComputeConfig,
    MoERuntimeConfig,
    MXFP8MoEComputeConfig,
    NVFP4MoEComputeConfig,
    TorchMoEDispatchConfig,
    mega_moe_precision,
)
from prime_rl.trainer.distributed.expert_parallel import ExpertWeightParallel
from prime_rl.trainer.distributed.token_dispatcher import (
    FusedTokenDispatcher,
    LocalTokenDispatcher,
    MXFP8TorchTokenDispatcher,
    TorchTokenDispatcher,
)
from prime_rl.trainer.models.layers.expert_compute import (
    BF16ExpertCompute,
    DeepGemmFP8ExpertCompute,
    ExpertCompute,
    MXFP8ExpertCompute,
    NVFP4ExpertCompute,
)
from prime_rl.trainer.models.layers.mega_moe import (
    MegaMoEExpertCompute,
    default_nccl_max_ctas,
    num_concurrent_nccl_kernels,
    reserve_sms_for_comm,
)
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.vlm import get_language_model
from prime_rl.utils.logger import get_logger


def setup_moe_comm_env(config: ModelConfig) -> None:
    """Environment the MoE dispatch needs before torch distributed creates NCCL communicators."""
    dispatch = config.moe.dispatch
    if isinstance(dispatch, MegaMoeMoEDispatchConfig):
        default_nccl_max_ctas(dispatch.num_reserved_sms, num_concurrent_nccl_kernels(config.dp_replicate))


def _resolve_expert_compute(config: ModelConfig) -> ExpertCompute:
    compute = config.moe.compute
    if isinstance(compute, BF16MoEComputeConfig):
        if compute.backend == "sonicmoe":
            from prime_rl.trainer.models.layers.sonic_moe import SonicMoEExpertCompute

            return SonicMoEExpertCompute()
        return BF16ExpertCompute()
    if isinstance(compute, DeepGemmFP8MoEComputeConfig):
        if importlib.util.find_spec("vllm.third_party.deep_gemm") is None:
            raise RuntimeError("DeepGEMM FP8 expert compute requires the DeepGEMM bundled with vLLM.")
        capability = torch.cuda.get_device_capability()
        if capability[0] not in (9, 10):
            raise RuntimeError(
                f"DeepGEMM FP8 expert compute requires SM90 or SM100, but this device is SM{capability[0]}{capability[1]}."
            )
        return DeepGemmFP8ExpertCompute()
    if isinstance(compute, MXFP8MoEComputeConfig):
        import prime_kernels

        kernel = prime_kernels.load("mxfp8_moe")
        return MXFP8ExpertCompute(
            kernel=kernel,
            high_precision_wgrad=compute.recipe == "mxfp8_rceil_wgrad_with_hp",
        )
    if isinstance(compute, NVFP4MoEComputeConfig):
        import prime_kernels

        if "nvfp4_moe" not in prime_kernels.KERNELS:
            raise RuntimeError("NVFP4 expert compute needs a prime-kernels build containing nvfp4_moe.")
        capability = torch.cuda.get_device_capability()
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(
                f"NVFP4 expert compute needs SM100 or SM103 (B200, GB200, B300), but this device is "
                f"SM{capability[0]}{capability[1]}. Use moe.compute.type = 'mxfp8' (SM100+) or 'deepgemm_fp8' (SM90)."
            )
        return NVFP4ExpertCompute(kernel=prime_kernels.load("nvfp4_moe"))
    raise TypeError(f"Unsupported MoE compute config: {type(compute).__name__}")


def _resolve_mega_moe_compute(
    moe: MoE, dispatch: MegaMoeMoEDispatchConfig, compute: MoEComputeConfig, group: ProcessGroup, dp_replicate: int
) -> MegaMoEExpertCompute:
    if dispatch.max_tokens_per_rank is None:
        raise ValueError(
            "Mega MoE dispatch needs `model.moe.dispatch.max_tokens_per_rank`; the trainer configs derive it from "
            "the micro batch and model.cp, set it explicitly when configuring the model on its own."
        )
    return MegaMoEExpertCompute(
        num_experts=moe.experts.num_experts,
        top_k=moe.router.top_k,
        hidden=moe.experts.down_proj.shape[1],
        intermediate_hidden=moe.experts.hidden_dim,
        group=group,
        max_tokens_per_rank=dispatch.max_tokens_per_rank,
        num_sms=reserve_sms_for_comm(dispatch.num_reserved_sms, num_concurrent_nccl_kernels(dp_replicate)),
        precision=mega_moe_precision(compute),
    )


def seed_expert_rounding(config: ModelConfig, step: int) -> None:
    """Seed the NVFP4 experts' stochastic rounding for ``step``: its backward then quantizes the same way on every
    replay, so a step is reproducible."""
    if isinstance(config.moe.compute, NVFP4MoEComputeConfig) and not config.moe.compute.selects_none:
        import prime_kernels

        prime_kernels.load("nvfp4_moe").stochastic.set_seed(step)


def configure_moe_runtime(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims) -> None:
    moe_layers = [module for module in model.modules() if isinstance(module, MoE)]
    if not moe_layers:
        if config.moe != MoERuntimeConfig():
            raise ValueError("A non-default model.moe runtime was configured, but the model has no custom MoE layers.")
        return

    selected_moes = set(moe_layers)
    if config.moe.compute.apply_to != "all":
        language_model = get_language_model(model)
        selected_layers = config.moe.compute.resolve_layers(len(language_model.layers))
        selected_moes = {
            module
            for index, layer in enumerate(language_model.layers.children())
            if index in selected_layers
            for module in layer.modules()
            if isinstance(module, MoE)
        }
        get_logger().debug(f"Selected model layers for MoE compute: {sorted(selected_layers)}")
    if config.moe.compute.remainder == "mxfp8":
        import prime_kernels

        remainder_compute = MXFP8ExpertCompute(kernel=prime_kernels.load("mxfp8_moe"), high_precision_wgrad=False)
    else:
        remainder_compute = BF16ExpertCompute()
    dispatch = config.moe.dispatch
    is_mega_moe = isinstance(dispatch, MegaMoeMoEDispatchConfig)
    ep_mesh = parallel_dims.get_mesh("ep") if parallel_dims.ep_enabled else None
    if is_mega_moe:
        if ep_mesh is None:
            raise ValueError(
                "Mega MoE dispatch requires an expert-parallel group (model.ep > 1); it has no single-rank mode."
            )
        # Mega MoE fuses the expert compute into dispatch, so `model.moe.compute` does not apply to it.
        selected_moes = set(moe_layers)
        selected_compute = _resolve_mega_moe_compute(
            moe_layers[0], dispatch, config.moe.compute, ep_mesh.get_group(), config.dp_replicate
        )
    else:
        selected_compute = _resolve_expert_compute(config) if selected_moes else remainder_compute

    for moe in moe_layers:
        if ep_mesh is not None and moe.experts.num_experts % parallel_dims.ep:
            raise ValueError(
                f"MoE expert count {moe.experts.num_experts} must be divisible by model.ep={parallel_dims.ep}."
            )
        if is_mega_moe and moe.score_before_experts:
            raise ValueError(
                "Mega MoE dispatch requires `score_before_experts=False` on the MoE layer "
                "(router weights are applied at combine time, after both expert GEMMs)."
            )
        if is_mega_moe:
            selected_compute.validate_top_k(moe.router.top_k)
        compute = selected_compute if moe in selected_moes else remainder_compute
        moe.experts.set_compute(compute)
        if ep_mesh is None:
            token_dispatcher = LocalTokenDispatcher(
                num_experts=moe.experts.num_experts,
                top_k=moe.router.top_k,
                token_group_alignment=compute.token_group_alignment,
            )
        elif isinstance(dispatch, TorchMoEDispatchConfig):
            if dispatch.transport == "mxfp8" and isinstance(compute, MXFP8ExpertCompute):
                token_dispatcher = MXFP8TorchTokenDispatcher(
                    num_experts=moe.experts.num_experts,
                    top_k=moe.router.top_k,
                    token_group_alignment=compute.token_group_alignment,
                    group=ep_mesh.get_group(),
                )
            else:
                token_dispatcher = TorchTokenDispatcher(
                    num_experts=moe.experts.num_experts,
                    top_k=moe.router.top_k,
                    token_group_alignment=compute.token_group_alignment,
                    group=ep_mesh.get_group(),
                )
        elif isinstance(dispatch, DeepEPMoEDispatchConfig):
            from prime_rl.trainer.distributed.deepep import DeepEPTokenDispatcher

            token_dispatcher = DeepEPTokenDispatcher(
                num_experts=moe.experts.num_experts,
                token_group_alignment=compute.token_group_alignment,
                group=ep_mesh.get_group(),
                num_sms=dispatch.num_sms,
                hidden_size=moe.experts.down_proj.shape[1],
            )
        elif isinstance(dispatch, MegaMoeMoEDispatchConfig):
            token_dispatcher = FusedTokenDispatcher()
        else:
            raise TypeError(f"Unsupported MoE dispatch config: {type(dispatch).__name__}")
        moe.set_token_dispatcher(token_dispatcher)

        if ep_mesh is not None:
            parallelize_module(moe.experts, device_mesh=ep_mesh, parallelize_plan=ExpertWeightParallel())

    get_logger().info(
        f"Configured {len(selected_moes)}/{len(moe_layers)} MoE layers with compute={type(selected_compute).__name__}, "
        f"apply_to={config.moe.compute.apply_to}, remainder={config.moe.compute.remainder}, dispatch={config.moe.dispatch.type}, ep={parallel_dims.ep}"
    )
