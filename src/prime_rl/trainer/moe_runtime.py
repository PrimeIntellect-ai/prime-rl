import importlib.util

import torch
from torch import nn
from torch.distributed.tensor.parallel import parallelize_module

from prime_rl.configs.trainer import (
    BF16MoEComputeConfig,
    DeepEPMoEDispatchConfig,
    DeepGemmFP8MoEComputeConfig,
    ModelConfig,
    MoERuntimeConfig,
    MXFP8MoEComputeConfig,
    TorchMoEDispatchConfig,
)
from prime_rl.trainer.distributed.expert_parallel import ExpertWeightParallel
from prime_rl.trainer.distributed.token_dispatcher import (
    LocalTokenDispatcher,
    MXFP8TorchTokenDispatcher,
    TorchTokenDispatcher,
)
from prime_rl.trainer.models.layers.grouped_gemm import (
    BF16GroupedGemm,
    DeepGemmFP8GroupedGemm,
    GroupedGemm,
    MXFP8GroupedGemm,
)
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.logger import get_logger


def _resolve_grouped_gemm(config: ModelConfig) -> GroupedGemm:
    compute = config.moe.compute
    if isinstance(compute, BF16MoEComputeConfig):
        return BF16GroupedGemm()
    if isinstance(compute, DeepGemmFP8MoEComputeConfig):
        if importlib.util.find_spec("deep_gemm") is None:
            raise RuntimeError("DeepGEMM FP8 expert compute requires the deep-gemm package.")
        capability = torch.cuda.get_device_capability()
        if capability < (9, 0):
            raise RuntimeError(
                f"DeepGEMM FP8 expert compute requires SM90 or newer, but this device is SM{capability[0]}{capability[1]}."
            )
        return DeepGemmFP8GroupedGemm()
    if isinstance(compute, MXFP8MoEComputeConfig):
        import prime_kernels

        kernel = prime_kernels.load("mxfp8_moe")
        return MXFP8GroupedGemm(
            kernel=kernel,
            high_precision_wgrad=compute.recipe == "mxfp8_rceil_wgrad_with_hp",
            token_group_alignment=kernel.TOKEN_GROUP_ALIGNMENT,
        )
    raise TypeError(f"Unsupported MoE compute config: {type(compute).__name__}")


def configure_moe_runtime(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims) -> None:
    moe_layers = [module for module in model.modules() if isinstance(module, MoE)]
    if not moe_layers:
        if config.moe != MoERuntimeConfig():
            raise ValueError("A non-default model.moe runtime was configured, but the model has no custom MoE layers.")
        return

    grouped_gemm = _resolve_grouped_gemm(config)
    ep_mesh = parallel_dims.get_mesh("ep") if parallel_dims.ep_enabled else None
    dispatch = config.moe.dispatch

    for moe in moe_layers:
        if ep_mesh is not None and moe.experts.num_experts % parallel_dims.ep:
            raise ValueError(
                f"MoE expert count {moe.experts.num_experts} must be divisible by model.ep={parallel_dims.ep}."
            )
        moe.experts.set_grouped_gemm(grouped_gemm)
        if ep_mesh is None:
            token_dispatcher = LocalTokenDispatcher(
                num_experts=moe.experts.num_experts,
                top_k=moe.router.top_k,
                token_group_alignment=grouped_gemm.token_group_alignment,
            )
        elif isinstance(dispatch, TorchMoEDispatchConfig):
            if dispatch.transport == "mxfp8":
                token_dispatcher = MXFP8TorchTokenDispatcher(
                    num_experts=moe.experts.num_experts,
                    top_k=moe.router.top_k,
                    token_group_alignment=grouped_gemm.token_group_alignment,
                    group=ep_mesh.get_group(),
                )
            else:
                token_dispatcher = TorchTokenDispatcher(
                    num_experts=moe.experts.num_experts,
                    top_k=moe.router.top_k,
                    token_group_alignment=grouped_gemm.token_group_alignment,
                    group=ep_mesh.get_group(),
                )
            moe.set_overlap_shared_expert(dispatch.overlap_shared_expert)
        elif isinstance(dispatch, DeepEPMoEDispatchConfig):
            from prime_rl.trainer.distributed.deepep import DeepEPTokenDispatcher

            token_dispatcher = DeepEPTokenDispatcher(
                num_experts=moe.experts.num_experts,
                token_group_alignment=grouped_gemm.token_group_alignment,
                group=ep_mesh.get_group(),
                num_sms=dispatch.num_sms,
                token_chunk_size=dispatch.token_chunk_size,
            )
        else:
            raise TypeError(f"Unsupported MoE dispatch config: {type(dispatch).__name__}")
        moe.set_token_dispatcher(token_dispatcher)

        if ep_mesh is not None:
            parallelize_module(moe.experts, device_mesh=ep_mesh, parallelize_plan=ExpertWeightParallel())

    get_logger().info(
        f"Configured {len(moe_layers)} MoE layers with compute={config.moe.compute.type}, "
        f"dispatch={config.moe.dispatch.type}, ep={parallel_dims.ep}"
    )
