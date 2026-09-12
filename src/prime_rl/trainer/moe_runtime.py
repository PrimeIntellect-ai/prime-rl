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
from prime_rl.trainer.models.layers.expert_compute import (
    BF16ExpertCompute,
    DeepGemmFP8ExpertCompute,
    ExpertCompute,
    MXFP8ExpertCompute,
)
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_language_model


def _resolve_expert_compute(config: ModelConfig) -> ExpertCompute:
    compute = config.moe.compute
    if isinstance(compute, BF16MoEComputeConfig):
        if compute.backend == "sonicmoe":
            from prime_rl.trainer.models.layers.sonic_moe import SonicMoEExpertCompute

            return SonicMoEExpertCompute()
        return BF16ExpertCompute()
    if isinstance(compute, DeepGemmFP8MoEComputeConfig):
        if importlib.util.find_spec("deep_gemm") is None:
            raise RuntimeError("DeepGEMM FP8 expert compute requires the deep-gemm package.")
        capability = torch.cuda.get_device_capability()
        if capability < (9, 0):
            raise RuntimeError(
                f"DeepGEMM FP8 expert compute requires SM90 or newer, but this device is SM{capability[0]}{capability[1]}."
            )
        return DeepGemmFP8ExpertCompute()
    if isinstance(compute, MXFP8MoEComputeConfig):
        import prime_kernels

        kernel = prime_kernels.load("mxfp8_moe")
        return MXFP8ExpertCompute(
            kernel=kernel,
            high_precision_wgrad=compute.recipe == "mxfp8_rceil_wgrad_with_hp",
        )
    raise TypeError(f"Unsupported MoE compute config: {type(compute).__name__}")


def configure_moe_runtime(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims) -> None:
    moe_layers = [module for module in model.modules() if isinstance(module, MoE)]
    if not moe_layers:
        if config.moe != MoERuntimeConfig():
            raise ValueError("A non-default model.moe runtime was configured, but the model has no custom MoE layers.")
        return

    selected_moes = set(moe_layers)
    if config.moe.compute.apply_to != "all":
        language_model = get_language_model(
            model, override=config.vlm.language_model_attr if config.vlm is not None else None
        )
        selected_layers = config.moe.compute.resolve_layers(len(language_model.layers))
        selected_moes = {
            module
            for index, layer in enumerate(language_model.layers.children())
            if index in selected_layers
            for module in layer.modules()
            if isinstance(module, MoE)
        }
        get_logger().debug(f"Selected model layers for MoE compute: {sorted(selected_layers)}")
    bf16_compute = BF16ExpertCompute()
    selected_compute = _resolve_expert_compute(config) if selected_moes else bf16_compute
    ep_mesh = parallel_dims.get_mesh("ep") if parallel_dims.ep_enabled else None
    dispatch = config.moe.dispatch

    for moe in moe_layers:
        compute = selected_compute if moe in selected_moes else bf16_compute
        if ep_mesh is not None and moe.experts.num_experts % parallel_dims.ep:
            raise ValueError(
                f"MoE expert count {moe.experts.num_experts} must be divisible by model.ep={parallel_dims.ep}."
            )
        moe.experts.compute = compute
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
                token_chunk_size=dispatch.token_chunk_size,
            )
        else:
            raise TypeError(f"Unsupported MoE dispatch config: {type(dispatch).__name__}")
        moe.set_token_dispatcher(token_dispatcher)

        if ep_mesh is not None:
            parallelize_module(moe.experts, device_mesh=ep_mesh, parallelize_plan=ExpertWeightParallel())

    get_logger().info(
        f"Configured {len(selected_moes)}/{len(moe_layers)} MoE layers with compute={type(selected_compute).__name__}, "
        f"apply_to={config.moe.compute.apply_to}, fallback=bf16, dispatch={config.moe.dispatch.type}, ep={parallel_dims.ep}"
    )
