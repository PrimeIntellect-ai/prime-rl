# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register Inkling2 with transformers and the vLLM model registry."""

ARCHITECTURE = "Inkling2ForConditionalGeneration"


def _patch_fp8_online_moe_backend() -> None:
    """Let the online block-FP8 MoE method use vLLM's CUTLASS FP8 experts.

    The online FP8 methods default to `allow_vllm_cutlass=False`, which leaves
    only the Triton experts on Hopper (DeepGEMM is not installed here). The
    CUTLASS FP8 experts drive the activation through `apply_moe_activation`, so
    they support Inkling2's SiTU exactly; allowing them lets vLLM pick the
    faster backend.
    """
    import os

    from vllm.model_executor.layers.quantization.online import fp8 as online_fp8

    if os.environ.get("INKLING2_DISABLE_CUTLASS") == "1":
        return

    original_init = online_fp8._Fp8OnlineMoEBase.__init__
    if getattr(original_init, "_prime_rl_allows_cutlass", False):
        return

    def __init__(
        self,
        *,
        weight_block_size,
        moe,
        weight_key=None,
        activation_key=None,
        allow_vllm_cutlass=False,
    ):
        original_init(
            self,
            weight_block_size=weight_block_size,
            moe=moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=True,
        )

    __init__._prime_rl_allows_cutlass = True
    online_fp8._Fp8OnlineMoEBase.__init__ = __init__


def _patch_deep_gemm_situ() -> None:
    """Let the FP8-block DeepGEMM experts run Inkling2's SiTU activation.

    `DeepGemmExperts._supports_activation` lists only the fused-kernel
    activations; its `_act_mul_quant` already has an unfused fallback (activation
    through `apply_moe_activation`, then FP8 requantization) that is exact for
    SiTU. Allowing SiTU here lets vLLM pick DeepGEMM when it is available.
    """
    import os

    if os.environ.get("INKLING2_DISABLE_DEEPGEMM") == "1":
        return
    # Opt-in: DeepGEMM's unfused SiTU path rounds activation scales differently
    # from the exact Triton/CUTLASS paths (max |dlogprob| 0.27 over 651 values on
    # a 6-layer probe, no token flips). CUTLASS is the default FP8 MoE backend;
    # enabling DeepGEMM here also requires `use_deep_gemm = true` (or
    # VLLM_USE_DEEP_GEMM=1/VLLM_MOE_USE_DEEP_GEMM=1).
    if os.environ.get("INKLING2_ENABLE_DEEPGEMM") != "1":
        return

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe

    for name in ("DeepGemmExperts", "BatchedDeepGemmExperts"):
        cls = getattr(deep_gemm_moe, name, None)
        if cls is None:
            continue
        original = cls._supports_activation
        if getattr(original, "_prime_rl_allows_situ", False):
            continue

        def _supports_activation(activation, _original=original):
            return _original(activation) or activation == MoEActivation.SITU

        _supports_activation._prime_rl_allows_situ = True
        cls._supports_activation = staticmethod(_supports_activation)


def _register_parsers() -> None:
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    from .parsers import Inkling2ReasoningParser, Inkling2ToolParser

    ReasoningParserManager.register_module(["inkling2"], module=Inkling2ReasoningParser)
    ToolParserManager.register_module(["inkling2"], module=Inkling2ToolParser)


def register() -> None:
    _patch_fp8_online_moe_backend()
    _patch_deep_gemm_situ()
    _register_parsers()
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry

    from .configs import Inkling2MMConfig, Inkling2ModelConfig
    from .model import Inkling2ForCausalLM

    for model_type, config_cls in (
        ("inkling2", Inkling2MMConfig),
        ("inkling2_model", Inkling2ModelConfig),
    ):
        AutoConfig.register(model_type, config_cls, exist_ok=True)
    ModelRegistry.register_model(ARCHITECTURE, Inkling2ForCausalLM)
