import pytest
import torch
import torch._dynamo
import torch._inductor.config
from torch import nn

from prime_rl.configs.trainer import CompileConfig
from prime_rl.trainer.model import apply_compile


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2)])


class _ModelWithDynamicLayerArgs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Decoder()

    @classmethod
    def cuda_graph_blocker(cls, config) -> str | None:
        return None

    @classmethod
    def dynamic_layer_arg_dims(cls) -> tuple[tuple[str, int], ...]:
        return (("routed_experts", 1),)


@pytest.fixture
def compiler_settings():
    """`apply_compile` sets process-global compiler options; put them back for the tests that follow."""
    triton = torch._inductor.config.triton
    saved = (
        triton.mix_order_reduction,
        triton.mix_order_reduction_non_strict_mode,
        torch.compiler.config.dynamic_sources,
        torch._dynamo.config.capture_scalar_outputs,
        torch._dynamo.config.capture_dynamic_output_shape_ops,
    )
    yield triton
    (
        triton.mix_order_reduction,
        triton.mix_order_reduction_non_strict_mode,
        torch.compiler.config.dynamic_sources,
        torch._dynamo.config.capture_scalar_outputs,
        torch._dynamo.config.capture_dynamic_output_shape_ops,
    ) = saved


def test_dynamic_layer_args_turn_mixed_order_fusion_off(compiler_settings):
    """A layer compiled for any row length must not get Inductor's fused mixed-order reduction kernel. Its size
    thresholds are guards on the row length, and without them (non-strict mode) torch 2.14 fuses the router's
    confidence sum into a kernel that also sums the rows past the end of the row: with `top_scores / (sum + 1e-20)`
    each of them adds `top_k * sigmoid(0) / 1e-20` to the logged routing confidence."""
    compiler_settings.mix_order_reduction = True
    compiler_settings.mix_order_reduction_non_strict_mode = False

    apply_compile(_ModelWithDynamicLayerArgs(), CompileConfig())

    assert compiler_settings.mix_order_reduction is False
    assert compiler_settings.mix_order_reduction_non_strict_mode is False
    assert r".*(\['routed_experts'\]|\.routed_experts):1" in torch.compiler.config.dynamic_sources.split(",")
