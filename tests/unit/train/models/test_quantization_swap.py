"""Module-swap logic for the FP8 / MXFP8 dense-linear paths.

The swap decides *which* layers train in low precision, and it fails silently in both
directions: an ignore pattern that is too broad leaves a layer in BF16 (drift against an
FP8 inference engine), one that is too narrow quantizes a layer that must stay high
precision (routers, LM head). Neither shows up as an error — only as worse numerics — so
the mapping from module name and shape to swap decision is pinned here.

No GPU needed: the replacement functions only build modules, they never launch a kernel.
"""

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import FP8Config, ModelConfig, MXFP8Config
from prime_rl.trainer.model import apply_quantization
from prime_rl.trainer.models.layers.fp8_linear import (
    Float8BlockwiseLinear,
    replace_linear_with_fp8_blockwise_linear,
)
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear, replace_linear_with_mxfp8_linear

DIM, HIDDEN, NUM_EXPERTS = 256, 512, 8

# Divisible by 32 but not by 128: MXFP8 takes it, FP8 leaves it in BF16.
MX_ONLY_DIM = 96
# Divisible by neither: both backends leave it in BF16.
UNALIGNED_DIM = 48


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM, bias=False)
        self.k_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.o_proj = nn.Linear(DIM, DIM, bias=False)


class _DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.up_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.down_proj = nn.Linear(HIDDEN, DIM, bias=False)


class _Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(DIM, NUM_EXPERTS, bias=False)


class _MoEMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _Router()
        self.shared_expert_gate = nn.Linear(DIM, 1, bias=False)
        self.mx_only = nn.Linear(DIM, MX_ONLY_DIM, bias=False)
        self.unaligned = nn.Linear(DIM, UNALIGNED_DIM, bias=False)


class _Layer(nn.Module):
    def __init__(self, moe: bool) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MoEMLP() if moe else _DenseMLP()


class _Model(nn.Module):
    """Module tree with the fully-qualified names the ignore patterns are written against."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(moe=False), _Layer(moe=True)])
        self.lm_head = nn.Linear(DIM, HIDDEN, bias=False)


ALWAYS_REPLACED = [
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.o_proj",
    "layers.1.self_attn.q_proj",
    # `mlp.gate_proj` is a dense MLP projection, not a router gate. The ignore pattern is
    # written `mlp\.gate\.` precisely so it does not match here — an unescaped `mlp.gate.`
    # matched `mlp.gate_proj` and left it in BF16 while inference quantized it.
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
]

ALWAYS_SKIPPED = [
    "lm_head",
    "layers.1.mlp.router.gate",
    "layers.1.mlp.shared_expert_gate",
    "layers.1.mlp.unaligned",
]


@pytest.fixture
def model() -> _Model:
    return _Model()


def _module_types(model: nn.Module) -> dict[str, type]:
    return {name: type(module) for name, module in model.named_modules()}


def test_fp8_swap_replaces_aligned_layers(model: _Model):
    replace_linear_with_fp8_blockwise_linear(model, ignore_modules=FP8Config().ignore_patterns)
    types = _module_types(model)

    for name in ALWAYS_REPLACED:
        assert types[name] is Float8BlockwiseLinear, f"{name} should train in FP8"
    for name in ALWAYS_SKIPPED + ["layers.1.mlp.mx_only"]:
        assert types[name] is nn.Linear, f"{name} should stay in BF16"


def test_mxfp8_swap_replaces_aligned_layers(model: _Model):
    replace_linear_with_mxfp8_linear(model, recipe="mxfp8_rceil", ignore_modules=MXFP8Config().ignore_patterns)
    types = _module_types(model)

    for name in ALWAYS_REPLACED:
        assert types[name] is MXFP8Linear, f"{name} should train in MXFP8"
    # MXFP8 scales over blocks of 32, so it takes shapes FP8 blockwise (128) has to skip.
    assert types["layers.1.mlp.mx_only"] is MXFP8Linear
    for name in ALWAYS_SKIPPED:
        assert types[name] is nn.Linear, f"{name} should stay in BF16"


@pytest.mark.parametrize(
    "replace",
    [
        pytest.param(
            lambda m: replace_linear_with_fp8_blockwise_linear(m, ignore_modules=FP8Config().ignore_patterns),
            id="fp8",
        ),
        pytest.param(
            lambda m: replace_linear_with_mxfp8_linear(
                m, recipe="mxfp8_rceil", ignore_modules=MXFP8Config().ignore_patterns
            ),
            id="mxfp8",
        ),
    ],
)
def test_swap_keeps_the_same_parameter_objects(model: _Model, replace):
    """The swap must rebind the existing parameters, not re-initialize the model.

    Weights are loaded from a checkpoint before quantization is applied, and the swapped
    modules are what FSDP later shards — a copy here would train from random init.
    """
    before = {name: param for name, param in model.named_parameters()}
    replace(model)
    after = {name: param for name, param in model.named_parameters()}

    assert before.keys() == after.keys()
    for name, param in before.items():
        assert after[name] is param, f"{name} was rebound to a different parameter"


def test_ignore_patterns_are_regexes_not_substrings(model: _Model):
    """`ignore_patterns` go through `re.search`, so callers can pass anchors and alternation."""
    replace_linear_with_mxfp8_linear(model, recipe="mxfp8_rceil", ignore_modules=[r"^layers\.0\.", "q_proj$"])
    types = _module_types(model)

    assert types["layers.0.self_attn.q_proj"] is nn.Linear
    assert types["layers.0.mlp.up_proj"] is nn.Linear
    assert types["layers.1.self_attn.q_proj"] is nn.Linear
    assert types["layers.1.self_attn.k_proj"] is MXFP8Linear


def test_empty_ignore_patterns_quantize_everything_aligned(model: _Model):
    replace_linear_with_mxfp8_linear(model, recipe="mxfp8_rceil", ignore_modules=[])
    types = _module_types(model)

    assert types["lm_head"] is MXFP8Linear
    assert types["layers.1.mlp.router.gate"] is nn.Linear  # 256 -> 8, not 32-divisible
    assert types["layers.1.mlp.unaligned"] is nn.Linear


def test_apply_quantization_without_config_is_a_noop(model: _Model):
    before = _module_types(model)
    apply_quantization(model, ModelConfig(quantization=None))

    assert _module_types(model) == before


def test_apply_quantization_dispatches_fp8(model: _Model):
    apply_quantization(model, ModelConfig(quantization=FP8Config()))

    assert _module_types(model)["layers.0.self_attn.q_proj"] is Float8BlockwiseLinear


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (10, 0),
    reason="checks the pre-Blackwell rejection, so it needs a GPU older than SM100",
)
def test_apply_quantization_rejects_mxfp8_below_blackwell(model: _Model):
    with pytest.raises(ValueError, match="MXFP8 quantization requires SM100"):
        apply_quantization(model, ModelConfig(quantization=MXFP8Config()))
