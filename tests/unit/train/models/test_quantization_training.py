"""A tiny training run per quantization backend, checked against the BF16 path.

The numerics tests cover single GEMMs. This covers the glue: that the swapped modules stay
in the autograd graph, that gradients reach every parameter, that the optimizer makes
progress, and that nothing goes non-finite over a sequence of steps.

Comparing two independent loss curves does *not* work here: overfitting a fixed batch is a
chaotic trajectory, so two runs that differ only by rounding drift apart on their own, and
a threshold loose enough not to flake is loose enough to pass a broken backward. So the
comparison keeps both models on the **same weights** — the BF16 model trains, the quantized
model is synced to it every step — and checks the gradient it produces there. That isolates
the quantization error from the trajectory divergence it would otherwise compound into.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.configs.trainer import FP8Config, ModelConfig, MXFP8Config, QuantizationConfig
from prime_rl.trainer.model import apply_quantization
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeForCausalLM as Qwen3Moe
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.test_quantization_numerics import (
    FP8_UNAVAILABLE,
    MXFP8_GROUPED_UNAVAILABLE,
    MXFP8_UNAVAILABLE,
)

pytestmark = [pytest.mark.gpu]

SEQ_LEN, VOCAB_SIZE = 256, 512
LEARNING_RATE = 5e-3
# At initialization the gradient is dominated by near-degenerate structure (a router that
# routes at random, an untrained LM head), and the two paths agree only to ~0.95 cosine
# there — through no fault of the kernels. A couple of optimizer steps settle it.
WARMUP_STEPS, COMPARE_STEPS = 2, 6
CONVERGENCE_STEPS = 60

# Every hidden size is a multiple of 128 so the FP8 blockwise swap accepts each projection —
# on a model with unaligned dims the layers would silently stay in BF16 and these tests
# would be comparing BF16 against BF16.
MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_experts=4,
    num_experts_per_tok=2,
    max_position_embeddings=SEQ_LEN,
    mlp_only_layers=[0],  # layer 0 dense, layer 1 MoE: both dense and expert GEMMs run
    use_grouped_mm=True,
    norm_topk_prob=True,
    rms_norm_eps=1e-6,
)

QUANTIZED_LINEARS = (Float8BlockwiseLinear, MXFP8Linear)

BACKENDS = [
    pytest.param(
        FP8Config(),
        id="fp8",
        marks=pytest.mark.skipif(FP8_UNAVAILABLE is not None, reason=str(FP8_UNAVAILABLE)),
    ),
    pytest.param(
        MXFP8Config(enable_grouped_gemm=False),
        id="mxfp8-dense",
        marks=pytest.mark.skipif(MXFP8_UNAVAILABLE is not None, reason=str(MXFP8_UNAVAILABLE)),
    ),
    pytest.param(
        MXFP8Config(),
        id="mxfp8-grouped",
        marks=pytest.mark.skipif(MXFP8_GROUPED_UNAVAILABLE is not None, reason=str(MXFP8_GROUPED_UNAVAILABLE)),
    ),
]

# Measured for MXFP8 (the looser of the two backends) over the compared steps: cosine
# similarity stays above 0.999 and the relative gradient error tracks the ~4e-2 e4m3
# quantization floor. The bounds sit an order of magnitude out from that in (1 - cos) terms.
MIN_GRADIENT_COSINE = 0.99
MAX_GRADIENT_ERROR = 0.15
MAX_LOSS_DEVIATION = 1e-2


def _build_model(quantization: QuantizationConfig | None) -> Qwen3Moe:
    """Build the same tiny model every call, quantized the way `get_model` would."""
    config = Qwen3MoeConfig(**MODEL_CONFIG)
    config._attn_implementation = "flash_attention_2"
    # Mirrors `get_model`: the MoE expert GEMM reads FP8 off the model config, while the
    # dense-linear swap (and the MXFP8 expert wrapping) go through `apply_quantization`.
    config.fp8 = isinstance(quantization, FP8Config) and quantization.enable_grouped_gemm

    torch.manual_seed(0)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3Moe._from_config(config)
        # `_from_config` leaves the fused expert weights at zero — HF's initializer does not
        # reach them. Without this the expert GEMMs compute nothing and the tests would be
        # judging the dense layers alone, whatever the grouped path did.
        for module in model.modules():
            if isinstance(module, GroupedExperts):
                module.init_weights(0.02)
    assert model.model.layers[1].mlp.experts.w1.any(), "expert weights are zero, the MoE GEMM path would be dead"

    inject_prime_lm_head(model, chunk_size=None)
    apply_quantization(model, ModelConfig(quantization=quantization))
    return model


def _backward(model: Qwen3Moe, input_ids: Tensor) -> float:
    """One forward + backward on the fixed batch; returns the loss."""
    logits = model(
        input_ids=input_ids,
        position_ids=torch.arange(SEQ_LEN, device="cuda").unsqueeze(0),
        seq_lens=torch.tensor([SEQ_LEN], device="cuda"),
    )["logits"]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE).float(), input_ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.item()


def _sync_weights(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        source_params = dict(source.named_parameters())
        for name, param in target.named_parameters():
            param.copy_(source_params[name])


def _flat_gradient(model: nn.Module, order: list[str]) -> Tensor:
    params = dict(model.named_parameters())
    return torch.cat([params[name].grad.float().flatten() for name in order])


@pytest.fixture(scope="module")
def input_ids() -> Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device="cuda")


@pytest.mark.parametrize("quantization", BACKENDS)
def test_gradients_track_bf16_along_a_training_trajectory(quantization: QuantizationConfig, input_ids: Tensor):
    """At every step of a real BF16 run, the quantized backward must point the same way."""
    reference, quantized = _build_model(None), _build_model(quantization)
    assert [name for name, mod in quantized.named_modules() if isinstance(mod, QUANTIZED_LINEARS)], (
        "no linear was swapped — this would compare BF16 against BF16"
    )
    optimizer = torch.optim.AdamW(reference.parameters(), lr=LEARNING_RATE)
    order = [name for name, _ in reference.named_parameters()]

    for step in range(WARMUP_STEPS + COMPARE_STEPS):
        _sync_weights(quantized, reference)
        reference_loss = _backward(reference, input_ids)
        quantized_loss = _backward(quantized, input_ids)

        if step >= WARMUP_STEPS:
            reference_grad, quantized_grad = _flat_gradient(reference, order), _flat_gradient(quantized, order)
            cosine = F.cosine_similarity(quantized_grad, reference_grad, dim=0).item()
            grad_error = ((quantized_grad - reference_grad).norm() / reference_grad.norm()).item()
            loss_deviation = abs(quantized_loss - reference_loss) / reference_loss
            print(
                f"{quantization.type} step {step}: loss {reference_loss:.4f} vs {quantized_loss:.4f} "
                f"(dev={loss_deviation:.2e}) grad cos={cosine:.5f} rel_err={grad_error:.3e}"
            )

            assert quantized_grad.isfinite().all(), f"step {step}: non-finite gradient"
            # A silent fallback to BF16 would score a perfect 1.0 cosine and zero error.
            assert grad_error > 0, f"step {step}: gradient is bit-identical to BF16, the quantized path did not run"
            assert cosine > MIN_GRADIENT_COSINE, f"step {step}: gradient cosine {cosine:.5f} — the backward is off"
            assert grad_error < MAX_GRADIENT_ERROR, f"step {step}: gradient relative error {grad_error:.3e}"
            assert loss_deviation < MAX_LOSS_DEVIATION, f"step {step}: loss deviates {loss_deviation:.2%} from BF16"

        optimizer.step()
        optimizer.zero_grad()
        quantized.zero_grad()


@pytest.mark.parametrize("quantization", BACKENDS)
def test_quantized_run_converges(quantization: QuantizationConfig, input_ids: Tensor):
    """The backend trains on its own: a long run stays finite and drives the loss down.

    Judged on its own trajectory, not against BF16's — the point here is that repeated
    quantized updates neither stall nor blow up, which no single-step check can show.
    """
    model = _build_model(quantization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    losses = []
    for step in range(CONVERGENCE_STEPS):
        losses.append(_backward(model, input_ids))
        for name, param in model.named_parameters():
            assert param.grad is None or param.grad.isfinite().all(), f"step {step}: non-finite gradient in {name}"
        optimizer.step()
        optimizer.zero_grad()

    print(f"{quantization.type}: {losses[0]:.4f} -> {losses[-1]:.4f} over {CONVERGENCE_STEPS} steps")
    assert all(loss == loss for loss in losses), f"non-finite loss: {losses}"
    assert losses[-1] < 0.5 * losses[0], f"training stalled ({losses[0]:.4f} -> {losses[-1]:.4f})"
    for name, param in model.named_parameters():
        assert param.isfinite().all(), f"non-finite weights in {name} after {CONVERGENCE_STEPS} steps"


@pytest.mark.parametrize("quantization", BACKENDS)
def test_every_parameter_receives_a_gradient(quantization: QuantizationConfig, input_ids: Tensor):
    """A quantized layer that drops out of the graph trains at its initial weights forever."""
    model = _build_model(quantization)
    _backward(model, input_ids)

    missing = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not missing, f"parameters left out of the backward pass: {missing}"
    dead = [name for name, param in model.named_parameters() if param.grad is not None and not param.grad.any()]
    assert not dead, f"parameters received an all-zero gradient: {dead}"
