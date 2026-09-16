import importlib.util

import pytest
import torch

import prime_rl.trainer.models.layers.fp8_grouped_gemm  # noqa: F401
import prime_rl.trainer.models.layers.fp8_linear  # noqa: F401

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="DeepGEMM fp8 GEMMs require SM90 or newer",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("deep_gemm") is None,
        reason="DeepGEMM fp8 GEMMs require the deep-gemm package",
    ),
]

OPCHECK_UTILS = ["test_schema", "test_faketensor"]

NUM_EXPERTS = 4
TOKENS_PER_EXPERT = 256
TOTAL_TOKENS = NUM_EXPERTS * TOKENS_PER_EXPERT
K_DIM = 512
N_DIM = 1024

BLOCK_SIZE = 128
NUM_ROWS = 256
IN_FEATURES = 512
OUT_FEATURES = 256


def make_activation(rows: int, cols: int, contiguous: bool) -> torch.Tensor:
    if contiguous:
        return torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    return torch.randn(cols, rows, device="cuda", dtype=torch.bfloat16).t()


def make_expert_weight(contiguous: bool) -> torch.Tensor:
    if contiguous:
        return torch.randn(NUM_EXPERTS, K_DIM, N_DIM, device="cuda", dtype=torch.bfloat16)
    return torch.randn(NUM_EXPERTS, N_DIM, K_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def offs() -> torch.Tensor:
    tokens_per_expert = torch.full((NUM_EXPERTS,), TOKENS_PER_EXPERT, device="cuda", dtype=torch.int32)
    return tokens_per_expert.cumsum(0).to(torch.int32)


@pytest.fixture
def linear_weight() -> torch.Tensor:
    return torch.randn(OUT_FEATURES, IN_FEATURES, device="cuda", dtype=torch.bfloat16)


def test_grouped_fp8_gemm_fake_matches_real(offs):
    x = make_activation(TOTAL_TOKENS, K_DIM, contiguous=True)
    weight = make_expert_weight(contiguous=True)

    torch.library.opcheck(
        torch.ops.prime_rl.grouped_fp8_gemm.default,
        (x, weight, offs),
        test_utils=OPCHECK_UTILS,
    )


@pytest.mark.parametrize("needs_grad_x", [True, False])
@pytest.mark.parametrize("needs_grad_weight", [True, False])
@pytest.mark.parametrize("weight_contiguous", [True, False])
@pytest.mark.parametrize("x_contiguous", [True, False])
def test_grouped_fp8_gemm_backward_fake_matches_real(
    offs, needs_grad_x, needs_grad_weight, weight_contiguous, x_contiguous
):
    x = make_activation(TOTAL_TOKENS, K_DIM, contiguous=x_contiguous)
    weight = make_expert_weight(contiguous=weight_contiguous)
    grad_output = make_activation(TOTAL_TOKENS, N_DIM, contiguous=True)

    torch.library.opcheck(
        torch.ops.prime_rl.grouped_fp8_gemm_backward.default,
        (grad_output, x, weight, offs, needs_grad_x, needs_grad_weight),
        test_utils=OPCHECK_UTILS,
    )


def test_fp8_blockwise_mm_fake_matches_real(linear_weight):
    x = make_activation(NUM_ROWS, IN_FEATURES, contiguous=True)

    torch.library.opcheck(
        torch.ops.prime_rl.fp8_blockwise_mm.default,
        (x, linear_weight, BLOCK_SIZE),
        test_utils=OPCHECK_UTILS,
    )


@pytest.mark.parametrize("needs_grad_x", [True, False])
@pytest.mark.parametrize("needs_grad_weight", [True, False])
@pytest.mark.parametrize("x_contiguous", [True, False])
def test_fp8_blockwise_mm_backward_fake_matches_real(linear_weight, needs_grad_x, needs_grad_weight, x_contiguous):
    x = make_activation(NUM_ROWS, IN_FEATURES, contiguous=x_contiguous)
    grad_output = make_activation(NUM_ROWS, OUT_FEATURES, contiguous=True)

    torch.library.opcheck(
        torch.ops.prime_rl.fp8_blockwise_mm_backward.default,
        (grad_output, x, linear_weight, BLOCK_SIZE, needs_grad_x, needs_grad_weight),
        test_utils=OPCHECK_UTILS,
    )
