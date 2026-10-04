import pytest
import torch
from dion.muon import zeropower_via_newtonschulz5

from prime_rl.trainer.optim.muon import NS_COEFFICIENTS, get_quack_gemm_symmetric, newton_schulz_symmetric

pytestmark = [pytest.mark.gpu]


def newton_schulz_fp32(G: torch.Tensor) -> torch.Tensor:
    X = G.mT if G.size(-2) > G.size(-1) else G
    X = X / X.norm(dim=(-2, -1), keepdim=True)
    for a, b, c in NS_COEFFICIENTS:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


@pytest.mark.parametrize("shape", [(1024, 3072), (3072, 1024), (8, 768, 2048)])
def test_newton_schulz_symmetric_is_as_accurate_as_dense(shape):
    if get_quack_gemm_symmetric() is None:
        pytest.skip("QuACK symmetric GEMM is unavailable")
    torch.manual_seed(0)
    G = torch.randn(shape, device="cuda")
    reference = newton_schulz_fp32(G)

    def error(X: torch.Tensor) -> float:
        return ((X.float() - reference).norm() / reference.norm()).item()

    assert error(newton_schulz_symmetric(G)) <= 1.25 * error(zeropower_via_newtonschulz5(G))
