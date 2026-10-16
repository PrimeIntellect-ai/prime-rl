from functools import partial

import torch
from torch import Tensor

# dion's default schedule
MUON_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
# 8 steps that converge fast, then 2 that settle the singular values at 1 (DeepSeek-V4 report, section 2.4)
DEEPSEEK_V4_COEFFICIENTS = ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2

COEFFICIENTS = {"muon": MUON_COEFFICIENTS, "deepseek_v4": DEEPSEEK_V4_COEFFICIENTS}


@torch.compile(fullgraph=True)
def _newton_schulz(
    G: Tensor, epsilon: Tensor, coefficients: tuple[tuple[float, float, float], ...], scale: float
) -> Tensor:
    """dion's bfloat16 Newton-Schulz with a given schedule, its output scaled by ``scale``."""
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in coefficients:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X * scale


def newton_schulz(coefficients: str, update_rms: float):
    """A ``newton_schulz_func`` for dion's Muon. dion's ``rms_norm`` learning-rate adjustment sets the update RMS to
    0.2 (Moonlight); scaling the orthogonalized output sets it to ``update_rms`` and leaves weight decay as it is."""
    return partial(_newton_schulz, coefficients=COEFFICIENTS[coefficients], scale=update_rms / 0.2)
