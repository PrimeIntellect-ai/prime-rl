import torch
from torch import Tensor


@torch.no_grad()
def parameter_change_counts(before: Tensor, after: Tensor) -> Tensor:
    """Count changed finite FP32 elements and nonfinite pairs in parameter snapshots."""
    if before.dtype != torch.float32 or after.dtype != torch.float32:
        raise TypeError("Parameter snapshots must be FP32")
    if before.shape != after.shape:
        raise ValueError("Parameter snapshots must have identical shapes")
    finite = torch.isfinite(before) & torch.isfinite(after)
    changed = before.contiguous().view(torch.int32) != after.contiguous().view(torch.int32)
    return torch.stack(((changed & finite).sum(), (~finite).sum()))


@torch.no_grad()
def mismatch_diagnostics(trainer_logprobs: Tensor, inference_logprobs: Tensor) -> dict[str, Tensor]:
    """Per-token diagnostics on already-selected sampled tokens, returned as CPU FP32 tensors."""
    trainer = trainer_logprobs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    inference = inference_logprobs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if trainer.shape != inference.shape:
        raise ValueError("Trainer and inference logprobs must have identical shapes")
    finite = torch.isfinite(trainer) & torch.isfinite(inference)
    delta = trainer.double() - inference.double()
    # expm1 avoids subtracting two numbers rounded to one for small logprob errors.
    k3 = torch.expm1(delta) - delta
    return {
        "logprob_abs_error": delta.abs().float(),
        "logprob_bit_mismatch": ((trainer.view(torch.int32) != inference.view(torch.int32)) | ~finite).float(),
        "logprob_nonfinite": (~finite).float(),
        "mismatch_k3_stable": k3.float(),
    }
