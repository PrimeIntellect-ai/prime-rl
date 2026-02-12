import torch


def force_eos(logits: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    logits.fill_(float("-inf"))
    logits[eos_token_id] = 0.0
    return logits
