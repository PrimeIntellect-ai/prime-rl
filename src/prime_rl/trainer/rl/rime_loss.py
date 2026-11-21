"""Token-summed asymmetric PPO plus initial-policy KL for RIME."""

import torch

from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def rime_loss(inputs: LossInputs, clip_low=0.2, clip_high=0.28, kl_coef=0.01) -> LossOutputs:
    if not 0 < clip_low < 1 or clip_high <= 0 or kl_coef < 0:
        raise ValueError("Invalid RIME clipping or KL parameters")
    if inputs.ref_logprobs is None:
        raise ValueError("RIME requires initial-policy reference log probabilities")
    mask = inputs.loss_mask
    current = inputs.trainer_logprobs[mask].float()
    old = inputs.inference_logprobs[mask].detach().float()
    reference = inputs.ref_logprobs[mask].detach().float()
    advantage = inputs.advantages[mask].detach().float()
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    delta = reference - current
    kl = torch.expm1(delta) - delta
    tokens = -surrogate + kl_coef * kl
    if inputs.loss_weights is not None:
        tokens = tokens * inputs.loss_weights[mask]
    return LossOutputs(
        loss=tokens.sum(),
        metrics={
            "rime/reference_kl": kl.detach(),
            "rime/importance_ratio": ratio.detach(),
            "rime/advantage": advantage.detach(),
        },
    )
