import torch

BLOCK = 32
E4M3_MAX = 448.0
E4M3_EMAX = 8  # 448 == 1.75 * 2**8


def _shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    bits = amax.float().contiguous().view(torch.int32)
    return (((bits >> 23) & 0xFF) - E4M3_EMAX).clamp_(0, 254)


def quantize_mx(t: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    k = t.shape[-1]
    assert k % block == 0, f"last dim {k} must be a multiple of {block}"
    g = t.float().reshape(*t.shape[:-1], k // block, block)
    e = _shared_exponent(g.abs().amax(-1))
    inv = torch.exp2((127 - e).float())
    q = (g * inv.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape(*t.shape[:-1], k).contiguous(), e.to(torch.uint8).view(torch.float8_e8m0fnu)


def dequantize_mx(q: torch.Tensor, scales: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    k = q.shape[-1]
    e = scales.view(torch.uint8).to(torch.int32)
    scale = torch.exp2((e - 127).float())
    g = q.float().reshape(*q.shape[:-1], k // block, block) * scale.unsqueeze(-1)
    return g.reshape(*q.shape[:-1], k)


def pack_scales_blocked(scales: torch.Tensor) -> torch.Tensor:
    u = scales.view(torch.uint8)
    lead = u.shape[:-2]
    mn, kb = u.shape[-2:]
    assert mn % 128 == 0, f"MN ({mn}) must be a multiple of 128"
    assert kb % 4 == 0, f"K ({kb * BLOCK}) must be a multiple of 128"
    t = u.reshape(-1, mn // 128, 4, 32, kb // 4, 4)
    t = t.permute(0, 1, 4, 3, 2, 5).contiguous()
    return t.reshape(*lead, -1).view(torch.float8_e8m0fnu)


def quantize_activation_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_mx(x)


def quantize_weight_mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, scales = quantize_mx(w)
    return q, pack_scales_blocked(scales)
