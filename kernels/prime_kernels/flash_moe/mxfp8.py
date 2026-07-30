import torch

BLOCK = 32


def pack_scales_blocked(scales: torch.Tensor) -> torch.Tensor:
    """Reorder row major `(..., MN, K // 32)` weight scales into the blocked layout the kernel reads.

    The kernel addresses weight scales as a sequence of 512 byte tiles, each covering a
    128 MN x 128 K block. Activation scales stay row major and need no packing.
    """
    u = scales.view(torch.uint8)
    lead = u.shape[:-2]
    mn, kb = u.shape[-2:]
    assert mn % 128 == 0, f"MN ({mn}) must be a multiple of 128"
    assert kb % 4 == 0, f"K ({kb * BLOCK}) must be a multiple of 128"
    t = u.reshape(-1, mn // 128, 4, 32, kb // 4, 4)  # (lead, mn_tile, s, d, k_tile, kb)
    t = t.permute(0, 1, 4, 3, 2, 5).contiguous()  # (lead, mn_tile, k_tile, d, s, kb)
    return t.reshape(*lead, -1).view(torch.float8_e8m0fnu)
