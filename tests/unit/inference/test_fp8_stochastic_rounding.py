import torch

from prime_rl.inference.patches import stochastic_round_fp8


def test_stochastic_round_fp8_preserves_on_grid_values():
    values = torch.tensor([-2.0, -1.125, -0.5, 0.0, 0.5, 1.125, 2.0])
    quantized = values.to(torch.float8_e4m3fn)

    rounded = stochastic_round_fp8(values, quantized)

    assert torch.equal(rounded.view(torch.uint8), quantized.view(torch.uint8))


def test_stochastic_round_fp8_is_unbiased_at_midpoint():
    torch.manual_seed(0)
    midpoint = torch.full((200_000,), 1.0625)
    nearest = midpoint.to(torch.float8_e4m3fn)

    rounded = stochastic_round_fp8(midpoint, nearest).float()

    assert abs(rounded.mean().item() - midpoint[0].item()) < 1e-3
