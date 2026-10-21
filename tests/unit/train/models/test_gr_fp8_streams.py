import torch

from prime_rl.trainer.models.layers.hyper_connection import fp8_round_trip


def test_fp8_round_trip_scales_each_stream_and_passes_the_gradient():
    stream_scales = torch.tensor([1e-3, 1.0, 1e2, 1e4]).repeat_interleave(64)
    states = (torch.randn(2, 5, 4 * 64) * stream_scales).requires_grad_()
    out = fp8_round_trip(states, 4)
    streams, quantized = states.detach().unflatten(-1, (4, 64)), out.detach().unflatten(-1, (4, 64))
    row_max = streams.abs().amax(dim=-1, keepdim=True)
    large = streams.abs() > row_max / 64
    # e4m3 keeps 3 mantissa bits: at most half an ulp (2^-4) relative error in each row's normal range.
    assert ((quantized - streams).abs() <= streams.abs() * 2**-4)[large].all()
    assert torch.equal(quantized.abs().amax(dim=-1), row_max.squeeze(-1))
    out.sum().backward()
    assert torch.equal(states.grad, torch.ones_like(states))
