import pytest
import torch

from prime_rl.trainer.models.inkling2.converting_inkling2 import InterleavedGateUp


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_interleaved_gate_up_rows_alternate_and_round_trip(dim):
    op = InterleavedGateUp(hf="w13", gate="gate", up="up", dim=dim)
    shape = [3, 4, 5]
    gate, up = torch.randn(shape), torch.randn(shape)
    sd = {"gate": gate.clone(), "up": up.clone()}

    op.prime_to_hf(sd)
    w13 = sd["w13"]
    index = (slice(None),) * dim
    assert w13.shape[dim] == 2 * shape[dim]
    torch.testing.assert_close(w13[index + (slice(0, None, 2),)], gate, rtol=0, atol=0)
    torch.testing.assert_close(w13[index + (slice(1, None, 2),)], up, rtol=0, atol=0)

    op.hf_to_prime(sd)
    torch.testing.assert_close(sd["gate"], gate, rtol=0, atol=0)
    torch.testing.assert_close(sd["up"], up, rtol=0, atol=0)
