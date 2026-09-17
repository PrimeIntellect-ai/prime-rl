import pytest
import torch

from prime_rl.experimental.fully_shard_caching.fp8_cast import (
    grouped_per_block_cast_to_fp8,
    grouped_per_block_cast_to_fp8_both_layouts,
    transposed_blockwise_fp8,
)
from prime_rl.trainer.models.kernels.fp8_utils import ue8m0_for_device

BLOCK_ALIGNED_SHAPE = (2, 256, 384)
RAGGED_SHAPE = (2, 200, 300)
SENTINEL_VALUE = 1e30
SENTINEL_BYTE = 0xA5


@pytest.mark.parametrize("shape", [BLOCK_ALIGNED_SHAPE, RAGGED_SHAPE])
@pytest.mark.parametrize("fill_out", [False, True])
def test_transposed_blockwise_fp8_matches_both_layouts_bitwise(shape, fill_out):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    use_ue8m0 = ue8m0_for_device(weight.device)
    _, _, expected_out_t, expected_sf_t = grouped_per_block_cast_to_fp8_both_layouts(weight, use_ue8m0)
    qdata, scales = grouped_per_block_cast_to_fp8(weight, use_ue8m0)

    if fill_out:
        destinations = [torch.empty_like(expected_out_t), torch.empty_like(expected_sf_t)]
        destinations[0].view(torch.uint8).fill_(SENTINEL_BYTE)
        destinations[1].fill_(SENTINEL_VALUE)
        out_t, sf_t = transposed_blockwise_fp8(qdata, scales, out=destinations[0], sf=destinations[1])
        for returned, destination in zip((out_t, sf_t), destinations):
            assert returned is destination
    else:
        out_t, sf_t = transposed_blockwise_fp8(qdata, scales)

    assert out_t.is_contiguous()
    assert sf_t.is_contiguous()
    assert torch.equal(out_t.view(torch.uint8), expected_out_t.view(torch.uint8))
    assert torch.equal(sf_t, expected_sf_t)
