from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import _encode_images_per_cp_rank, _partition_images_for_cp
from prime_rl.trainer.models.qwen3_5.vision import Qwen3_5VisionModel
from prime_rl.utils.cp import CPContext


def test_partition_is_deterministic():
    thw = torch.tensor([[1, 64, 64]] * 8 + [[1, 32, 32]] * 8)
    assert _partition_images_for_cp(thw, 4) == _partition_images_for_cp(thw, 4)


def test_partition_covers_every_image_once():
    thw = torch.tensor([[1, 64, 64]] * 5 + [[1, 32, 32]] * 3 + [[1, 16, 16]] * 2)
    buckets = _partition_images_for_cp(thw, 4)
    flat = sorted(i for b in buckets for i in b)
    assert flat == list(range(thw.shape[0]))


def test_partition_is_balanced_by_patches():
    thw = torch.tensor([[1, 64, 64]] * 8 + [[1, 32, 32]] * 8)
    buckets = _partition_images_for_cp(thw, 4)
    loads = [sum(int(thw[i].prod()) for i in b) for b in buckets]
    assert max(loads) - min(loads) <= 32 * 32


def test_partition_more_ranks_than_images_leaves_empty_buckets():
    thw = torch.tensor([[1, 32, 32]])
    buckets = _partition_images_for_cp(thw, 4)
    assert sum(len(b) for b in buckets) == 1
    assert sum(1 for b in buckets if not b) == 3


def _check_empty_rank_backward(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    torch.manual_seed(42)
    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=32,
        patch_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    config._attn_implementation = "flash_attention_2"
    visual = Qwen3_5VisionModel(config).cuda()
    fully_shard(
        visual,
        mesh=init_device_mesh("cuda", (2,)),
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    )
    pixels = torch.randn(16, 12, device="cuda", dtype=torch.bfloat16)
    grid = torch.tensor([[1, 4, 4]], device="cuda")
    baseline = visual(pixels, grid).pooler_output
    baseline[rank::2].float().square().sum().backward()
    reference = {name: parameter.grad.to_local().clone() for name, parameter in visual.named_parameters()}
    visual.zero_grad(set_to_none=True)

    output = _encode_images_per_cp_rank(visual, pixels, grid, 2, CPContext(dist.group.WORLD, rank, 2), True)
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    assert output.requires_grad, "An empty CP rank must participate in autograd gather and FSDP backward"
    output[rank::2].float().square().sum().backward()
    error = torch.zeros(2, dtype=torch.float64, device="cuda")
    for name, parameter in visual.named_parameters():
        assert parameter.grad is not None, name
        gradient = parameter.grad.to_local()
        assert torch.isfinite(gradient).all(), name
        error[0] += (gradient - reference[name]).double().square().sum()
        error[1] += reference[name].double().square().sum()
    dist.all_reduce(error)
    assert error[1] > 0
    assert (error[0] / error[1]).sqrt() < 0.02
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_empty_cp_rank_retains_fsdp_backward(free_port):
    mp.spawn(_check_empty_rank_backward, args=(free_port,), nprocs=2)
