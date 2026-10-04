import torch

from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import _partition_images_for_cp


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
