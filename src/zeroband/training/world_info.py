from typing import Dict

import torch

import zeroband.training.envs as envs


class WorldInfo:
    """
    This class retrieves topology information for distributed training and inference settings by parsing environment variables, typically set by torchrun.
    """

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    def __init__(
        self, rank: int | None = None, world_size: int | None = None, local_rank: int | None = None, local_world_size: int | None = None
    ):
        """
        Initialize the WorldInfo object either manually or by parsing environment variables.
        """
        self.rank = rank or envs.RANK
        self.world_size = world_size or envs.WORLD_SIZE
        self.local_rank = local_rank or envs.LOCAL_RANK
        self.local_world_size = local_world_size or envs.LOCAL_WORLD_SIZE
        self.gpu_ids = envs.CUDA_VISIBLE_DEVICES or list(range(torch.cuda.device_count()))
        self._check_world_info()
        self.num_gpus = len(self.gpu_ids)
        self.num_nodes = self.world_size // self.local_world_size

    def _check_world_info(self):
        assert 0 <= self.local_rank < self.local_world_size
        assert 0 <= self.rank < self.world_size
        assert self.local_world_size <= self.world_size
        # TODO: This is only true if we have evenly distributed node groups, which is probably a fair assumption (maybe at some point we want to run uneven node groups for pipelined inference)
        assert self.world_size % self.local_world_size == 0

    def __repr__(self):
        return f"WorldInfo(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}, local_world_size={self.local_world_size}, num_nodes={self.num_nodes}, num_gpus={self.num_gpus}, gpu_ids={self.gpu_ids})"

    def json(self) -> Dict[str, int]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "num_nodes": self.num_nodes,
        }


# Singleton instance of WorldInfo
_WORLD_INFO: WorldInfo | None = None


def get_world_info(
    rank: int | None = None, world_size: int | None = None, local_rank: int | None = None, local_world_size: int | None = None
) -> WorldInfo:
    """Returns the WorldInfo. If not initialized, it will initialize."""
    global _WORLD_INFO
    if _WORLD_INFO is None:
        _WORLD_INFO = WorldInfo(rank=rank, world_size=world_size, local_rank=local_rank, local_world_size=local_world_size)
    return _WORLD_INFO


def reset_world_info() -> None:
    global _WORLD_INFO
    _WORLD_INFO = None


if __name__ == "__main__":
    # Used in tests/units/test_world_info.py to test init with torchrun
    import torch.distributed as dist

    print(get_world_info())
    if dist.is_initialized():
        dist.destroy_process_group()
