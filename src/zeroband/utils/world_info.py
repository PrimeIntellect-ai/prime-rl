from typing import Dict

from zeroband.utils import envs


class WorldInfo:
    """
    This class parses enviornment variables (usually set by torchrun) to get topology information for distributed settings training and inference.
    """

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    def __init__(self):
        """
        Initialize the WorldInfo object by parsing enviornment variables. If not set,
        it will set variables for single-node single-GPU training.
        """
        self.rank = envs.RANK
        self.world_size = envs.WORLD_SIZE
        self.local_rank = envs.LOCAL_RANK
        self.local_world_size = envs.LOCAL_WORLD_SIZE
        self._check_world_info()
        self.num_nodes = self.world_size // self.local_world_size

    def _check_world_info(self):
        assert 0 <= self.local_rank < self.local_world_size
        assert 0 <= self.rank < self.world_size
        assert self.local_world_size <= self.world_size
        # TODO: This is only true if we have evenly distributed node groups, which is probably a fair assumption (maybe at some point we want to run uneven node groups for pipelined inference)
        assert self.world_size % self.local_world_size == 0

    def __repr__(self):
        return f"WorldInfo(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}, local_world_size={self.local_world_size}, num_nodes={self.num_nodes})"

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


def get_world_info() -> WorldInfo:
    """Returns the WorldInfo. If not initialized, it will initialize."""
    global _WORLD_INFO
    if _WORLD_INFO is None:
        _WORLD_INFO = WorldInfo()
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
