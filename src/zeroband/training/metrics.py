import torch
import torch.distributed as dist
from torch import Tensor

from zeroband.training.world import get_world


class BatchMetrics:
    """
    Utility class to accumulate local metrics over multiple gradient
    accumulation steps and synchronize them across all ranks. For each call to
    `update` the tensor is accumulated and the count is incremented.  When
    `sync` is called, the tensor and counts are summed across all ranks ensuring
    that we compute an average over all ranks.
    """

    def __init__(self):
        self.metrics: dict[str, list[Tensor]] = {}
        self.count: dict[str, int] = {}
        self.world = get_world()

    @torch.no_grad()
    def update(self, key: str, value: Tensor):
        if key not in self.metrics:
            self.metrics[key] = value
            self.count[key] = 1
        else:
            self.metrics[key] += value
            self.count[key] += 1

    @torch.no_grad()
    def sync(self):
        for key in self.metrics:
            value = self.metrics[key].clone()
            count = torch.tensor(self.count[key])

            dist.all_reduce(value.to("cuda"), op=dist.ReduceOp.SUM)
            dist.all_reduce(count.to("cuda"), op=dist.ReduceOp.SUM)

            value = value / count

            self.metrics[key] = value

    def __getitem__(self, key):
        return self.metrics[key]

    def items(self):
        return self.metrics.items()
