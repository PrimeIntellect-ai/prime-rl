import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


class ExpertWeightParallel(ParallelStyle):
    @staticmethod
    def _partition_fn(_name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        for parameter_name, parameter in module.named_parameters(recurse=False):
            sharded = distribute_tensor(parameter, device_mesh, [Shard(0)])
            module.register_parameter(parameter_name, nn.Parameter(sharded))

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)
