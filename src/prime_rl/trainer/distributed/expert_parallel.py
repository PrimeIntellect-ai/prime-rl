import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
from torchtitan.distributed.expert_parallel import ExpertParallel


class MXFP8ExpertParallel(ExpertParallel):
    """Expert-parallel style that quantizes the token all-to-all to MXFP8.

    Extends TorchTitan's ``ExpertParallel`` (which transfers bf16 over the dispatch/combine
    all-to-all) by routing the collectives through torchao's MXFP8 a2a kernels. The forward
    dispatch and the backward combine transfer (1x32 e8m0-scaled) FP8 payloads, roughly
    halving the EP communication volume; the dispatched activations are dequantized to bf16
    locally before the expert grouped GEMM consumes them.
    """

    def __init__(self, scaling_mode: str = "rceil"):
        super().__init__()
        from torchao.prototype.mx_formats.config import ScaleCalculationMode

        self._scaling_mode = {"rceil": ScaleCalculationMode.RCEIL, "floor": ScaleCalculationMode.FLOOR}[scaling_mode]

    def _token_dispatch(self, mod, inputs, device_mesh):
        from prime_rl.trainer.distributed.mxfp8_a2a import mxfp8_dispatch_all_to_all

        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()

        # Exchange per-expert token counts so each rank knows the all-to-all splits.
        with torch.no_grad():
            num_tokens_per_expert_group = num_tokens_per_expert.new_empty(num_tokens_per_expert.shape[0])
            dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert, group=group)
            # NOTE: this incurs a device-to-host sync.
            self.input_splits = num_tokens_per_expert.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
            self.output_splits = num_tokens_per_expert_group.view(device_mesh.shape[0], -1).sum(dim=1).tolist()

        routed_input = mxfp8_dispatch_all_to_all(
            routed_input, self.output_splits, self.input_splits, group, self._scaling_mode
        )
        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):
        from prime_rl.trainer.distributed.mxfp8_a2a import mxfp8_combine_all_to_all

        group = device_mesh.get_group()
        return mxfp8_combine_all_to_all(
            routed_output, self.input_splits, self.output_splits, group, self._scaling_mode
        )


class DeepEPExpertParallel(ParallelStyle):
    """Expert-parallel style backed by DeepEP dispatch/combine.

    Only handles weight sharding (Shard(0) on expert dim) and stores the EP
    process group on the module. PrimeRL drives DeepEP dispatch/combine from
    `MoE.forward()` so communication stays outside the selective-AC checkpoint
    boundary while local expert matmuls remain checkpointable.
    """

    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(param_name, nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))
        mod._ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)


def get_ep_group(experts: nn.Module) -> ProcessGroup:
    return experts._ep_group
