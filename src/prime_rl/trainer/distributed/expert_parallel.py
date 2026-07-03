import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchao.prototype.moe_training.ep import a2a_combine_hp_fwd_mxfp8_bwd


class MXFP8ExpertParallel(ExpertParallel):
    def _token_dispatch(self, mod, inputs, device_mesh):
        from torchao.prototype.moe_training.ep import a2a_dispatch_mxfp8_fwd_hp_bwd

        routed_input, num_tokens_per_expert = inputs
        with torch.no_grad():
            num_tokens_per_expert_group = num_tokens_per_expert.new_empty(num_tokens_per_expert.shape[0])
            dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert, group=device_mesh.get_group())
            self.input_splits = num_tokens_per_expert.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
            self.output_splits = num_tokens_per_expert_group.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
        mx_routed_input = a2a_dispatch_mxfp8_fwd_hp_bwd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group().group_name,
        )
        routed_input = mx_routed_input.dequantize(routed_input.dtype)
        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):

        return a2a_combine_hp_fwd_mxfp8_bwd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group().group_name,
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
