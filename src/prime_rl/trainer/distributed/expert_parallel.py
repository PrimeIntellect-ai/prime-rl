import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
from torchao.prototype.moe_training.ep import a2a_combine_hp_fwd_mxfp8_bwd, a2a_dispatch_mxfp8_fwd_hp_bwd
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchtitan.distributed.expert_parallel import ExpertParallel


class _MXFP8Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed_input, output_splits, input_splits, group_name):
        mx_routed_input = a2a_dispatch_mxfp8_fwd_hp_bwd(routed_input, output_splits, input_splits, group_name)
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = _resolve_process_group(group_name)
        return mx_routed_input.dequantize(routed_input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty((sum(ctx.input_splits),) + tuple(grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input, grad_output.contiguous(), ctx.input_splits, ctx.output_splits, group=ctx.group
        )
        return grad_input, None, None, None


class _DequantMXGradInBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hp_dtype):
        ctx.hp_dtype = hp_dtype
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if isinstance(grad_output, MXTensor):
            grad_output = grad_output.dequantize(ctx.hp_dtype)
        return grad_output, None


class MXFP8ExpertParallel(ExpertParallel):
    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        with torch.no_grad():
            num_tokens_per_expert_group = num_tokens_per_expert.new_empty(num_tokens_per_expert.shape[0])
            dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert, group=device_mesh.get_group())
            self.input_splits = num_tokens_per_expert.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
            self.output_splits = num_tokens_per_expert_group.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
        routed_input = _MXFP8Dispatch.apply(
            routed_input, self.output_splits, self.input_splits, device_mesh.get_group().group_name
        )
        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):
        routed_output = _DequantMXGradInBwd.apply(routed_output, routed_output.dtype)
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
