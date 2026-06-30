import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


class MXFP8ExpertParallel(ParallelStyle):
    """Expert parallel with MXFP8 all-to-all, via torchao's ``moe_training.ep`` pipeline.

    Functionally equivalent to torchao's reference ``MXFP8ExpertParallel`` but shaped like
    prime-rl's other EP styles. The dispatch all-to-all and the token permute quantize to
    (1x32 e8m0-scaled) MXFP8; the permuted ``MXTensor`` flows straight into the expert grouped
    GEMM (consumed pre-quantized), and the unpermute + combine all-to-all run in bf16:

        forward:  mxfp8 a2a-dispatch -> mxfp8 permute -> mxfp8 grouped GEMM -> bf16 unpermute -> bf16 a2a-combine
        backward: bf16  a2a-dispatch <- bf16  permute <- mxfp8 grouped GEMM <- mxfp8 unpermute <- mxfp8 a2a-combine

    The permute pads each local expert's token group to a multiple of 32 (the MXFP8 scaling
    block size), so ``GroupedExperts`` must run its grouped GEMM directly on the dispatched
    ``MXTensor`` (no ``@expert_parallel`` re-permute) when this style is applied — see
    ``GroupedExperts._forward_mxfp8_ep``.
    """

    def __init__(self):
        super().__init__()
        self.input_splits: list[int] | None = None
        self.output_splits: list[int] | None = None
        self.input_shape: torch.Size | None = None
        self.permuted_indices: torch.Tensor | None = None

    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(param_name, nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))

    def _token_dispatch(self, mod, inputs, device_mesh):
        from torchao.prototype.moe_training.ep import (
            a2a_dispatch_mxfp8_fwd_hp_bwd,
            permute_mxfp8_fwd_hp_bwd,
        )

        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        group = device_mesh.get_group()

        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(num_tokens_per_expert, None, None, group=group)
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(num_tokens_per_expert_group)
            # NOTE: these incur a device-to-host sync.
            self.input_splits = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1).tolist()
            self.output_splits = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1).tolist()

        # mxfp8-quantized all-to-all dispatch; returns an MXTensor.
        routed_input = a2a_dispatch_mxfp8_fwd_hp_bwd(
            routed_input, output_splits=self.output_splits, input_splits=self.input_splits, group_name=group.group_name
        )
        # Shuffle to local-expert-contiguous layout and pad each group to a multiple of 32.
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
            _,
        ) = permute_mxfp8_fwd_hp_bwd(routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts)
        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):
        from torchao.prototype.moe_training.ep import (
            a2a_combine_hp_fwd_mxfp8_bwd,
            unpermute_hp_fwd_mxfp8_bwd,
        )

        routed_output = unpermute_hp_fwd_mxfp8_bwd(routed_output, self.permuted_indices, self.input_shape)
        # Reverse the dispatch all-to-all (swap input/output splits).
        return a2a_combine_hp_fwd_mxfp8_bwd(
            routed_output,
            output_splits=self.input_splits,
            input_splits=self.output_splits,
            group_name=device_mesh.get_group().group_name,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
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
