from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.parallel import RowwiseParallel


class EmbeddingParallel(RowwiseParallel):
    """Vocabulary-parallel embedding with local backward and uneven token batches.

    Lookup runs on plain local weights so dense backward never constructs the
    global table gradient. Outputs are reduce-scattered over the token dimension.
    """

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Embedding):
            raise TypeError("EmbeddingParallel requires nn.Embedding")
        if module.max_norm is not None or module.scale_grad_by_freq or module.sparse:
            raise ValueError("EmbeddingParallel requires a dense embedding without max_norm or scale_grad_by_freq")
        distribute_module(module, device_mesh, self._partition_embedding_fn)
        module.forward = partial(self._forward, module, device_mesh)
        return module

    def _forward(self, module: torch.nn.Embedding, device_mesh: DeviceMesh, input: torch.Tensor) -> torch.Tensor:
        local_tokens = input.shape[0]
        max_tokens = input.new_tensor(local_tokens)
        dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=device_mesh.get_group())
        # TODO: Avoid the D2H sync in max_tokens.item(); F.pad needs the dynamic
        # cross-rank token count on the host to size the padded tensor.
        padded = F.pad(input, (0, 0) * (input.ndim - 1) + (0, max_tokens.item() - local_tokens))
        indices = funcol.all_gather_single(padded.contiguous(), 0, device_mesh)

        weight = module.weight.to_local(grad_placements=module.weight.placements)
        shard_rows = (module.num_embeddings + device_mesh.size() - 1) // device_mesh.size()
        start = device_mesh.get_local_rank() * shard_rows
        owned = (indices >= start) & (indices < start + weight.shape[0])
        if weight.shape[0]:
            local_indices = (indices - start).clamp(0, weight.shape[0] - 1)
            padding_idx = module.padding_idx
            local_padding_idx = (
                padding_idx - start
                if padding_idx is not None and start <= padding_idx < start + weight.shape[0]
                else None
            )
            output = F.embedding(local_indices, weight, padding_idx=local_padding_idx)
            output = output.masked_fill(~owned.unsqueeze(-1), 0)
        else:
            output = weight.new_zeros((*indices.shape, module.embedding_dim)) + weight.sum()

        output = funcol.reduce_scatter_single_autograd(output, "sum", 0, device_mesh)
        return output[:local_tokens]
