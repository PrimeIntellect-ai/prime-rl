from functools import partial

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.parallel import RowwiseParallel


class EmbeddingParallel(RowwiseParallel):
    """Vocabulary-parallel embedding with local backward and uneven token batches.

    Lookup runs on plain local weights so dense backward never constructs the
    global table gradient. IDs and vectors are routed only to their owning rank.
    """

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Embedding):
            raise TypeError("EmbeddingParallel requires nn.Embedding")
        if module.padding_idx is not None or module.max_norm is not None or module.scale_grad_by_freq or module.sparse:
            raise ValueError(
                "EmbeddingParallel requires padding_idx=None, max_norm=None, scale_grad_by_freq=False and sparse=False"
            )
        distribute_module(module, device_mesh, self._partition_embedding_fn)
        module.forward = partial(self._forward, module, device_mesh)
        return module

    def _forward(self, module: torch.nn.Embedding, device_mesh: DeviceMesh, input: torch.Tensor) -> torch.Tensor:
        weight = module.weight.to_local(grad_placements=module.weight.placements)
        shard_rows = max(1, (module.num_embeddings + device_mesh.size() - 1) // device_mesh.size())
        start = device_mesh.get_local_rank() * shard_rows

        sorted_ids, order = input.flatten().sort(stable=True)
        send_counts = torch.bincount(sorted_ids // shard_rows, minlength=device_mesh.size())
        recv_counts = funcol.all_to_all_single(send_counts, None, None, device_mesh)
        # Stacking waits on the asynchronous counts; transfer both lists in one D2H copy.
        send, recv = torch.stack((send_counts, recv_counts)).cpu().tolist()
        recv_ids = funcol.all_to_all_single(sorted_ids, recv, send, device_mesh)
        vectors = F.embedding(recv_ids - start, weight)
        sorted_vectors = funcol.all_to_all_single_autograd(vectors, send, recv, device_mesh)

        # Restore local token/head order; each destination is written exactly once.
        output = torch.empty_like(sorted_vectors)
        output[order] = sorted_vectors
        return output.view(*input.shape, module.embedding_dim)
