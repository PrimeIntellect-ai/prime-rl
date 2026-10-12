import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import RowwiseParallel


class EmbeddingParallel(RowwiseParallel):
    """Vocabulary-parallel embedding with uneven token batches on each rank."""

    def __init__(self) -> None:
        super().__init__(input_layouts=Shard(0), output_layouts=Shard(0))

    def _prepare_input_fn(self, input_layouts, desired_input_layouts, module, inputs, device_mesh):
        indices = inputs[0]
        self.local_tokens = indices.shape[0]
        max_tokens = indices.new_tensor(self.local_tokens)
        dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=device_mesh.get_group())
        # TODO: Avoid the D2H sync in max_tokens.item(); F.pad needs the dynamic
        # cross-rank token count on the host to size the padded tensor.
        indices = F.pad(indices, (0, 0, 0, max_tokens.item() - self.local_tokens))
        return super()._prepare_input_fn(input_layouts, desired_input_layouts, module, (indices,), device_mesh)

    def _prepare_output_fn(self, output_layouts, use_local_output, module, outputs, device_mesh):
        output = super()._prepare_output_fn(output_layouts, use_local_output, module, outputs, device_mesh)
        return output[: self.local_tokens]
