from torch import nn
from torch.distributed.tensor.parallel import parallelize_module

from prime_rl.trainer.distributed.ngram_parallel import NGramRowParallel
from prime_rl.trainer.models.layers.ngram import NGramTable, RowShardedLookup
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.logger import get_logger


def configure_ngram_runtime(model: nn.Module, parallel_dims: ParallelDims) -> None:
    tables = [module for module in model.modules() if isinstance(module, NGramTable)]
    if not tables:
        return
    head_mesh = parallel_dims.get_mesh("head")
    num_ranks = head_mesh.size()
    if num_ranks == 1:
        return

    for table in tables:
        rows_per_rank = max(1, (table.weight.shape[0] + num_ranks - 1) // num_ranks)
        parallelize_module(table, device_mesh=head_mesh, parallelize_plan=NGramRowParallel())
        table.set_lookup(RowShardedLookup(head_mesh, rows_per_rank))

    get_logger().info(f"Row-sharded {len(tables)} n-gram tables over the head mesh ({num_ranks} ranks)")
