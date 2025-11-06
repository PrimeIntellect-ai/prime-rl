from typing import Literal, Sequence

import torch
from torch.distributed.tensor.experimental._attention import _context_parallel_shard, context_parallel_unshard
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from prime_rl.trainer.parallel_dims import ParallelDims

attn_type = Literal["doc_causal", "causal", "none"]
t_shardable = torch.Tensor | BlockMask | torch.nn.Buffer


class CPSharder:
    def __init__(self, parallel_dims: ParallelDims):
        self.parallel_dims = parallel_dims
        if parallel_dims.cp > 1:
            self.cp_mesh = parallel_dims.world_mesh["cp"]
        else:
            self.cp_mesh = None

    def shard(self, *args: Sequence[t_shardable], **kwargs) -> Sequence[torch.Tensor | BlockMask | torch.nn.Buffer]:
        return args

    def _shard_arg(
        self,
        arg: t_shardable,
        shard_dim: tuple[int],
    ) -> t_shardable:
        if isinstance(arg, torch.nn.Buffer):
            self._buffer_count += 1

        return _context_parallel_shard(
            mesh=self.cp_mesh,
            buffers=[arg],
            seq_dims=shard_dim,
            load_balancer=self._load_balancer,
        )[0]
    
    def _unshard_arg(
        self,
        arg: t_shardable,
        shard_dim: tuple[int],
    ) -> t_shardable:
        return context_parallel_unshard(
            mesh=self.cp_mesh,
            buffers=[arg],
            seq_dims=shard_dim,
            load_balancer=self._load_balancer,
        )[0]

    def _shard_or_unshard_args(self, args: Sequence[t_shardable], buffer_seq_dims: Sequence[int], unshard: bool = False) -> Sequence[t_shardable]:
        new_args = []

        if unshard:
            self._function = self._unshard_arg
        else:
            self._function = self._shard_arg

        self._buffer_count = 0

        self._shard_dim_map = {
            torch.Tensor: lambda idx: (1,),
            BlockMask: lambda idx: (2,),
            torch.nn.Buffer: lambda idx: buffer_seq_dims[idx],
        }

        for arg in args:
            if not isinstance(arg, t_shardable):
                raise ValueError(f"Unsupported argument type: {type(arg)}")

            shard_dim = self._shard_dim_map[type(arg)](self._buffer_count)
            new_args.append(self._function(arg, shard_dim))

        return tuple(new_args)

    def unshard(self, *args: Sequence[t_shardable], buffer_seq_dims: Sequence[int] = None) -> Sequence[t_shardable]:
        if buffer_seq_dims is None:
            buffer_seq_dims = []

        assert self._load_balancer is not None, "Load balancer not set"

        return self._shard_or_unshard_args(args, buffer_seq_dims, unshard=True)


class DocCausalCPSharder(CPSharder):
    def __init__(self, parallel_dims: ParallelDims):
        super().__init__(parallel_dims)

    def shard(self, *args: Sequence[t_shardable], **kwargs) -> Sequence[t_shardable]:
        seq_length_per_doc = kwargs.get("seq_length_per_doc", None)
        if seq_length_per_doc is None:
            raise ValueError("seq_length_per_doc is required for DocCausalCPSharder")

        self._load_balancer = _PerDocumentHeadTailLoadBalancer(seq_length_per_doc, self.cp_mesh.size(0), args[0].device)

        return self._shard_or_unshard_args(args, kwargs.get("buffer_seq_dims", []), unshard=False)


class CausalCPSharder(CPSharder):
    def __init__(self, parallel_dims: ParallelDims):
        super().__init__(parallel_dims)

    def shard(self, *args: Sequence[t_shardable], **kwargs) -> Sequence[t_shardable]:
        seq_length = kwargs.get("seq_length", None)
        if seq_length is None:
            raise ValueError("seq_length is required for CausalCPSharder")

        self._load_balancer = _HeadTailLoadBalancer(seq_length, self.cp_mesh.size(0), args[0].device)

        return self._shard_or_unshard_args(args, kwargs.get("buffer_seq_dims", []))


def setup_cp_sharder(parallel_dims: ParallelDims, attn_mask_type: attn_type) -> CPSharder:
    if parallel_dims.cp <= 1:
        return CPSharder(parallel_dims)
    if attn_mask_type == "doc_causal":
        return CausalCPSharder(parallel_dims)
    elif attn_mask_type == "causal":
        return CausalCPSharder(parallel_dims)
    else:
        raise ValueError(f"Unsupported attention mask type: {attn_mask_type}")
