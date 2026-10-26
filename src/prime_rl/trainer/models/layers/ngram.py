import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from prime_rl.utils.cp import CPContext, gather_for_cp_wo_grad


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime

    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime(candidate):
            candidate += 2
        prime = candidate
    return prime


def owner_counts(sorted_ids: torch.Tensor, rows_per_rank: int, num_ranks: int) -> torch.Tensor:
    first_rows = torch.arange(num_ranks + 1, device=sorted_ids.device) * rows_per_rank
    return torch.searchsorted(sorted_ids, first_rows).diff()


class LocalLookup:
    def __call__(self, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, weight)


class RowShardedLookup:
    """Look up a table sharded `Shard(0)` over a 1-D mesh by sending each id to the rank that owns its row."""

    def __init__(self, mesh: DeviceMesh, rows_per_rank: int) -> None:
        self.mesh = mesh
        self.rows_per_rank = rows_per_rank
        self.first_local_row = mesh.get_local_rank() * rows_per_rank

    def __call__(self, weight: DTensor, ids: torch.Tensor) -> torch.Tensor:
        local_weight = weight.to_local(grad_placements=weight.placements)
        sorted_ids, order = ids.flatten().sort(stable=True)
        send_counts = owner_counts(sorted_ids, self.rows_per_rank, self.mesh.size())
        recv_counts = funcol.all_to_all_single(send_counts, None, None, self.mesh)
        send_splits, recv_splits = torch.stack((send_counts, recv_counts)).cpu().tolist()
        recv_ids = funcol.all_to_all_single(sorted_ids, recv_splits, send_splits, self.mesh)
        rows = F.embedding(recv_ids - self.first_local_row, local_weight)
        sorted_rows = funcol.all_to_all_single_autograd(rows, send_splits, recv_splits, self.mesh)

        output = torch.empty_like(sorted_rows)
        output[order] = sorted_rows
        return output.view(*ids.shape, local_weight.shape[-1])


class NGramTable(nn.Module):
    """One hashed n-gram embedding table and its hash layout, looked up through a swappable strategy."""

    def __init__(self, num_rows: int, head_dim: int, max_ngram_size: int, num_columns: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_rows, head_dim))
        self.register_buffer("layer_multipliers", torch.zeros(max_ngram_size, dtype=torch.long))
        self.register_buffer("ngram_heads_vocab_sizes", torch.zeros(num_columns, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.zeros(num_columns, dtype=torch.long))
        self.lookup: LocalLookup | RowShardedLookup = LocalLookup()

    def set_lookup(self, lookup: LocalLookup | RowShardedLookup) -> None:
        self.lookup = lookup

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lookup(self.weight, ids).flatten(-2)


class NGramHasher(nn.Module, ABC):
    """Hash every token's preceding n-grams into row ids of each n-gram table, with per-model override points."""

    def __init__(
        self,
        *,
        max_ngram_size: int,
        heads_per_ngram: int,
        min_rows_per_head: int,
        rows_multiple_of: int,
    ) -> None:
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.num_columns = (max_ngram_size - 1) * heads_per_ngram
        self.min_rows_per_head = min_rows_per_head
        self.rows_multiple_of = rows_multiple_of
        self.cp_context = CPContext()

    def build_layout(self, table_index: int) -> tuple[list[int], list[int], list[int]]:
        sizes = self._head_sizes(table_index)
        offsets = list(itertools.accumulate(sizes[:-1], initial=0))
        return self.table_multipliers(table_index), sizes, offsets

    def table_rows(self, table_index: int) -> int:
        total_rows = sum(self._head_sizes(table_index))
        return (total_rows + self.rows_multiple_of - 1) // self.rows_multiple_of * self.rows_multiple_of

    def _head_sizes(self, table_index: int) -> list[int]:
        return [
            nth_prime_after(self.min_rows_per_head - 1, table_index * self.num_columns + column + 1)
            for column in range(self.num_columns)
        ]

    def map_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens

    def segment_starts(self, positions: torch.Tensor, cu_seqlens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        document_indices = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        return cu_seqlens[:-1][document_indices].long()

    @abstractmethod
    def pad_token(self) -> int: ...

    @abstractmethod
    def table_multipliers(self, table_index: int) -> list[int]: ...

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        multipliers: torch.Tensor,
        sizes: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        local_token_count = input_ids.numel()
        if self.cp_context.cp_enabled:
            input_ids = gather_for_cp_wo_grad(input_ids, self.cp_context.cp_world_size, self.cp_context.cp_group)

        tokens = input_ids.reshape(-1).long()
        positions = torch.arange(tokens.shape[0], device=tokens.device)
        local_start = self.cp_context.cp_rank * local_token_count
        local_slice = slice(local_start, local_start + local_token_count)
        local_positions = positions[local_slice]
        local_segment_starts = self.segment_starts(positions, cu_seqlens, tokens)[local_slice]
        mapped_tokens = self.map_tokens(tokens)
        pad_token = self.pad_token()
        shifted_tokens = []
        for shift in range(self.max_ngram_size):
            source_positions = local_positions - shift
            source_tokens = mapped_tokens[source_positions.clamp_min(0)]
            shifted_tokens.append(torch.where(source_positions >= local_segment_starts, source_tokens, pad_token))

        mixed = shifted_tokens[0][:, None] * multipliers[:, 0]
        id_blocks = []
        for ngram in range(2, self.max_ngram_size + 1):
            mixed = torch.bitwise_xor(mixed, shifted_tokens[ngram - 1][:, None] * multipliers[:, ngram - 1])
            columns = slice((ngram - 2) * self.heads_per_ngram, (ngram - 1) * self.heads_per_ngram)
            id_blocks.append(torch.remainder(mixed[..., None], sizes[:, columns]) + offsets[:, columns])
        return torch.cat(id_blocks, dim=-1)


class NGramEmbeddings(nn.Module):
    """The model's n-gram tables, keyed by consuming decoder index and hashed together once per forward."""

    def __init__(
        self,
        hasher: NGramHasher,
        table_decoder_indices: Sequence[int],
        num_layers: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if list(table_decoder_indices) != sorted(set(table_decoder_indices)):
            raise ValueError(f"table_decoder_indices must be sorted and distinct, got {list(table_decoder_indices)}")
        self.hasher = hasher
        self.table_indices = {
            decoder_index: table_index
            for table_index, decoder_index in enumerate(table_decoder_indices)
            if decoder_index < num_layers
        }
        self.hash_positions = {decoder_index: position for position, decoder_index in enumerate(self.table_indices)}
        self.tables = nn.ModuleDict(
            {
                str(decoder_index): NGramTable(
                    hasher.table_rows(table_index), head_dim, hasher.max_ngram_size, hasher.num_columns
                )
                for decoder_index, table_index in self.table_indices.items()
            }
        )
        self.reset_layouts()

    def reset_layouts(self) -> None:
        for decoder_index, table_index in self.table_indices.items():
            table = self.tables[str(decoder_index)]
            layout_buffers = (table.layer_multipliers, table.ngram_heads_vocab_sizes, table.ngram_heads_offsets)
            for buffer, values in zip(layout_buffers, self.hasher.build_layout(table_index), strict=True):
                buffer.copy_(buffer.new_tensor(values))

    def hash(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor | None:
        if not self.tables:
            return None
        tables = [self.tables[str(decoder_index)] for decoder_index in self.hash_positions]
        return self.hasher(
            input_ids,
            cu_seqlens,
            torch.stack([table.layer_multipliers for table in tables]),
            torch.stack([table.ngram_heads_vocab_sizes for table in tables]),
            torch.stack([table.ngram_heads_offsets for table in tables]),
        )

    def embed(self, ngram_ids: torch.Tensor | None, decoder_index: int) -> torch.Tensor | None:
        if decoder_index not in self.hash_positions:
            return None
        table = self.tables[str(decoder_index)]
        return table(ngram_ids[:, self.hash_positions[decoder_index]]).unsqueeze(0)


__all__ = [
    "LocalLookup",
    "NGramEmbeddings",
    "NGramHasher",
    "NGramTable",
    "RowShardedLookup",
    "is_prime",
    "nth_prime_after",
    "owner_counts",
]
