import itertools
from abc import ABC, abstractmethod
from collections.abc import Mapping

import sympy
import torch
from jaxtyping import Int
from torch import Tensor, nn

from prime_rl.trainer.distributed.ngram_lookup import LocalLookup, NGramLookup
from prime_rl.utils.cp import CPContext, gather_for_cp_wo_grad

LAYOUT_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


def consecutive_primes_from(minimum: int, skip: int, count: int) -> list[int]:
    return [sympy.nextprime(minimum - 1, ith=skip + index + 1) for index in range(count)]


class NGramTable(nn.Module):
    """One hashed n-gram embedding table and its hash layout, looked up through a swappable strategy."""

    def __init__(self, num_rows: int, head_dim: int, max_ngram_size: int, num_heads: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_rows, head_dim))
        for name, length in zip(LAYOUT_BUFFERS, (max_ngram_size, num_heads, num_heads), strict=True):
            self.register_buffer(name, torch.zeros(length, dtype=torch.long))
        self.lookup: NGramLookup = LocalLookup()

    def set_lookup(self, lookup: NGramLookup) -> None:
        self.lookup = lookup

    def set_layout(self, multipliers: list[int], sizes: list[int], offsets: list[int]) -> None:
        for name, values in zip(LAYOUT_BUFFERS, (multipliers, sizes, offsets), strict=True):
            buffer = getattr(self, name)
            buffer.copy_(buffer.new_tensor(values))

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lookup(self.weight, ids)


class NGramHasher(nn.Module, ABC):
    """Hash every token's preceding n-grams into row ids of each n-gram table, with per-model override points."""

    def __init__(
        self,
        *,
        max_ngram_size: int,
        heads_per_ngram: int,
        rows_multiple_of: int,
    ) -> None:
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (max_ngram_size - 1) * heads_per_ngram
        self.rows_multiple_of = rows_multiple_of
        self.cp_context = CPContext()

    def build_layout(self, table_index: int) -> tuple[list[int], list[int], list[int]]:
        sizes = self.head_sizes(table_index)
        offsets = list(itertools.accumulate(sizes[:-1], initial=0))
        return self.table_multipliers(table_index), sizes, offsets

    def num_table_rows(self, table_index: int) -> int:
        total_rows = sum(self.head_sizes(table_index))
        return (total_rows + self.rows_multiple_of - 1) // self.rows_multiple_of * self.rows_multiple_of

    def map_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens

    def segment_starts(self, positions: torch.Tensor, cu_seqlens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        document_indices = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        return cu_seqlens[:-1][document_indices].long()

    @abstractmethod
    def head_sizes(self, table_index: int) -> list[int]: ...

    @abstractmethod
    def pad_token(self) -> int: ...

    @abstractmethod
    def table_multipliers(self, table_index: int) -> list[int]: ...

    @torch.no_grad()
    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"],
        cu_seqlens: torch.Tensor,
        multipliers: Int[Tensor, "tables max_ngram_size"],
        sizes: Int[Tensor, "tables heads"],
        offsets: Int[Tensor, "tables heads"],
    ) -> Int[Tensor, "tokens tables heads"]:
        sequence_tokens, local_positions = self.gather_sequence(input_ids)
        ngram_tokens = self.ngram_tokens(sequence_tokens, local_positions, cu_seqlens)
        ngram_hashes = self.hash_ngrams(ngram_tokens, multipliers)
        return self.table_rows(ngram_hashes, sizes, offsets)

    def gather_sequence(
        self, input_ids: Int[Tensor, "batch seq"]
    ) -> tuple[Int[Tensor, "sequence"], Int[Tensor, "tokens"]]:
        local_token_count = input_ids.numel()
        if self.cp_context.cp_enabled:
            input_ids = gather_for_cp_wo_grad(input_ids, self.cp_context.cp_world_size, self.cp_context.cp_group)
        local_start = self.cp_context.cp_rank * local_token_count
        local_positions = torch.arange(local_start, local_start + local_token_count, device=input_ids.device)
        return input_ids.reshape(-1).long(), local_positions

    def ngram_tokens(
        self, sequence_tokens: Int[Tensor, "sequence"], local_positions: Int[Tensor, "tokens"], cu_seqlens: torch.Tensor
    ) -> Int[Tensor, "tokens max_ngram_size"]:
        sequence_positions = torch.arange(sequence_tokens.shape[0], device=sequence_tokens.device)
        segment_starts = self.segment_starts(sequence_positions, cu_seqlens, sequence_tokens)[local_positions]
        source_positions = local_positions[:, None] - torch.arange(self.max_ngram_size, device=sequence_tokens.device)
        source_tokens = self.map_tokens(sequence_tokens)[source_positions.clamp_min(0)]
        return torch.where(source_positions >= segment_starts[:, None], source_tokens, self.pad_token())

    def hash_ngrams(
        self, ngram_tokens: Int[Tensor, "tokens max_ngram_size"], multipliers: Int[Tensor, "tables max_ngram_size"]
    ) -> Int[Tensor, "tokens tables ngram_sizes"]:
        products = ngram_tokens[:, None, :] * multipliers
        rolling_hash = products[..., 0]
        hashes = []
        for token_index in range(1, self.max_ngram_size):
            rolling_hash = rolling_hash ^ products[..., token_index]
            hashes.append(rolling_hash)
        return torch.stack(hashes, dim=-1)

    def table_rows(
        self,
        ngram_hashes: Int[Tensor, "tokens tables ngram_sizes"],
        sizes: Int[Tensor, "tables heads"],
        offsets: Int[Tensor, "tables heads"],
    ) -> Int[Tensor, "tokens tables heads"]:
        head_hashes = ngram_hashes.repeat_interleave(self.heads_per_ngram, dim=-1)
        return torch.remainder(head_hashes, sizes) + offsets


class NGramEmbeddings(nn.Module):
    """The model's n-gram tables, keyed by consuming layer index and hashed together once per forward."""

    def __init__(
        self,
        hasher: NGramHasher,
        table_indices: Mapping[int, int],
        head_dim: int,
    ) -> None:
        super().__init__()
        self.hasher = hasher
        self.table_indices = dict(sorted(table_indices.items()))
        self.hash_positions = {layer_index: position for position, layer_index in enumerate(self.table_indices)}
        self.tables = nn.ModuleDict(
            {
                str(layer_index): NGramTable(
                    hasher.num_table_rows(table_index), head_dim, hasher.max_ngram_size, hasher.num_heads
                )
                for layer_index, table_index in self.table_indices.items()
            }
        )
        self.reset_layouts()

    def reset_layouts(self) -> None:
        for layer_index, table_index in self.table_indices.items():
            self.tables[str(layer_index)].set_layout(*self.hasher.build_layout(table_index))

    def hash(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor | None:
        if not self.tables:
            return None
        tables = [self.tables[str(layer_index)] for layer_index in self.hash_positions]
        multipliers, sizes, offsets = (
            torch.stack([getattr(table, name) for table in tables]) for name in LAYOUT_BUFFERS
        )
        return self.hasher(input_ids, cu_seqlens, multipliers, sizes, offsets)

    def embed(self, ngram_ids: torch.Tensor | None, layer_index: int) -> torch.Tensor | None:
        if layer_index not in self.hash_positions:
            return None
        table = self.tables[str(layer_index)]
        return table(ngram_ids[:, self.hash_positions[layer_index]]).flatten(-2).unsqueeze(0)


__all__ = [
    "LAYOUT_BUFFERS",
    "NGramEmbeddings",
    "consecutive_primes_from",
    "NGramHasher",
    "NGramTable",
]
