import torch
from torch import nn
from torch.distributed import ProcessGroup

from prime_rl.trainer.models.layers.head_sharded_embedding import HeadShardedEmbedding
from prime_rl.utils.cp import gather_for_cp

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
NGRAM_LAYER_PRIME = 10007


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_MULTIPLIER_1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_MULTIPLIER_2) & MASK64
    return (value ^ (value >> 31)) & MASK64


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


def build_hash_layout(
    *,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size: int,
    token_vocab_size: int,
    ngram_layer_index: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    max_multiplier = ((1 << 63) - 1) // token_vocab_size
    multiplier_bound = max(1, max_multiplier // 2)
    layer_seed = seed + NGRAM_LAYER_PRIME * ngram_layer_index
    multipliers = [
        2 * (splitmix64(layer_seed + SPLITMIX_GAMMA * (index + 1)) % multiplier_bound) + 1
        for index in range(ngram_size)
    ]

    num_heads = (ngram_size - 1) * heads_per_ngram
    sizes = [
        nth_prime_after(ngram_vocab_size - 1, ngram_layer_index * num_heads + head + 1) for head in range(num_heads)
    ]
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += size
    return multipliers, sizes, offsets


class NGramEmbedding(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        ngram_size: int,
        heads_per_ngram: int,
        ngram_vocab_size: int,
        token_vocab_size: int,
        eos_token_id: int,
        vocab_size_divisor: int,
        ngram_layer_index: int,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (ngram_size - 1) * heads_per_ngram
        self.head_dim = embedding_dim // self.num_heads
        self.ngram_vocab_size = ngram_vocab_size
        self.token_vocab_size = token_vocab_size
        self.eos_token_id = eos_token_id
        self.ngram_layer_index = ngram_layer_index
        self.seed = seed

        multipliers, sizes, offsets = build_hash_layout(
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            ngram_vocab_size=ngram_vocab_size,
            token_vocab_size=token_vocab_size,
            ngram_layer_index=ngram_layer_index,
            seed=seed,
        )
        self.register_buffer("layer_multipliers", torch.tensor(multipliers, dtype=torch.long))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.tensor(offsets, dtype=torch.long))

        total_vocab_size = offsets[-1] + sizes[-1]
        padded_vocab_size = (total_vocab_size + vocab_size_divisor - 1) // vocab_size_divisor * vocab_size_divisor
        self.ngram_embedding = HeadShardedEmbedding(padded_vocab_size, self.head_dim, sizes)

        self.context_parallel_group: ProcessGroup | None = None
        self.context_parallel_rank = 0

    def reset_parameters(self) -> None:
        multipliers, sizes, offsets = build_hash_layout(
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
            ngram_vocab_size=self.ngram_vocab_size,
            token_vocab_size=self.token_vocab_size,
            ngram_layer_index=self.ngram_layer_index,
            seed=self.seed,
        )
        self.layer_multipliers.copy_(self.layer_multipliers.new_tensor(multipliers))
        self.ngram_heads_vocab_sizes.copy_(self.ngram_heads_vocab_sizes.new_tensor(sizes))
        self.ngram_heads_offsets.copy_(self.ngram_heads_offsets.new_tensor(offsets))

    def set_context_parallel_attributes(self, process_group: ProcessGroup, rank: int, world_size: int) -> None:
        self.context_parallel_group = process_group
        self.context_parallel_rank = rank

    def compute_ngram_ids(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        tokens = input_ids.reshape(-1).long()
        positions = torch.arange(tokens.shape[0], device=tokens.device)
        sequence_indices = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        sequence_starts = cu_seqlens[:-1][sequence_indices].long()

        eos_positions = torch.where(tokens == self.eos_token_id, positions, -1)
        latest_eos = torch.cummax(eos_positions, dim=0).values
        previous_eos = torch.cat((latest_eos.new_full((1,), -1), latest_eos[:-1]))
        segment_starts = torch.maximum(sequence_starts, previous_eos + 1)

        shifted_tokens = [tokens]
        for shift in range(1, self.ngram_size):
            source_positions = positions - shift
            source_tokens = tokens[source_positions.clamp_min(0)]
            shifted_tokens.append(
                torch.where(
                    source_positions >= segment_starts,
                    source_tokens,
                    source_tokens.new_full((), self.eos_token_id),
                )
            )

        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            mixed = shifted_tokens[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted_tokens[index] * self.layer_multipliers[index])

            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            id_blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(id_blocks, dim=-1)

    def forward(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        local_tokens = input_ids.numel()
        if self.context_parallel_group is None:
            full_input_ids = input_ids
        else:
            full_input_ids = gather_for_cp(input_ids, self.context_parallel_group)

        ngram_ids = self.compute_ngram_ids(full_input_ids, cu_seqlens)
        if self.context_parallel_group is not None:
            local_start = self.context_parallel_rank * local_tokens
            ngram_ids = ngram_ids[local_start : local_start + local_tokens]

        embeddings = self.ngram_embedding(ngram_ids).flatten(-2)
        return embeddings.reshape(*input_ids.shape, self.embedding_dim)


__all__ = ["NGramEmbedding"]
