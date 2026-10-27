import torch

from prime_rl.trainer.models.layers.ngram import NGramHasher
from prime_rl.trainer.models.qwen3_8_flash_next.configuration_qwen3_8_flash_next import Qwen3_8FlashNextTextConfig

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
NGRAM_LAYER_PRIME = 10007
NGRAM_HASH_SEED = 1234


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_MULTIPLIER_1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_MULTIPLIER_2) & MASK64
    return (value ^ (value >> 31)) & MASK64


class Qwen3_8FlashNextNGramHasher(NGramHasher):
    def __init__(self, config: Qwen3_8FlashNextTextConfig) -> None:
        super().__init__(
            max_ngram_size=config.ngram_size,
            heads_per_ngram=config.heads_per_ngram,
            min_rows_per_head=config.ngram_vocab_size_base,
            rows_multiple_of=config.make_ngram_vocab_size_divisible_by,
        )
        self.token_vocab_size = config.vocab_size
        self.eos_token_id = config.eos_token_id

    def table_multipliers(self, table_index: int) -> list[int]:
        max_multiplier = ((1 << 63) - 1) // self.token_vocab_size
        multiplier_bound = max(1, max_multiplier // 2)
        table_seed = NGRAM_HASH_SEED + NGRAM_LAYER_PRIME * table_index
        return [
            2 * (splitmix64(table_seed + SPLITMIX_GAMMA * (index + 1)) % multiplier_bound) + 1
            for index in range(self.max_ngram_size)
        ]

    def pad_token(self) -> int:
        return self.eos_token_id

    def segment_starts(self, positions: torch.Tensor, cu_seqlens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        document_starts = super().segment_starts(positions, cu_seqlens, tokens)
        eos_positions = torch.where(tokens == self.eos_token_id, positions, -1)
        latest_eos = torch.cummax(eos_positions, dim=0).values
        previous_eos = torch.cat((latest_eos.new_full((1,), -1), latest_eos[:-1]))
        return torch.maximum(document_starts, previous_eos + 1)


__all__ = ["Qwen3_8FlashNextNGramHasher"]
