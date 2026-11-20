"""DeepSeek-V4.1 Engram: hashed n-gram lookups written into the residual streams.

Every position is hashed as the `max_ngram_size - 1` n-grams ending at it (2-gram ..
`max_ngram_size`-gram), each over `n_heads` prime-sized buckets, giving one row id per hash column
into one huge table per engram layer (~384M rows of 256 channels). The tables are the model's
shared `NGramEmbeddings`; each engram layer's `DeepseekV41Engram` turns its fetched rows into one
key per hyper-connection stream plus a shared value, and a gate built from how well each stream
matches its key decides how much of the value goes into that stream.
"""

import numpy as np
import torch
from torch import Tensor, nn

from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
from prime_rl.trainer.models.layers.ngram import NGramHasher, consecutive_primes_from

LAYER_SEED_PRIME = 10007


def build_compressed_token_map(tokenizer_path: str) -> tuple[list[int], int]:
    """Map every token id onto the normalized id space the n-grams hash over.

    Tokens that normalize alike (" The", "the", "THE") share one compressed id. Returns the lookup
    and the compressed vocabulary size, which every hash multiplier is derived from.
    """
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # A private-use char, so a token that is exactly one space survives Strip().
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class DeepseekV41NGramHasher(NGramHasher):
    """DeepSeek's engram hash: n-grams over a normalized vocabulary, never looking back past a document start.

    The token map is built from the checkpoint's tokenizer (`tokenizer_path`), and it has to be the one
    the tables were hashed with: the compressed vocabulary size it yields is checked against
    `engram_compressed_vocab_size`.
    """

    def __init__(self, config: DeepseekV41TextConfig, tokenizer_path: str | None):
        super().__init__(
            max_ngram_size=config.engram_max_ngram_size, heads_per_ngram=config.engram_n_heads, rows_multiple_of=1
        )
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.register_buffer("compressed_token_map", torch.zeros(config.vocab_size, dtype=torch.long), persistent=False)
        self.pad_id = 0

    def init_buffers_post_meta(self) -> None:
        if self.tokenizer_path:
            self.set_token_map(*build_compressed_token_map(self.tokenizer_path))
        else:
            # A model built from a bare config has no pretrained tables to match, so every token keeps its id.
            self.set_token_map(list(range(self.config.vocab_size)), compressed_vocab_size=None)

    def set_token_map(self, token_map: list[int], compressed_vocab_size: int | None) -> None:
        if compressed_vocab_size is not None and compressed_vocab_size != self.config.engram_compressed_vocab_size:
            raise ValueError(
                f"the tokenizer compresses to {compressed_vocab_size} ids, but the checkpoint was hashed over "
                f"{self.config.engram_compressed_vocab_size}: every n-gram would land in the wrong bucket"
            )
        self.compressed_token_map.zero_()
        self.compressed_token_map[: len(token_map)].copy_(torch.tensor(token_map))
        self.pad_id = token_map[self.config.engram_pad_token_id]

    def num_table_rows(self, table_index: int) -> int:
        num_rows = self.config.engram_num_embeddings[table_index]
        if sum(self.head_sizes(table_index)) > num_rows:
            raise ValueError(
                f"engram table {table_index} hashes into {sum(self.head_sizes(table_index))} rows, but has {num_rows}"
            )
        return num_rows

    def head_sizes(self, table_index: int) -> list[int]:
        return consecutive_primes_from(self.config.engram_vocab_size, table_index * self.num_heads, self.num_heads)

    def table_multipliers(self, table_index: int) -> list[int]:
        bound = max(1, (np.iinfo(np.int64).max // self.config.engram_compressed_vocab_size) // 2)
        seed = LAYER_SEED_PRIME * self.config.engram_layer_ids[table_index]
        values = np.random.default_rng(seed).integers(low=0, high=bound, size=(self.max_ngram_size,), dtype=np.int64)
        return (values * 2 + 1).tolist()

    def map_tokens(self, tokens: Tensor) -> Tensor:
        return self.compressed_token_map[tokens]

    def pad_token(self) -> int:
        return self.pad_id


@torch.compile
def _engram_gate(
    h: Tensor, kv: Tensor, q_weight: Tensor, k_weight: Tensor, hc_mult: int, eps: float, clamp_value: float
) -> Tensor:
    """`h + gate * value`, the gate a signed-sqrt sigmoid of each stream's normalized match to its key."""
    dim = h.shape[-1]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    h32 = h.float()
    # Normalized per (token, stream) over `dim`, not jointly over the streams.
    rstd = torch.rsqrt(h32.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    # q * k in fp32, as in DeepSeek's reference: rounding the product to bf16 first moves the gate.
    dot = (h32 * (q_weight.float() * k_weight.float()) * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h32 + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


class DeepseekV41Engram(nn.Module):
    """Gates one engram layer's looked-up n-gram rows into the `(b, t, hc_mult, hidden)` residual streams."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = nn.Linear(
            n_hash_cols * config.engram_head_dim, config.hidden_size * (config.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(self, mhc_states: Tensor, rows: Tensor) -> Tensor:
        """`rows` is `(1, t, n_hash_cols * head_dim)`, the row's table rows from `NGramEmbeddings.embed`."""
        kv = self.wkv(rows.to(mhc_states.dtype))
        return _engram_gate(mhc_states, kv, self.q_weight, self.k_weight, self.hc_mult, self.eps, self.clamp_value)

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.wkv.weight, mean=0.0, std=init_std)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)


__all__ = ["DeepseekV41Engram", "DeepseekV41NGramHasher", "build_compressed_token_map"]
