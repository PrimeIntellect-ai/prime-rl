"""DeepSeek-V4.1 Engram: hashed n-gram lookups written into the residual stream.

Every position is hashed as the `max_ngram_size - 1` n-grams ending at it (2-gram ..
`max_ngram_size`-gram), each over `n_heads` prime-sized buckets, giving `n_hash_cols` row ids into
one huge table per engram layer (~384M rows of 256 channels). The fetched rows become one key per
hyper-connection stream plus a shared value, and a gate built from how well each stream matches
its key decides how much of the value goes into that stream.

The tables are far too large for FSDP, which all-gathers a whole parameter to compute with it.
The trainer row-shards each table over the FSDP mesh with `AllToAllEmbeddingParallel`, which serves
a lookup by sending the row ids to the ranks owning them and receiving only those rows.
"""

import numpy as np
import torch
from torch import Tensor, nn

from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_embedding import is_prime

LAYER_SEED_PRIME = 10007


def _next_unseen_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_bucket_primes(config: DeepseekV41TextConfig) -> list[list[list[int]]]:
    """`[engram layer][n-gram size][head]` bucket moduli: primes above `engram_vocab_size`, never reused."""
    primes, seen = [], set()
    for _ in config.engram_layer_ids:
        per_ngram = []
        for _ in range(config.engram_max_ngram_size - 1):
            sizes, current = [], config.engram_vocab_size - 1
            for _ in range(config.engram_n_heads):
                current = _next_unseen_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def build_hash_multipliers(config: DeepseekV41TextConfig) -> np.ndarray:
    """One odd multiplier per (engram layer, lookback), bounded so `token * multiplier` fits in int64."""
    bound = max(1, (np.iinfo(np.int64).max // config.engram_compressed_vocab_size) // 2)
    rows = []
    for layer_idx in config.engram_layer_ids:
        values = np.random.default_rng(LAYER_SEED_PRIME * layer_idx).integers(
            low=0, high=bound, size=(config.engram_max_ngram_size,), dtype=np.int64
        )
        rows.append(values * 2 + 1)
    return np.stack(rows)


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


class EngramHasher(nn.Module):
    """Maps every token of a packed row to its row ids in each engram layer's table.

    Look-back stops at the start of the token's document: the missing history is filled with the
    compressed pad token, as at the start of a sequence in the reference.

    The token map is built from the checkpoint's tokenizer (`tokenizer_path`), and it has to be
    the one the tables were hashed with: the compressed vocabulary size it yields is checked
    against `engram_compressed_vocab_size`.
    """

    def __init__(self, config: DeepseekV41TextConfig, tokenizer_path: str | None):
        super().__init__()
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.max_ngram_size = config.engram_max_ngram_size
        n_layers = len(config.engram_layer_ids)
        n_cols = (self.max_ngram_size - 1) * config.engram_n_heads
        self.register_buffer("token_map", torch.zeros(config.vocab_size, dtype=torch.long), persistent=False)
        self.register_buffer(
            "primes",
            torch.zeros(n_layers, self.max_ngram_size - 1, config.engram_n_heads, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("offsets", torch.zeros(n_layers, n_cols, dtype=torch.long), persistent=False)
        self.register_buffer(
            "multipliers", torch.zeros(n_layers, self.max_ngram_size, dtype=torch.long), persistent=False
        )
        self.pad_id = 0

    def init_buffers_post_meta(self) -> None:
        if not self.tokenizer_path:
            raise ValueError("the engram hash needs the checkpoint's tokenizer, but the config has no name_or_path")
        token_map, compressed_vocab_size = build_compressed_token_map(self.tokenizer_path)
        self.set_token_map(token_map, compressed_vocab_size)

    def set_token_map(self, token_map: list[int], compressed_vocab_size: int) -> None:
        """Fill every buffer from `token_map`, the compressed id of each token id."""
        config = self.config
        if compressed_vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"the tokenizer compresses to {compressed_vocab_size} ids, but the checkpoint was hashed over "
                f"{config.engram_compressed_vocab_size}: every n-gram would land in the wrong bucket"
            )
        primes = build_bucket_primes(config)
        bucket_sizes = [[p for ngram in layer for p in ngram] for layer in primes]
        for layer_idx, sizes, num_embeddings in zip(
            config.engram_layer_ids, bucket_sizes, config.engram_num_embeddings, strict=True
        ):
            if sum(sizes) > num_embeddings:
                raise ValueError(
                    f"engram layer {layer_idx} hashes into {sum(sizes)} rows, but its table has {num_embeddings}"
                )
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in bucket_sizes]
        self.token_map.zero_()
        self.token_map[: len(token_map)].copy_(torch.tensor(token_map))
        self.primes.copy_(torch.tensor(primes))
        self.offsets.copy_(torch.tensor(np.stack(offsets)))
        self.multipliers.copy_(torch.from_numpy(build_hash_multipliers(config)))
        self.pad_id = token_map[config.engram_pad_token_id]

    @torch.no_grad()
    def forward(self, input_ids: Tensor, tok_doc_start: Tensor) -> Tensor:
        """`input_ids` is the `(1, t)` packed row, `tok_doc_start` each token's document start.
        Returns `(t, n_engram_layers, n_hash_cols)` int64 row ids."""
        compressed = self.token_map[input_ids[0]]
        positions = torch.arange(compressed.shape[0], device=compressed.device)

        tokens = []
        for shift in range(self.max_ngram_size):
            source = positions - shift
            tokens.append(torch.where(source >= tok_doc_start, compressed[source.clamp_min(0)], self.pad_id))
        tokens = torch.stack(tokens, dim=-1)  # (t, max_ngram_size)

        # XOR the multiplied ids together one lookback at a time: after step i the running value is
        # the hash of the (i + 1)-gram, which lands in its own prime-sized bucket range per head.
        products = tokens[:, None, :] * self.multipliers  # (t, n_layers, max_ngram_size)
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


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
    """Adds one engram layer's n-gram lookup into the `(b, t, hc_mult, hidden)` residual streams.

    `embed` is a plain `nn.Embedding`; the trainer row-shards it (see the module docstring).
    """

    def __init__(self, config: DeepseekV41TextConfig, engram_idx: int):
        super().__init__()
        self.engram_idx = engram_idx
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.embed = nn.Embedding(config.engram_num_embeddings[engram_idx], config.engram_head_dim)
        self.wkv = nn.Linear(
            n_hash_cols * config.engram_head_dim, config.hidden_size * (config.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(self, mhc_states: Tensor, hash_ids: Tensor) -> Tensor:
        """`hash_ids` is `(t, n_hash_cols)` for the row's `t` tokens."""
        rows = self.embed(hash_ids).to(mhc_states.dtype).view(*mhc_states.shape[:2], -1)
        # Recomputed in backward, as the decoder layers are under activation checkpointing. Besides the
        # memory, this keeps the compiled gate's saved views of `kv` away from activation offloading,
        # which would hand them back contiguous and break the strides the compiled backward expects.
        return torch.utils.checkpoint.checkpoint(self._mix, mhc_states, rows, use_reentrant=False)

    def _mix(self, mhc_states: Tensor, rows: Tensor) -> Tensor:
        kv = self.wkv(rows)
        return _engram_gate(mhc_states, kv, self.q_weight, self.k_weight, self.hc_mult, self.eps, self.clamp_value)

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.wkv.weight, mean=0.0, std=init_std)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)


__all__ = ["DeepseekV41Engram", "EngramHasher", "build_bucket_primes", "build_hash_multipliers"]
