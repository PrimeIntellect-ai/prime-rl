import re
from collections import Counter

import numpy as np
import pytest
import torch
from huggingface_hub.errors import StrictDataclassClassValidationError

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.deepseek_v41 import DeepseekV41Config, DeepseekV41TextConfig
from prime_rl.trainer.models.deepseek_v41.converting_deepseek_v41 import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.deepseek_v41.engram import EngramHasher, build_bucket_primes, build_hash_multipliers

# The published `deepseek-ai/DeepSeek-V4.1-Flash` config.json, vision tower abridged.
PUBLISHED_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 129264,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
    "text_config": {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "tie_word_embeddings": False,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "compress_rope_theta": 160000,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    },
    "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024},
}

# Small enough to enumerate every n-gram by hand: two engram layers, 2- and 3-grams, two heads each.
HASH_CONFIG = dict(
    num_hidden_layers=6,
    compress_ratios=[0, 2, 2, 1, 1, 1],
    kv_source_layer_ids=[1, 3],
    index_source_layer_ids=[1, 3, 5],
    candidate_source_layer_id=3,
    vocab_size=64,
    engram_layer_ids=[1, 4],
    engram_num_embeddings=[240, 306],
    engram_max_ngram_size=3,
    engram_vocab_size=50,
    engram_n_heads=2,
    engram_compressed_vocab_size=16,
    engram_pad_token_id=2,
)


def test_deepseek_v41_config_parses_the_published_config():
    config = DeepseekV41Config(**PUBLISHED_CONFIG)
    text = config.text_config

    assert isinstance(text, DeepseekV41TextConfig)
    assert config.get_text_config() is text
    assert not config.tie_word_embeddings
    assert (config.pad_token_id, config.eos_token_id) == (2, 1)
    assert not hasattr(config, "vision_config")
    assert text.qk_rope_head_dim == 64
    assert text.rope_parameters["main"] == {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.125}
    compress = text.rope_parameters["compress"]
    assert (compress["rope_type"], compress["rope_theta"], compress["factor"]) == ("yarn", 160000, 16)
    assert compress["original_max_position_embeddings"] == 65536
    # The DSpark draft layers' entries ride along but are past `num_hidden_layers`.
    assert len(text.compress_ratios) == 43
    assert text.compress_rates == {1, 2}
    assert text.active_engram_layer_ids == [1, 14]
    # The fixed attributes V4's MoE reads.
    assert (text.num_hash_layers, text.mlp_bias) == (0, False)


def test_deepseek_v41_shared_sources_follow_the_schedule():
    text = DeepseekV41Config(**PUBLISHED_CONFIG).text_config

    assert [text.kv_source_of(layer) for layer in (0, 1)] == [None, None]
    assert [text.kv_source_of(layer) for layer in (2, 7, 8, 19)] == [2, 2, 8, 14]
    # Layer 20 starts the ratio-1 layers, so they never read the ratio-2 entries before it.
    assert [text.kv_source_of(layer) for layer in (20, 39)] == [20, 20]
    assert [text.index_source_of(layer) for layer in (2, 13, 14, 20, 23, 24, 39)] == [2, 8, 14, 20, 20, 24, 36]


def test_deepseek_v41_config_rejects_an_unsourced_compressed_layer():
    with pytest.raises(StrictDataclassClassValidationError, match="no KV source layer"):
        DeepseekV41TextConfig(**{**HASH_CONFIG, "kv_source_layer_ids": [3]})
    with pytest.raises(StrictDataclassClassValidationError, match="does not compress"):
        DeepseekV41TextConfig(**{**HASH_CONFIG, "index_source_layer_ids": [0, 1, 3]})


def _published_keys(text: DeepseekV41TextConfig) -> list[str]:
    """Every bf16 key of the published checkpoint for `text`'s layers, plus keys the trainer drops."""
    keys = ["embed.weight", "head.weight", "norm.weight", "mtp.0.hc_attn_base", "vision.blocks.0.w", "image_start"]
    for layer in range(text.num_hidden_layers):
        p = f"layers.{layer}"
        keys += [f"{p}.{name}" for name in ("attn_norm.weight", "ffn_norm.weight")]
        keys += [f"{p}.hc_{sub}_{part}" for sub in ("attn", "ffn") for part in ("fn", "base", "scale")]
        keys += [
            f"{p}.attn.{name}"
            for name in ("wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight")
            + ("wo_a.weight", "wo_b.weight", "attn_sink")
        ]
        keys += [f"{p}.ffn.gate.{name}" for name in ("weight", "bias", "bias_vl")]
        keys += [f"{p}.ffn.shared_experts.w{i}.weight" for i in (1, 2, 3)]
        keys += [f"{p}.ffn.experts.{e}.w{i}.weight" for e in range(text.n_routed_experts) for i in (1, 2, 3)]
        if layer in text.kv_source_layer_ids:
            keys += [f"{p}.attn.compressor.{name}.weight" for name in ("wkv", "norm")]
            if text.compress_ratios[layer] == 2:
                keys.append(f"{p}.attn.compressor.wgate.weight")
            if layer in text.index_source_layer_ids:
                keys += [f"{p}.attn.indexer.{name}.weight" for name in ("wk", "k_norm")]
        if layer in text.index_source_layer_ids:
            keys += [f"{p}.attn.indexer.{name}.weight" for name in ("wq_b", "weights_proj")]
        if layer in text.engram_layer_ids:
            keys += [f"{p}.engram.{name}" for name in ("embed.weight", "wkv.weight", "q_weight", "k_weight")]
    return keys


def test_deepseek_v41_conversion_round_trips_the_published_names():
    """Every published key a layer carries gets a prime-rl name; only the untrained parts are dropped."""
    config = DeepseekV41Config(
        **{**PUBLISHED_CONFIG, "text_config": {**PUBLISHED_CONFIG["text_config"], "n_routed_experts": 4}}
    )
    text = config.text_config
    hf = {key: torch.zeros(1) for key in _published_keys(text)}
    # The chain stacks the routed experts, which needs real shapes.
    for key in hf:
        if ".ffn.experts." in key:
            hf[key] = torch.full((3, 2), float(key.split(".")[4]))
    assert is_hf_state_dict(hf) and not is_prime_state_dict(hf)

    prime = apply_hf_to_prime(dict(hf), conversion_chain(config))
    assert is_prime_state_dict(prime) and not is_hf_state_dict(prime)
    assert all(key.startswith(("model.", "lm_head.")) for key in prime), sorted(prime)[:5]
    patterns = Counter(re.sub(r"\.\d+\.", ".N.", key) for key in prime)
    assert patterns["model.engrams.N.embed.weight"] == 2
    assert patterns["model.layers.N.self_attn.compressor.gate_proj.weight"] == 3
    assert patterns["model.layers.N.self_attn.indexer.k_proj.weight"] == 4
    assert patterns["model.layers.N.self_attn.indexer.q_b_proj.weight"] == 8
    assert patterns["model.layers.N.mlp.experts.gate_proj"] == 40
    assert not any("bias_vl" in key or "mtp" in key or "vision" in key or "image" in key for key in prime)

    back = apply_prime_to_hf(prime, conversion_chain(config))
    dropped = {key for key in hf if key.startswith(("mtp.", "vision.", "image_")) or key.endswith("bias_vl")}
    assert set(back) == set(hf) - dropped
    for key in back:
        assert torch.equal(back[key], hf[key]), key


def test_engram_bucket_primes_and_multipliers_are_deterministic():
    config = DeepseekV41TextConfig(**HASH_CONFIG)

    primes = build_bucket_primes(config)
    assert primes == build_bucket_primes(config)
    # Drawn in order above `engram_vocab_size` and never reused, so every bucket range is disjoint.
    assert primes == [[[53, 59], [61, 67]], [[71, 73], [79, 83]]]
    # Each table is exactly as long as its buckets.
    assert [sum(p for ngram in layer for p in ngram) for layer in primes] == config.engram_num_embeddings

    multipliers = build_hash_multipliers(config)
    np.testing.assert_array_equal(multipliers, build_hash_multipliers(config))
    assert multipliers.shape == (2, 3) and (multipliers % 2 == 1).all()
    # Seeded per engram layer id, as DeepSeek's reference does.
    bound = (np.iinfo(np.int64).max // config.engram_compressed_vocab_size) // 2
    expected = np.random.default_rng(10007 * 4).integers(0, bound, size=(3,), dtype=np.int64) * 2 + 1
    np.testing.assert_array_equal(multipliers[1], expected)


def _reference_hash_ids(hasher: EngramHasher, tokens: list[int], doc_lens: list[int]) -> torch.Tensor:
    """One token at a time: look back within the document, pad before its start, XOR the products."""
    config = hasher.config
    primes, offsets, multipliers = hasher.primes.tolist(), hasher.offsets.tolist(), hasher.multipliers.tolist()
    starts = [start for start, length in zip(np.cumsum([0, *doc_lens[:-1]]), doc_lens) for _ in range(length)]
    rows = []
    for t in range(len(tokens)):
        window = [
            hasher.token_map[tokens[t - s]].item() if t - s >= starts[t] else hasher.pad_id
            for s in range(config.engram_max_ngram_size)
        ]
        per_layer = []
        for layer in range(len(config.engram_layer_ids)):
            rolling, ids = window[0] * multipliers[layer][0], []
            for n in range(1, config.engram_max_ngram_size):
                rolling ^= window[n] * multipliers[layer][n]
                ids += [rolling % prime for prime in primes[layer][n - 1]]
            per_layer.append([i + o for i, o in zip(ids, offsets[layer])])
        rows.append(per_layer)
    return torch.tensor(rows)


def test_engram_hasher_matches_a_per_token_reference_and_stops_at_documents():
    config = DeepseekV41TextConfig(**HASH_CONFIG)
    hasher = EngramHasher(config, tokenizer_path=None)
    token_map = [token % 16 for token in range(64)]
    hasher.set_token_map(token_map, compressed_vocab_size=16)

    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 64, (1, 12), generator=generator)
    doc_lens = [5, 7]
    tok_doc_start = torch.tensor([0] * 5 + [5] * 7)
    hash_ids = hasher(tokens, tok_doc_start)

    assert hash_ids.shape == (12, 2, 4)
    assert torch.equal(hash_ids, hasher(tokens, tok_doc_start))
    assert torch.equal(hash_ids, _reference_hash_ids(hasher, tokens[0].tolist(), doc_lens))
    for layer, rows in enumerate(config.engram_num_embeddings):
        assert int(hash_ids[:, layer].max()) < rows
    # A document's first token hashes as if the sequence started there, whatever precedes it.
    alone = hasher(tokens[:, 5:], torch.zeros(7, dtype=torch.long))
    assert torch.equal(hash_ids[5:], alone)


def test_engram_hasher_rejects_a_mismatched_tokenizer():
    hasher = EngramHasher(DeepseekV41TextConfig(**HASH_CONFIG), tokenizer_path=None)
    with pytest.raises(ValueError, match="compresses to 17 ids"):
        hasher.set_token_map(list(range(17)), compressed_vocab_size=17)
