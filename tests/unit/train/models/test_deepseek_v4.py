"""DeepSeek V4 checks that need a GPU.

There is no HF oracle here. `transformers.models.deepseek_v4` only exists from transformers 5.15
and the repo pins an older version, so every assertion is either self-consistency (packed against
unpacked), a closed form written out by hand, or vLLM's own loader. That rules out the parity
archetype the other models in this directory use, where a tiny `HF<X>ForCausalLM` supplies the
expected logits and gradients.
"""

import math
import re
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM, eager_reference
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers import norms
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

# Deliberately heterogeneous: one layer of every attention type, hash-routed bootstrap
# layers ahead of standard MoE ones, and a sliding window narrow enough that the compressed
# branches are what carries any long-range signal.
_MODEL = dict(
    vocab_size=64,
    hidden_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    q_lora_rank=64,
    partial_rotary_factor=0.5,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    max_position_embeddings=256,
    sliding_window=6,
    o_groups=2,
    o_lora_rank=16,
    layer_types=[
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "sliding_attention",
    ],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    index_n_heads=4,
    index_head_dim=24,
    # Smaller than the number of compressed entries the sequence yields, so the Lightning
    # Indexer's selection has to actually discard some of them.
    index_topk=2,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=2,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
)

_MODEL_BATCH, _MODEL_SEQ = 2, 32
_MODULE_BATCH = 2

_CSA_LAYER, _HCA_LAYER = 1, 2
_COMPRESS_RATE = _MODEL["compress_rates"]["compressed_sparse_attention"]
_HCA_COMPRESS_RATE = _MODEL["compress_rates"]["heavily_compressed_attention"]


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


@pytest.fixture
def _torch_rms_norm(monkeypatch):
    """Make the shared `RMSNorm` take its PyTorch path instead of the quack kernel.

    The kernel is a project-wide choice that predates this model and drifts from a fp32
    reference by up to ~1e-2 in bf16, which would swamp what the V4-specific math contributes.
    """
    monkeypatch.setattr(norms, "_get_quack_rmsnorm", lambda: None)


def _tid2eid(vocab_size: int, num_experts: int, top_k: int) -> torch.Tensor:
    """A frozen token id -> expert ids table, distinct experts per row as a real one has."""
    rows = [torch.randperm(num_experts)[:top_k] for _ in range(vocab_size)]
    return torch.stack(rows).to(device="cuda", dtype=torch.long)


def _randomize(module: nn.Module) -> None:
    """Draw non-degenerate values for every parameter and routing buffer.

    These modules allocate with `torch.empty`, and the values `init_weights` would write are
    themselves degenerate for testing: norm gains default to ones and the sinks, position biases,
    load-balancing bias and hash table all default to zeros, each of which leaves the path it
    controls indistinguishable from a no-op. The position bias is drawn wide because it is a
    softmax logit over a pooling window; at the projections' std the gate would stay near uniform.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("scale"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            elif name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("sinks") or name.endswith("position_bias"):
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)

    with torch.no_grad():
        for name, buffer in module.named_buffers():
            # HF and prime-rl both hang the aux-loss-free load-balancing bias off the router,
            # under different names. Draw whichever this module carries.
            if name.endswith("e_score_correction_bias") or name.endswith("selection_bias"):
                buffer.normal_(mean=0.0, std=0.1)
            elif name.endswith("tid2eid"):
                router = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
                buffer.copy_(_tid2eid(buffer.shape[0], router.num_experts, router.top_k))


def _prime_config(attn_impl: str = "eager") -> DeepseekV4Config:
    """The toy config. It defaults to eager because the kernel cannot tile 4 attention heads."""
    return DeepseekV4Config(**_MODEL, _attn_impl=attn_impl)


def get_prime_model(dtype: torch.dtype = torch.bfloat16, attn_impl: str = "eager") -> nn.Module:
    """A prime-rl model with non-degenerate weights and the LM head training code wraps it in."""
    with torch.device("cuda"), default_dtype(dtype):
        model = DeepseekV4ForCausalLM._from_config(_prime_config(attn_impl))
    _randomize(model)
    inject_prime_lm_head(model, chunk_size=None)
    return model


def prime_attention(layer_idx: int, dtype: torch.dtype = torch.bfloat16, attn_impl: str = "eager") -> nn.Module:
    """One attention layer of the same config the whole-model tests use.

    `DeepseekV4Attention` reads no MoE or hyper-connection field, so the layer this builds is
    bit-identical to one from a config carrying only the attention keys.
    """
    with torch.device("cuda"), default_dtype(dtype):
        module = DeepseekV4Attention(_prime_config(attn_impl), layer_idx=layer_idx)
    _randomize(module)
    return module


def _single_doc(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`position_ids` and `seq_lens` for a row holding one document per batch entry."""
    batch, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
    return position_ids, torch.tensor([seq_len], device=input_ids.device)


def _assert_relative(prime: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    prime, reference = prime.float(), reference.float()
    deviation = (prime - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _packed_context(
    doc_lens: tuple[int, ...], dtype: torch.dtype, config: DeepseekV4Config | None = None
) -> PackedContext:
    """The context `DeepseekV4Model` would hand its attention layers for a row of `doc_lens`.

    A single-element `doc_lens` gives back the single-document context, which is what the unpacked
    half of a packing comparison runs at. `dtype` types the mask and the rotary tables, and has to
    be the one the caller runs at. `config` defaults to the toy `_MODEL`; the Flash-shaped tests
    at the bottom of this file pass their own.
    """
    config = config if config is not None else _prime_config()
    with torch.device("cuda"), default_dtype(dtype):
        rotary = DeepseekV4RotaryEmbedding(config)
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(doc_lens, device="cuda"),
        dtype=dtype,
        device=torch.device("cuda"),
    )


def test_deepseek_v4_hash_layers_route_on_token_ids():
    """The bootstrap layers read `input_ids`, so identical hidden states still route apart."""
    prime_model = get_prime_model()
    hash_layers = prime_model.model.layers[: _MODEL["num_hash_layers"]]
    assert hash_layers, "config must contain a hash-routed layer"

    counts = []
    for token_id in (0, 1):
        input_ids = torch.full((_MODEL_BATCH, _MODEL_SEQ), token_id, device="cuda", dtype=torch.long)
        for layer in hash_layers:
            layer.mlp.tokens_per_expert.zero_()
        position_ids, seq_lens = _single_doc(input_ids)
        prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)
        counts.append(torch.stack([layer.mlp.tokens_per_expert.clone() for layer in hash_layers]))

    table = hash_layers[0].mlp.router.tid2eid
    assert set(table[0].tolist()) != set(table[1].tolist()), "the two table rows must differ for this to bite"
    assert not torch.equal(counts[0], counts[1]), "a hash layer must route the two token ids to different experts"
    expected = torch.zeros_like(counts[0][0])
    expected[table[0]] = _MODEL_BATCH * _MODEL_SEQ
    torch.testing.assert_close(counts[0][0], expected)


def test_deepseek_v4_backward():
    """Every parameter that can train does, and the Lightning Indexer's still cannot."""
    prime_config = _prime_config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = DeepseekV4ForCausalLM(prime_config)
    _randomize(model)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, _MODEL["vocab_size"], (_MODEL_BATCH, _MODEL_SEQ), device="cuda")
    position_ids, seq_lens = _single_doc(input_ids)
    output = model(input_ids, position_ids=position_ids, seq_lens=seq_lens)
    output["logits"].sum().backward()

    dead, unexpectedly_alive = [], []
    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue
        has_grad = param.grad is not None and param.grad.norm().item() > 0
        # The indexer reaches the loss only through integer top-k indices, so nothing
        # differentiates back into it. DeepSeek trains it with a separate auxiliary loss
        # that prime-rl does not implement.
        if ".indexer." in name:
            if has_grad:
                unexpectedly_alive.append(name)
        elif not has_grad:
            dead.append(name)

    assert not dead, f"Parameters with zero/no gradients: {dead}"
    assert not unexpectedly_alive, f"Lightning Indexer parameters received a gradient: {unexpectedly_alive}"


def test_deepseek_v4_weight_conversion_roundtrip():
    prime_config = _prime_config()
    model = DeepseekV4ForCausalLM(prime_config).to("cuda")
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    state_dict = model.state_dict()
    model.convert_to_hf(state_dict)
    assert DeepseekV4ForCausalLM.is_hf_state_dict(state_dict)
    assert not DeepseekV4ForCausalLM.is_prime_state_dict(state_dict)
    model.convert_to_prime(state_dict)
    assert DeepseekV4ForCausalLM.is_prime_state_dict(state_dict)

    assert set(state_dict) == set(original)
    for name, tensor in original.items():
        assert torch.equal(state_dict[name], tensor), f"Value mismatch for {name}"


def _fill_hash_tables(model: nn.Module) -> dict[int, torch.Tensor]:
    """Give every bootstrap layer its own non-degenerate table, and hand them back by layer."""
    tables = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if not layer.mlp.is_hash:
            continue
        table = _tid2eid(_MODEL["vocab_size"], _MODEL["n_routed_experts"], _MODEL["num_experts_per_tok"])
        with torch.no_grad():
            layer.mlp.router.tid2eid.copy_(table)
        tables[layer_idx] = table
    assert len(tables) == _MODEL["num_hash_layers"], "config must contain hash-routed layers"
    return tables


def test_deepseek_v4_hash_table_survives_the_load_path(tmp_path, monkeypatch):
    """The frozen table has to reach `mlp.router.tid2eid` from the name a real checkpoint ships.

    It is the one buffer no `init_weights` can reconstruct: an all-zero table is a valid tensor
    that routes every token to expert 0, so a rename anywhere along the loading path degrades the
    model in silence rather than raising. `test_deepseek_v4_weight_conversion_roundtrip` only ever
    roundtrips the zeros the constructor leaves behind, and `_VLLM_MAPPED_NAMES` pins the on-disk
    name without saying where it lands, so this carries real values across the whole path.

    Three things in sequence, because they are three links in one chain. The conversion emits only
    the bootstrap layers' tables, under the on-disk name; reversing it lands them back on
    `mlp.router.tid2eid`; and nothing between `dcp_load` and the end of loading resets what the
    checkpoint supplied. `init_buffers_post_meta` walks every MoE and runs just before the load,
    so getting it to reset one buffer too many is an easy mistake with no symptom other than every
    bootstrap token routing to expert 0.
    """
    prime_config = _prime_config()
    model = DeepseekV4ForCausalLM(prime_config).to("cuda")
    tables = _fill_hash_tables(model)

    state_dict = model.convert_to_hf(dict(model.state_dict()))

    assert {key for key in state_dict if key.endswith("tid2eid")} == {
        f"layers.{layer_idx}.ffn.gate.tid2eid" for layer_idx in tables
    }, "only the bootstrap layers carry a table, under the name the real checkpoint ships"
    for layer_idx, table in tables.items():
        assert torch.equal(state_dict[f"layers.{layer_idx}.ffn.gate.tid2eid"], table)

    model.convert_to_prime(state_dict)
    reloaded = DeepseekV4ForCausalLM(prime_config).to("cuda")
    reloaded.load_state_dict(state_dict)
    for layer_idx, table in tables.items():
        assert torch.equal(reloaded.model.layers[layer_idx].mlp.router.tid2eid, table)

    # And now the loading path itself, on a meta-device model, with the name `load_dcp_from_hf`
    # asks the checkpoint for raising a `KeyError` in the stub below if the buffer ever moves.
    with torch.device("meta"):
        meta_model = DeepseekV4ForCausalLM(prime_config)
    expected = tables[0]

    def fake_dcp_load(state_dict, storage_reader=None):
        buffer = state_dict["model.layers.0.mlp.router.tid2eid"]
        buffer.copy_(expected.to(device=buffer.device))

    monkeypatch.setattr("prime_rl.trainer.model.dcp_load", fake_dcp_load)
    monkeypatch.setattr("prime_rl.trainer.model.load_state_dict_keys", lambda path: meta_model.state_dict().keys())
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)

    load_dcp_from_hf(meta_model, ModelConfig(name=str(tmp_path)), parallel_dims=MagicMock())

    torch.testing.assert_close(meta_model.model.layers[0].mlp.router.tid2eid, expected)


# What vLLM's DeepSeek V4 loader sees, with the layer and expert indices folded away. Written
# out rather than derived: this file has no independent implementation to derive it from, and a
# rename on either side of the boundary is exactly what it exists to catch.
_VLLM_MAPPED_NAMES = {
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.hc_head_base",
    "model.hc_head_fn",
    "model.hc_head_scale",
    "model.layers.{i}.attn.attn_sink",
    "model.layers.{i}.attn.compressor.ape",
    "model.layers.{i}.attn.compressor.norm.weight",
    "model.layers.{i}.attn.compressor.wgate.weight",
    "model.layers.{i}.attn.compressor.wkv.weight",
    "model.layers.{i}.attn.indexer.compressor.ape",
    "model.layers.{i}.attn.indexer.compressor.norm.weight",
    "model.layers.{i}.attn.indexer.compressor.wgate.weight",
    "model.layers.{i}.attn.indexer.compressor.wkv.weight",
    "model.layers.{i}.attn.indexer.weights_proj.weight",
    "model.layers.{i}.attn.indexer.wq_b.weight",
    "model.layers.{i}.attn.kv_norm.weight",
    "model.layers.{i}.attn.q_norm.weight",
    "model.layers.{i}.attn.wkv.weight",
    "model.layers.{i}.attn.wo_a.weight",
    "model.layers.{i}.attn.wo_b.weight",
    "model.layers.{i}.attn.wq_a.weight",
    "model.layers.{i}.attn.wq_b.weight",
    "model.layers.{i}.attn_norm.weight",
    "model.layers.{i}.ffn.experts.{i}.w1.weight",
    "model.layers.{i}.ffn.experts.{i}.w2.weight",
    "model.layers.{i}.ffn.experts.{i}.w3.weight",
    "model.layers.{i}.ffn.gate.e_score_correction_bias",
    "model.layers.{i}.ffn.gate.tid2eid",
    "model.layers.{i}.ffn.gate.weight",
    "model.layers.{i}.ffn.shared_experts.down_proj.weight",
    "model.layers.{i}.ffn.shared_experts.w1.weight",
    "model.layers.{i}.ffn.shared_experts.w3.weight",
    "model.layers.{i}.ffn_norm.weight",
    "model.layers.{i}.hc_attn_base",
    "model.layers.{i}.hc_attn_fn",
    "model.layers.{i}.hc_attn_scale",
    "model.layers.{i}.hc_ffn_base",
    "model.layers.{i}.hc_ffn_fn",
    "model.layers.{i}.hc_ffn_scale",
    "model.norm.weight",
}


def test_deepseek_v4_on_disk_keys_map_to_the_names_vllm_expects():
    """Pin the names prime-rl's checkpoint arrives under on vLLM's side of the boundary.

    `convert_to_hf` emits the compact DeepSeek-native naming a real checkpoint ships (the
    conversion chain runs on-disk -> prime, so reversing it lands back on the on-disk names).
    That is what `utils/weights.py` broadcasts during a run and what `scripts/mini_moe.py`
    writes, and `_make_deepseek_v4_weights_mapper` is what vLLM puts it through on the way in.

    Asserted against a written-out set because the mapper cannot serve as its own oracle: it
    returns unrecognized keys unchanged rather than `None`, so "every key maps to something" is
    true for any input whatsoever, including a misspelling. Pinning the mapped names instead
    fails on a rename on either side: prime-rl renaming a weight, or vLLM's mapper moving under
    a bump. It does not prove vLLM's loader has a parameter of that name, which would take
    instantiating vLLM's model; only a real serving run covers that end to end.

    `_make_deepseek_v4_weights_mapper` is private API in a URL-pinned wheel (`vllm==0.28.0`).
    Imported inside the test because importing vLLM at module scope would run during collection,
    including in the CPU CI job that deselects this file.
    """
    from vllm.models.deepseek_v4.nvidia.model import _make_deepseek_v4_weights_mapper

    with torch.device("meta"):
        model = DeepseekV4ForCausalLM._from_config(_prime_config())
    on_disk_state_dict = model.convert_to_hf(dict(model.state_dict()))
    assert on_disk_state_dict, "vacuous probe: the model produced no weights to map"

    # Both expert dtypes, since the mapper is built per dtype and only one of them ships fp4.
    for expert_dtype in ("fp8", "fp4"):
        mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
        mapped = {re.sub(r"\.\d+\.", ".{i}.", mapper._map_name(key)) for key in on_disk_state_dict}
        assert mapped == _VLLM_MAPPED_NAMES, (
            f"{expert_dtype}: unexpected {sorted(mapped - _VLLM_MAPPED_NAMES)}, "
            f"missing {sorted(_VLLM_MAPPED_NAMES - mapped)}"
        )


def test_deepseek_v4_init_buffers_post_meta_restores_every_rotary():
    """Rotary tables are non-persistent and computed eagerly, so meta loading loses them."""
    prime_config = _prime_config()
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(prime_config)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    reference = 1.0 / (prime_config.rope_theta ** (torch.arange(0, 16, 2, device="cuda", dtype=torch.float) / 16))
    torch.testing.assert_close(model.model.rotary_emb.main_inv_freq, reference)
    compressors = [layer.self_attn.compressor for layer in model.model.layers if layer.self_attn.compressor]
    assert compressors, "config must contain a compressed attention layer"
    for compressor in compressors:
        assert torch.isfinite(compressor.rotary_emb.compress_inv_freq).all()
        # The compress branch runs a different base, so it must not collapse onto `main`.
        assert not torch.equal(compressor.rotary_emb.compress_inv_freq, reference)


# DeepSeek ships its own inference code inside the checkpoint, at
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference. Its
# `model.py:481-485` is what decides which frequencies each layer gets:
#
#     if self.compress_ratio:
#         original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
#     else:
#         # disable YaRN and use base rope_theta in pure sliding-window attention
#         original_seq_len, rope_theta = 0, args.rope_theta
#
# `precompute_freqs_cis` (`model.py:206-235`) applies the NTK-by-parts ramp only when
# `original_seq_len > 0`, so a pure sliding-window layer gets plain RoPE at `rope_theta` and every
# compressed layer gets YaRN at `compress_rope_theta`. vLLM's `build_deepseek_v4_rope` branches the
# base but not the scaling, which `monkey_patch_deepseek_v4_per_layer_rope` corrects.
#
# The real checkpoint's beta range and factor, but a reduced `original_max_position_embeddings`:
# the cos/sin cache is `original_max * factor` rows of fp32, which at the checkpoint's 65536 would
# allocate 268 MB per rope. 4096 still places the correction range at channels 10 to 23, well
# inside the 32 channel pairs, so the ramp is exercised.
_ROPE_FACTOR = 16
_ROPE_ORIGINAL_MAX_POSITION = 4096
_ROPE_BETA_FAST, _ROPE_BETA_SLOW = 32, 1
_ROPE_THETA, _COMPRESS_ROPE_THETA = 10000.0, 160000.0
_ROPE_HEAD_DIM, _ROPE_ROTARY_DIM = 512, 64
_ROPE_MAX_POSITION = _ROPE_ORIGINAL_MAX_POSITION * _ROPE_FACTOR

_ROPE_SCALING = {
    "type": "yarn",
    "factor": _ROPE_FACTOR,
    "beta_fast": _ROPE_BETA_FAST,
    "beta_slow": _ROPE_BETA_SLOW,
    "original_max_position_embeddings": _ROPE_ORIGINAL_MAX_POSITION,
}

# The nested `main`/`compress` schema, which HF's own `DeepseekV4Config` and this repo's port both
# write and which vLLM's config shim cannot read. A config.json can also carry the flat legacy
# `rope_scaling` the real checkpoint ships, or no YaRN parameters at all; the nested scaled form
# is asserted here because it is the one the patch has the most normalization to do on.
_ROPE_NESTED_PLAIN = {"rope_type": "default", "partial_rotary_factor": 0.125}
_ROPE_PARAMETERS = {
    "main": dict(_ROPE_NESTED_PLAIN),
    "compress": {**_ROPE_SCALING, "partial_rotary_factor": 0.125},
}


def _reference_rope_freqs(original_seq_len: int, base: float) -> torch.Tensor:
    """Verbatim from the checkpoint's `inference/model.py:206-231`, minus the unused `t` outer.

    Only the frequency vector is kept; the reference then does `torch.polar(ones, outer(t, freqs))`,
    whose magnitude is 1, i.e. no mscale on cos/sin.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    dim = _ROPE_ROTARY_DIM
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(_ROPE_BETA_FAST, _ROPE_BETA_SLOW, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / _ROPE_FACTOR * (1 - smooth) + freqs * smooth
    return freqs


def _vllm_rope_freqs(rotary_emb) -> torch.Tensor:
    """Recover inv_freq from row 1 of vLLM's `[position, rotary_dim]` cos-then-sin cache."""
    cos, sin = rotary_emb.cos_sin_cache[1].chunk(2)
    return torch.atan2(sin, cos)


@pytest.fixture
def vllm_rope_builder():
    """`build_deepseek_v4_rope`, patched, under the live vLLM config its `CustomOp` base asserts on.

    The builder is resolved off the module at call time rather than bound at import, because the
    patch rebinds that attribute. The config goes through vLLM's own `patch_rope_parameters`, the
    same normalization the engine runs: it renames the legacy `type` key and, for the nested
    schema, injects a top-level `rope_type="default"` beside the sub-dicts.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding
    from vllm.models.deepseek_v4.common import rope as dsv4_rope
    from vllm.transformers_utils.config import patch_rope_parameters
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config as VllmDeepseekV4Config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from prime_rl.inference.patches import monkey_patch_deepseek_v4_per_layer_rope

    monkey_patch_deepseek_v4_per_layer_rope()
    rotary_embedding._ROPE_DICT.clear()

    def build(compress_ratio: int):
        config = VllmDeepseekV4Config(
            rope_theta=_ROPE_THETA,
            compress_rope_theta=_COMPRESS_ROPE_THETA,
            max_position_embeddings=_ROPE_MAX_POSITION,
            rope_parameters={key: dict(value) for key, value in _ROPE_PARAMETERS.items()},
        )
        patch_rope_parameters(config)
        # Model init runs under the model dtype (`vllm/model_executor/model_loader/base_loader.py`),
        # which is what would leave an unscaled config with a bf16 cache.
        with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(VllmConfig()):
            return dsv4_rope.build_deepseek_v4_rope(
                config,
                head_dim=_ROPE_HEAD_DIM,
                rope_head_dim=_ROPE_ROTARY_DIM,
                max_position_embeddings=_ROPE_MAX_POSITION,
                compress_ratio=compress_ratio,
            )

    return build


def test_deepseek_v4_vllm_rope_matches_the_reference(vllm_rope_builder):
    """Sliding-window layers take plain RoPE at `rope_theta`, compressed layers YaRN at theirs.

    Both are asserted together: neutralizing YaRN on the sliding layers must not disturb the
    compressed layers, which vLLM already gets right.
    """
    from vllm.model_executor.layers import rotary_embedding

    sliding = vllm_rope_builder(compress_ratio=1)
    compressed = vllm_rope_builder(compress_ratio=4)

    # `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` asserts fp32.
    assert sliding.cos_sin_cache.dtype is torch.float32
    assert compressed.cos_sin_cache.dtype is torch.float32

    torch.testing.assert_close(_vllm_rope_freqs(sliding), _reference_rope_freqs(0, _ROPE_THETA), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        _vllm_rope_freqs(compressed),
        _reference_rope_freqs(_ROPE_ORIGINAL_MAX_POSITION, _COMPRESS_ROPE_THETA),
        atol=1e-6,
        rtol=0,
    )

    # One rope per distinct `rope_theta`, and no more: `get_rope`'s cache key has to stay
    # hashable. An unhashable key silently defeats memoization, which costs one 256 MB fp32
    # cos/sin cache per attention layer on the real checkpoint instead of two in total.
    assert len(rotary_embedding._ROPE_DICT) == 2
    assert vllm_rope_builder(compress_ratio=1) is sliding


# Everything below exercises a *packed* batch: rollouts concatenated into one row, `position_ids`
# restarting per document, per-document lengths handed over as `seq_lens`, exactly as
# `trainer/batch.py` builds them. A packed row must equal running each document alone, because
# that is how vLLM serves them.

# Neither length is a multiple of a compress rate, so both compressors have to drop a trailing
# partial window instead of pooling across the boundary.
_DOC_LENS = (14, 18)

# The bf16 expert floor, which four hyper-connected layers amplify into every gradient. Measured
# worst case is 7.4e-3 against each tensor's own scale; treating the packed row as one long
# document instead of two moves the gradients 35x further than that.
_MODEL_GRAD_RTOL = 8e-2

# The per-mechanism cases run in float32. `kv_proj` sees a different number of rows packed than
# alone and cuBLAS may tile the two differently, so they never match bit for bit, and in bfloat16
# that floor would swallow the cross-document leakage these tests exist to catch.
_PACKED_RTOL, _PACKED_ATOL = 1e-5, 1e-6
# Gradients are bounded against the tensor's own scale instead: they are sums over the whole row,
# so their near-zero entries are the ones whose summands cancelled, and an element-wise relative
# bound would read out that cancellation noise rather than a document leak.
_PACKED_GRAD_RTOL = 1e-5

# One row folding together every length regime the per-document layout has to get right: 3
# compresses to nothing, 8 is a whole number of windows at both rates, and 13 ends mid-window at
# both. At both rates the per-document entry count lands below the row-global one, so neither
# case is vacuous.
_MIXED_DOCS = (3, 8, 13)

# The boundary falls inside a window of both rates, so both drop tokens at it.
_MID_WINDOW_DOCS = (7, 9)
# Whole windows everywhere, so only the numbering, and with it the RoPE position, moves.
_EXACT_MULTIPLE_DOCS = (8, 8)


def _packed_inputs(doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One packed row: token ids, `position_ids` restarting per document, and flat `seq_lens`."""
    total = sum(doc_lens)
    input_ids = torch.randint(0, _MODEL["vocab_size"], (1, total), device="cuda")
    position_ids = torch.cat([torch.arange(length, device="cuda") for length in doc_lens]).unsqueeze(0)
    return input_ids, position_ids, torch.tensor(doc_lens, device="cuda")


def _doc_ids(doc_lens: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([torch.full((length,), index, device="cuda") for index, length in enumerate(doc_lens)])


def _doc_slice(doc_lens: tuple[int, ...], index: int) -> slice:
    start = sum(doc_lens[:index])
    return slice(start, start + doc_lens[index])


def _entry_counts(doc_lens: tuple[int, ...], compress_rate: int) -> list[int]:
    return [length // compress_rate for length in doc_lens]


def _fp32_hidden_states(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two leaves carrying identical values, one for the packed run and one for the lone runs."""
    with torch.device("cuda"):
        hidden = torch.randn(_MODULE_BATCH, seq_len, _MODEL["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _take_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    """Detach whatever gradients have accumulated and clear them for the next run."""
    grads = {name: None if param.grad is None else param.grad.clone() for name, param in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    return grads


def _compare_accumulated_grads(
    module: nn.Module, expected: dict[str, torch.Tensor | None], rtol: float = _PACKED_GRAD_RTOL
) -> None:
    """Compare the gradients now on `module` against a snapshot taken from an earlier backward.

    Allows for a parameter that legitimately receives nothing: the Lightning Indexer reaches the
    loss only through integer top-k indices, so neither run may hand its parameters a gradient.
    """
    for name, param in module.named_parameters():
        if expected[name] is None:
            assert param.grad is None, f"{name} received a gradient per document but not packed"
            continue
        assert param.grad is not None, f"{name} received no gradient per document"
        _assert_relative(param.grad, expected[name], rtol, name)


def test_packed_sliding_window_mask_respects_documents(_torch_rms_norm, monkeypatch):  # noqa: F811
    """The local window stops at document boundaries, on every layer.

    Guards the `cu_seqlens` term in `build_sliding_window_mask`: without it the distances come
    from `torch.arange(seq_len)` over the whole packed row and a query reaches the previous
    `sliding_window` packed positions whatever document they belong to, which at the production
    `sliding_window = 128` against 77-token rollouts spans roughly two neighbours on all 43 layers.

    Captured from the mask the model actually applies rather than by calling the builder, so it
    keeps holding if the masking ever moves.
    """
    recorded = []
    real_attention = dsv4_attention.eager_attention_with_sinks

    def record(query, key, value, sinks, attention_mask, **kwargs):
        recorded.append(attention_mask)
        return real_attention(query, key, value, sinks, attention_mask, **kwargs)

    monkeypatch.setattr(dsv4_attention, "eager_attention_with_sinks", record)

    # The dense mask is an eager-path artifact; the count below is one call per layer.
    prime_model = get_prime_model(torch.float32)
    input_ids, position_ids, seq_lens = _packed_inputs(_DOC_LENS)
    prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)

    assert len(recorded) == _MODEL["num_hidden_layers"]
    total = sum(_DOC_LENS)
    doc_ids = _doc_ids(_DOC_LENS)
    positions = position_ids[0]
    distance = positions[:, None] - positions[None, :]
    expected = (doc_ids[:, None] == doc_ids[None, :]) & (distance >= 0) & (distance < _MODEL["sliding_window"])
    for layer_idx, mask in enumerate(recorded):
        # Compressed layers append their own entries as extra columns; those are covered
        # separately, so only the local window is compared here.
        local = mask[0, 0, :, :total] == 0
        assert torch.equal(local, expected), f"layer {layer_idx}: the local window crosses a document boundary"


def test_deepseek_v4(_torch_rms_norm):  # noqa: F811
    """The invariant that makes the trainer agree with vLLM, which serves each rollout alone.

    End to end over every pathway at once: the local sliding window, the CSA compressor with its
    indexer, and HCA. Each document's logits have to come out the same whether it is packed beside
    another rollout or served on its own. Gradients are compared too, since a leak that barely
    moves the logits can still move the update.

    This is the whole-model test, and it stands in for the HF-parity test the other models in this
    directory get: with no reference implementation available, packing is the only oracle that
    covers the assembled stack.
    """
    prime_model = get_prime_model(torch.float32)
    input_ids, position_ids, seq_lens = _packed_inputs(_DOC_LENS)

    packed = prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]
    # One random weight drawn over the packed logits and sliced per document, so the packed loss
    # and the summed per-document losses are the same function of the same numbers.
    with torch.device("cuda"):
        weight = torch.randn_like(packed)
    (packed * weight).sum().backward()
    packed_grads = _take_grads(prime_model)

    for index, length in enumerate(_DOC_LENS):
        span = _doc_slice(_DOC_LENS, index)
        alone = prime_model(
            input_ids[:, span],
            position_ids=torch.arange(length, device="cuda").unsqueeze(0),
            seq_lens=torch.tensor([length], device="cuda"),
        )["logits"]
        # `GroupedExperts` runs the routed experts through `torch._grouped_mm` in bfloat16
        # whatever dtype the model runs in, and packing changes which tokens share an expert
        # matmul, so the two runs agree only to the bf16 floor. Worst relative deviation
        # measured over six seeds is 1.2e-3; a document actually reading its neighbour moves
        # the logits by a fraction of their own scale, orders of magnitude above this.
        _assert_relative(packed[:, span], alone, 1e-2, f"document {index}")
        (alone * weight[:, span]).sum().backward()

    _compare_accumulated_grads(prime_model, packed_grads, rtol=_MODEL_GRAD_RTOL)


def _assert_layout_is_consistent(packed: PackedContext, doc_lens: tuple[int, ...], compress_rate: int) -> None:
    """Pin the layout against the per-document construction it claims to describe.

    Document `d` of length `L_d` gets `L_d // compress_rate` entries; entry `j` covers that
    document's local tokens `j * compress_rate` through `(j + 1) * compress_rate - 1` and is
    rotated at local position `j * compress_rate`. The entries are ordered document by document,
    which is what lets every comparison below slice them per document.
    """
    counts = _entry_counts(doc_lens, compress_rate)
    starts = [sum(doc_lens[:index]) for index in range(len(doc_lens))]
    expected_doc = [index for index, count in enumerate(counts) for _ in range(count)]
    expected_local = [entry for count in counts for entry in range(count)]
    expected_src = [
        [starts[doc] + local * compress_rate + offset for offset in range(compress_rate)]
        for doc, local in zip(expected_doc, expected_local)
    ]

    def as_tensor(values: list, dtype: torch.dtype = torch.long) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device="cuda")

    layout = packed.compression_layouts[compress_rate]
    assert torch.equal(layout.entry_doc_idx, as_tensor(expected_doc)), "entries are not ordered document by document"
    assert torch.equal(layout.entry_local_idx, as_tensor(expected_local)), "entries are not numbered within a document"
    assert torch.equal(layout.entry_tok_idx, as_tensor(expected_src).reshape(-1, compress_rate)), (
        "an entry pools source tokens outside its own document's window"
    )
    assert torch.equal(packed.tok_doc_idx, _doc_ids(doc_lens)), "tok_doc_idx must follow the document lengths"


@pytest.mark.parametrize(
    ("layer_idx", "compress_rate", "expected_counts"),
    [(_CSA_LAYER, _COMPRESS_RATE, [0, 2, 3]), (_HCA_LAYER, _HCA_COMPRESS_RATE, [0, 1, 1])],
    ids=["csa", "hca"],
)
def test_compressor_packed_matches_per_document(layer_idx, compress_rate, expected_counts):
    """Compressing a packed row must equal compressing each of its documents on its own.

    Forward and backward both: entry `n` of the packed run must pool the same source tokens, at
    the same compress-RoPE position, as the corresponding entry of its own document's run, and
    the gradient the packed run sends into the weights must equal the one the per-document runs
    accumulate. One random weight tensor is drawn over the packed entries and sliced per
    document, so the packed loss and the summed per-document losses are literally the same
    function of the same numbers.

    `_MIXED_DOCS` folds every length regime that matters into one row; see its definition.
    """
    module = prime_attention(layer_idx, dtype=torch.float32)
    compressor = module.compressor
    doc_lens = _MIXED_DOCS
    packed = _packed_context(doc_lens, torch.float32)

    counts = _entry_counts(doc_lens, compress_rate)
    assert counts == expected_counts
    assert (
        packed.compression_layouts[compress_rate].entry_tok_idx.shape[0]
        == sum(counts)
        < (sum(doc_lens) // compress_rate)
    ), "a row-global compression would emit more entries than this, so the probe is not vacuous"

    _assert_layout_is_consistent(packed, doc_lens, compress_rate)
    packed_input, alone_input = _fp32_hidden_states(sum(doc_lens))

    packed_entries = compressor.compress(packed_input, packed)
    assert packed_entries.shape == (_MODULE_BATCH, sum(counts), compressor.head_dim)

    with torch.device("cuda"):
        weight = torch.randn_like(packed_entries)
    (packed_entries * weight).sum().backward()
    packed_grads = _take_grads(compressor)

    for index, count in enumerate(counts):
        # The entry axis is laid out document by document exactly as the token axis is.
        entries = _doc_slice(tuple(counts), index)
        alone = compressor.compress(
            alone_input[:, _doc_slice(doc_lens, index)], _packed_context((doc_lens[index],), torch.float32)
        )
        assert alone.shape == (_MODULE_BATCH, count, compressor.head_dim), (
            f"document {index} compressed to the wrong count"
        )
        torch.testing.assert_close(
            packed_entries[:, entries],
            alone,
            rtol=_PACKED_RTOL,
            atol=_PACKED_ATOL,
            msg=lambda m, i=index: f"document {i} compresses differently packed than alone: {m}",
        )
        (alone * weight[:, entries]).sum().backward()

    _compare_accumulated_grads(compressor, packed_grads)
    torch.testing.assert_close(alone_input.grad, packed_input.grad, rtol=_PACKED_RTOL, atol=_PACKED_ATOL)


@pytest.mark.parametrize(
    ("layer_idx", "doc_lens"),
    [(_CSA_LAYER, _MID_WINDOW_DOCS), (_HCA_LAYER, _EXACT_MULTIPLE_DOCS)],
    ids=["csa", "hca"],
)
def test_attention_packed_matches_unpacked(layer_idx, doc_lens, _torch_rms_norm):  # noqa: F811
    """The same invariant, one whole attention layer at a time rather than one compressor.

    Everything the layer reads past its local window comes through the compressor, so a leaking
    entry, a misnumbered pick and a misrotated entry all show up here at once. The HCA case has no
    indexer to narrow the damage, and at rate 8 both documents own an entry, so the second one's
    has to be rotated at its own position rather than at its packed one.

    Sharper than `test_deepseek_v4`, which asserts the same property through the logits at a bf16
    floor: this runs in float32 and compares the layer's own output, forward and backward.
    """
    # The kernel's own packing invariant is asserted at the Flash shapes further down.
    module = prime_attention(layer_idx, dtype=torch.float32)
    packed_input, alone_input = _fp32_hidden_states(sum(doc_lens))
    packed = _packed_context(doc_lens, torch.float32)

    q_residual = module.q_a_norm(module.q_a_proj(packed_input.detach()))
    _, picks = module.compressor(packed_input.detach(), q_residual, packed)
    # (batch, seq_len, n_picks), with `-1` where the query had no entry left to pick.
    assert (picks[:, _doc_slice(doc_lens, 1)] >= 0).any(), (
        "vacuous probe: no query of the second document picks a compressed entry"
    )

    packed_output, _ = module(packed_input, packed=packed)
    with torch.device("cuda"):
        weight = torch.randn_like(packed_output)
    (packed_output * weight).sum().backward()
    packed_grads = _take_grads(module)

    for index, length in enumerate(doc_lens):
        span = _doc_slice(doc_lens, index)
        alone_output, _ = module(alone_input[:, span], packed=_packed_context((length,), torch.float32))
        torch.testing.assert_close(
            packed_output[:, span],
            alone_output,
            rtol=_PACKED_RTOL,
            atol=_PACKED_ATOL,
            msg=lambda m, i=index: f"document {i} attends differently packed than alone: {m}",
        )
        (alone_output * weight[:, span]).sum().backward()

    _compare_accumulated_grads(module, packed_grads)
    torch.testing.assert_close(alone_input.grad, packed_input.grad, rtol=_PACKED_RTOL, atol=_PACKED_ATOL)


# Everything below runs the real DeepSeek V4 Flash attention shapes instead of the toy `_MODEL`
# above, written out as a literal so nothing here depends on a local HF cache. The kernel path
# only exists at those shapes: a CSA query reads `sliding_window + index_topk = 640` keys
# over 512 channels with 64 heads, and the slot padding, the top-k saturation and the index
# arithmetic that the sparse representation has to get right are all invisible at toy sizes. The
# MoE fields are shrunk to nothing, since `DeepseekV4Attention` reads none of them.
_FLASH_MODEL = dict(
    vocab_size=64,
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=1,
    head_dim=512,
    q_lora_rank=1024,
    o_groups=8,
    o_lora_rank=1024,
    qk_rope_head_dim=64,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    sliding_window=128,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
    layer_types=["compressed_sparse_attention", "heavily_compressed_attention"],
    num_hidden_layers=2,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    max_position_embeddings=65536,
    moe_intermediate_size=64,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=1,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
)

_FLASH_CSA_LAYER = 0
_FLASH_COMPRESS_RATE = _FLASH_MODEL["compress_rates"]["compressed_sparse_attention"]

# Document layouts for the sparse path, at `compress_rate = 4`. The first four leave every query
# short of `index_topk = 512` readable entries, so the sentinel padding of the pick slots carries
# the difference; `(2600,)` saturates the picks instead, which the toy shapes cannot express at
# all. `(3,)` compresses to no entries whatsoever, leaving the local window alone to answer.
_FLASH_DOC_LENS = [(517, 1019), (3,), (300,), (3, 129, 1021), (2600,)]
_FLASH_DOC_IDS = ["two-docs", "no-entries", "one-short-doc", "three-docs", "saturated-topk"]


def _flash_config(attn_impl: str = "kernel") -> DeepseekV4Config:
    return DeepseekV4Config(**_FLASH_MODEL, _attn_impl=attn_impl)


def flash_attention(layer_idx: int, dtype: torch.dtype = torch.float32, attn_impl: str = "kernel") -> nn.Module:
    """One attention layer at the real DeepSeek V4 Flash shapes, 126M parameters of it."""
    with torch.device("cuda"), default_dtype(dtype):
        module = DeepseekV4Attention(_flash_config(attn_impl), layer_idx=layer_idx)
    _randomize(module)
    return module


def _flash_hidden_states(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two leaves carrying identical values, one per attention path.

    Batch 1: at these shapes the score tensors are the bulk of the memory and a second batch
    entry repeats the first without covering anything new.
    """
    with torch.device("cuda"):
        hidden = torch.randn(1, seq_len, _FLASH_MODEL["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _record_attention(monkeypatch) -> dict[str, torch.Tensor]:
    """Capture what each attention path is handed and what it returns, from real forwards.

    Both implementations are module-level functions the layer looks up by name, so patching them
    records the call the layer actually made rather than a reconstruction of it.
    """
    recorded: dict[str, torch.Tensor] = {}
    real_eager = dsv4_attention.eager_attention_with_sinks
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def eager(query, key, value, sinks, attention_mask, **kwargs):
        recorded["mask"] = attention_mask
        recorded["eager"] = real_eager(query, key, value, sinks, attention_mask, **kwargs)
        return recorded["eager"]

    def kernel(q, kv_buf, indices, sinks, scale):
        recorded["kv_buf"], recorded["indices"] = kv_buf, indices
        recorded["kernel"] = real_kernel(q, kv_buf, indices, sinks, scale)
        return recorded["kernel"]

    monkeypatch.setattr(dsv4_attention, "eager_attention_with_sinks", eager)
    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", kernel)
    return recorded


def _selected_positions(indices: torch.Tensor, n_positions: int) -> torch.Tensor:
    """`(seq_len, n_positions)` bool: which KV positions each query's gather slots address."""
    selected = torch.zeros((indices.shape[1], n_positions), dtype=torch.bool, device=indices.device)
    return selected.scatter_(1, indices[0, :, 0, :].long(), True)


@pytest.mark.skipif(dsv4_attention.dsv4_sparse_attn is None, reason="the sparse attention kernel needs tilelang")
@pytest.mark.parametrize("doc_lens", _FLASH_DOC_LENS, ids=_FLASH_DOC_IDS)
def test_sparse_indices_address_exactly_the_keys_the_dense_mask_admits(doc_lens, monkeypatch):
    """A CSA layer's gather slots must reach the keys the dense rules admit, key for key.

    One selection rendered two independent ways: the dense rendering concatenates the indexer's
    picks onto a sliding mask built straight from the document boundaries rather than from
    `window_indices`; the sparse one writes the window and the picks into a single index tensor
    over a gathered KV buffer. Nothing in the layer compares them, and every way of getting the
    sparse side wrong (a window base off by one, an entry index not offset by the token count, a
    `-1` pick surviving, a stale sentinel) still produces a finite output.

    Pure set equality on integers, so no tolerance enters; bfloat16 is only what the kernel these
    indices are recorded from insists on. Both the index tensor and the picks the dense rendering
    is built from come from a real forward of the module on the input.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(doc_lens, torch.bfloat16, _flash_config())
    hidden_states = _flash_hidden_states(sum(doc_lens))[0].detach().to(torch.bfloat16)
    recorded = _record_attention(monkeypatch)

    real_compressor = module.compressor.forward

    def compressor(hidden_states, q_residual, packed):
        compressed_kv, recorded["picks"] = real_compressor(hidden_states, q_residual, packed)
        return compressed_kv, recorded["picks"]

    monkeypatch.setattr(module.compressor, "forward", compressor)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    n_positions = recorded["kv_buf"].shape[1]
    seq_len, n_entries = sum(doc_lens), n_positions - 1 - sum(doc_lens)
    assert n_entries == sum(length // _FLASH_COMPRESS_RATE for length in doc_lens)
    block_bias = eager_reference.block_bias_from_indices(recorded["picks"], n_entries, torch.float32)
    sliding_mask = eager_reference.build_sliding_window_mask(
        tok_doc_idx=packed.tok_doc_idx, sliding_window=_FLASH_MODEL["sliding_window"], dtype=torch.float32
    )
    admitted = torch.cat([sliding_mask[0, 0], block_bias[0, 0]], dim=-1) == 0
    if n_entries:
        assert admitted[:, seq_len:].any(), "vacuous probe: no query reads a compressed entry"

    # The trailing position is the sentinel, which is "no key" on the sparse side and has no
    # column at all on the dense one.
    selected = _selected_positions(recorded["indices"], n_positions)
    assert torch.equal(selected[:, :-1], admitted), "the sparse and dense paths select different keys"


@pytest.mark.skipif(dsv4_attention.dsv4_sparse_attn is None, reason="the sparse attention kernel needs tilelang")
@pytest.mark.parametrize("doc_lens", _FLASH_DOC_LENS, ids=_FLASH_DOC_IDS)
def test_sparse_indices_are_in_range_and_never_repeat_a_key(doc_lens, monkeypatch):
    """Every gather slot addresses a real KV position, and no query counts a key twice.

    A negative or out-of-range index reads whatever lies next to the buffer instead of raising,
    which the kernel that will consume these has no way to detect. A repeat is worse than
    wasteful: the duplicated key takes twice its share of the softmax, silently reweighting the
    output. The sentinel is exempt from uniqueness, since padding every query out to a fixed slot
    count is exactly what it is for.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(doc_lens, torch.bfloat16, _flash_config())
    hidden_states = _flash_hidden_states(sum(doc_lens))[0].detach().to(torch.bfloat16)
    recorded = _record_attention(monkeypatch)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    indices, n_positions = recorded["indices"], recorded["kv_buf"].shape[1]
    # The minimum width: the window plus the picks the row actually affords, tile-aligned. A row
    # with fewer entries than `index_topk` gets a narrower slot count, not a sentinel-padded one.
    n_entries = n_positions - sum(doc_lens) - 1
    n_picks = min(_FLASH_MODEL["index_topk"], n_entries)
    tile = dsv4_attention._SLOT_TILE
    n_slots = indices.shape[-1]
    assert n_slots == ((_FLASH_MODEL["sliding_window"] + n_picks + tile - 1) // tile) * tile
    assert (indices >= 0).all(), "a gather slot addresses a negative KV position"
    assert (indices <= n_positions - 1).all(), "a gather slot addresses past the end of the KV buffer"

    slot_idx = indices[0, :, 0, :].long()
    counts = torch.zeros((slot_idx.shape[0], n_positions), dtype=torch.int32, device="cuda")
    counts.scatter_add_(1, slot_idx, torch.ones_like(slot_idx, dtype=torch.int32))
    assert (counts[:, :-1] <= 1).all(), "a query gathers the same key twice"


# One CSA layer in bfloat16, so `_PACKED_RTOL` (float32, and three orders of magnitude tighter
# than a kernel accumulating bfloat16 inputs) does not apply, but neither does `_MODEL_GRAD_RTOL`,
# which is sized for four hyper-connected layers amplifying a bf16 expert floor. Each bound below
# is the tightest round number holding over 30 seeds; the worst is 1.4e-3 on the output and
# 7.6e-3 on a gradient, against 6.9e-4 and 6.2e-3 on the fixed seed the test actually runs. The
# gradient bound is the tighter fit of the two, at 1.3x: every seed lands between 5.8e-3 and
# 7.6e-3, so the bound sits just above a well-sampled ceiling rather than above a long tail.
_KERNEL_RTOL, _KERNEL_GRAD_RTOL = 5e-3, 1e-2

# `compress_rate = 4` yields 129 + 254 = 383 compressed entries, under `index_topk = 512`, so
# every readable entry is picked and the indexer's ordering cannot differ packed from alone. A
# saturated layout would let a bfloat16 tie flip a pick and move the output for a reason that has
# nothing to do with document independence.
_KERNEL_DOC_LENS = (517, 1019)


@pytest.mark.skipif(dsv4_attention.dsv4_sparse_attn is None, reason="the sparse attention kernel needs tilelang")
def test_sparse_attention_kernel_packed_matches_unpacked(monkeypatch):
    """The fused kernel path, end to end through one CSA layer, must respect documents.

    The same invariant its float32 neighbours assert, run in bfloat16 because that is the only
    dtype `dsv4_sparse_attn` accepts. Numerics belong to
    `test_dsv4_sparse_attn.py`, which compares the kernel against a float32 gather oracle on
    hand-built tensors; what is covered here is that the modeling code feeds the kernel inputs it
    can act on, and that nothing in `q`, the KV buffer or the indices carries the packed row's
    layout into a document's own answer.

    The call count is load-bearing, not decoration: `dsv4_sparse_attn` raises today rather than
    demoting a dtype it cannot run, but without counting the calls a regression that reintroduced
    a fallback would leave this test asserting a property of the eager path instead.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(_KERNEL_DOC_LENS, torch.bfloat16, _flash_config())
    with torch.device("cuda"):
        hidden = torch.randn(1, sum(_KERNEL_DOC_LENS), _FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    packed_input, alone_input = hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, kv_buf, indices, sinks, scale):
        calls.append(q.shape[1])
        return real_kernel(q, kv_buf, indices, sinks, scale)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    packed_output, _ = module(packed_input, packed=packed)
    assert calls == [sum(_KERNEL_DOC_LENS)], f"the packed forward never reached the kernel, calls={calls}"
    with torch.device("cuda"):
        weight = torch.randn_like(packed_output)
    (packed_output * weight).sum().backward()
    packed_grads = _take_grads(module)

    for index, length in enumerate(_KERNEL_DOC_LENS):
        span = _doc_slice(_KERNEL_DOC_LENS, index)
        alone_output, _ = module(
            alone_input[:, span], packed=_packed_context((length,), torch.bfloat16, _flash_config())
        )
        _assert_relative(packed_output[:, span], alone_output, _KERNEL_RTOL, f"document {index}")
        (alone_output * weight[:, span]).sum().backward()

    assert calls == [sum(_KERNEL_DOC_LENS), *_KERNEL_DOC_LENS], f"a forward never reached the kernel, calls={calls}"
    _compare_accumulated_grads(module, packed_grads, rtol=_KERNEL_GRAD_RTOL)
    _assert_relative(alone_input.grad, packed_input.grad, _KERNEL_GRAD_RTOL, "hidden states gradient")


@pytest.mark.skipif(dsv4_attention.dsv4_sparse_attn is None, reason="the sparse attention kernel needs tilelang")
def test_sparse_attention_kernel_trains_every_parameter(monkeypatch):
    """Every parameter of a CSA layer that can train does, with the kernel in the path.

    `test_deepseek_v4_backward` makes this assertion through the assembled model, but only on the
    eager path: the kernel does not tile the toy `_MODEL` shapes. This is the same assertion at
    module level and at the real Flash shapes, and it is not implied by its neighbour above, which
    compares two runs of the same path and would pass unchanged if both left a parameter at zero.

    The call count is load-bearing rather than decoration: `dsv4_sparse_attn` raises today
    instead of falling back, but a regression that reintroduced a fallback would leave this
    asserting a property of the eager path.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(_KERNEL_DOC_LENS, torch.bfloat16, _flash_config())
    with torch.device("cuda"):
        hidden_states = torch.randn(1, sum(_KERNEL_DOC_LENS), _FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    hidden_states.requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, kv_buf, indices, sinks, scale):
        calls.append(q.shape[1])
        return real_kernel(q, kv_buf, indices, sinks, scale)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    output, _ = module(hidden_states, packed=packed)
    assert calls == [sum(_KERNEL_DOC_LENS)], f"the forward never reached the kernel, calls={calls}"
    with torch.device("cuda"):
        weight = torch.randn_like(output)
    (output * weight).sum().backward()

    dead, unexpectedly_alive = [], []
    for name, param in module.named_parameters():
        has_grad = param.grad is not None and param.grad.norm().item() > 0
        # The same expectation `test_deepseek_v4_backward` and `_compare_accumulated_grads` carry:
        # the indexer reaches the loss only through integer top-k indices, so nothing
        # differentiates back into it.
        if ".indexer." in name:
            if has_grad:
                unexpectedly_alive.append(name)
        elif not has_grad:
            dead.append(name)

    assert not dead, f"Parameters with zero/no gradients: {dead}"
    assert not unexpectedly_alive, f"Lightning Indexer parameters received a gradient: {unexpectedly_alive}"
    assert hidden_states.grad is not None and hidden_states.grad.norm() > 0, (
        "the hidden states received no gradient, so nothing reached the layer's inputs"
    )
