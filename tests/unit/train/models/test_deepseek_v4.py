"""Whole-model checks for the DeepSeek V4 port that need no reference implementation.

Hash routing, gradient reachability, the weight-conversion chain against vLLM's own loader,
meta-device buffer restoration, the packed-batch invariant the trainer needs in order to agree
with vLLM, and the RoPE vLLM builds on the other side of that boundary. The parity half, which uses HF's own DeepSeek V4 implementation as its oracle,
lives in `test_deepseek_v4_hf.py`.
"""

import math
import re

import pytest
import torch
from huggingface_hub.errors import StrictDataclassClassValidationError
from torch import nn

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import (
    CompressionLayout,
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    build_compression_layout,
)
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
from prime_rl.utils.utils import default_dtype

from .deepseek_v4_helpers import (
    _BASE,
    _BATCH,
    _SEQ,
    _assert_relative,
    _inputs,
    _prime_config,
    _randomize,
    _seed_rng,  # noqa: F401 -- pytest fixture, applied by name
    _seq_lens,
    _torch_rms_norm,  # noqa: F401 -- pytest fixture, applied by name
    get_prime_model,
)

pytestmark = [pytest.mark.gpu]


def test_deepseek_v4_hash_layers_route_on_token_ids():
    """The bootstrap layers read `input_ids`, so identical hidden states still route apart."""
    prime_model = get_prime_model()
    hash_layers = prime_model.model.layers[: _BASE["num_hash_layers"]]
    assert hash_layers, "config must contain a hash-routed layer"

    counts = []
    for token_id in (0, 1):
        input_ids = torch.full((_BATCH, _SEQ), token_id, device="cuda", dtype=torch.long)
        for layer in hash_layers:
            layer.mlp.tokens_per_expert.zero_()
        prime_model(input_ids, seq_lens=_seq_lens(input_ids))
        counts.append(torch.stack([layer.mlp.tokens_per_expert.clone() for layer in hash_layers]))

    table = hash_layers[0].mlp.tid2eid
    assert set(table[0].tolist()) != set(table[1].tolist()), "the two table rows must differ for this to bite"
    assert not torch.equal(counts[0], counts[1]), "a hash layer must route the two token ids to different experts"
    expected = torch.zeros_like(counts[0][0])
    expected[table[0]] = _BATCH * _SEQ
    torch.testing.assert_close(counts[0][0], expected)


def test_deepseek_v4_backward():
    """Every parameter that can train does, and the Lightning Indexer's still cannot."""
    prime_config = _prime_config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = DeepseekV4ForCausalLM(prime_config)
    _randomize(model)
    inject_prime_lm_head(model)

    input_ids, _ = _inputs()
    output = model(input_ids, seq_lens=_seq_lens(input_ids))
    output["logits"].sum().backward()

    dead, unexpectedly_alive = [], []
    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue
        has_grad = param.grad is not None and param.grad.norm().item() > 0
        # The indexer reaches the loss only through integer top-k indices, so nothing
        # differentiates back into it. DeepSeek trains it with a separate auxiliary loss
        # that prime-rl does not implement; see TODO.md.
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


def test_deepseek_v4_config_rejects_a_foreign_layer_type():
    """V4's own attention vocabulary, not the generic one transformers checks against.

    `PretrainedConfig.validate_layer_type` runs first, from `super().__init__()`, and accepts
    anything in transformers' generic layer-type list; only `DeepseekV4Config`'s own override
    narrows that to the three V4 variants. `compress_rates` carries a rate for the foreign type
    on purpose, so `validate_architecture` cannot be what rejects it.
    """
    kwargs = _BASE | {"layer_types": ["full_attention"] * 5, "compress_rates": {"full_attention": 4}}

    with pytest.raises(StrictDataclassClassValidationError, match="layer_types entries must be one of"):
        DeepseekV4Config(**kwargs)


def test_deepseek_v4_config_translates_legacy_compress_ratios():
    """Real checkpoints ship the V3-flavoured legacy `compress_ratios`/`num_hash_layers` schema
    instead of `layer_types`/`mlp_layer_types` (see NOTES-ds-v4-inference-preflight.md). prime-rl's
    model code reads `layer_types`/`mlp_layer_types` directly, so the config has to translate them.

    The expected schedules are written out here rather than read off HF's own config;
    `test_deepseek_v4_hf.py` holds the version that checks them against it.
    """
    config = DeepseekV4Config(num_hidden_layers=6, compress_ratios=[0, 0, 4, 128, 4, 128], num_hash_layers=2)

    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config.mlp_layer_types == ["hash_moe", "hash_moe", "moe", "moe", "moe", "moe"]


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
    instantiating vLLM's model; `examples/advanced/deepseek-v4-flash/kl-check.toml` is what
    covers that end to end.

    `_make_deepseek_v4_weights_mapper` is private API in a URL-pinned wheel (`vllm==0.26.0+cu129`).
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


# Everything below exercises a *packed* batch: several rollouts concatenated into one row, with
# `position_ids` restarting at 0 per document and the per-document lengths handed over as
# `seq_lens`, exactly as `trainer/batch.py` builds them. `DeepseekV4Model.forward` consumes
# `seq_lens` for the sliding-window mask and for the compression layout both compressors pool
# and number their entries against, so every pathway a query reads through stops at the
# boundaries of that query's own document.
#
# The assertions deliberately avoid naming which compressed entry index belongs to which
# document, and go through `forward` rather than the internals it calls. Entry numbering is a
# detail of the layout, so an index-based assertion would pin the numbering scheme rather than
# the property that matters. They state the invariant instead: redraw the first document and
# nothing the second document reads may move, and what a query reads must not depend on whether
# its document was packed or run alone.

# Neither length is a multiple of a compress rate, so each document ends mid-window and both
# compressors have to drop a trailing partial window instead of pooling across the boundary.
_DOC_LENS = (14, 18)
# Below the HCA compress rate of 8, where HCA yields no entries and contributes nothing either way.
_SHORT_DOC_LENS = (6, 6)


def _packed_inputs(doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One packed row: token ids, `position_ids` restarting per document, and flat `seq_lens`."""
    total = sum(doc_lens)
    input_ids = torch.randint(0, _BASE["vocab_size"], (1, total), device="cuda")
    position_ids = torch.cat([torch.arange(length, device="cuda") for length in doc_lens]).unsqueeze(0)
    return input_ids, position_ids, torch.tensor(doc_lens, device="cuda")


def _layout(doc_lens: tuple[int, ...], compress_rate: int) -> CompressionLayout:
    """The layout `DeepseekV4Model` would build for a row laid out as `doc_lens`."""
    total = sum(doc_lens)
    cu_seqlens, _ = get_cu_seqlens_from_seq_lens(torch.tensor(doc_lens, device="cuda"), total_tokens=total)
    return build_compression_layout(cu_seqlens, compress_rate, total)


def _doc_ids(doc_lens: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([torch.full((length,), index, device="cuda") for index, length in enumerate(doc_lens)])


def _doc_slice(doc_lens: tuple[int, ...], index: int) -> slice:
    start = sum(doc_lens[:index])
    return slice(start, start + doc_lens[index])


def _compressor_inputs(doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Random tensors of the shapes a decoder layer hands its compressor."""
    total = sum(doc_lens)
    hidden_states = torch.randn(1, total, _BASE["hidden_size"], device="cuda")
    q_residual = torch.randn(1, total, _BASE["q_lora_rank"], device="cuda")
    return hidden_states, q_residual


def _resample_first_document(tensors: tuple[torch.Tensor, ...], doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Copies with document 0's rows redrawn and every later document left byte-identical."""
    first = _doc_slice(doc_lens, 0)
    resampled = []
    for tensor in tensors:
        clone = tensor.clone()
        clone[:, first] = torch.randn_like(clone[:, first])
        resampled.append(clone)
    return tuple(resampled)


def _compressor_of_type(model: nn.Module, compressor_class: type) -> nn.Module:
    compressors = [
        layer.self_attn.compressor
        for layer in model.model.layers
        if isinstance(layer.self_attn.compressor, compressor_class)
    ]
    assert compressors, f"config must contain a {compressor_class.__name__} layer"
    return compressors[0]


def _assert_reads_are_document_local(compressor: nn.Module, doc_lens: tuple[int, ...]) -> None:
    """The second document's readable entries, and their values, must not depend on the first."""
    hidden_states, q_residual = _compressor_inputs(doc_lens)
    _, position_ids, _ = _packed_inputs(doc_lens)

    layout = _layout(doc_lens, compressor.compress_rate)
    compressed_kv, block_bias = compressor(hidden_states, q_residual, position_ids, layout)
    other_hidden, other_q = _resample_first_document((hidden_states, q_residual), doc_lens)
    other_kv, other_bias = compressor(other_hidden, other_q, position_ids, layout)

    second = _doc_slice(doc_lens, 1)
    readable = block_bias[0, 0, second] == 0
    assert readable.any(), "vacuous probe: the second document reads no compressed entry at all"
    assert torch.equal(readable, other_bias[0, 0, second] == 0), (
        "which entries the second document may read changed when only the first document did"
    )
    for row, entries in enumerate(readable):
        assert torch.equal(compressed_kv[0, 0][entries], other_kv[0, 0][entries]), (
            f"query {row} of the second document reads an entry built from the first document"
        )


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

    prime_model = get_prime_model(torch.float32)
    input_ids, position_ids, seq_lens = _packed_inputs(_DOC_LENS)
    prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)

    assert len(recorded) == _BASE["num_hidden_layers"]
    total = sum(_DOC_LENS)
    doc_ids = _doc_ids(_DOC_LENS)
    positions = position_ids[0]
    distance = positions[:, None] - positions[None, :]
    expected = (doc_ids[:, None] == doc_ids[None, :]) & (distance >= 0) & (distance < _BASE["sliding_window"])
    for layer_idx, mask in enumerate(recorded):
        # Compressed layers append their own entries as extra columns; those are covered
        # separately, so only the local window is compared here.
        local = mask[0, 0, :, :total] == 0
        assert torch.equal(local, expected), f"layer {layer_idx}: the local window crosses a document boundary"


def test_packed_logits_match_unpacked(_torch_rms_norm):  # noqa: F811
    """The invariant that makes the trainer agree with vLLM, which serves each rollout alone.

    End to end over every pathway at once: the local sliding window, the CSA compressor with its
    indexer, and HCA. Each document's logits have to come out the same whether it is packed beside
    another rollout or served on its own.
    """
    prime_model = get_prime_model(torch.float32)
    input_ids, position_ids, seq_lens = _packed_inputs(_DOC_LENS)

    packed = prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]

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


def test_packed_csa_reads_only_own_document_entries(_torch_rms_norm):  # noqa: F811
    """CSA's long-range pathway stays inside the querying token's own document.

    `causal_threshold` counts entries per document and `build_compression_layout` numbers and
    pools them per document to match, so the entries a query at local position 4 of the second
    document may read are its own document's, not the ones pooled from the first.
    """
    prime_model = get_prime_model(torch.float32)
    _assert_reads_are_document_local(_compressor_of_type(prime_model, DeepseekV4CSACompressor), _DOC_LENS)


def test_packed_hca_reads_only_own_document_entries(_torch_rms_norm):  # noqa: F811
    """The same invariant for HCA, which has no indexer and reads every entry under its threshold.

    The documents here are longer than the compress rate of 8 on purpose: below it the threshold
    floors to zero, HCA contributes nothing at all, and the probe would be vacuous. That regime is
    covered by `test_hca_inert_below_compress_rate_matches_unpacked` instead.
    """
    prime_model = get_prime_model(torch.float32)
    _assert_reads_are_document_local(_compressor_of_type(prime_model, DeepseekV4HCACompressor), _DOC_LENS)


def test_packed_csa_selection_matches_unpacked(_torch_rms_norm):  # noqa: F811
    """What the Lightning Indexer hands a query must not depend on how the row was packed.

    The indexer masks its candidates to the querying document's own entries, and those entries
    pool only that document's tokens, so a query's top-k comes out the same either way.

    This is the oracle `forward(pack([A, B])) == concat(forward(A), forward(B))` applied one level
    down, so unlike the perturbation probes above it also catches pooling that blends two
    documents, which no mask can repair. Compared as the gathered entry *values* per query, not as
    indices, since entries are numbered per document and the packed row numbers the second
    document's entries differently than the lone run does. It goes through `forward` rather than
    `compress` and `indexer` directly, so the whole path a query takes is covered.
    """
    prime_model = get_prime_model(torch.float32)
    compressor = _compressor_of_type(prime_model, DeepseekV4CSACompressor)
    rate = compressor.compress_rate

    hidden_states, q_residual = _compressor_inputs(_DOC_LENS)
    _, position_ids, _ = _packed_inputs(_DOC_LENS)
    packed_kv, packed_bias = compressor(hidden_states, q_residual, position_ids, _layout(_DOC_LENS, rate))

    second = _doc_slice(_DOC_LENS, 1)
    length = _DOC_LENS[1]
    alone_kv, alone_bias = compressor(
        hidden_states[:, second],
        q_residual[:, second],
        torch.arange(length, device="cuda").unsqueeze(0),
        _layout((length,), rate),
    )

    packed_reads = packed_bias[0, 0, second] == 0
    assert packed_reads.any(), "vacuous probe: the second document selects no entry at all"
    for row in range(length):
        selected = packed_kv[0, 0][packed_reads[row]]
        alone_selected = alone_kv[0, 0][alone_bias[0, 0, row] == 0]
        assert selected.shape == alone_selected.shape and torch.equal(selected, alone_selected), (
            f"query {row} of the second document reads different entries packed than on its own"
        )


def test_hca_inert_below_compress_rate_matches_unpacked(_torch_rms_norm):  # noqa: F811
    """HCA contributes nothing either way when every document is shorter than its rate.

    Its threshold `(position_ids + 1) // 8` floors to zero for every query in a 6-token document,
    so nothing is readable and `exp(-inf) = 0`; run alone, the document is too short to fill a
    window and there are no entries at all. Packed and unpacked agree, so this is a capability
    loss rather than a packing mismatch and must not be "fixed" here. It matters because a
    whole-model packing test built on documents this short would report green without ever
    exercising HCA.

    Asserted as an entry count: compressing per document, a row of documents this short yields no
    entries at all, so there is nothing for a query to read rather than entries it may not reach.
    The layout is handed in, so the entry count restates what the layout already says; the
    property that a short document compresses to nothing is pinned directly by
    `test_hca_compressor_emits_no_entries_when_every_document_is_short`. What is left here is that
    the compressor honours a zero-entry layout identically packed and alone.
    """
    prime_model = get_prime_model(torch.float32)
    compressor = _compressor_of_type(prime_model, DeepseekV4HCACompressor)
    rate = compressor.compress_rate
    assert rate == 8

    hidden_states, q_residual = _compressor_inputs(_SHORT_DOC_LENS)
    _, position_ids, _ = _packed_inputs(_SHORT_DOC_LENS)
    _, packed_bias = compressor(hidden_states, q_residual, position_ids, _layout(_SHORT_DOC_LENS, rate))
    assert packed_bias.shape[-1] == 0, "every document is too short to fill a window, so the row has no entries"

    for index, length in enumerate(_SHORT_DOC_LENS):
        span = _doc_slice(_SHORT_DOC_LENS, index)
        alone_kv, alone_bias = compressor(
            hidden_states[:, span],
            q_residual[:, span],
            torch.arange(length, device="cuda").unsqueeze(0),
            _layout((length,), rate),
        )
        assert alone_kv.shape[2] == 0, f"document {index} is too short to fill a window"
        assert alone_bias.shape[-1] == 0


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
# base but not the scaling, which `monkey_patch_deepseek_v4_rope_disable_yarn_on_sliding_layers`
# corrects.
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
    patch rebinds that attribute.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4.common import rope as dsv4_rope
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config as VllmDeepseekV4Config

    from prime_rl.inference.patches import monkey_patch_deepseek_v4_rope_disable_yarn_on_sliding_layers

    monkey_patch_deepseek_v4_rope_disable_yarn_on_sliding_layers()

    def build(compress_ratio: int, rope_parameters: dict | None = None):
        config = VllmDeepseekV4Config(
            rope_scaling=dict(_ROPE_SCALING) if rope_parameters is None else None,
            rope_parameters=rope_parameters,
            rope_theta=_ROPE_THETA,
            compress_rope_theta=_COMPRESS_ROPE_THETA,
            max_position_embeddings=_ROPE_MAX_POSITION,
        )
        with set_current_vllm_config(VllmConfig()):
            return dsv4_rope.build_deepseek_v4_rope(
                config,
                head_dim=_ROPE_HEAD_DIM,
                rope_head_dim=_ROPE_ROTARY_DIM,
                max_position_embeddings=_ROPE_MAX_POSITION,
                compress_ratio=compress_ratio,
            )

    return build


def _assert_reference_rope(builder, rope_parameters=None):
    """Sliding-window layers take plain RoPE at `rope_theta`, compressed layers YaRN at theirs.

    Both are asserted together: neutralizing YaRN on the sliding layers must not disturb the
    compressed layers, which vLLM already gets right.
    """
    sliding = _vllm_rope_freqs(builder(compress_ratio=1, rope_parameters=rope_parameters))
    compressed = _vllm_rope_freqs(builder(compress_ratio=4, rope_parameters=rope_parameters))

    torch.testing.assert_close(sliding, _reference_rope_freqs(0, _ROPE_THETA), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        compressed,
        _reference_rope_freqs(_ROPE_ORIGINAL_MAX_POSITION, _COMPRESS_ROPE_THETA),
        atol=1e-6,
        rtol=0,
    )


def test_deepseek_v4_vllm_rope_matches_the_reference(vllm_rope_builder):
    """The flat legacy `rope_scaling` the real checkpoint ships."""
    _assert_reference_rope(vllm_rope_builder)


def test_deepseek_v4_vllm_rope_matches_the_reference_from_nested_parameters(vllm_rope_builder):
    """A `save_pretrained` round trip nests `rope_parameters` under `main`/`compress` keys.

    vLLM's config shim assumes a flat dict, so transformers injects a top-level
    `rope_type="default"` beside the sub-dicts and `build_deepseek_v4_rope` drops YaRN from every
    layer. The patch reads the `compress` sub-dict instead.
    """
    nested = {
        "main": {"rope_type": "default", "rope_theta": _ROPE_THETA, "partial_rotary_factor": 0.125},
        "compress": {
            **_ROPE_SCALING,
            "rope_type": "yarn",
            "rope_theta": _COMPRESS_ROPE_THETA,
            "partial_rotary_factor": 0.125,
            "attention_factor": 1.0,
        },
    }
    _assert_reference_rope(vllm_rope_builder, rope_parameters=nested)
