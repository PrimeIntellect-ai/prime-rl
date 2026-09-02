"""DeepSeek V4 checks that need a GPU and no reference implementation.

The first half works on the assembled model: hash routing, gradient reachability, the
weight-conversion chain against vLLM's own loader, meta-device buffer restoration, the
packed-batch invariant the trainer needs in order to agree with vLLM, and the RoPE vLLM builds on
the other side of that boundary. The second half works one mechanism at a time, on the smaller
per-module configs, and is marked as such where it begins.

There is no HF oracle here. `transformers.models.deepseek_v4` only exists from transformers 5.15
and the repo pins an older version, so every assertion is either self-consistency (packed against
unpacked), a closed form written out by hand, or vLLM's own loader. The CPU-only checks live in
`test_deepseek_v4_cpu.py`.
"""

import math
import re
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    PackedContext,
)
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

from .deepseek_v4_helpers import (
    _ATTN,
    _COMPRESS_RATE,
    _CSA_LAYER,
    _HC,
    _HCA_COMPRESS_RATE,
    _HCA_LAYER,
    _MODEL,
    _MODEL_BATCH,
    _MODEL_SEQ,
    _MODULE_BATCH,
    _MODULE_SEQ,
    _MOE,
    _MOE_TOKENS,
    _SINGLE_DOC,
    _SLIDING_LAYER,
    _assert_relative,
    _hidden_states,
    _hidden_streams,
    _input_ids,
    _inputs,
    _moe_hidden_states,
    _packed_context,
    _prime_config,
    _randomize,
    _seed_rng,  # noqa: F401 -- pytest fixture, applied by name
    _seq_lens,
    _seq_positions,
    _tid2eid,
    _torch_rms_norm,  # noqa: F401 -- pytest fixture, applied by name
    get_prime_model,
    prime_attention,
    prime_clamped_moe,
    prime_hyper_connection,
    prime_moe,
)

pytestmark = [pytest.mark.gpu]


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
        prime_model(input_ids, position_ids=_seq_positions(input_ids), seq_lens=_seq_lens(input_ids))
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

    input_ids, position_ids = _inputs()
    output = model(input_ids, position_ids=position_ids, seq_lens=_seq_lens(input_ids))
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


def test_deepseek_v4_hash_table_survives_the_on_disk_roundtrip():
    """The frozen table has to reach `mlp.router.tid2eid` from the name a real checkpoint ships.

    It is the one buffer no `init_weights` can reconstruct: an all-zero table is a valid tensor
    that routes every token to expert 0, so a rename on either side of the conversion chain
    degrades the model in silence rather than raising. `test_deepseek_v4_weight_conversion_roundtrip`
    only ever roundtrips the zeros the constructor leaves behind, and `_VLLM_MAPPED_NAMES` pins the
    on-disk name without saying where it lands, so this carries real values across both.
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


def test_deepseek_v4_load_dcp_from_hf_keeps_the_hash_table(tmp_path, monkeypatch):
    """Checkpoint values for the frozen table must survive the loading path itself.

    Two things at once: the name `load_dcp_from_hf` asks the checkpoint for (a `KeyError` in the
    stub below if the buffer moves), and that nothing between `dcp_load` and the end of loading
    resets it. `init_buffers_post_meta` walks every MoE and runs just before the load, so getting
    it to reset one buffer too many is an easy mistake with no symptom other than every bootstrap
    token routing to expert 0.
    """
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(_prime_config())
    expected = _tid2eid(_MODEL["vocab_size"], _MODEL["n_routed_experts"], _MODEL["num_experts_per_tok"])

    def fake_dcp_load(state_dict, storage_reader=None):
        buffer = state_dict["model.layers.0.mlp.router.tid2eid"]
        buffer.copy_(expected.to(device=buffer.device))

    monkeypatch.setattr("prime_rl.trainer.model.dcp_load", fake_dcp_load)
    monkeypatch.setattr("prime_rl.trainer.model.load_state_dict_keys", lambda path: model.state_dict().keys())
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)

    load_dcp_from_hf(model, ModelConfig(name=str(tmp_path)), parallel_dims=MagicMock())

    torch.testing.assert_close(model.model.layers[0].mlp.router.tid2eid, expected)


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
# Four hyper-connected layers amplify the bf16 expert floor described on the logits comparison
# below into every parameter's gradient, not just the experts' own, and which tokens share an
# expert matmul (and so what cancels) moves with the seed. Measured worst case here is 7.4e-3
# relative to each tensor's own scale; telling the packed run it is one long document instead of
# two moves the gradients by 2.8, which is 35x this bound.
_MODEL_GRAD_RTOL = 8e-2

_DOC_LENS = (14, 18)
# Below the HCA compress rate of 8, where HCA yields no entries and contributes nothing either way.
_SHORT_DOC_LENS = (6, 6)


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


def _compressor_inputs(doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Random tensors of the shapes a decoder layer hands its compressor."""
    total = sum(doc_lens)
    hidden_states = torch.randn(1, total, _MODEL["hidden_size"], device="cuda")
    q_residual = torch.randn(1, total, _MODEL["q_lora_rank"], device="cuda")
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
    packed = _packed_context(doc_lens, torch.float32, config=_prime_config())

    compressed_kv, block_bias = compressor(hidden_states, q_residual, packed)
    other_hidden, other_q = _resample_first_document((hidden_states, q_residual), doc_lens)
    other_kv, other_bias = compressor(other_hidden, other_q, packed)

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


def test_packed_logits_match_unpacked(_torch_rms_norm):  # noqa: F811
    """The invariant that makes the trainer agree with vLLM, which serves each rollout alone.

    End to end over every pathway at once: the local sliding window, the CSA compressor with its
    indexer, and HCA. Each document's logits have to come out the same whether it is packed beside
    another rollout or served on its own. Gradients are compared too, since a leak that barely
    moves the logits can still move the update.
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


def test_packed_csa_reads_only_own_document_entries(_torch_rms_norm):  # noqa: F811
    """CSA's long-range pathway stays inside the querying token's own document.

    `causal_threshold` counts entries per document and `CompressionLayout.build` numbers and
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

    hidden_states, q_residual = _compressor_inputs(_DOC_LENS)
    packed_kv, packed_bias = compressor(
        hidden_states, q_residual, _packed_context(_DOC_LENS, torch.float32, config=_prime_config())
    )

    second = _doc_slice(_DOC_LENS, 1)
    length = _DOC_LENS[1]
    alone_kv, alone_bias = compressor(
        hidden_states[:, second],
        q_residual[:, second],
        _packed_context((length,), torch.float32, config=_prime_config()),
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
    property that a short document compresses to nothing is pinned directly by the
    `hca_short_first_document` case of `test_compressor_packed_matches_per_document`. What is left
    here is that the compressor honours a zero-entry layout identically packed and alone.
    """
    prime_model = get_prime_model(torch.float32)
    compressor = _compressor_of_type(prime_model, DeepseekV4HCACompressor)
    rate = compressor.compress_rate
    assert rate == 8

    hidden_states, q_residual = _compressor_inputs(_SHORT_DOC_LENS)
    packed_kv, packed_bias = compressor(
        hidden_states, q_residual, _packed_context(_SHORT_DOC_LENS, torch.float32, config=_prime_config())
    )
    assert packed_bias.shape[-1] == 0, "every document is too short to fill a window, so the row has no entries"
    # The projections run whether or not the layout selects anything, so every parameter still
    # takes part in the backward, with a zero gradient rather than none at all.
    assert packed_kv.requires_grad, "an empty compression must still carry a graph back to the row"
    packed_kv.sum().backward()
    for name, param in compressor.named_parameters():
        assert param.grad is not None, f"{name} took no part in the backward of an empty compression"
        assert not param.grad.any(), f"{name} received a non-zero gradient from an empty compression"

    for index, length in enumerate(_SHORT_DOC_LENS):
        span = _doc_slice(_SHORT_DOC_LENS, index)
        alone_kv, alone_bias = compressor(
            hidden_states[:, span],
            q_residual[:, span],
            _packed_context((length,), torch.float32, config=_prime_config()),
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


# Everything below works one mechanism at a time, below the assembled-model level the tests above
# work at, on the smaller `_ATTN`/`_MOE`/`_HC` configs. These carry the only coverage of the
# compressors' window structure, the Lightning Indexer's selection contract, the router's scoring,
# and the per-document packing invariant, none of which the whole-model tests can see sharply
# enough to be worth asserting there.
#
# Every test in this half stands alone: each builds its own module and its only shared machinery
# is the helper module's builders, so any one of them can be deleted without touching another. The
# exceptions are noted on the helpers in the packed section further down.


def test_hyperconnection_gates_are_doubly_stochastic_and_bounded():
    """mHC's mixing matrix is a Sinkhorn iterate, so it must sum to one along both axes.

    This is the property the fp32 weight-transfer exception exists to protect: handed bf16
    hyper-connection parameters, the normalization produces NaN rather than a slightly wrong
    gate. The pre-gate is `2 * sigmoid(.)`, so it is bounded independently.
    """
    prime_module = prime_hyper_connection()
    _, streams = _hidden_streams()

    post, comb, collapsed = prime_module(streams)

    assert collapsed.shape == (_MODULE_BATCH, _MODULE_SEQ, _HC["hidden_size"])
    assert collapsed.dtype == streams.dtype
    assert (post >= 0).all() and (post <= 2).all()
    assert (comb > 0).all()
    ones = torch.ones_like(comb.sum(dim=-1))
    torch.testing.assert_close(comb.sum(dim=-1), ones, rtol=0, atol=1e-5)
    torch.testing.assert_close(comb.sum(dim=-2), ones, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("layer_idx", "reaches_past_the_window"),
    [(_SLIDING_LAYER, False), (_CSA_LAYER, True), (_HCA_LAYER, True)],
    ids=["sliding", "csa", "hca"],
)
def test_attention_long_range_reach_by_layer_type(layer_idx, reaches_past_the_window):
    """Perturb token 0 and read off how far forward the change is still visible.

    A sliding layer must forget it the moment it leaves the local window; the two compressed
    layers must still reach it through their pooled entries. HCA is the sharpest of the three:
    its first entry covers tokens `0 .. rate - 1` and is unreadable until a query reaches the
    last of them, so between leaving the window and that point nothing carries token 0 at all.
    """
    prime_module = prime_attention(layer_idx)
    _, hidden = _hidden_states()
    packed = _packed_context(_SINGLE_DOC, torch.bfloat16)
    window = _ATTN["sliding_window"]

    baseline, _ = prime_module(hidden, packed=packed)
    perturbed_input = hidden.clone()
    perturbed_input[:, 0] += 1.0
    perturbed, _ = prime_module(perturbed_input, packed=packed)

    # Token 0 is the last key inside the window of query `window - 1`, so every layer moves there.
    assert not torch.equal(perturbed[:, window - 1], baseline[:, window - 1])

    if not reaches_past_the_window:
        torch.testing.assert_close(perturbed[:, window:], baseline[:, window:], rtol=0, atol=0)
        return

    if layer_idx == _HCA_LAYER:
        first_readable = _HCA_COMPRESS_RATE - 1
        assert first_readable > window, "config must leave a gap between the window and the first entry"
        torch.testing.assert_close(
            perturbed[:, window:first_readable], baseline[:, window:first_readable], rtol=0, atol=0
        )
        assert not torch.equal(perturbed[:, first_readable:], baseline[:, first_readable:])
    else:
        assert not torch.equal(perturbed[:, window:], baseline[:, window:])


@pytest.mark.parametrize(
    ("layer_idx", "compress_rate", "extra_entries_touched"),
    [(_CSA_LAYER, _COMPRESS_RATE, {1}), (_HCA_LAYER, _HCA_COMPRESS_RATE, set())],
    ids=["csa_overlapping", "hca_non_overlapping"],
)
def test_compressor_pooling_window_structure(layer_idx, compress_rate, extra_entries_touched):
    """Which entries a single token feeds is the whole structural difference between CSA and HCA.

    CSA runs a dual series: a token feeds its own window's entry through `Cb` and the *next*
    window's through `Ca`. HCA's windows do not overlap, so a token feeds its own entry and no
    other. Perturbing one token and collecting which entries moved states both at once.
    """
    compressor = prime_attention(layer_idx).compressor
    _, hidden = _hidden_states()
    packed = _packed_context(_SINGLE_DOC, torch.bfloat16)

    compressed = compressor.compress(hidden, packed)
    assert compressed.shape == (_MODULE_BATCH, _MODULE_SEQ // compress_rate, _ATTN["head_dim"])

    token = compress_rate + 1
    perturbed_input = hidden.clone()
    perturbed_input[:, token] += 1.0
    perturbed = compressor.compress(perturbed_input, packed)

    changed = {w for w in range(compressed.shape[1]) if not torch.equal(perturbed[:, w], compressed[:, w])}
    own = token // compress_rate
    assert changed == {own} | {own + offset for offset in extra_entries_touched}


def test_csa_indexer_keeps_only_readable_entries():
    """The Lightning Indexer may never name an entry the query cannot causally read.

    Entry `w` pools tokens up to `(w + 1) * compress_rate - 1`, so query `t` may read
    `(t + 1) // compress_rate` of them. Queries with fewer readable entries than `index_topk`
    get their picks padded with `-1` rather than with a real index, and the count of real picks
    has to track the readable count exactly.
    """
    prime_module = prime_attention(_CSA_LAYER)
    indexer = prime_module.compressor.indexer
    _, hidden = _hidden_states()
    packed = _packed_context(_SINGLE_DOC, torch.bfloat16)
    q_residual = prime_module.q_a_norm(prime_module.q_a_proj(hidden))

    top_k_indices = indexer(hidden, q_residual, packed)

    top_k = _ATTN["index_topk"]
    assert top_k_indices.shape == (_MODULE_BATCH, _MODULE_SEQ, top_k)
    readable = (packed.position_ids + 1) // _COMPRESS_RATE
    assert readable.max() > top_k, "config must leave the indexer something to discard"
    assert (top_k_indices < readable.unsqueeze(-1)).all(), "an unreadable entry was selected"
    kept = (top_k_indices >= 0).sum(dim=-1)
    torch.testing.assert_close(kept, readable.clamp(max=top_k).expand_as(kept))


def test_hca_compressor_masks_unreadable_entries():
    """HCA has no indexer, so its block bias is the whole gate and must be the exact closed form.

    Every readable entry is unbiased and every unreadable one is `-inf`, at the
    `(pos + 1) // rate` threshold. Asserted against the formula rather than across two runs,
    which is what makes this catch an off-by-one in the threshold itself.
    """
    prime_module = prime_attention(_HCA_LAYER)
    compressor = prime_module.compressor
    _, hidden = _hidden_states()
    packed = _packed_context(_SINGLE_DOC, torch.bfloat16)
    q_residual = prime_module.q_a_norm(prime_module.q_a_proj(hidden))

    compressed_kv, block_bias = compressor(hidden, q_residual, packed)

    n_windows = _MODULE_SEQ // _HCA_COMPRESS_RATE
    assert compressed_kv.shape == (_MODULE_BATCH, 1, n_windows, _ATTN["head_dim"])
    assert block_bias.shape == (_MODULE_BATCH, 1, _MODULE_SEQ, n_windows)
    readable = (packed.position_ids + 1) // _HCA_COMPRESS_RATE
    entries = torch.arange(n_windows, device=block_bias.device).view(1, 1, 1, -1)
    expected = torch.where(entries < readable.unsqueeze(1).unsqueeze(-1), 0.0, float("-inf"))
    torch.testing.assert_close(block_bias, expected.to(block_bias.dtype).expand_as(block_bias), rtol=0, atol=0)


def test_moe_router_scoring_and_selection_bias():
    """The router scores with `sqrt(softplus(.))`, normalizes, then scales, in that order.

    The ordering is what the second half pins: `selection_bias` steers which experts win the
    top-k but must not reach the gating values, which stay the unbiased scores. Getting that
    backwards is a one-character change in the router and would leave every weight subtly wrong.
    """
    prime_module = prime_moe()
    router = prime_module.router
    _, hidden = _moe_hidden_states()
    x = hidden.detach().reshape(-1, _MOE["hidden_size"])

    def expected_gates(indices: torch.Tensor) -> torch.Tensor:
        gathered = F.softplus(F.linear(x, router.gate.weight)).sqrt().gather(dim=1, index=indices)
        return gathered / gathered.sum(dim=-1, keepdim=True) * _MOE["routed_scaling_factor"]

    router.selection_bias.zero_()
    top_scores, indices, num_tokens_per_expert, _ = router(x)

    scores = F.softplus(F.linear(x, router.gate.weight)).sqrt()
    _, expected_indices = torch.topk(scores, _MOE["num_experts_per_tok"], dim=1)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(top_scores, expected_gates(indices), rtol=0, atol=0)
    # Normalization happens before the scale, so every token's weights sum to it.
    torch.testing.assert_close(
        top_scores.sum(dim=-1), torch.full((x.shape[0],), _MOE["routed_scaling_factor"], device=x.device)
    )
    assert num_tokens_per_expert.sum().item() == _MOE_TOKENS * _MOE["num_experts_per_tok"]

    favored = 5
    router.selection_bias[favored] = 100.0
    biased_scores, biased_indices, biased_counts, _ = router(x)

    assert not torch.equal(biased_indices, indices), "the bias must change the selection"
    assert biased_counts[favored].item() == _MOE_TOKENS, "every token must reach the favored expert"
    torch.testing.assert_close(biased_scores, expected_gates(biased_indices), rtol=0, atol=0)


def test_moe_shared_expert_clamps_the_swiglu():
    """`ClampedSwiglu` clips the gate above and the up projection both ways before the product."""
    prime_module = prime_clamped_moe()
    shared_expert = prime_module.shared_expert
    _, hidden = _moe_hidden_states()
    x = hidden.detach()

    output = shared_expert(x)

    gate, up = shared_expert.gate_proj(x), shared_expert.up_proj(x)
    limit = shared_expert.limit
    assert (gate > limit).any() and (up.abs() > limit).any(), "config must push the pre-activations past the clamp"
    clamped = shared_expert.down_proj(F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit))
    torch.testing.assert_close(output, clamped, rtol=0, atol=0)
    assert not torch.equal(output, shared_expert.down_proj(F.silu(gate) * up))


def test_moe_ignores_input_ids_when_not_hash_routed():
    """The decoder layer hands `input_ids` to every MLP, so a score-routed one must ignore them.

    Not visible at model level, where `input_ids` also drive the embeddings and so change the
    hidden states the router sees.
    """
    prime_module = prime_moe()
    _, hidden = _moe_hidden_states()

    prime_module.tokens_per_expert.zero_()
    with_ids = prime_module(hidden, input_ids=_input_ids())
    counts_with_ids = prime_module.tokens_per_expert.clone()
    prime_module.tokens_per_expert.zero_()
    without_ids = prime_module(hidden)

    # The selection is identical down to the token count per expert. The outputs only match
    # to the float32 floor: the scatter-add over experts is not bitwise reproducible.
    torch.testing.assert_close(counts_with_ids, prime_module.tokens_per_expert, rtol=0, atol=0)
    torch.testing.assert_close(with_ids, without_ids, rtol=1e-5, atol=1e-8)


# The packed module-level section. A packed row must equal running each of its documents alone:
# the trainer packs, vLLM serves each rollout alone, and a trainer that disagrees optimizes a
# model the sampler does not implement. The whole-model tests further up assert the same thing
# through the logits at a bf16 floor; these assert it on the compressor and the attention layer
# in float32, forward and backward, which is roughly a thousand times sharper.
#
# Float32 throughout. `kv_proj` sees a different number of rows packed than alone and cuBLAS is
# free to tile the two differently, so the runs can never be bit-identical; in bfloat16 that floor
# would sit around 1e-2 and swallow exactly the cross-document leakage these tests exist to catch.

# The boundary falls inside a window of both rates, so both drop tokens at it.
_MID_WINDOW_DOCS = (7, 9)
# Every document is a whole number of windows at both rates, so the per-document entry count
# equals the row-global one and only the numbering (and with it the RoPE position) moves.
_EXACT_MULTIPLE_DOCS = (8, 8)
# The first document is shorter than either rate, so it compresses to nothing while the rest of
# the row still carries entries.
_SHORT_FIRST_DOCS = (3, 13)
# Shorter than the HCA rate everywhere, so HCA compresses the whole row to zero entries.
_ALL_SHORT_DOCS = (5, 5, 6)
# One CSA entry in the first document and three in the second: `index_topk` of 2 has to discard,
# and a pick renumbered into the second document cannot coincide with its own local index.
_INDEXER_DOCS = (4, 12)

# Packed and unpacked run the same arithmetic through differently shaped matmuls, so they agree
# to the float32 rounding floor and no further. A query reading another document's tokens moves
# an output by a fraction of its own scale, orders of magnitude above this.
_PACKED_RTOL, _PACKED_ATOL = 1e-5, 1e-6
# Gradients get a bound against the tensor's own scale instead. They are sums over the whole row,
# and their near-zero entries are the ones whose summands cancelled, so an element-wise relative
# bound reads out cancellation noise rather than anything a document leak would move: one entry
# of `kv_proj`'s gradient lands 3e-4 off in relative terms while being 2e-6 off in absolute ones.
# Against the tensor's own scale the worst deviation measured over these tests is 8e-7.
_PACKED_GRAD_RTOL = 1e-5


def _entry_counts(doc_lens: tuple[int, ...], compress_rate: int) -> list[int]:
    return [length // compress_rate for length in doc_lens]


def _entry_slice(doc_lens: tuple[int, ...], compress_rate: int, index: int) -> slice:
    """Where one document's compressed entries sit, the entry axis being laid out document by
    document exactly as the token axis is."""
    return _doc_slice(tuple(_entry_counts(doc_lens, compress_rate)), index)


def _fp32_hidden_states(seq_len: int = _MODULE_SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two leaves carrying identical values, one for the packed run and one for the lone runs."""
    with torch.device("cuda"):
        hidden = torch.randn(_MODULE_BATCH, seq_len, _ATTN["hidden_size"])
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


def _assert_compress_matches_per_document(
    compressor: nn.Module, doc_lens: tuple[int, ...], compress_rate: int, packed: PackedContext
) -> None:
    """Compressing a packed row must equal compressing each of its documents on its own.

    Forward and backward both: entry `n` of the packed run must pool the same source tokens, at
    the same compress-RoPE position, as the corresponding entry of its own document's run, and
    the gradient the packed run sends into the weights must equal the one the per-document runs
    accumulate. One random weight tensor is drawn over the packed entries and sliced per
    document, so the packed loss and the summed per-document losses are literally the same
    function of the same numbers.
    """
    _assert_layout_is_consistent(packed, doc_lens, compress_rate)
    counts = _entry_counts(doc_lens, compress_rate)
    packed_input, alone_input = _fp32_hidden_states(sum(doc_lens))

    packed_entries = compressor.compress(packed_input, packed)
    assert packed_entries.shape == (_MODULE_BATCH, sum(counts), compressor.head_dim)

    with torch.device("cuda"):
        weight = torch.randn_like(packed_entries)
    (packed_entries * weight).sum().backward()
    packed_grads = _take_grads(compressor)

    for index, count in enumerate(counts):
        entries = _entry_slice(doc_lens, compress_rate, index)
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
    ("layer_idx", "compress_rate", "doc_lens", "expected_counts"),
    [
        (_CSA_LAYER, _COMPRESS_RATE, _SINGLE_DOC, [4]),
        (_CSA_LAYER, _COMPRESS_RATE, _MID_WINDOW_DOCS, [1, 2]),
        (_CSA_LAYER, _COMPRESS_RATE, _EXACT_MULTIPLE_DOCS, [2, 2]),
        (_CSA_LAYER, _COMPRESS_RATE, _SHORT_FIRST_DOCS, [0, 3]),
        (_CSA_LAYER, _COMPRESS_RATE, _ALL_SHORT_DOCS, [1, 1, 1]),
        (_HCA_LAYER, _HCA_COMPRESS_RATE, _SINGLE_DOC, [2]),
        (_HCA_LAYER, _HCA_COMPRESS_RATE, _MID_WINDOW_DOCS, [0, 1]),
        (_HCA_LAYER, _HCA_COMPRESS_RATE, _EXACT_MULTIPLE_DOCS, [1, 1]),
        (_HCA_LAYER, _HCA_COMPRESS_RATE, _SHORT_FIRST_DOCS, [0, 1]),
    ],
    ids=[
        "csa_single_document",
        "csa_mid_window_boundary",
        "csa_exact_multiple",
        "csa_short_first_document",
        "csa_every_document_short",
        "hca_single_document",
        "hca_mid_window_boundary",
        "hca_exact_multiple",
        "hca_short_first_document",
    ],
)
def test_compressor_packed_matches_per_document(layer_idx, compress_rate, doc_lens, expected_counts):
    """Each case is a length regime the per-document layout has to get right on its own.

    `single_document` is the unpacked case, which packing must leave exactly where it was.
    `mid_window_boundary` puts the split inside a window, so a row-global compression would pool
    the tail of one document with the head of the next. `exact_multiple` drops nothing, isolating
    the numbering: the second document's entries have to be rotated at its own local positions and
    its first entry marked as first, so CSA's backward-looking series is gated off instead of
    reaching into the previous document. `short_first_document` is the path that did not exist
    before per-document compression, one empty document among non-empty ones.
    `csa_every_document_short` is the degenerate end, every entry the first of its own document. A
    row that compresses to nothing at all belongs to `test_attention_survives_a_zero_entry_document`
    instead: both sides of the comparison here would be zeros, so nothing but the shapes could fail.
    """
    module = prime_attention(layer_idx, dtype=torch.float32)
    packed = _packed_context(doc_lens, torch.float32)

    assert _entry_counts(doc_lens, compress_rate) == expected_counts
    if sum(expected_counts) < sum(doc_lens) // compress_rate:
        assert packed.compression_layouts[compress_rate].entry_tok_idx.shape[0] == sum(expected_counts), (
            "a row-global compression would emit more entries than this, so the probe is not vacuous"
        )
    _assert_compress_matches_per_document(module.compressor, doc_lens, compress_rate, packed)


def test_csa_indexer_marks_every_pick_invalid_with_no_readable_entry():
    """A query whose own document compressed to nothing must pick nothing, not the row's entries.

    `_SHORT_FIRST_DOCS` at the CSA rate is the regime the whole-model tests never reach: a
    zero-entry document sitting beside one that carries three.
    """
    module = prime_attention(_CSA_LAYER, dtype=torch.float32)
    indexer = module.compressor.indexer
    hidden, _ = _fp32_hidden_states(sum(_SHORT_FIRST_DOCS))
    hidden = hidden.detach()
    q_residual = module.q_a_norm(module.q_a_proj(hidden))

    packed_picks = indexer(hidden, q_residual, _packed_context(_SHORT_FIRST_DOCS, torch.float32))

    first, second = (_doc_slice(_SHORT_FIRST_DOCS, index) for index in (0, 1))
    assert (packed_picks[:, first] < 0).all(), "a query whose document compressed to nothing was given a pick"
    assert (packed_picks[:, second] >= 0).any(), "vacuous probe: the second document picks nothing either"
    alone_picks = indexer(
        hidden[:, first], q_residual[:, first], _packed_context((_SHORT_FIRST_DOCS[0],), torch.float32)
    )
    assert (alone_picks >= 0).sum().item() == 0, "run alone the same document has nothing to pick from"


def _assert_attention_matches_per_document(module: nn.Module, doc_lens: tuple[int, ...]) -> None:
    """A packed attention layer must equal the same layer run on each document separately.

    Forward and backward, with one random weight drawn over the packed output and sliced per
    document so the two losses are the same function. The lone runs each get a single-document
    context, which is the shape a rollout arrives in at inference time.
    """
    packed_input, alone_input = _fp32_hidden_states(sum(doc_lens))
    packed = _packed_context(doc_lens, torch.float32)

    q_residual = module.q_a_norm(module.q_a_proj(packed_input.detach()))
    _, block_bias = module.compressor(packed_input.detach(), q_residual, packed)
    assert (block_bias[:, :, _doc_slice(doc_lens, 1)] == 0).any(), (
        "vacuous probe: no query of the second document reads a compressed entry"
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


def _assert_attention_is_finite(module: nn.Module, doc_lens: tuple[int, ...]) -> None:
    """Run one packed forward and backward and require every number that comes out to be finite."""
    packed_input, _ = _fp32_hidden_states(sum(doc_lens))

    output, _ = module(packed_input, packed=_packed_context(doc_lens, torch.float32))
    assert torch.isfinite(output).all(), "the attention output is not finite"

    with torch.device("cuda"):
        weight = torch.randn_like(output)
    (output * weight).sum().backward()
    assert torch.isfinite(packed_input.grad).all(), "the input gradient is not finite"
    for name, param in module.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), f"{name} received a non-finite gradient"


@pytest.mark.parametrize(
    ("layer_idx", "doc_lens"),
    [(_CSA_LAYER, _MID_WINDOW_DOCS), (_HCA_LAYER, _EXACT_MULTIPLE_DOCS)],
    ids=["csa", "hca"],
)
def test_attention_packed_matches_unpacked(layer_idx, doc_lens, _torch_rms_norm):  # noqa: F811
    """The invariant that makes the trainer agree with vLLM, one whole attention layer at a time.

    Everything the layer reads past its local window comes through the compressor, so a leaking
    entry, a misnumbered pick and a misrotated entry all show up here at once. The HCA case has no
    indexer to narrow the damage, and at rate 8 both documents own an entry, so the second one's
    has to be rotated at its own position rather than at its packed one.
    """
    _assert_attention_matches_per_document(prime_attention(layer_idx, dtype=torch.float32), doc_lens)


@pytest.mark.parametrize(
    ("layer_idx", "compress_rate", "doc_lens", "n_entries"),
    [(_CSA_LAYER, _COMPRESS_RATE, _SHORT_FIRST_DOCS, 3), (_HCA_LAYER, _HCA_COMPRESS_RATE, _ALL_SHORT_DOCS, 0)],
    ids=["csa_empty_document_beside_a_full_one", "hca_empty_row"],
)
def test_attention_survives_a_zero_entry_document(layer_idx, compress_rate, doc_lens, n_entries, _torch_rms_norm):  # noqa: F811
    """A document that compressed to nothing must not turn the softmax into NaN.

    Two degenerate shapes. In the CSA case the row still carries the second document's entries,
    so the first document's queries get a compressed logit row that is masked off in full; in the
    HCA case the whole row compressed to nothing and the block bias has zero columns, which the
    attention block still has to concatenate onto its local mask. The local window and the
    attention sink are what keep the softmax normalizable either way.
    """
    module = prime_attention(layer_idx, dtype=torch.float32)
    hidden, _ = _fp32_hidden_states(sum(doc_lens))
    hidden = hidden.detach()
    q_residual = module.q_a_norm(module.q_a_proj(hidden))

    compressed_kv, block_bias = module.compressor(hidden, q_residual, _packed_context(doc_lens, torch.float32))

    assert block_bias.shape[-1] == n_entries, "vacuous probe: the row did not compress to the expected entry count"
    if n_entries == 0:
        assert compressed_kv.shape[2] == 0
    else:
        assert (block_bias[:, :, _doc_slice(doc_lens, 0)] == float("-inf")).all(), (
            "vacuous probe: the first document was allowed to read an entry"
        )

    _assert_attention_is_finite(module, doc_lens)

    for name, param in module.compressor.named_parameters():
        # The Lightning Indexer reaches the loss only through integer top-k indices, so no
        # gradient can arrive here through attention at all.
        if "indexer" in name:
            continue
        assert param.grad is not None, f"{name} took no part in the backward"
        if n_entries == 0:
            assert not param.grad.any(), f"{name} received a non-zero gradient from an empty compression"


def test_model_segments_by_seq_lens_on_a_padded_row(_torch_rms_norm, monkeypatch):  # noqa: F811
    """`seq_lens` decides the document layout, and `position_ids` restarting does not.

    `pad_micro_batch` folds its padding into the last document: it extends `seq_lens[-1]` while
    restarting `position_ids` at 0 over the pad block, so on a padded micro-batch the two disagree
    by construction. Following the restart would cut the last rollout in two and drop the entries
    that straddle the cut, a real capability loss on every padded step; following `seq_lens` treats
    the padding as a continuation, which costs nothing, since causality keeps it away from every
    real token and it is loss-masked. This is a design decision, not an accident, so it is asserted
    directly rather than through a packing oracle.

    The positions every rotation and threshold reads come from `seq_lens` too, so the pad block
    continues its document instead of restarting. That is the same decision applied to the same
    disagreement, and it is why the restart the caller passes below is accepted rather than
    rejected by `PackedContext.check_position_ids`: padding sits mid-document, never at a start.

    Read off the width of the key stream each layer actually attends over, which is the local
    window plus that layer's compressed entries, rather than off the layout builder, so it keeps
    holding if the threading moves.
    """
    doc_len, pad_size = 7, 5
    total = doc_len + pad_size
    csa_rate = _MODEL["compress_rates"]["compressed_sparse_attention"]
    hca_rate = _MODEL["compress_rates"]["heavily_compressed_attention"]
    by_seq_lens = {
        "sliding_attention": 0,
        "compressed_sparse_attention": total // csa_rate,
        "heavily_compressed_attention": total // hca_rate,
    }
    by_restart = {
        "compressed_sparse_attention": doc_len // csa_rate + pad_size // csa_rate,
        "heavily_compressed_attention": doc_len // hca_rate + pad_size // hca_rate,
    }
    assert by_restart != {key: by_seq_lens[key] for key in by_restart}, "the two segmentations must disagree"

    recorded: list[int] = []
    real_attention = dsv4_attention.eager_attention_with_sinks

    def record(query, key, value, sinks, attention_mask, **kwargs):
        recorded.append(key.shape[2])
        return real_attention(query, key, value, sinks, attention_mask, **kwargs)

    monkeypatch.setattr(dsv4_attention, "eager_attention_with_sinks", record)

    model = get_prime_model(torch.float32)
    input_ids = torch.randint(0, _MODEL["vocab_size"], (1, total), device="cuda")
    position_ids = torch.cat([torch.arange(doc_len, device="cuda"), torch.arange(pad_size, device="cuda")])
    model(input_ids, position_ids=position_ids.unsqueeze(0), seq_lens=torch.tensor([total], device="cuda"))

    assert len(recorded) == _MODEL["num_hidden_layers"]
    for layer_idx, layer_type in enumerate(_MODEL["layer_types"]):
        expected = total + by_seq_lens[layer_type]
        assert recorded[layer_idx] == expected, (
            f"layer {layer_idx} ({layer_type}) attends over {recorded[layer_idx]} keys, not {expected}: "
            "the compressed entries follow the position_ids restart rather than seq_lens"
        )


def test_model_rejects_position_ids_that_disagree_with_seq_lens(_torch_rms_norm):  # noqa: F811
    """A sequence-global `position_ids` over a packed row is refused, not silently followed.

    Followed, it would hand `causal_threshold` a count of entries measured from the start of the
    row while the entries are numbered from the start of their own document, so the second
    document's opening query would read an entry pooling tokens at and after itself. The
    positions the layers use are derived from `seq_lens`, which makes that unreachable, and a
    caller whose own positions contradict the boundaries it asked for is working from a different
    segmentation and should hear about it.
    """
    model = get_prime_model(torch.float32)
    input_ids, _, seq_lens = _packed_inputs(_DOC_LENS)
    global_position_ids = torch.arange(sum(_DOC_LENS), device="cuda").unsqueeze(0)

    with pytest.raises(ValueError, match="restart at 0"):
        model(input_ids, position_ids=global_position_ids, seq_lens=seq_lens)


@pytest.mark.xfail(
    strict=True,
    reason="inject_prime_lm_head fills an absent position_ids with a 1-based arange; make that "
    "0-based and this passes, then drop the marker",
)
def test_model_accepts_a_call_with_no_position_ids(_torch_rms_norm):  # noqa: F811
    """Passing no `position_ids` should leave the model on the positions it derives itself.

    It does not: `inject_prime_lm_head` rebinds this model's forward and substitutes a 1-based
    `arange(1, N + 1)` before the model sees it, which is wrong for an architecture that numbers
    and rotates its compressed entries from 0, so the check refuses it. Strict, so correcting
    that default reports here as an unexpected pass rather than going unnoticed.
    """
    model = get_prime_model(torch.float32)
    input_ids, _ = _inputs()

    model(input_ids, seq_lens=_seq_lens(input_ids))
