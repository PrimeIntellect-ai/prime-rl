import re

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.deepseek_v4_markers import requires_datacenter_gpu

# Every layer is built through `DeepseekV4Attention.__init__`, which refuses to construct without
# the kernel it dispatches to, so the whole file needs tilelang.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        dsv4_attention.dsv4_sparse_attn is None,
        reason="the fused sparse attention kernel did not import; tilelang ships in the `gpu` extra, on linux only",
    ),
]

# Deliberately heterogeneous: one layer of every attention type, hash-routed bootstrap
# layers ahead of standard MoE ones, and a sliding window narrow enough that the compressed
# branches are what carries any long-range signal.
MODEL = dict(
    vocab_size=64,
    hidden_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=5,
    # The smallest head count `DeepseekV4Attention.__init__` accepts: the kernel's tiler pads the
    # head axis to a power of two and its backward GEMM needs 32 rows.
    num_attention_heads=32,
    num_key_value_heads=1,
    dsa_backend="tilelang",
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
    # The real V4-Flash Lightning Indexer shapes. `fp8_indexer` is the only indexer path and it
    # does `tl.arange(0, index_head_dim)`, so the dimension has to be a power of two.
    index_n_heads=64,
    index_head_dim=128,
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

# Shared: model construction only writes transformers' private `_attn_implementation_internal`
# and `_experts_implementation_internal`, idempotently, so deep-copy before any other mutation.
MODEL_CONFIG = DeepseekV4Config(**MODEL)

# The Lightning Indexer scores one packed row, which makes every batch axis here 1.
BATCH = 1
MODEL_SEQ = 32

CSA_LAYER, HCA_LAYER = 1, 2
COMPRESS_RATE = MODEL["compress_rates"]["compressed_sparse_attention"]
HCA_COMPRESS_RATE = MODEL["compress_rates"]["heavily_compressed_attention"]


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


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


def get_prime_model(dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """A prime-rl model with non-degenerate weights and the LM head training code wraps it in."""
    with torch.device("cuda"), default_dtype(dtype):
        model = DeepseekV4ForCausalLM._from_config(MODEL_CONFIG)
    _randomize(model)
    inject_prime_lm_head(model, chunk_size=None)
    return model


def prime_attention(layer_idx: int, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """One attention layer of the same config the whole-model tests use.

    `DeepseekV4Attention` reads no MoE or hyper-connection field, so the layer this builds is
    bit-identical to one from a config carrying only the attention keys.
    """
    with torch.device("cuda"), default_dtype(dtype):
        module = DeepseekV4Attention(MODEL_CONFIG, layer_idx, DeepseekV4RotaryEmbedding(MODEL_CONFIG))
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


def _packed_context(doc_lens: tuple[int, ...], dtype: torch.dtype) -> PackedContext:
    """The context `DeepseekV4Model` would hand its attention layers for a row of `doc_lens`.

    A single-element `doc_lens` gives back the single-document context, which is what the unpacked
    half of a packing comparison runs at. `dtype` is the default dtype the rotary embedding is built
    under; the RoPE tables themselves are always fp32.
    """
    with torch.device("cuda"), default_dtype(dtype):
        rotary = DeepseekV4RotaryEmbedding(MODEL_CONFIG)
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(doc_lens, device="cuda"),
        device=torch.device("cuda"),
    )


@requires_datacenter_gpu
def test_deepseek_v4_hash_layers_route_on_token_ids():
    """The bootstrap layers read `input_ids`, so identical hidden states still route apart."""
    prime_model = get_prime_model()
    hash_layers = prime_model.model.layers[: MODEL["num_hash_layers"]]
    assert hash_layers, "config must contain a hash-routed layer"

    counts = []
    for token_id in (0, 1):
        input_ids = torch.full((BATCH, MODEL_SEQ), token_id, device="cuda", dtype=torch.long)
        for layer in hash_layers:
            layer.mlp.tokens_per_expert.zero_()
        position_ids, seq_lens = _single_doc(input_ids)
        prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)
        counts.append(torch.stack([layer.mlp.tokens_per_expert.clone() for layer in hash_layers]))

    table = hash_layers[0].mlp.router.tid2eid
    assert set(table[0].tolist()) != set(table[1].tolist()), "the two table rows must differ for this to bite"
    assert not torch.equal(counts[0], counts[1]), "a hash layer must route the two token ids to different experts"
    expected = torch.zeros_like(counts[0][0])
    expected[table[0]] = BATCH * MODEL_SEQ
    torch.testing.assert_close(counts[0][0], expected)


@requires_datacenter_gpu
def test_deepseek_v4_backward():
    """Every parameter that can train does, and the Lightning Indexer's still cannot."""
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = DeepseekV4ForCausalLM(MODEL_CONFIG)
    _randomize(model)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, MODEL["vocab_size"], (BATCH, MODEL_SEQ), device="cuda")
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
    model = DeepseekV4ForCausalLM(MODEL_CONFIG).to("cuda")
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
        table = _tid2eid(MODEL["vocab_size"], MODEL["n_routed_experts"], MODEL["num_experts_per_tok"])
        with torch.no_grad():
            layer.mlp.router.tid2eid.copy_(table)
        tables[layer_idx] = table
    assert len(tables) == MODEL["num_hash_layers"], "config must contain hash-routed layers"
    return tables


def test_deepseek_v4_hash_table_survives_the_load_path(tmp_path, monkeypatch):
    """The frozen table has to reach `mlp.router.tid2eid` from the name a real checkpoint ships.

    It is the one buffer no `init_weights` can reconstruct: an all-zero table is a valid tensor
    that routes every token to expert 0, so a rename anywhere along the loading path degrades the
    model in silence rather than raising. `test_deepseek_v4_weight_conversion_roundtrip` only ever
    roundtrips the zeros the constructor leaves behind, and `VLLM_MAPPED_NAMES` pins the on-disk
    name without saying where it lands, so this carries real values across the whole path.

    Three things in sequence, because they are three links in one chain. The conversion emits only
    the bootstrap layers' tables, under the on-disk name; reversing it lands them back on
    `mlp.router.tid2eid`; and nothing between `dcp_load` and the end of loading resets what the
    checkpoint supplied. `init_buffers_post_meta` walks every MoE and runs just before the load,
    so getting it to reset one buffer too many is an easy mistake with no symptom other than every
    bootstrap token routing to expert 0.
    """
    model = DeepseekV4ForCausalLM(MODEL_CONFIG).to("cuda")
    tables = _fill_hash_tables(model)

    state_dict = model.convert_to_hf(dict(model.state_dict()))

    assert {key for key in state_dict if key.endswith("tid2eid")} == {
        f"layers.{layer_idx}.ffn.gate.tid2eid" for layer_idx in tables
    }, "only the bootstrap layers carry a table, under the name the real checkpoint ships"
    for layer_idx, table in tables.items():
        assert torch.equal(state_dict[f"layers.{layer_idx}.ffn.gate.tid2eid"], table)

    model.convert_to_prime(state_dict)
    reloaded = DeepseekV4ForCausalLM(MODEL_CONFIG).to("cuda")
    reloaded.load_state_dict(state_dict)
    for layer_idx, table in tables.items():
        assert torch.equal(reloaded.model.layers[layer_idx].mlp.router.tid2eid, table)

    with torch.device("meta"):
        meta_model = DeepseekV4ForCausalLM(MODEL_CONFIG)
    expected = tables[0]

    def fake_dcp_load(state_dict, storage_reader=None, process_group=None):
        buffer = state_dict["model.layers.0.mlp.router.tid2eid"]
        buffer.copy_(expected.to(device=buffer.device))

    monkeypatch.setattr("prime_rl.trainer.model.dcp_load", fake_dcp_load)
    monkeypatch.setattr("prime_rl.trainer.model.load_state_dict_keys", lambda path: meta_model.state_dict().keys())
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)

    load_dcp_from_hf(meta_model, ModelConfig(name=str(tmp_path)))

    torch.testing.assert_close(meta_model.model.layers[0].mlp.router.tid2eid, expected)


# What vLLM's DeepSeek V4 loader sees, with the layer and expert indices folded away. Written
# out rather than derived: this file has no independent implementation to derive it from, and a
# rename on either side of the boundary is exactly what it exists to catch.
VLLM_MAPPED_NAMES = {
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
        model = DeepseekV4ForCausalLM._from_config(MODEL_CONFIG)
    on_disk_state_dict = model.convert_to_hf(dict(model.state_dict()))
    assert on_disk_state_dict, "vacuous probe: the model produced no weights to map"

    # Both expert dtypes, since the mapper is built per dtype and only one of them ships fp4.
    for expert_dtype in ("fp8", "fp4"):
        mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
        mapped = {re.sub(r"\.\d+\.", ".{i}.", mapper._map_name(key)) for key in on_disk_state_dict}
        assert mapped == VLLM_MAPPED_NAMES, (
            f"{expert_dtype}: unexpected {sorted(mapped - VLLM_MAPPED_NAMES)}, "
            f"missing {sorted(VLLM_MAPPED_NAMES - mapped)}"
        )


def test_deepseek_v4_init_buffers_post_meta_restores_every_rotary():
    """Rotary tables are non-persistent and computed eagerly, so meta loading loses them."""
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(MODEL_CONFIG)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    reference = 1.0 / (MODEL_CONFIG.rope_theta ** (torch.arange(0, 16, 2, device="cuda", dtype=torch.float) / 16))
    torch.testing.assert_close(model.model.rotary_emb.main_inv_freq, reference)
    compressors = [layer.self_attn.compressor for layer in model.model.layers if layer.self_attn.compressor]
    assert compressors, "config must contain a compressed attention layer"
    for compressor in compressors:
        assert torch.isfinite(compressor.rotary_emb.compress_inv_freq).all()
        # The compress branch runs a different base, so it must not collapse onto `main`.
        assert not torch.equal(compressor.rotary_emb.compress_inv_freq, reference)


# Everything below exercises a *packed* batch: rollouts concatenated into one row, `position_ids`
# restarting per document, per-document lengths handed over as `seq_lens`, exactly as
# `trainer/batch.py` builds them. A packed row must equal running each document alone, because
# that is how vLLM serves them.

# The compressor case runs in float32. `kv_proj` sees a different number of rows packed than
# alone and cuBLAS may tile the two differently, so they never match bit for bit, and in bfloat16
# that floor would swallow the cross-document leakage the test exists to catch.
PACKED_RTOL, PACKED_ATOL = 1e-5, 1e-6
# Gradients are bounded against the tensor's own scale instead: they are sums over the whole row,
# so their near-zero entries are the ones whose summands cancelled, and an element-wise relative
# bound would read out that cancellation noise rather than a document leak.
PACKED_GRAD_RTOL = 1e-5

# One row folding together every length regime the per-document layout has to get right: 3
# compresses to nothing, 8 is a whole number of windows at both rates, and 13 ends mid-window at
# both. At both rates the per-document entry count lands below the row-global one, so neither
# case is vacuous.
MIXED_DOCS = (3, 8, 13)


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
        hidden = torch.randn(BATCH, seq_len, MODEL["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _take_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    """Detach whatever gradients have accumulated and clear them for the next run."""
    grads = {name: None if param.grad is None else param.grad.clone() for name, param in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    return grads


def _compare_accumulated_grads(
    module: nn.Module, expected: dict[str, torch.Tensor | None], rtol: float = PACKED_GRAD_RTOL
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


@pytest.mark.parametrize(
    ("layer_idx", "compress_rate", "expected_counts"),
    [(CSA_LAYER, COMPRESS_RATE, [0, 2, 3]), (HCA_LAYER, HCA_COMPRESS_RATE, [0, 1, 1])],
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

    `MIXED_DOCS` folds every length regime that matters into one row; see its definition.
    """
    module = prime_attention(layer_idx, dtype=torch.float32)
    compressor = module.compressor
    doc_lens = MIXED_DOCS
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
    assert packed_entries.shape == (BATCH, sum(counts), compressor.head_dim)

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
        assert alone.shape == (BATCH, count, compressor.head_dim), f"document {index} compressed to the wrong count"
        torch.testing.assert_close(
            packed_entries[:, entries],
            alone,
            rtol=PACKED_RTOL,
            atol=PACKED_ATOL,
            msg=lambda m, i=index: f"document {i} compresses differently packed than alone: {m}",
        )
        (alone * weight[:, entries]).sum().backward()

    _compare_accumulated_grads(compressor, packed_grads)
    torch.testing.assert_close(alone_input.grad, packed_input.grad, rtol=PACKED_RTOL, atol=PACKED_ATOL)
