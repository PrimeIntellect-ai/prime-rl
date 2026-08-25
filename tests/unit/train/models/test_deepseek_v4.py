import pytest
import torch
from torch import nn
from transformers.core_model_loading import revert_weight_conversion
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4CSACompressor, DeepseekV4HCACompressor
from prime_rl.trainer.models.deepseek_v4.converting_deepseek_v4 import to_on_disk_naming
from prime_rl.trainer.models.layers import norms
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

# Deliberately heterogeneous: one layer of every attention type, hash-routed bootstrap
# layers ahead of standard MoE ones, and a sliding window narrow enough that the compressed
# branches are what carries any long-range signal.
_BASE = dict(
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
    mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe", "moe"],
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
)

_BATCH, _SEQ = 2, 32


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


@pytest.fixture
def _torch_rms_norm(monkeypatch):
    """Make the shared `RMSNorm` take its PyTorch path instead of the quack kernel.

    The kernel is a project-wide choice that predates this model and drifts from HF's fp32
    reference by up to ~1e-2 in bf16, which would swamp what the V4-specific math
    contributes to the comparison.
    """
    monkeypatch.setattr(norms, "_get_quack_rmsnorm", lambda: None)


def _tid2eid(vocab_size: int, num_experts: int, top_k: int) -> torch.Tensor:
    """A frozen token id -> expert ids table, distinct experts per row as a real one has."""
    rows = [torch.randperm(num_experts)[:top_k] for _ in range(vocab_size)]
    return torch.stack(rows).to(device="cuda", dtype=torch.long)


def _randomize(model: nn.Module) -> None:
    """Draw non-degenerate values for every parameter and routing buffer.

    Norm gains default to ones and the sinks, position biases, load-balancing bias and hash
    table all default to zeros, each of which would leave the path it controls
    indistinguishable from a no-op. The position bias is drawn wide because it is a softmax
    logit over a pooling window; at the projections' std the gate would stay near uniform.
    """
    for name, param in model.named_parameters():
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
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.normal_(mean=0.0, std=0.1)
            elif name.endswith("tid2eid"):
                buffer.copy_(_tid2eid(_BASE["vocab_size"], _BASE["n_routed_experts"], _BASE["num_experts_per_tok"]))


def _configs() -> tuple[HFDeepseekV4Config, DeepseekV4Config]:
    hf_config = HFDeepseekV4Config(**_BASE)
    # Force the eager path so HF actually runs its sink softmax, and keep the compressors'
    # rolling-window caches out of a training-shaped single forward.
    hf_config._attn_implementation = "eager"
    hf_config.use_cache = False
    # The for-loop expert path keeps the routed experts in the activation dtype; the
    # grouped-mm kernel casts to bfloat16 internally and is covered in test_deepseek_v4_temp.
    return hf_config, DeepseekV4Config(**_BASE, use_grouped_mm=False)


def _on_disk_state_dict(hf_model: nn.Module) -> dict[str, torch.Tensor]:
    """An HF model's weights under the key naming a real DeepSeek V4 checkpoint uses.

    `conversion_chain` converts *on-disk* names, which for this model are not the names
    `hf_model.state_dict()` returns: transformers carries a conversion registry entry for
    deepseek_v4 and applies it inside `from_pretrained` / `save_pretrained`, so the on-disk
    names are the compact DeepSeek-native ones (`attn`, `ffn`, `wkv`, per-expert `w1`/`w2`/`w3`,
    no `model.` prefix). The trainer reads raw on-disk state dicts in `load_dcp_from_hf` and
    never goes through `from_pretrained`, so that is the naming the chain has to handle.

    `revert_weight_conversion` is transformers' own reverse pass, the one `save_pretrained`
    runs, so this stays authoritative rather than restating the mapping here.
    """
    reverted = revert_weight_conversion(hf_model, dict(hf_model.state_dict()))
    return to_on_disk_naming(reverted)


def get_model_pairs(dtype: torch.dtype = torch.bfloat16) -> tuple[nn.Module, nn.Module]:
    """Build an HF and a prime-rl model carrying identical weights."""
    hf_config, prime_config = _configs()
    with torch.device("cuda"), default_dtype(dtype):
        hf_model = HFDeepseekV4ForCausalLM._from_config(hf_config)
        prime_model = DeepseekV4ForCausalLM._from_config(prime_config)
    _randomize(hf_model)

    with torch.no_grad():
        state_dict = _on_disk_state_dict(hf_model)
        prime_state_keys = set(prime_model.state_dict())
        prime_model.convert_to_prime(state_dict)
        assert set(state_dict) == prime_state_keys, "the converted HF key set must equal prime-rl's exactly"
        prime_model.load_state_dict(state_dict)

    # Training code wraps the LM head; tests mirror that so forward takes labels/temperature.
    inject_prime_lm_head(prime_model, chunk_size=None)
    return hf_model, prime_model


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, _BASE["vocab_size"], (_BATCH, _SEQ), device="cuda")
    position_ids = torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)
    return input_ids, position_ids


def _seq_lens(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.tensor([input_ids.shape[1]], device=input_ids.device)


def _run_pair(hf_model: nn.Module, prime_model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, position_ids = _inputs()
    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids, seq_lens=_seq_lens(input_ids))

    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()
    return hf_output.logits, prime_output["logits"]


def _assert_relative(prime: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    prime, reference = prime.float(), reference.float()
    deviation = (prime - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _assert_close(
    prime_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    hf_model: nn.Module,
    prime_model: nn.Module,
    *,
    logits_rtol: float,
    grad_rtol: float,
) -> None:
    assert prime_logits.shape == (_BATCH, _SEQ, _BASE["vocab_size"])
    _assert_relative(prime_logits, hf_logits, logits_rtol, "logits")
    _assert_relative(
        prime_model.model.embed_tokens.weight.grad,
        hf_model.model.embed_tokens.weight.grad,
        grad_rtol,
        "embedding gradient",
    )


class _IdentityMLP(nn.Module):
    """Stands in for a decoder layer's MoE block: same shape in, same shape out.

    It has to swallow `input_ids` (and prime-rl's `routed_experts`): the decoder layer
    passes them to every layer, hash-routed or not.
    """

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


def _identity_attention(hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
    return hidden_states, None


def test_deepseek_v4_attn_only(_torch_rms_norm):
    hf_model, prime_model = get_model_pairs()
    for model in (hf_model, prime_model):
        for layer in model.model.layers:
            layer.mlp = _IdentityMLP()

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=0.02, grad_rtol=0.02)


def test_deepseek_v4_mlp_only(_torch_rms_norm):
    hf_model, prime_model = get_model_pairs()
    for model in (hf_model, prime_model):
        for layer in model.model.layers:
            layer.self_attn.forward = _identity_attention

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=0.02, grad_rtol=0.02)


def test_deepseek_v4(_torch_rms_norm):
    hf_model, prime_model = get_model_pairs()

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    # Loose by design, and the loosest assertion in this file. prime-rl's router scores in
    # float32 (`TokenChoiceTopKRouter` upcasts to keep the training loss from exploding)
    # while HF scores in the activation dtype, so in bfloat16 a few percent of the tokens
    # in the deeper layers pick a different expert set and their outputs then legitimately
    # diverge. `test_deepseek_v4_float32` runs the same comparison with that one difference
    # removed and holds to 1e-5; the isolation tests above carry the tight bfloat16 bound.
    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=0.2, grad_rtol=0.1)


def test_deepseek_v4_float32(_torch_rms_norm):
    """Full-model parity with the router's dtype difference removed."""
    hf_model, prime_model = get_model_pairs(dtype=torch.float32)

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=1e-5, grad_rtol=1e-5)


def test_deepseek_v4_hash_layers_route_on_token_ids():
    """The bootstrap layers read `input_ids`, so identical hidden states still route apart."""
    _, prime_model = get_model_pairs()
    hash_layers = [
        layer for layer, mlp_type in zip(prime_model.model.layers, _BASE["mlp_layer_types"]) if mlp_type == "hash_moe"
    ]
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
    _, prime_config = _configs()
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
    _, prime_config = _configs()
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


def test_deepseek_v4_conversion_matches_the_hf_key_set():
    """The converted HF checkpoint has to land on prime-rl's keys with nothing left over."""
    hf_config, prime_config = _configs()
    with torch.device("meta"):
        hf_model = HFDeepseekV4ForCausalLM._from_config(hf_config)
        prime_model = DeepseekV4ForCausalLM._from_config(prime_config)

    state_dict = _on_disk_state_dict(hf_model)
    # A real checkpoint ships multi-token-prediction heads that neither side instantiates,
    # at the top level (`mtp.0.hc_attn_base`, ...) rather than nested inside a layer.
    state_dict["mtp.0.embed.weight"] = torch.empty(0, device="meta")
    prime_model.convert_to_prime(state_dict)

    assert set(state_dict) == set(prime_model.state_dict())


def test_deepseek_v4_init_buffers_post_meta_restores_every_rotary():
    """Rotary tables are non-persistent and computed eagerly, so meta loading loses them."""
    _, prime_config = _configs()
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
# `seq_lens`, exactly as `trainer/batch.py` builds them. `DeepseekV4Model.forward` ignores
# `seq_lens` today (`modeling_deepseek_v4.py:223-228`), and the three resulting defects do not
# cost the same to fix. The sliding-window mask (defect 1) is a port regression: HF builds it
# with `create_sliding_window_causal_mask(..., position_ids=position_ids)` and recovers the
# boundaries from the restarts, so there is an upstream reference to copy. The compressors'
# coordinate mismatch (defect 2) and their boundary-straddling pooling windows (defect 3) are
# inherited from HF, which never runs this model on packed input, so those have to be designed.
#
# The assertions deliberately avoid naming which compressed entry index belongs to which
# document, and go through `forward` rather than the internals it calls. A fix that compresses
# per document renumbers the entries, and an index-based assertion would then stay red for the
# wrong reason, which `strict=True` cannot detect. They state the invariant instead: redraw the
# first document and nothing the second document reads may move, and what a query reads must not
# depend on whether its document was packed or run alone.

# Neither length is a multiple of a compress rate, so the boundary falls mid-window for both
# compressors and defect 3's blending is reached on each.
_DOC_LENS = (14, 18)
# Below the HCA compress rate of 8, where packed and unpacked agree for a reason that is not a fix.
_SHORT_DOC_LENS = (6, 6)


def _packed_inputs(doc_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One packed row: token ids, `position_ids` restarting per document, and flat `seq_lens`."""
    total = sum(doc_lens)
    input_ids = torch.randint(0, _BASE["vocab_size"], (1, total), device="cuda")
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

    compressed_kv, block_bias = compressor(hidden_states, q_residual, position_ids)
    other_hidden, other_q = _resample_first_document((hidden_states, q_residual), doc_lens)
    other_kv, other_bias = compressor(other_hidden, other_q, position_ids)

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


@pytest.mark.xfail(
    strict=True,
    reason="defect 1: build_sliding_window_mask ignores document boundaries (attention.py:85-89)",
)
def test_packed_sliding_window_mask_respects_documents(_torch_rms_norm, monkeypatch):
    """The local window bleeds across documents.

    `build_sliding_window_mask` (`attention.py:85-89`) derives its distances from
    `torch.arange(seq_len)` over the whole packed row, so a query attends to the previous
    `sliding_window` packed positions no matter which document they belong to. This hits every
    layer, sliding and compressed alike.

    Captured from the mask the model actually applies, not from the builder's signature, so the
    test stays valid whether the fix threads document boundaries into that builder or replaces it
    with HF's `create_sliding_window_causal_mask`.
    """
    recorded = []
    real_attention = dsv4_attention.eager_attention_with_sinks

    def record(query, key, value, sinks, attention_mask, **kwargs):
        recorded.append(attention_mask)
        return real_attention(query, key, value, sinks, attention_mask, **kwargs)

    monkeypatch.setattr(dsv4_attention, "eager_attention_with_sinks", record)

    _, prime_model = get_model_pairs(dtype=torch.float32)
    input_ids, position_ids, seq_lens = _packed_inputs(_DOC_LENS)
    prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)

    assert len(recorded) == _BASE["num_hidden_layers"]
    total = sum(_DOC_LENS)
    doc_ids = _doc_ids(_DOC_LENS)
    positions = position_ids[0]
    distance = positions[:, None] - positions[None, :]
    expected = (doc_ids[:, None] == doc_ids[None, :]) & (distance >= 0) & (distance < _BASE["sliding_window"])
    for layer_idx, mask in enumerate(recorded):
        # Compressed layers append their own entries as extra columns; those carry defects 2 and 3
        # and are covered separately, so only the local window is compared here.
        local = mask[0, 0, :, :total] == 0
        assert torch.equal(local, expected), f"layer {layer_idx}: the local window crosses a document boundary"


@pytest.mark.xfail(strict=True, reason="defects 1-3: a packed forward does not reproduce per-document forwards")
def test_packed_logits_match_unpacked(_torch_rms_norm):
    """The invariant that makes the trainer agree with vLLM, which serves each rollout alone.

    All three defects land here at once: the bleeding sliding window (`attention.py:85-89`), the
    compressed entries addressed in packed coordinates while their threshold counts per document
    (`attention.py:159`, `:226-232`, `:329-332`), and pooling windows that straddle boundaries
    (`attention.py:129-151`, `:301-317`).
    """
    _, prime_model = get_model_pairs(dtype=torch.float32)
    input_ids, position_ids, seq_lens = _packed_inputs(_DOC_LENS)

    packed = prime_model(input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]

    for index, length in enumerate(_DOC_LENS):
        span = _doc_slice(_DOC_LENS, index)
        alone = prime_model(
            input_ids[:, span],
            position_ids=torch.arange(length, device="cuda").unsqueeze(0),
            seq_lens=torch.tensor([length], device="cuda"),
        )["logits"]
        _assert_relative(packed[:, span], alone, 1e-4, f"document {index}")


@pytest.mark.xfail(strict=True, reason="defect 2: CSA thresholds a per-document counter against packed entries")
def test_packed_csa_reads_only_own_document_entries(_torch_rms_norm):
    """CSA points the second document's long-range pathway at the start of the row.

    `causal_threshold` returns `(position_ids + 1) // compress_rate` (`attention.py:159`), and
    `position_ids` restart per document, but entry `w` pools *packed* positions
    (`attention.py:129-151`). A query at local position 4 of the second document therefore reads
    entry 0, which pools the first document's opening tokens, while its own document's entries sit
    above the threshold and are masked off as future.
    """
    _, prime_model = get_model_pairs(dtype=torch.float32)
    _assert_reads_are_document_local(_compressor_of_type(prime_model, DeepseekV4CSACompressor), _DOC_LENS)


@pytest.mark.xfail(strict=True, reason="defect 2: HCA thresholds a per-document counter against packed entries")
def test_packed_hca_reads_only_own_document_entries(_torch_rms_norm):
    """The same coordinate mismatch in HCA, which has no indexer to narrow the damage.

    `attention.py:329-332` compares `(position_ids + 1) // compress_rate` against entries indexed
    from the start of the packed row, and every entry under that threshold is readable rather than
    a selected few. The documents here are longer than the compress rate of 8 on purpose: below it
    the threshold floors to zero and the defect hides, which
    `test_hca_inert_below_compress_rate_matches_unpacked` pins.
    """
    _, prime_model = get_model_pairs(dtype=torch.float32)
    _assert_reads_are_document_local(_compressor_of_type(prime_model, DeepseekV4HCACompressor), _DOC_LENS)


@pytest.mark.xfail(strict=True, reason="defects 2 and 3: CSA hands a query different entries packed than alone")
def test_packed_csa_selection_matches_unpacked(_torch_rms_norm):
    """What the Lightning Indexer hands a query must not depend on how the row was packed.

    The indexer masks its candidates with the same mismatched threshold (`attention.py:226-232`),
    so the second document's top-k is drawn from the first document's entries, and the entries
    themselves pool across the boundary (`attention.py:129-151`).

    This is the oracle `forward(pack([A, B])) == concat(forward(A), forward(B))` applied one level
    down, so unlike the perturbation probes above it also catches the blending, which no mask can
    repair. Compared as the gathered entry *values* per query, not as indices, since a fix that
    compresses per document renumbers the entries. It goes through `forward` rather than
    `compress` and `indexer` directly, so a fix living in `forward` reaches it.
    """
    _, prime_model = get_model_pairs(dtype=torch.float32)
    compressor = _compressor_of_type(prime_model, DeepseekV4CSACompressor)

    hidden_states, q_residual = _compressor_inputs(_DOC_LENS)
    _, position_ids, _ = _packed_inputs(_DOC_LENS)
    packed_kv, packed_bias = compressor(hidden_states, q_residual, position_ids)

    second = _doc_slice(_DOC_LENS, 1)
    length = _DOC_LENS[1]
    alone_kv, alone_bias = compressor(
        hidden_states[:, second],
        q_residual[:, second],
        torch.arange(length, device="cuda").unsqueeze(0),
    )

    packed_reads = packed_bias[0, 0, second] == 0
    assert packed_reads.any(), "vacuous probe: the second document selects no entry at all"
    for row in range(length):
        selected = packed_kv[0, 0][packed_reads[row]]
        alone_selected = alone_kv[0, 0][alone_bias[0, 0, row] == 0]
        assert selected.shape == alone_selected.shape and torch.equal(selected, alone_selected), (
            f"query {row} of the second document reads different entries packed than on its own"
        )


def test_hca_inert_below_compress_rate_matches_unpacked(_torch_rms_norm):
    """Not a defect: HCA contributes nothing either way when documents are shorter than its rate.

    The threshold `(position_ids + 1) // 8` (`attention.py:329`) floors to zero for every query in
    a 6-token document, so nothing is readable and `exp(-inf) = 0`; run alone, the document is too
    short to fill a window and there are no entries at all. Packed and unpacked agree, so this is
    a capability loss rather than a packing mismatch and must not be "fixed" here. It matters
    because a whole-model packing test built on documents this short would report green while HCA
    is broken.

    Asserted as "no query reads anything", not as an entry count: the 12-token row happens to fill
    one global window today, but a fix that compresses per document leaves zero entries, and both
    satisfy the invariant.
    """
    _, prime_model = get_model_pairs(dtype=torch.float32)
    compressor = _compressor_of_type(prime_model, DeepseekV4HCACompressor)
    assert compressor.compress_rate == 8

    hidden_states, q_residual = _compressor_inputs(_SHORT_DOC_LENS)
    _, position_ids, _ = _packed_inputs(_SHORT_DOC_LENS)
    _, packed_bias = compressor(hidden_states, q_residual, position_ids)
    assert not (packed_bias == 0).any(), "no query may read a compressed entry, so none contributes"

    for index, length in enumerate(_SHORT_DOC_LENS):
        span = _doc_slice(_SHORT_DOC_LENS, index)
        alone_kv, alone_bias = compressor(
            hidden_states[:, span],
            q_residual[:, span],
            torch.arange(length, device="cuda").unsqueeze(0),
        )
        assert alone_kv.shape[2] == 0, f"document {index} is too short to fill a window"
        assert alone_bias.shape[-1] == 0
