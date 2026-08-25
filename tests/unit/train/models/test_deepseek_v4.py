import pytest
import torch
from torch import nn
from transformers.core_model_loading import revert_weight_conversion
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
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
