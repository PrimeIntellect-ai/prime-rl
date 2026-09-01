"""Shared fixtures, configs and prime-rl-only builders for the DeepSeek V4 tests.

Two scales live here. `_MODEL` builds the whole assembled model, at `_MODEL_BATCH`/`_MODEL_SEQ`;
`_HC`, `_ATTN` and `_MOE` build one mechanism at a time, at the smaller
`_MODULE_BATCH`/`_MODULE_SEQ`. The two sequence lengths are not interchangeable: every
`entry_pos` literal in the packed tests is written against `_MODULE_SEQ`, and that value is what
makes `index_topk` smaller than the entry count and leaves a gap between the sliding window and
HCA's first readable entry.

Nothing here needs `transformers.models.deepseek_v4`, which the pinned transformers version does
not ship, and nothing touches CUDA at import time, so `test_deepseek_v4_cpu.py` can import it.
"""

import pytest
import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers import norms
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
from prime_rl.utils.utils import default_dtype

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

_HC = dict(
    hidden_size=128,
    num_hidden_layers=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
)

_ATTN = dict(
    hidden_size=128,
    num_hidden_layers=4,
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
        "sliding_attention",
    ],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    index_n_heads=4,
    index_head_dim=24,
    # Smaller than the four compressed entries a 16-token sequence yields, so the
    # Lightning Indexer's selection has to actually discard some of them.
    index_topk=2,
    rms_norm_eps=1e-6,
)

_MOE = dict(
    hidden_size=64,
    num_hidden_layers=2,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    # The real default, which leaves the clamp idle on the spread `_randomize` draws. That is
    # deliberate: `GroupedExperts` runs the routed experts in bfloat16, and next to a saturating
    # clamp bf16 rounding flips which entries get clipped, so a clipped entry's gradient jumps
    # between `silu'(gate) * up` and exactly zero. `_CLAMPED_MOE` covers the clamp instead.
    swiglu_limit=10.0,
    num_hash_layers=0,
    rms_norm_eps=1e-6,
)

# Small enough that the parameter spread `_randomize` draws actually reaches the clamp. Only the
# shared expert is compared under it, and that path stays in float32, so the comparison is exact
# and the bf16 boundary flipping described on `_MOE` does not arise.
_CLAMPED_MOE = dict(_MOE, swiglu_limit=0.1)

# A vocabulary small enough that the batch hits most rows of the hash table several times.
_HASH_MOE = dict(_MOE, num_hash_layers=1, vocab_size=16)

_MODEL_BATCH, _MODEL_SEQ = 2, 32
_MODULE_BATCH, _MODULE_SEQ = 2, 16
_MOE_TOKENS = _MODULE_BATCH * _MODULE_SEQ

_SLIDING_LAYER, _CSA_LAYER, _HCA_LAYER = 0, 1, 2
_HASH_LAYER = 0
_COMPRESS_RATE = _ATTN["compress_rates"]["compressed_sparse_attention"]
_HCA_COMPRESS_RATE = _ATTN["compress_rates"]["heavily_compressed_attention"]

# One document filling the row: the unpacked case, and the shape a rollout arrives in at
# inference time.
_SINGLE_DOC = (_MODULE_SEQ,)


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
                # Sized off the owning router rather than off a config constant, so this works
                # for both the whole model and a standalone hash-routed MoE block.
                router = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
                buffer.copy_(_tid2eid(buffer.shape[0], router.num_experts, router.top_k))


def _prime_config() -> DeepseekV4Config:
    return DeepseekV4Config(**_MODEL)


def get_prime_model(dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """A prime-rl model with non-degenerate weights and the LM head training code wraps it in."""
    with torch.device("cuda"), default_dtype(dtype):
        model = DeepseekV4ForCausalLM._from_config(_prime_config())
    _randomize(model)
    inject_prime_lm_head(model, chunk_size=None)
    return model


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, _MODEL["vocab_size"], (_MODEL_BATCH, _MODEL_SEQ), device="cuda")
    position_ids = torch.arange(_MODEL_SEQ, device="cuda").unsqueeze(0).expand(_MODEL_BATCH, -1)
    return input_ids, position_ids


def _seq_lens(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.tensor([input_ids.shape[1]], device=input_ids.device)


def _assert_relative(prime: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    prime, reference = prime.float(), reference.float()
    deviation = (prime - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _hidden_streams() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        streams = torch.randn(_MODULE_BATCH, _MODULE_SEQ, _HC["hc_mult"], _HC["hidden_size"])
    return streams.clone().requires_grad_(True), streams.clone().requires_grad_(True)


def prime_hyper_connection() -> nn.Module:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        module = DeepseekV4HyperConnection(DeepseekV4Config(**_HC))
    _randomize(module)
    return module


def prime_attention_config() -> DeepseekV4Config:
    return DeepseekV4Config(**_ATTN)


def _position_ids() -> torch.Tensor:
    return torch.arange(_MODULE_SEQ, device="cuda").unsqueeze(0).expand(_MODULE_BATCH, -1)


def _hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hidden = torch.randn(_MODULE_BATCH, _MODULE_SEQ, _ATTN["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _position_embeddings(
    position_ids: torch.Tensor | None = None, dtype: torch.dtype = torch.bfloat16
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The `main` and `compress` rotary tables for `position_ids`, defaulting to the unpacked row.

    `dtype` is the tables' and has to match the module the caller runs, so the fp32 packing
    comparisons are not silently handed bf16 cosines.
    """
    position_ids = _position_ids() if position_ids is None else position_ids
    prime_config = prime_attention_config()
    with torch.device("cuda"), default_dtype(dtype):
        rotary = DeepseekV4RotaryEmbedding(prime_config)
        probe = torch.zeros(*position_ids.shape, _ATTN["hidden_size"])
    return {rope_type: rotary(probe, position_ids, rope_type) for rope_type in ("main", "compress")}


def prime_attention(layer_idx: int = _SLIDING_LAYER, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    with torch.device("cuda"), default_dtype(dtype):
        module = DeepseekV4Attention(prime_attention_config(), layer_idx=layer_idx)
    _randomize(module)
    return module


def _moe_hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"):
        hidden = torch.randn(_MODULE_BATCH, _MODULE_SEQ, _MOE["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _input_ids() -> torch.Tensor:
    return torch.randint(_HASH_MOE["vocab_size"], (_MODULE_BATCH, _MODULE_SEQ), device="cuda")


def prime_moe() -> nn.Module:
    """A float32 MoE block with non-degenerate weights, for the reason `_MOE` documents."""
    with torch.device("cuda"):
        module = DeepseekV4MoE(DeepseekV4Config(**_MOE), layer_idx=0)
    _randomize(module)
    return module


def prime_clamped_moe() -> nn.Module:
    """A MoE block whose SwiGLU clamp actually bites, for the shared expert's clamp test."""
    with torch.device("cuda"):
        module = DeepseekV4MoE(DeepseekV4Config(**_CLAMPED_MOE), layer_idx=0)
    _randomize(module)
    return module


def prime_hash_moe() -> nn.Module:
    """The same, hash-routed. `_randomize` fills `tid2eid`, which starts all-zero and would
    otherwise send every token to expert 0."""
    with torch.device("cuda"):
        module = DeepseekV4MoE(DeepseekV4Config(**_HASH_MOE), layer_idx=_HASH_LAYER)
    _randomize(module)
    return module


def _cu_seqlens(doc_lens: tuple[int, ...]) -> torch.Tensor:
    return get_cu_seqlens_from_seq_lens(torch.tensor(doc_lens, device="cuda"), total_tokens=sum(doc_lens))[0]


def _packed_position_ids(doc_lens: tuple[int, ...], batch: int = _MODULE_BATCH) -> torch.Tensor:
    positions = torch.cat([torch.arange(length, device="cuda") for length in doc_lens])
    return positions.unsqueeze(0).expand(batch, -1)


def _compress_rates(module: nn.Module) -> set[int]:
    """The rates an attention layer needs a layout for: its compressor's, or none at all."""
    return set() if module.compressor is None else {module.compressor.compress_rate}


def _packed_context(module: nn.Module, doc_lens: tuple[int, ...], dtype: torch.dtype) -> PackedContext:
    """The context `DeepseekV4Model` would hand `module` for a row laid out as `doc_lens`.

    `_SINGLE_DOC` gives back the single-document context, which is what the unpacked half of a
    packing comparison runs at. `dtype` is the mask's, and has to be the one the caller runs at.
    """
    return PackedContext.build(
        cu_seqlens=_cu_seqlens(doc_lens),
        position_ids=_packed_position_ids(doc_lens),
        total_tokens=sum(doc_lens),
        compress_rates=_compress_rates(module),
        sliding_window=_ATTN["sliding_window"],
        dtype=dtype,
        device=torch.device("cuda"),
    )
