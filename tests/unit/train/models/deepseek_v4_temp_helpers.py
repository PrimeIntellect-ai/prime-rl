"""Shared fixtures, configs and prime-rl-only builders for the DeepSeek V4 module tests.

Imported by `test_deepseek_v4_temp.py` and by its HF-oracle counterpart
`test_deepseek_v4_temp_hf.py`. Nothing here needs `transformers.models.deepseek_v4`, so the
HF-free half of the suite keeps running under the pinned transformers version.
"""

import pytest
import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers import norms
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
from prime_rl.utils.utils import default_dtype

_BASE = dict(
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

_BATCH, _SEQ = 2, 16
_SLIDING_LAYER, _CSA_LAYER, _HCA_LAYER = 0, 1, 2
_COMPRESS_RATE = _ATTN["compress_rates"]["compressed_sparse_attention"]
_HCA_COMPRESS_RATE = _ATTN["compress_rates"]["heavily_compressed_attention"]


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


@pytest.fixture
def _torch_rms_norm(monkeypatch):
    """Make the shared `RMSNorm` take its PyTorch path instead of the quack kernel.

    The kernel is a project-wide choice that predates this model and drifts from HF's
    fp32 reference by up to ~1e-2 in bf16, which would swamp everything the V4-specific
    math contributes. Disabling it is what lets the parity assertions stay exact.
    """
    monkeypatch.setattr(norms, "_get_quack_rmsnorm", lambda: None)


def _randomize(module: nn.Module) -> None:
    """Draw non-degenerate values for every parameter.

    These modules allocate with `torch.empty`, so a test must fill them. `init_weights`
    zeros the biases and ones the scales, which would leave those paths untested, hence
    the explicit spread here.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("scale"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            else:
                param.normal_(mean=0.0, std=0.02)


def _hidden_streams():
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        streams = torch.randn(_BATCH, _SEQ, _BASE["hc_mult"], _BASE["hidden_size"])
    return streams.clone().requires_grad_(True), streams.clone().requires_grad_(True)


def prime_hyper_connection() -> nn.Module:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        module = DeepseekV4HyperConnection(DeepseekV4Config(**_BASE))
    _randomize(module)
    return module


def _randomize_attention(module: nn.Module) -> None:
    """Draw non-degenerate values for every attention parameter.

    Norm gains default to ones, the sinks and the compressors' position biases to zeros,
    which would leave all three paths indistinguishable from an identity, hence the
    explicit spread. The position bias is drawn wide because it is a softmax logit: at the
    projections' std it would leave the pooling gate all but uniform.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name == "sinks" or name.endswith("position_bias"):
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)


def prime_attention_config() -> DeepseekV4Config:
    return DeepseekV4Config(**_ATTN)


def _position_ids() -> torch.Tensor:
    return torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)


def _hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hidden = torch.randn(_BATCH, _SEQ, _ATTN["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _position_embeddings() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    prime_config = prime_attention_config()
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        rotary = DeepseekV4RotaryEmbedding(prime_config)
        probe = torch.zeros(_BATCH, _SEQ, _ATTN["hidden_size"])
    position_ids = _position_ids()
    return {rope_type: rotary(probe, position_ids, rope_type) for rope_type in ("main", "compress")}


def prime_attention(layer_idx: int = _SLIDING_LAYER) -> nn.Module:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        module = DeepseekV4Attention(prime_attention_config(), layer_idx=layer_idx)
    _randomize_attention(module)
    return module


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
    # between `silu'(gate) * up` and exactly zero. Measured against HF that swings individual
    # routed-expert gradients by 39%, against 0.66% with the clamp idle, which would leave the
    # parity comparisons unable to see anything else. `_CLAMPED_MOE` covers the clamp instead.
    swiglu_limit=10.0,
    num_hash_layers=0,
    rms_norm_eps=1e-6,
)
_MOE_TOKENS = _BATCH * _SEQ


def _moe_hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("cuda"):
        hidden = torch.randn(_BATCH, _SEQ, _MOE["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


# A vocabulary small enough that a 32-token batch hits most rows of the table several times.
# Small enough that the parameter spread `_randomize` draws actually reaches the clamp. Only the
# shared expert is compared under it, and that path stays in float32, so the comparison is exact
# and the bf16 boundary flipping described on `_MOE` does not arise.
_CLAMPED_MOE = dict(_MOE, swiglu_limit=0.1)

_HASH_MOE = dict(_MOE, num_hash_layers=1, vocab_size=16)
_HASH_LAYER = 0


def _tid2eid() -> torch.Tensor:
    """A frozen token id -> expert ids table, distinct experts per row as a real one has."""
    rows = [
        torch.randperm(_HASH_MOE["n_routed_experts"])[: _HASH_MOE["num_experts_per_tok"]]
        for _ in range(_HASH_MOE["vocab_size"])
    ]
    return torch.stack(rows).to(device="cuda", dtype=torch.long)


def _input_ids() -> torch.Tensor:
    return torch.randint(_HASH_MOE["vocab_size"], (_BATCH, _SEQ), device="cuda")


def prime_moe() -> nn.Module:
    """A float32 MoE block with non-degenerate weights, for the reason `_MOE` documents."""
    with torch.device("cuda"):
        module = DeepseekV4MoE(DeepseekV4Config(**_MOE), layer_idx=0)
    _randomize(module)
    # The aux-loss-free load-balancing bias is a buffer, so `_randomize` leaves it at zero and
    # the biased selection path would go untested.
    with torch.no_grad():
        module.router.selection_bias.normal_(mean=0.0, std=0.1)
    return module


def prime_clamped_moe() -> nn.Module:
    """A MoE block whose SwiGLU clamp actually bites, for the shared expert's clamp test."""
    with torch.device("cuda"):
        module = DeepseekV4MoE(DeepseekV4Config(**_CLAMPED_MOE), layer_idx=0)
    _randomize(module)
    return module


def prime_hash_moe() -> nn.Module:
    """The same, hash-routed. The table starts all-zero, which would send every token to
    expert 0, so it is drawn here."""
    with torch.device("cuda"):
        module = DeepseekV4MoE(DeepseekV4Config(**_HASH_MOE), layer_idx=_HASH_LAYER)
    _randomize(module)
    with torch.no_grad():
        module.tid2eid.copy_(_tid2eid())
    return module


# One document filling the row: the unpacked case, and the shape a rollout arrives in at
# inference time.
_SINGLE_DOC = (_SEQ,)


def _cu_seqlens(doc_lens: tuple[int, ...]) -> torch.Tensor:
    return get_cu_seqlens_from_seq_lens(torch.tensor(doc_lens, device="cuda"), total_tokens=sum(doc_lens))[0]


def _packed_position_ids(doc_lens: tuple[int, ...]) -> torch.Tensor:
    positions = torch.cat([torch.arange(length, device="cuda") for length in doc_lens])
    return positions.unsqueeze(0).expand(_BATCH, -1)


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
