"""Shared fixtures and prime-rl-only builders for the whole-model DeepSeek V4 tests.

Imported by `test_deepseek_v4.py`, by its HF-oracle counterpart `test_deepseek_v4_hf.py`, and by
`test_deepseek_v4_dequantize_e2e_hf.py`. Nothing here needs `transformers.models.deepseek_v4`, so
the HF-free half of the suite keeps running under the pinned transformers version.
"""

import pytest
import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.layers import norms
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

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
    num_hash_layers=2,
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
            # HF and prime-rl both hang the aux-loss-free load-balancing bias off the router,
            # under different names. Draw whichever this model carries.
            if name.endswith("e_score_correction_bias") or name.endswith("selection_bias"):
                buffer.normal_(mean=0.0, std=0.1)
            elif name.endswith("tid2eid"):
                buffer.copy_(_tid2eid(_BASE["vocab_size"], _BASE["n_routed_experts"], _BASE["num_experts_per_tok"]))


def _to_on_disk_naming(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""`save_pretrained`'s key naming -> the naming a real DeepSeek V4 checkpoint ships.

    `transformers`' deepseek_v4 conversion registry mis-reverts four key families: its
    `^embed\.weight$` / `^hc_head_*$` patterns never match `state_dict()` keys, which carry the
    `model.` prefix that on-disk names do not, and its broad `.norm.` rule (meant for the
    compressor's norm) turns attention's `kv_norm` into `norm`. Repaired here so
    `conversion_chain` is exercised against the naming a real checkpoint actually has.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key.removeprefix("model.")
        new_key = new_key.replace("embed_tokens.weight", "embed.weight")
        new_key = new_key.replace("hc_head.hc_fn", "hc_head_fn")
        new_key = new_key.replace("hc_head.hc_base", "hc_head_base")
        new_key = new_key.replace("hc_head.hc_scale", "hc_head_scale")
        new_key = new_key.replace(".attn.norm.weight", ".attn.kv_norm.weight")
        renamed[new_key] = tensor
    return renamed


def _prime_config() -> DeepseekV4Config:
    # The for-loop expert path keeps the routed experts in the activation dtype; the
    # grouped-mm kernel casts to bfloat16 internally and is covered in test_deepseek_v4_temp.
    return DeepseekV4Config(**_BASE)


def get_prime_model(dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """A prime-rl model with non-degenerate weights and the LM head training code wraps it in."""
    with torch.device("cuda"), default_dtype(dtype):
        model = DeepseekV4ForCausalLM._from_config(_prime_config())
    _randomize(model)
    inject_prime_lm_head(model, chunk_size=None)
    return model


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
