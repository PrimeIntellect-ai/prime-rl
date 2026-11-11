"""Load prime-rl's nanochat model files on a CPU-only env, without importing the (GPU-heavy) prime_rl package.

The three nanochat files are executed from the worktree with tiny stand-ins for the handful of prime_rl symbols they
import (base class, attention FLOPs helper, cu_seqlens helper)."""
import importlib.util
import sys
import types

import torch
from transformers import PreTrainedModel

import os
PRIME = os.environ.get("PRIME_RL_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[4]))
NANO = f"{PRIME}/src/prime_rl/trainer/models/nanochat"


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def quadratic_attention_flops(*, num_heads, qk_head_dim, v_head_dim, seq_len, attended_tokens=None):
    attended_tokens = seq_len if attended_tokens is None else attended_tokens
    return 6 * num_heads * (qk_head_dim + v_head_dim) * seq_len * attended_tokens


def get_cu_seqlens_from_seq_lens(seq_lens, total_tokens=None):
    seq_lens = seq_lens.to(dtype=torch.int32)
    cu = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32)
    cu[1:] = seq_lens.cumsum(0)
    return cu, int(seq_lens.max())


class PreTrainedModelPrimeRL(PreTrainedModel):
    pass


_loaded = None


def load_nanochat(prime=PRIME):
    global _loaded
    if _loaded is not None:
        return _loaded
    nano = f"{prime}/src/prime_rl/trainer/models/nanochat"
    for name in ("prime_rl", "prime_rl.trainer", "prime_rl.trainer.models", "prime_rl.trainer.models.layers",
                 "prime_rl.utils", "prime_rl.trainer.models.nanochat"):
        _module(name)
    _module("prime_rl.trainer.models.base", PreTrainedModelPrimeRL=PreTrainedModelPrimeRL)
    _module("prime_rl.trainer.models.layers.attn", flash_attn_2_varlen_op=None, flash_attn_3_varlen_op=None,
            quadratic_attention_flops=quadratic_attention_flops)
    _module("prime_rl.trainer.models.layers.lm_head", PrimeLmOutput=dict)
    _module("prime_rl.utils.sequence", get_cu_seqlens_from_seq_lens=get_cu_seqlens_from_seq_lens)
    mods = {}
    for short in ("configuration_nanochat", "residual", "modeling_nanochat"):
        name = f"prime_rl.trainer.models.nanochat.{short}"
        spec = importlib.util.spec_from_file_location(name, f"{nano}/{short}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[short] = mod
    _loaded = types.SimpleNamespace(**mods)
    return _loaded


def build(depth, residual=None, device="meta", vocab=32768, seq_len=2048, **kw):
    m = load_nanochat()
    dim = depth * 64
    dim = ((dim + 127) // 128) * 128
    cfg = m.configuration_nanochat.NanochatConfig(
        vocab_size=vocab, hidden_size=dim, num_hidden_layers=depth, num_attention_heads=dim // 128,
        num_key_value_heads=dim // 128, max_position_embeddings=seq_len, residual=residual or {"type": "plain"}, **kw)
    with torch.device(device):
        model = m.modeling_nanochat.NanochatForCausalLM(cfg)
    return model


def counts(model):
    """Parameter counts in nanochat's conventions plus FLOPs/token (incl. connection extras)."""
    total = sum(p.numel() for p in model.parameters())
    wte = model.model.embed_tokens.weight.numel()
    lm_head = model.lm_head.weight.numel()
    ve = sum(b.value_embed.weight.numel() for b in model.model.layers if b.value_embed is not None)
    scalars = sum(b.resid_lambda.numel() + b.x0_lambda.numel() for b in model.model.layers)
    transformer = total - wte - lm_head - ve - scalars  # includes connection params (blocks + final norm)
    return dict(total=total, wte=wte, value_embeds=ve, lm_head=lm_head, transformer_matrices=transformer,
                scalars=scalars, scaling=transformer + lm_head, flops_per_token=model.estimate_flops())
