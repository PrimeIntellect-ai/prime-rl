"""Shared eager forward arithmetic for the dense Qwen3 alignment experiment."""

from functools import lru_cache
from types import MethodType

import torch
from torch import Tensor, nn
from vllm.model_executor.layers.batch_invariant import log_softmax, matmul_persistent, rms_norm_batch_invariant


def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return matmul_persistent(x.reshape(-1, x.shape[-1]), weight.T, bias).reshape(*x.shape[:-1], weight.shape[0])


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return linear_forward(x, weight, bias)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, grad.shape[-1])
        return (
            (flat_grad @ weight).reshape_as(x),
            flat_grad.T @ x.reshape(-1, x.shape[-1]),
            flat_grad.sum(0) if ctx.has_bias else None,
        )


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rms_norm_batch_invariant(x, weight, eps)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, weight = ctx.saved_tensors
        xf = x.float()
        rstd = (xf.square().mean(-1, keepdim=True) + ctx.eps).rsqrt()
        normalized = xf * rstd
        weighted_grad = grad.float() * weight.float()
        dx = (weighted_grad - normalized * (weighted_grad * normalized).mean(-1, keepdim=True)) * rstd
        dw = (grad.float() * normalized).reshape(-1, weight.numel()).sum(0)
        return dx.to(x.dtype), dw.to(weight.dtype), None


class _LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:
        output = log_softmax(logits.float())
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad: Tensor):
        (output,) = ctx.saved_tensors
        return grad - output.exp() * grad.sum(-1, keepdim=True)


@lru_cache(maxsize=8)
def rope_table(dim: int, length: int, base: float) -> Tensor:
    with torch.device("cpu"):
        inverse = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        frequencies = torch.outer(torch.arange(length, dtype=torch.float32), inverse)
        return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


def rotate(q: Tensor, k: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
    from vllm import _custom_ops as ops

    head_dim = q.shape[-1]
    sequence = q.shape[-2]
    query = q.transpose(1, 2).reshape(sequence, -1).contiguous().clone()
    key = k.transpose(1, 2).reshape(sequence, -1).contiguous().clone()
    positions = torch.arange(sequence, device=q.device)
    ops.rotary_embedding(positions, query, key, head_dim, cache, True)
    return query.reshape(1, sequence, -1, head_dim).transpose(1, 2), key.reshape(1, sequence, -1, head_dim).transpose(
        1, 2
    )


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, cache: Tensor):
        ctx.save_for_backward(cache)
        return rotate(q, k, cache)

    @staticmethod
    def backward(ctx, dq: Tensor, dk: Tensor):
        (cache,) = ctx.saved_tensors
        cosine, sine = cache.chunk(2, dim=-1)
        dq, dk = rotate(dq, dk, torch.cat((cosine, -sine), dim=-1))
        return dq, dk, None


def _linear_forward(self, x):
    return _Linear.apply(x, self.weight, self.bias)


def _norm_forward(self, x):
    return _Norm.apply(x, self.weight, self.variance_epsilon)


def _rope_forward(self, x, position_ids):
    key = (x.device, x.dtype)
    if self._alignment_cache_key != key:
        self._alignment_cache = self._alignment_cpu_cache.to(device=x.device, dtype=x.dtype)
        self._alignment_cache_key = key
    values = self._alignment_cache[position_ids]
    cosine, sine = values.chunk(2, dim=-1)
    return torch.cat((cosine, cosine), -1), torch.cat((sine, sine), -1)


def _apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if q.shape[0] != 1 or unsqueeze_dim != 1:
        raise ValueError("Dense alignment currently requires the packed single-row Qwen3 layout")
    half = cos.shape[-1] // 2
    return _Rotary.apply(q, k, torch.cat((cos[0, :, :half], sin[0, :, :half]), dim=-1))


def _head_forward(self, hidden_states, labels=None, temperature=None, sampling_mask=None):
    if labels is None or temperature is None or sampling_mask is not None:
        raise ValueError("Dense alignment requires labels, temperatures, and full-vocabulary sampling")
    shape = hidden_states.shape[:-1]
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if self._alignment_fp32_head:
        from prime_rl.trainer.models.layers.aligned_head import FP32HeadLinear

        logits = FP32HeadLinear.apply(hidden, self.weight)
    else:
        logits = _Linear.apply(hidden, self.weight, None)
    distribution = _LogSoftmax.apply(logits.float() / temperature.reshape(-1, 1))
    selected = distribution.gather(1, labels.reshape(-1, 1)).reshape(shape).clone()
    with torch.no_grad():
        entropy = -(distribution.exp() * distribution).sum(-1).reshape(shape)
    return {"logprobs": selected, "entropy": entropy}


def enable_trainer_alignment(model: nn.Module, *, fp32_head: bool = False, allow_moe: bool = False) -> None:
    from vllm.platforms import current_platform
    from vllm.vllm_flash_attn.cute.interface import flash_attn_varlen_func

    from prime_rl.trainer.models.layers import attn
    from prime_rl.trainer.models.layers.inference_swiglu import enable_inference_swiglu
    from prime_rl.trainer.models.layers.mlp import FeedForward
    from prime_rl.trainer.models.layers.norms import RMSNorm
    from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding

    if model.config.model_type != "qwen3" and not (allow_moe and model.config.model_type in ("qwen3_moe", "glm4_moe")):
        raise ValueError("Dense alignment currently supports dense Qwen3 only")
    current_platform.import_kernels()
    if any(isinstance(module, FeedForward) for module in model.modules()):
        enable_inference_swiglu(model)
    for module in model.modules():
        if type(module) is nn.Linear:
            module.forward = MethodType(_linear_forward, module)
        elif isinstance(module, RMSNorm):
            module.forward = MethodType(_norm_forward, module)
        elif isinstance(module, RotaryEmbedding):
            if module.rope_type != "default":
                raise ValueError("Dense alignment requires default RoPE")
            config = module.config
            rotary_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
            module._alignment_cpu_cache = rope_table(
                rotary_dim, config.max_position_embeddings, config.rope_parameters["rope_theta"]
            )
            module._alignment_cache_key = None
            module.forward = MethodType(_rope_forward, module)
        elif isinstance(module, attn.FlashAttention):
            module._flash_attn_version = 4
            module._flash_attn_call = flash_attn_varlen_func
    attn.apply_rotary_pos_emb = _apply_rotary
    model.lm_head._alignment_fp32_head = fp32_head
    model.lm_head.forward = MethodType(_head_forward, model.lm_head)


def enable_serving_alignment(*, fp32_head: bool = False) -> None:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends import fa_utils, flash_attn
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample import logprob as v2_logprob

    def norm(self, x, residual=None):
        if self.variance_size_override is not None:
            raise ValueError("Dense alignment does not support partial RMS normalization")
        if residual is None:
            return rms_norm_batch_invariant(x, self.weight, self.variance_epsilon)
        residual = x + residual
        return rms_norm_batch_invariant(residual, self.weight, self.variance_epsilon), residual

    def linear(self, layer, x, bias=None):
        return linear_forward(x, layer.weight, bias)

    def table(self):
        return rope_table(self.rotary_dim, self.max_position_embeddings, self.base).clone()

    initialize_logits_processor = LogitsProcessor.__init__

    def logits_processor_init(self, *args, **kwargs):
        initialize_logits_processor(self, *args, **kwargs)
        if getattr(self, "_fp32_lm_head_enabled", False) and not fp32_head:
            raise ValueError("Dense alignment requires inference.enable_fp32_lm_head=false for its shared BF16 head")

    def fp32_logits(self, hidden_states, lm_head, embedding_bias):
        from prime_rl.trainer.models.layers.aligned_head import fixed_k_fp32

        logits = self._gather_logits(fixed_k_fp32(hidden_states, lm_head.weight, embedding_bias))
        return logits[..., : self.org_vocab_size] if logits is not None else None

    def attention_version(requires_alibi=False, head_size=None, head_size_v=None, has_sinks=False):
        capability = current_platform.get_device_capability()
        if capability is None or capability.major != 9:
            raise ValueError("Dense alignment FA4 batch invariance has only been validated on Hopper")
        if requires_alibi or has_sinks or head_size not in (None, 128) or head_size_v not in (None, 128):
            raise ValueError("Dense alignment requires Qwen3's 128-dimensional dense attention")
        return 4

    RMSNorm.forward_cuda = norm
    RMSNorm.forward_native = norm
    UnquantizedLinearMethod.apply = linear
    RotaryEmbeddingBase._compute_cos_sin_cache = table
    LogitsProcessor.__init__ = logits_processor_init
    if fp32_head:
        LogitsProcessor._get_logits = fp32_logits
    fa_utils.get_flash_attn_version = attention_version
    flash_attn.get_flash_attn_version = attention_version
    Sampler.compute_logprobs = staticmethod(lambda logits: log_softmax(logits.float()))
    v2_logprob.compute_token_logprobs = lambda logits, token_ids: log_softmax(logits.float()).gather(
        1, token_ids.to(torch.int64)
    )
