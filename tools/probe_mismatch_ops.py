"""Compare dense forward operators on identical BF16 inputs; requires one CUDA GPU."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.batch_invariant import matmul_persistent, rms_norm_batch_invariant
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding as ServingRotaryEmbedding

from prime_rl.trainer.models.layers.activations import Silu
from prime_rl.trainer.models.layers.aligned_head import FP32HeadLinear, fixed_k_fp32
from prime_rl.trainer.models.layers.dense_alignment import (
    _Linear,
    _LogSoftmax,
    _Norm,
    _Rotary,
    enable_serving_alignment,
    rope_table,
)
from prime_rl.trainer.models.layers.inference_swiglu import InferenceSilu
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig, apply_rotary_pos_emb


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    assert left.shape == right.shape and left.dtype == right.dtype
    difference = (left.float() - right.float()).abs()
    bit_dtype = torch.int16 if left.element_size() == 2 else torch.int32
    return {
        "elements": left.numel(),
        "bit_mismatch_fraction": (left.contiguous().view(bit_dtype) != right.contiguous().view(bit_dtype))
        .float()
        .mean()
        .item(),
        "abs_error_max": difference.max().item(),
        "abs_error_mean": difference.mean().item(),
    }


@torch.no_grad()
def probe() -> dict:
    torch.manual_seed(42)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn(257, 4096, dtype=dtype, device=device)
    weight = torch.randn(4096, 4096, dtype=dtype, device=device) / 64
    reference = matmul_persistent(x, weight.T)
    results = {}
    for rows in (1, 17, 128, 257):
        candidate = F.linear(x[:rows], weight)
        invariant = matmul_persistent(x[:rows], weight.T)
        results[f"linear/torch_vs_vllm_invariant/rows_{rows}"] = compare(candidate, invariant)
        results[f"linear/vllm_batch_shape/rows_{rows}"] = compare(reference[:rows], invariant)

    norm = RMSNorm(RMSNormConfig(4096)).to(device=device, dtype=dtype)
    norm.weight.copy_(torch.randn_like(norm.weight))
    compiled_norm = torch.compile(norm)
    for rows in (1, 17, 257):
        expected = rms_norm_batch_invariant(x[:rows], norm.weight, 1e-6)
        results[f"rmsnorm/eager_vs_vllm_invariant/rows_{rows}"] = compare(norm(x[:rows]), expected)
        results[f"rmsnorm/compiled_vs_vllm_invariant/rows_{rows}"] = compare(compiled_norm(x[:rows]), expected)

    gate = torch.randn(257, 12288, dtype=dtype, device=device)
    up = torch.randn_like(gate)
    with set_current_vllm_config(VllmConfig()):
        expected = SiluAndMul().forward_cuda(torch.cat((gate, up), dim=-1))
    results["swiglu/eager_vs_vllm_cuda"] = compare(Silu.apply(gate, up), expected)
    results["swiglu/compiled_vs_vllm_cuda"] = compare(torch.compile(Silu.apply)(gate, up), expected)
    results["swiglu/aligned_vs_vllm_cuda"] = compare(torch.compile(InferenceSilu.apply)(gate, up), expected)
    with torch.enable_grad():
        grad_output = torch.randn_like(gate)
        gate = gate.requires_grad_()
        up = up.requires_grad_()
        reference_gradients = torch.autograd.grad(Silu.apply(gate, up), (gate, up), grad_output)
        aligned_gradients = torch.autograd.grad(InferenceSilu.apply(gate, up), (gate, up), grad_output)
        for name, reference_gradient, aligned_gradient in zip(
            ("gate", "up"), reference_gradients, aligned_gradients, strict=True
        ):
            results[f"swiglu/backward/{name}"] = compare(reference_gradient, aligned_gradient)
            torch.testing.assert_close(reference_gradient, aligned_gradient, rtol=0, atol=0)
    assert results["swiglu/aligned_vs_vllm_cuda"]["bit_mismatch_fraction"] == 0

    positions = torch.arange(8192, device=device)
    q = torch.randn(1, 32, 8192, 128, device=device, dtype=dtype)
    k = torch.randn(1, 8, 8192, 128, device=device, dtype=dtype)
    rope_config = Qwen3Config(head_dim=128, rope_theta=1_000_000)
    training_rope = RotaryEmbedding(RotaryEmbeddingConfig(8192, "default", rope_config), device=device)
    cos, sin = training_rope(q, positions.unsqueeze(0))
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        serving_rope = ServingRotaryEmbedding(128, 128, 8192, 1_000_000, True, dtype)
    serving_q = q.squeeze(0).transpose(0, 1).contiguous().flatten(1)
    serving_k = k.squeeze(0).transpose(0, 1).contiguous().flatten(1)
    serving_q, serving_k = serving_rope.forward_cuda(positions, serving_q, serving_k)
    expected_q = serving_q.reshape(8192, 32, 128).transpose(0, 1).unsqueeze(0)
    expected_k = serving_k.reshape(8192, 8, 128).transpose(0, 1).unsqueeze(0)
    results["rope/cos_table"] = compare(cos[0, :, :64], serving_rope.cos_sin_cache[:, :64])
    results["rope/sin_table"] = compare(sin[0, :, :64], serving_rope.cos_sin_cache[:, 64:])
    for name, function in (("eager", apply_rotary_pos_emb), ("compiled", torch.compile(apply_rotary_pos_emb))):
        actual_q, actual_k = function(q, k, cos, sin)
        results[f"rope/{name}/query"] = compare(actual_q, expected_q)
        results[f"rope/{name}/key"] = compare(actual_k, expected_k)
        results[f"rope/{name}/query_after_593"] = compare(actual_q[:, :, 594:], expected_q[:, :, 594:])
    torch.cuda.synchronize()
    results.update(probe_alignment())
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": results}


@torch.enable_grad()
def probe_alignment(query_heads: int = 32, kv_heads: int = 8) -> dict:
    from vllm.model_executor.layers.batch_invariant import log_softmax
    from vllm.vllm_flash_attn.cute.interface import flash_attn_varlen_func
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as serving_attention

    results = {}
    x = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    actual = _Linear.apply(x, weight, None)
    results["aligned/linear"] = compare(actual, matmul_persistent(x, weight.T))
    expected_grads = torch.autograd.grad(F.linear(x, weight), (x, weight), grad)
    actual_grads = torch.autograd.grad(actual, (x, weight), grad)
    for expected, actual in zip(expected_grads, actual_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    norm_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = _Norm.apply(x, norm_weight, 1e-6)
    results["aligned/norm"] = compare(actual, rms_norm_batch_invariant(x, norm_weight, 1e-6))
    xf = x.float()
    expected = (xf * (xf.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * norm_weight.float()).to(x.dtype)
    grad = torch.randn_like(x)
    expected_grads = torch.autograd.grad(expected, (x, norm_weight), grad)
    actual_grads = torch.autograd.grad(actual, (x, norm_weight), grad)
    for expected, actual in zip(expected_grads, actual_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)

    logits = torch.randn(17, 151936, device="cuda", requires_grad=True)
    actual = _LogSoftmax.apply(logits)
    results["aligned/logsoftmax"] = compare(actual, log_softmax(logits))
    labels = torch.randint(logits.shape[-1], (17, 1), device="cuda")
    actual_grad = torch.autograd.grad(actual.gather(1, labels).sum(), logits)[0]
    expected_grad = torch.autograd.grad(logits.log_softmax(-1).gather(1, labels).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-7)

    cache = rope_table(128, 8192, 1_000_000).cuda().to(torch.bfloat16)
    q = torch.randn(1, query_heads, 8192, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, 8192, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual_q, actual_k = _Rotary.apply(q, k, cache)
    cosine, sine = cache.chunk(2, dim=-1)
    cosine = torch.cat((cosine, cosine), -1)[None]
    sine = torch.cat((sine, sine), -1)[None]
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cosine, sine)
    results["aligned/rope_eager/query"] = compare(actual_q, expected_q)
    results["aligned/rope_eager/key"] = compare(actual_k, expected_k)
    enable_serving_alignment()
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        serving_rope = ServingRotaryEmbedding(128, 128, 8192, 1_000_000, True, q.dtype)
    serving_q, serving_k = serving_rope.forward_cuda(
        torch.arange(8192, device="cuda"),
        q.detach()[0].transpose(0, 1).contiguous().flatten(1),
        k.detach()[0].transpose(0, 1).contiguous().flatten(1),
    )
    for name, actual, serving, heads in (
        ("query", actual_q, serving_q, query_heads),
        ("key", actual_k, serving_k, kv_heads),
    ):
        expected = serving.reshape(8192, heads, 128).transpose(0, 1)[None]
        results[f"aligned/rope_serving/{name}"] = compare(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_q, grad_k = torch.randn_like(q), torch.randn_like(k)
    actual_grads = torch.autograd.grad((actual_q, actual_k), (q, k), (grad_q, grad_k))
    expected_grads = torch.autograd.grad((expected_q, expected_k), (q, k), (grad_q, grad_k))
    for expected, actual in zip(expected_grads, actual_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.032)

    with torch.no_grad():
        q = q[0].transpose(0, 1).contiguous()
        k = k[0].transpose(0, 1).contiguous()
        v = torch.randn_like(k)
        for length in (257, 1024, 8192):
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            prefill = flash_attn_varlen_func(
                q[:length],
                k[:length],
                v[:length],
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=length,
                max_seqlen_k=length,
                causal=True,
                num_splits=1,
            )
            decode = flash_attn_varlen_func(
                q[length - 1 : length],
                k[:length],
                v[:length],
                cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                cu_seqlens_k=cu,
                max_seqlen_q=1,
                max_seqlen_k=length,
                causal=True,
                num_splits=1,
            )
            if isinstance(prefill, tuple):
                prefill = prefill[0]
            if isinstance(decode, tuple):
                decode = decode[0]
            results[f"aligned/attention/prefill_vs_decode/{length}"] = compare(prefill[-1:], decode)
            pages = (length + 15) // 16
            padding = torch.zeros(pages * 16 - length, kv_heads, 128, device=k.device, dtype=k.dtype)
            key_cache = torch.cat((k[:length], padding)).reshape(pages, 16, kv_heads, 128)
            value_cache = torch.cat((v[:length], padding)).reshape(pages, 16, kv_heads, 128)
            paged = serving_attention(
                q=q[length - 1 : length],
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
                max_seqlen_q=1,
                max_seqlen_k=length,
                causal=True,
                num_splits=1,
                block_table=torch.arange(pages, device="cuda", dtype=torch.int32)[None],
                fa_version=4,
            )
            results[f"aligned/attention/prefill_vs_paged_decode/{length}"] = compare(prefill[-1:], paged)
            torch.testing.assert_close(prefill[-1:], paged, rtol=0, atol=0)
    q = q[:128].detach().requires_grad_()
    k = k[:128].detach().requires_grad_()
    v = v[:128].detach().requires_grad_()
    cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True, num_splits=1)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).float(),
            k.transpose(0, 1).repeat_interleave(query_heads // kv_heads, dim=0).float(),
            v.transpose(0, 1).repeat_interleave(query_heads // kv_heads, dim=0).float(),
            is_causal=True,
        )
        .transpose(0, 1)
        .to(q.dtype)
    )
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    for name, actual, expected in zip(("q", "k", "v"), actual_grads, expected_grads, strict=True):
        results[f"aligned/attention/backward/{name}"] = compare(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.02)
    return results


def probe_fp32_head() -> dict:
    from vllm.model_executor.layers.batch_invariant import log_softmax
    from vllm.v1.worker.gpu.sample import logprob as v2_logprob

    torch.manual_seed(42)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(151936, 4096, device="cuda", dtype=torch.bfloat16) / 64
    results = {}
    with torch.no_grad():
        reference = fixed_k_fp32(x, weight)
        reference_lp = log_softmax(reference)
        token_ids = reference.argmax(-1, keepdim=True)
        expected_lp = reference_lp.gather(1, token_ids)
        results["head/v2_original_selected_logprob"] = compare(
            expected_lp, v2_logprob.compute_token_logprobs(reference, token_ids)
        )
        for dtype in (torch.bfloat16, torch.float32):
            peaked = reference[:17].to(dtype).clone()
            peaked[:, 0] = 16
            ids = torch.zeros(17, 1, device=x.device, dtype=torch.int64)
            expected = log_softmax(peaked.float()).gather(1, ids)
            results[f"head/v2_original_peaked/{dtype}"] = compare(
                expected, v2_logprob.compute_token_logprobs(peaked, ids)
            )
        enable_serving_alignment(fp32_head=True)
        aligned_lp = v2_logprob.compute_token_logprobs(reference, token_ids)
        results["head/v2_aligned_selected_logprob"] = compare(expected_lp, aligned_lp)
        torch.testing.assert_close(aligned_lp, expected_lp, rtol=0, atol=0)
        for rows in (1, 17, 128):
            actual = fixed_k_fp32(x[:rows], weight)
            results[f"head/fp32_batch_invariance/{rows}"] = compare(reference[:rows], actual)
            results[f"head/logprob_batch_invariance/{rows}"] = compare(reference_lp[:rows], log_softmax(actual))
            torch.testing.assert_close(reference[:rows], actual, rtol=0, atol=0)
            torch.testing.assert_close(reference_lp[:rows], log_softmax(actual), rtol=0, atol=0)
        expected = F.linear(x[:17].double(), weight[:512].double()).float()
        torch.testing.assert_close(reference[:17, :512], expected, rtol=1e-4, atol=1e-5)
        results["head/fp64_reference"] = compare(reference[:17, :512], expected)
    x = x[:17].detach().requires_grad_()
    weight = weight[:512].detach().requires_grad_()
    grad = torch.randn(17, 512, device="cuda")
    actual_grads = torch.autograd.grad(FP32HeadLinear.apply(x, weight), (x, weight), grad)
    expected_grads = torch.autograd.grad(F.linear(x.float(), weight.float()), (x, weight), grad)
    for name, actual, expected in zip(("input", "weight"), actual_grads, expected_grads, strict=True):
        results[f"head/backward/{name}"] = compare(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fp32-head-only", action="store_true")
    mode.add_argument("--moe-layouts", action="store_true")
    args = parser.parse_args()
    if args.moe_layouts:
        torch.manual_seed(42)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        result = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "layouts": {f"{q}q_{kv}kv": probe_alignment(q, kv) for q, kv in ((32, 4), (96, 8))},
        }
    else:
        result = probe_fp32_head() if args.fp32_head_only else probe()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
