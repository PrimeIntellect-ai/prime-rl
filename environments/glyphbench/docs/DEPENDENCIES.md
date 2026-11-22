# Dependency notes

The optional GPU runtime pins Prime-RL v0.9, vLLM 0.26.0, and PyTorch 2.11.0
as a coupled stack. The release review identified the following remaining
advisories. Dependency updates should preserve the training and serving
interfaces and be checked together with the model configuration.
The audit could not verify every source-built dependency or custom CUDA package.

## vLLM activation kernel: CVE-2026-73558

An integer overflow in an activation kernel can mix outputs between requests
in a sufficiently large inference batch. Upstream identifies vLLM 0.27.0 as
patched; [PR #49660](https://github.com/vllm-project/vllm/pull/49660) widens the
kernel's index arithmetic. See the [upstream advisory](https://github.com/vllm-project/vllm/security/advisories/GHSA-7m6h-x95x-82q5).

Moving to vLLM 0.27.0 also requires PyTorch 2.13.0 in its
[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.27.0/requirements/cuda.txt).
The current recipes retain their pinned runtime and apply these settings for
the dense Qwen3.5-4B model and checkpoints with the same architecture:

```text
--max-num-batched-tokens 8192
--enable-chunked-prefill
--compilation-config '{"custom_ops":["-silu_and_mul"]}'
```

Disabling `silu_and_mul` selects its native PyTorch implementation in both
eager and compiled execution, based on the upstream
[activation implementation](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/activation.py)
and [dispatch logic](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/custom_op.py).
The token limit also keeps the affected offset arithmetic below its overflow
threshold for the [model's intermediate width of 9,216](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json);
chunked prefill preserves support for longer prompts.

This is a mitigation for the documented model and settings, not a general
patch to vLLM. Other models, MoE implementations, or direct CUDA operator calls
may take different paths. Review model changes and serving overrides before
reusing it. The release checks verified configuration and dispatch behavior;
they did not run an end-to-end GPU training or inference workload.

## setuptools source distributions: PYSEC-2026-3447

This issue can bypass `MANIFEST.in` exclusions when a source distribution is
built with differently normalized Unicode filenames, notably on macOS APFS
or HFS+. See the [setuptools advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c).
It is fixed in [setuptools 83.0.0](https://setuptools.pypa.io/en/latest/history.html#v83-0-0),
while [vLLM 0.26 requires setuptools below 81](https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/common.txt).

This affected build path is not used by GlyphBench's release: the locked uv
runtime targets Linux, and GlyphBench builds with Hatchling and an explicit
source-distribution inclusion list in [pyproject.toml](../pyproject.toml).
That assessment does not extend to unrelated setuptools projects built in
the same environment.

## PyTorch TorchScript compilation: CVE-2025-3000

PyTorch 2.11.0 remains affected by a compiler crash involving bare `list` or
`tuple` annotations passed to `torch.jit.script`. The
[reported reproducer](https://github.com/pytorch/pytorch/issues/149623) and
[fix](https://github.com/pytorch/pytorch/commit/b90c949) identify the annotation
handling problem; the [advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
lists PyTorch 2.13.0 as patched.

Source inspection found no direct `torch.jit.script` or `CompilationUnit` use
in GlyphBench, Prime-RL v0.9, or vLLM 0.26. The bundled FlashAttention activation
decorators inspected did not contain the affected annotations. No route from
benchmark prompts or actions to this compiler input was identified. The pin
is still vulnerable; this assessment does not cover untrusted Python, model,
or checkpoint code.

## FlashAttention checkpoint examples: CVE-2026-31253

The [advisory](https://github.com/advisories/GHSA-7g5w-pq96-8c5w) flags unsafe
checkpoint deserialization in FlashAttention's upstream training framework,
including [its checkpoint helper](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/training/src/utils/checkpoint.py).
No fixed version is listed in the advisory as of September 14, 2026.

The inspected locked x86_64 `flash_attn` 2.8.3 CUDA wheel contains neither that
helper nor the upstream `training/` tree. Prime-RL uses the attention kernel
API, not those example checkpoint loaders. This finding is specific to that
wheel and execution path. Prime-RL's own training-state resume uses pickle-capable
loading and requires trusted checkpoints; the assessment does not make
arbitrary checkpoint loading safe.
