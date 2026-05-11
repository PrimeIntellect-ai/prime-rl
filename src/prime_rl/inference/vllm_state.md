# vLLM State

Last updated: 2026-05-12

prime-rl's routed-experts replay path requires a patched vLLM fork. This note
records the vLLM PR state the prime-rl wheel pin expects, what the fork changes,
and how the wheel was produced.

## Required vLLM

Use:

```text
repo: https://github.com/S1ro1/vllm.git
pr: https://github.com/S1ro1/vllm/pull/3
branch: feat/routed-experts-prefix-replay
head: a1c9051bc490be4200137735c4a806d22a184121
```

The branch is based on upstream vLLM `main` at:

```text
879a8c318032ed32716e5e0b10af355a4ddbced2
```

## Wheel Build

The x86_64 wheel pinned by prime-rl was built from the fork PR head with
vLLM's precompiled build path, so the Python changes come from the fork while
native/CUDA objects are taken from an already-built vLLM wheel:

```bash
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu129
export VLLM_USE_PRECOMPILED=1
uv build --wheel --out-dir dist
```

The resulting wheel was uploaded to the prime-rl `v0.5.0` release and pinned in
`pyproject.toml` for x86_64 installs:

```text
https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.5.0/vllm-0.20.2rc1.dev209%2Bga1c9051bc.precompiled-cp312-cp312-linux_x86_64.whl
```

SHA256:

```text
9f9fd0f24f6c93082e2449c37d6fa3e18a8ced838e0831db7563f358776db1d2
```

## Required Changes

The fork must keep upstream routed-experts response support from
`vllm-project/vllm#39917`. prime-rl depends on vLLM returning prompt routing at
the response level and completion routing on each generated choice.

The fork adds routed-experts replay support for prefix caching and chunked
prefill. Without this, cached prompt blocks can be missing their routed experts,
which makes trainer-side replay either fail or compare against the wrong
experts. The PR adds the replay cache and prefix-cache plumbing needed to
reconstruct prompt routing for cached blocks, including the DPEP/GLM MoE-layer
shape expected by the trainer. It also avoids extra routed replay prefill work
on P/D decode requests, where decode uses external KV and should only emit
completion routing.

The fork also reverts upstream `vllm-project/vllm#39366` to keep NCCL weight
transfer working. That upstream PR introduced a two-phase DP pause protocol that
conflicts with prime-rl's pause/resume flow during NCCL transfer. Keeping it
reverted preserves the older pause behavior that prime-rl's DP pause/deadlock
patch expects, so weight transfer can pause and resume generation reliably.

## prime-rl Contract

prime-rl's trainer consumes routed experts in the shape emitted by vLLM:

```text
[batch, sequence_length, num_moe_layers, topk]
```

`num_moe_layers` is the sparse MoE-layer count, not the total decoder-layer
count. The trainer maps decoder layers to sparse MoE-layer indices when replaying
experts.
