# vLLM State

Last updated: 2026-05-13

prime-rl's routed-experts replay path requires a patched vLLM fork. This note
records the vLLM PR state expected by the pinned wheel, the required fork
behavior, and how the wheel was produced.

## Required vLLM

Use:

```text
repo: https://github.com/S1ro1/vllm.git
pr: https://github.com/S1ro1/vllm/pull/3
branch: feat/routed-experts-prefix-replay
head: cff9b14e3f28310128ddbacb708f7bf87768ea86
```

The native objects in the pinned wheel are taken from the vLLM precompiled
nightly for:

```text
879a8c318032ed32716e5e0b10af355a4ddbced2
```

## Wheel Build

The x86_64 wheel pinned by prime-rl was built from the fork PR head with
vLLM's precompiled build path, so the Python changes come from the fork while
native/CUDA objects are taken from an already-built vLLM wheel:

```bash
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly
export VLLM_USE_PRECOMPILED=1
uv build --python 3.12 --wheel --out-dir dist
```

The resulting wheel was uploaded to the prime-rl `v0.5.0` release and pinned in
`pyproject.toml` for x86_64 installs:

```text
https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.5.0/vllm-0.20.2rc1.dev213%2Bgcff9b14e3.precompiled-cp312-cp312-linux_x86_64.whl
```

SHA256:

```text
5b594e78118d7e6d16adf63a6c8ac4cba059d9f8a1a37ec7e57bbbc70a7acbac
```

## Required Changes

The fork keeps upstream routed-experts response support from
`vllm-project/vllm#39917`. prime-rl depends on vLLM returning prompt routing at
the response level and completion routing on each generated choice.

The fork adds routed-experts replay support for prefix caching and chunked
prefill. Without this, cached prompt blocks can be missing their routed experts,
which makes trainer-side replay either fail or compare against the wrong
experts. The PR adds the replay cache and prefix-cache plumbing needed to
reconstruct prompt routing for cached blocks, including the DPEP/GLM MoE-layer
shape expected by the trainer.

For P/D decode requests, the fork avoids extra routed replay prefill work. The
decode side uses external KV and should only emit completion routing, while the
paired prefill response supplies prompt routing.

The fork reverts upstream `vllm-project/vllm#39366` to keep NCCL weight transfer
working. That upstream PR introduced a two-phase DP pause protocol that
conflicts with prime-rl's pause/resume flow during NCCL transfer. Keeping it
reverted preserves the older pause behavior needed by prime-rl's DP
pause/deadlock patch.

The fork reduces routed-experts overhead in the capture and response path. It
serializes routed experts as compact `(shape, bytes)` payloads instead of nested
Python lists, grows per-request host buffers less aggressively during decode,
avoids an extra device-to-device staging copy before routed-experts D2H, and
only publishes replay-cache blocks when a block can newly become complete.

## HTTP Payload Contract

Routed-experts HTTP responses use the compact payload only:

```json
{
  "encoding": "base64",
  "dtype": "int16",
  "shape": [1, 1024, 75, 8],
  "data": "..."
}
```

The old nested-list routed-experts response format is not supported by this
PR stack.

## prime-rl Contract

prime-rl's trainer consumes routed experts in the shape emitted by vLLM:

```text
[batch, sequence_length, num_moe_layers, topk]
```

`num_moe_layers` is the sparse MoE-layer count, not the total decoder-layer
count. The trainer maps decoder layers to sparse MoE-layer indices when replaying
experts.
