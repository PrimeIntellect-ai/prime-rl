# Image build plan — `prime-rl-mx-on-nixl:v0.7.2-kavin-mx-v2`

> Companion to [`post-pr2389-mx-v2.md`](./post-pr2389-mx-v2.md) (Workstream B).

## Goal

Ship a single deployable image that contains everything `weight_broadcast.type = "mx_v2"` needs:

- Phase 2 source (heartbeat + same-rank + freshest-per-rank) — *already in* `v0.7.1-kavin-phase2-phase3`
- Phase 3 source (conversion-registry extensions, compile_target tagging) — *already in* `v0.7.1-kavin-phase2-phase3`
- **NEW:** Phase 4 + `MxWeightTransferEngine` from MX PR #349 (`kavink/post-2389-phase3-4` branch)
- **NEW:** the v2 prime-rl source files from this RFC (`nixl_mx_v2.py` × 2, updated `__init__.py`, updated `configs/trainer.py`)

## Strategy: overlay, not from-scratch

The build-notes ([`build-notes-2026-05-28.md`](./build-notes-2026-05-28.md) §2) measured **6h 45min** for a from-scratch `Dockerfile.cuda` build on QEMU arm64. The overlay for v0.7.1 was **~3 min**. We overlay.

```dockerfile
# Dockerfile.cuda.mx-v2
FROM nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.7.1-kavin-phase2-phase3

# ── 1. Update modelexpress to the PR #349 branch (Phase 4 + engine adapter) ──
RUN --mount=type=cache,target=/app/.cache/uv \
    uv pip install --no-deps --reinstall \
        "modelexpress @ git+https://github.com/ai-dynamo/modelexpress.git@kavink/post-2389-phase3-4#subdirectory=modelexpress_client/python"

# ── 2. Overlay the v2 prime-rl files ─────────────────────────────────────────
COPY src/prime_rl/transport/                                       /app/src/prime_rl/transport/
COPY src/prime_rl/inference/vllm/worker/nixl_mx_v2.py              /app/src/prime_rl/inference/vllm/worker/nixl_mx_v2.py
COPY src/prime_rl/trainer/rl/broadcast/nixl_mx_v2.py               /app/src/prime_rl/trainer/rl/broadcast/nixl_mx_v2.py
COPY src/prime_rl/trainer/rl/broadcast/__init__.py                 /app/src/prime_rl/trainer/rl/broadcast/__init__.py
COPY packages/prime-rl-configs/src/prime_rl/configs/trainer.py     /app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py
```

Build:

```bash
docker buildx build \
    --platform linux/arm64 \
    --file Dockerfile.cuda.mx-v2 \
    --tag nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.7.2-kavin-mx-v2 \
    --push \
    .
```

Estimated: ~5 min (one git clone + uv install for modelexpress, then a 5-file COPY).

## What about the `flash_attn` ABI issue?

The earlier `flash_attn.ops` `ModuleNotFoundError` we saw was inside the *MX overlay benchmark pod* — that pod was using the v0.5.2 image with a Python overlay, and the v0.5.2 image's vLLM expects an older flash_attn layout.

**`v0.7.1-kavin-phase2-phase3` does NOT have that problem** — it was built from a fresh `Dockerfile.cuda` that pins `flash-attn==2.8.3+cu128torch2.11` (via the `flash-attn` extra in `pyproject.toml`) and rebuilds vLLM against it. Since we're overlaying on top of that, we inherit the fixed pin and never touch the ABI.

**Validation step:** confirm by running `python -c "import flash_attn; import flash_attn.ops"` inside the new image before doing anything else.

## Deployment

Same configmap pattern as `v0.7.1`. New trainer + inference manifests with `weight_broadcast.type = "mx_v2"`:

```yaml
# configmap delta for v2 deployment
weight_broadcast:
  type: mx_v2                         # was: nixl_mx
  host: modelexpress-server.kavin.svc.cluster.local
  port: 8001
  same_rank_only: true                # Phase 2 default
  dedup_freshest_per_rank: true       # Phase 2 default
  publish_compile_target: true        # Phase 3 default
  publish_self_as_replica: true       # tree fan-out default
  inference_world_size: 4
  inference_model_name: Qwen/Qwen3-30B-A3B-Instruct-2507
```

For A/B against PR #2389, run **two parallel deployments** in the kavin namespace under separate Job names — `prime-rl-nixl-mx-v0-7-1` (baseline) vs `prime-rl-mx-v2-kavin` (this work). Both use the same MX server (different `mx_source_id`s by content hash).

## What to measure

The validation matrix from `post-pr2389-mx-v2.md` §Validation plan, against the same workload PR #2389 was validated with:

| Config | Refit cycle | Bandwidth | Notes |
|---|---|---|---|
| `nixl_mx` baseline | target ~10 s | ~80 ms NIXL push | PR #2389 push |
| `mx_v2` defaults | target ≤ 10 s | should match | Pull semantics, single-receiver |
| `mx_v2` + 4 receivers | target ≤ 10 s | `fanout_factor > 1.0` | Tree fan-out engages |
| `mx_v2` + filter mismatch | refuse at discovery, 0 RDMA bytes | — | Phase 3 safety net in production |
| `mx_v2` elastic scale-up | 2 → 4 replicas mid-training | new receivers join under 2 s | Phase 2 same-rank routing in production |

## Open items

| # | Item | Owner / status |
|---|---|---|
| 1 | Confirm v0.7.1 image has `flash_attn.ops` importable | Smoke test, ~30 s |
| 2 | Build + push `v0.7.2-kavin-mx-v2` | ~5 min after Phase A code lands |
| 3 | Deploy parallel A/B Jobs in kavin namespace | Cluster booking — ~5 hours of GPU node time |
| 4 | Capture per-cycle timing + bandwidth JSONs into `pensieve/RL/PrimeRL/results/` | Direct mirror of Slide 9 / Table B in the presentation |
| 5 | Replace Slide 9 Table A's synthetic numbers with end-to-end numbers | Once #4 is in hand |
