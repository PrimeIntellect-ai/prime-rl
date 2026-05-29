# Build notes — Phase-2 + Phase-3 source-baked image (2026-05-28)

> **Companion to**: [`docs/proposals/post-pr2389-kernel-compile-plan.md`](./post-pr2389-kernel-compile-plan.md)
> **Status**: empirical findings from baking Phase 2 (rendezvous fixes) + Phase 3 (conversion-registry extensions) into an ARM64 GB200 image and running it against the live `kavin` namespace. Updates the RFC's framing where the build experience contradicted assumptions in the original RFC.

This document captures **what we learned producing a usable image** containing the two follow-up PRs ([phase-2 rendezvous fixes](https://github.com/KavinKrishnan/prime-rl/pull/1) and [conversion-registry extensions](https://github.com/KavinKrishnan/prime-rl/pull/2)) on top of PR #2389 (HEAD `79ea824d8`). The unit tests for both PRs were already green; this doc records the cluster + image surface area that the unit tests don't cover.

## 1. What we built

Two images, in order:

| Tag | Base | What's added | Status |
|---|---|---|---|
| `prime-rl-mx-on-nixl:v0.7.0-kavin-phase2-phase3` | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` (full Dockerfile.cuda rebuild) | Phase 2 + Phase 3 source merged in. Built from `kavink/post-2389-image-build-2026-05-28` (which merges PR #1 + PR #2 on top of `79ea824d8`). | **Pushed to nvcr** |
| `prime-rl-mx-on-nixl:v0.7.1-kavin-phase2-phase3` | `v0.7.0-kavin-phase2-phase3` | Adds the `disagg` extra (modelexpress + nixl-cu12 + vllm-router). Fixes the import error from v0.7.0. | **Pushed to nvcr** |

`v0.7.1` is the one to deploy. `v0.7.0` is kept as a reference of the from-scratch build artifact.

## 2. Build mechanics (ARM64 GB200 / QEMU)

The from-scratch ARM64 build of `v0.7.0` took **6h 45min on x86 host with QEMU arm64 emulation** (buildkit `multi-arch` builder). Breakdown:

| Stage | Time | Notes |
|---|---|---|
| Pull `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` ARM64 base | ~5 min | First time only; cached after |
| `apt-get install` builder + final stages | ~3 min total | Both stages of multi-stage Dockerfile |
| `COPY src/ + packages/ + deps/` | seconds | Trivial |
| **`uv sync --extra ... --locked --no-dev`** | **45 min** | Resolves + downloads + installs ~350 packages including torch 2.7+cu130 (~5 GB), nvidia-cudnn-cu12 (738 MiB), flashinfer-cubin (large), tilelang, xgrammar, vllm 0.21.0+cu129 etc. Under QEMU emulation. |
| **`docker-arm64-post-install.sh` (flash-attn from source for sm_100 / GB200)** | **~3h 45min** | 73 CUDA kernel `.o` files, each compiled via emulated `nvcc` for sm_80 + sm_90 + sm_100. Most expensive kernels are `hdim192_bf16_causal` and `hdim256_bf16` for backward pass (15-40 min each). |
| Final stage `COPY --from=builder /app` + image export | ~7 min | 15.9 GB final image, 6.5 GB of which is one big layer (the venv) |

`v0.7.1` overlay on top of `v0.7.0` was **~3 min** (the `uv sync` with `disagg` extra reuses every cached layer except the new modelexpress/nixl-cu12/vllm-router wheels).

**Practical implication**: every meaningful rebuild from the Dockerfile.cuda base is ~7 hours on a non-ARM host. Use overlay Dockerfiles for additive changes. Reserve from-scratch only for `pyproject.toml` / `uv.lock` updates or major source restructuring.

## 3. Three real issues the build surfaced that aren't in the RFC

### 3.1 `Dockerfile.cuda` is missing `--extra disagg` for nixl_mx use

[`Dockerfile.cuda`](../../Dockerfile.cuda) line 52:

```dockerfile
RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --extra flash-attn --extra flash-attn-3 --extra flash-attn-cute --extra envs --extra gpt-oss --group mamba-ssm --locked --no-dev
```

The `disagg` extra ([`pyproject.toml` line 90](../../pyproject.toml#L90)) contains:

```toml
disagg = [
    "deep-ep ; platform_machine == 'x86_64'",
    "deep-gemm ; platform_machine == 'x86_64'",
    "nixl",
    "nixl-cu12 ; platform_machine == 'x86_64'",
    "vllm-router ; platform_machine == 'x86_64'",
    "modelexpress",
]
```

Without it, **`modelexpress` is not installed**, and the inference worker crashes at the first import of `prime_rl.inference.vllm.worker.nixl_mx`:

```
File "/app/src/prime_rl/inference/vllm/worker/nixl_mx.py", line 7, in <module>
    from modelexpress import p2p_pb2
ModuleNotFoundError: No module named 'modelexpress'
```

The pre-PR-#2389 `Dockerfile.cuda` predates the `disagg` extra so this is an accidental gap, not an intentional opt-out. **Suggested change**: add `--extra disagg` (or rely on `--extra all`) for any image targeting `weight_broadcast.type=nixl_mx`. We've shipped `v0.7.1` as a one-line overlay that does this until the change can land in `Dockerfile.cuda` itself.

### 3.2 `LD_PRELOAD` path for libcudart.so.12 moved

The existing configmap's three run-scripts (`run_trainer.sh`, `run_inference.sh`, `run_orchestrator.sh`) all preload libcudart for ARM64 NIXL compatibility:

```bash
export LD_PRELOAD="/usr/local/cuda/lib64/libcudart.so.12:${LD_PRELOAD:-}"
```

`/usr/local/cuda` exists in the v0.5.2 image (which appears to have been built from a Dockerfile variant that retained the CUDA tooling in the final stage). In `v0.7.0` (built from the upstream `Dockerfile.cuda` as-is), the final stage is `python:3.12-slim` which **does not** have `/usr/local/cuda`. `libcudart.so.12` lives only inside the pip-installed `nvidia-cuda-runtime` wheel:

```
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
```

Symptom on v0.7.0 with the unmodified configmap:

```
ERROR: ld.so: object '/usr/local/cuda/lib64/libcudart.so.12' from LD_PRELOAD cannot be preloaded
```

**Fix applied**: the three run-scripts now use the wheel-internal path. Alternative: symlink `/usr/local/cuda/lib64/libcudart.so.12 -> /app/.venv/.../libcudart.so.12` in the Dockerfile's final stage. Either works; we picked the env-var path because it's a configmap edit, no image rebuild.

### 3.3 The configmap `patch_nixl_mx.py` and Phase 2 source coexist

The kavin namespace runs a configmap-injected monkeypatch at container start (`patch_nixl_mx.py`) that rewrites `src/prime_rl/trainer/rl/broadcast/nixl_mx.py` to add same-rank-only peer filter + freshest-per-rank dedup *at TransportPlan construction time*.

Phase 2 ([PR #1](https://github.com/KavinKrishnan/prime-rl/pull/1)) adds the same semantic guarantees but at a different layer — inside `src/prime_rl/transport/mx_rendezvous.py:wait_for_all_peers_ready`. The two patches are **complementary, not redundant**:

| Code path | Bug class | Covered by |
|---|---|---|
| `trainer/rl/broadcast/nixl_mx.py:lazy_init` → `TransportPlan(peer_metadata=…)` | Trainer adds dead peers as NIXL remote agents during the per-step broadcast | `patch_nixl_mx.py` (runtime monkeypatch) |
| `transport/mx_rendezvous.py:wait_for_all_peers_ready(role="trainer")` | Orchestrator counts historical trainer entries in Redis and times out waiting for `n_historical` to all reach READY when only `n_alive` exist | Phase 2 PR (source-level) |

On v0.7.1 + the existing configmap, both fire. The trainer log shows `[patch_nixl_mx] PATCHED v2 (kavin_freshest_per_rank)` from the configmap script; the rendezvous wait methods get the Phase 2 dedup automatically because the source is in the baked image. Empirically the orchestrator restart pattern we saw on v0.5.2 (~once per 30-66 min on this workload) should go away on v0.7.1. **Validation pending** — image just deployed at time of writing.

When PR #1 merges upstream, the configmap monkeypatch becomes redundant for the trainer-side path too and should be removed. Until then, both layers complement each other.

## 4. Cluster observations under v0.5.2 + configmap monkeypatch

For the record, the v0.5.2 + configmap-monkeypatch combination we ran for 8+ hours before v0.7.1 deploy:

- Workload: Qwen3-30B-A3B-Instruct-2507, FSDP 2×2, EP=4 (32/128 experts per rank), FLASHINFER attention, gsm8k env
- Trainer steady state: ~10–21 s/step (varies with sequence length 280–500 tokens)
- Reward signal: variance 0.5–1.0 per orchestrator step — **real learning gradient**, not just reward=1.0 collapse
- Off-policy level: 0 across all observed steps (in-lockstep refit)
- Best uninterrupted window: **183 successful RL refit cycles over 66 min** between orchestrator restarts
- Zero NIXL data-plane errors (no `REMOTE_DISCONNECT`, no `NOT_ALLOWED`, no stale-READY) — confirms the same-rank-only + freshest-per-rank patches are correct
- Recurring orchestrator timeout pattern: `TimeoutError: timed out after 1200.0s waiting for 12 'trainer' peers to reach status 1 (saw 4)` — exactly what Phase 2's rendezvous-level dedup fixes

That last bullet is the bug class v0.7.1 is meant to eliminate. The configmap monkeypatch couldn't fix it because the relevant call site is in the orchestrator's rendezvous, which is in a different module from the trainer-side broadcast the monkeypatch was rewriting.

## 5. Branches + image artifacts pushed

| Branch | What's in it | Where |
|---|---|---|
| [`kavink/post-2389-kernel-compile-plan`](https://github.com/KavinKrishnan/prime-rl/tree/kavink/post-2389-kernel-compile-plan) | RFC document + this build-notes doc | `KavinKrishnan/prime-rl` |
| [`kavink/post-2389-phase2-rendezvous-fixes`](https://github.com/KavinKrishnan/prime-rl/tree/kavink/post-2389-phase2-rendezvous-fixes) | Phase 2 source (heartbeat + dedup + same-rank), 11/11 unit tests green, plus the `modelexpress.heartbeat` module-path tolerance fix | [Draft PR #1](https://github.com/KavinKrishnan/prime-rl/pull/1) |
| [`kavink/post-2389-conversion-registry-extensions`](https://github.com/KavinKrishnan/prime-rl/tree/kavink/post-2389-conversion-registry-extensions) | Phase 3 conversion-registry extensions (`compile_target` + `compile_metadata` + `cutlass_fp8_e4m3_per_channel`), 19/19 unit tests green | [Draft PR #2](https://github.com/KavinKrishnan/prime-rl/pull/2) |
| [`kavink/post-2389-image-build-2026-05-28`](https://github.com/KavinKrishnan/prime-rl/tree/kavink/post-2389-image-build-2026-05-28) | Merge of Phase 2 + Phase 3 + the import-tolerance fix; this is the exact source tree v0.7.0 / v0.7.1 was built from | `KavinKrishnan/prime-rl` (this push) |

Image artifacts on `nvcr.io/nvidian/dynamo-dev/`:

- `prime-rl-mx-on-nixl:v0.7.0-kavin-phase2-phase3` — full from-scratch ARM64 build (broken — missing `disagg`)
- `prime-rl-mx-on-nixl:v0.7.1-kavin-phase2-phase3` — overlay that adds `disagg` extra

MX side ([`ai-dynamo/modelexpress#349`](https://github.com/ai-dynamo/modelexpress/pull/349)) updated with the graduation glue commit that plumbs `ConversionEntry.compile_target` + `ConversionEntry.compile_metadata` through `MxV2TrainingPublisher.add_tensor(compile_target=…, compile_metadata=…)`. Wire round-trip is unit-tested.

## 6. What to update in the RFC (`post-pr2389-kernel-compile-plan.md`) — but not yet

These are the four edits queued in [`pensieve/RL/PrimeRL/09_rfc_updates_needed.md`](https://github.com/ai-dynamo/modelexpress/) (internal), augmented by what we learned from the build:

1. **Reframe Phase 3** — trainer-side post-processed direct is primary; receiver-side compile passes are v4+ (scratch buffers are a fallback only, not the primary v3 design as the original RFC implied).
2. **Add Phase 0** — Phase B UCX/dma-buf env profile as cluster prerequisite (`UCX_TLS=rc,cuda_copy`, `NIXL_UCX_TLS=rc,cuda_copy`, `UCX_CUDA_COPY_DMABUF=yes`, etc.) — from NeMo-RL + Dynamo's empirical 380 Gbps validation.
3. **Mark Phases 2/3/4 as shipped, not paper** (with PR + commit references).
4. **Add a sub-section on conversion registry extensions** documenting the `ConversionEntry` schema extension + how to add a new kernel (~80 LOC per kernel).

**New from this build experience**:

5. **Document the `disagg` extra requirement in §0** alongside the env profile — easy gotcha that costs an entire rebuild to discover.
6. **Document the `LD_PRELOAD` path** for libcudart in §0 — pre-existing run-scripts assumed v0.5.2's `/usr/local/cuda` layout.
7. **§4 / §5 on the "fallback path"** — describe vLLM PR #43375 (Ray Direct Transport, Anyscale) as the canonical receiver-pull-via-load_weights instance of the fallback path. Our positioning of trainer-side post-processed direct as "primary path, zero receive-side compute" is unchanged; RDT is the upstream-stamped instance of the alternative receiver-pull path.

## 7. Open follow-ups

- **Validate v0.7.1 end-to-end** on kavin (pending — v0.7.1 just deployed). Expected: zero NIXL errors AND zero orchestrator-`wait_for_all_peers_ready` timeouts. If both hold for a long uninterrupted window (>3 hours), we declare the source-baked Phase 2 + Phase 3 production-ready.
- **Send a one-line PR to upstream `Dockerfile.cuda`** adding `--extra disagg` (or `--extra all`). Tiny patch, unblocks every other team trying to bake `nixl_mx` mode into an image.
- **MX side: roll the server with the `SourceIdentity` round-trip fix** (proto change committed but not deployed). After that, the `__mx_v2_meta__` sidecar transport workaround can be dropped from `MxV2TrainingPublisher` (it's already filtered before NIXL register via [PR #295](https://github.com/ai-dynamo/modelexpress/pull/295)).
- **`pull_one(name)` semantic on MX** — inspired by vLLM PR #43375's RDT contract. Would let MX expose Ray-like per-tensor elasticity without abandoning the trainer-side compile model. ~50 LOC; not on the critical path but a clean addition for the post-Phase-4 work.
