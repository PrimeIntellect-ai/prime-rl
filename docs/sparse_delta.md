# Sparse Delta Weight Synchronization

PrimeRL can synchronize dense trainer weights to remote vLLM deployments using
exact sparse deltas instead of transferring a full checkpoint after every
optimizer step. The path is intended for bandwidth-constrained or
geographically separated trainer and inference fleets.

## Configuration

Sparse deltas use the filesystem weight transport for trainer-side artifact
production. Transactional HTTP staging removes the requirement that inference
servers share the trainer filesystem.

A small local example is available at [`configs/debug/sparse-delta.toml`](../configs/debug/sparse-delta.toml).

```toml
[weight_broadcast]
type = "filesystem"
mode = "delta"
update_protocol = "stage_commit"
stage_transport = "streaming_upload"
background_stage = true
retain_all_deltas = true
delta_stream_group_size = 4

[orchestrator.model.client]
base_url = "http://inference-router:8000/v1"
admin_base_url = [
  "http://inference-a:8100/v1",
  "http://inference-b:8100/v1",
]
lease_enabled = true
lease_recovery_enabled = true
lease_cooldown_s = 20.0
lease_recovery_poll_interval_s = 5.0
```

`stage_transport` accepts:

- `shared_fs`: inference records the trainer-visible path.
- `http_upload`: one multipart upload per endpoint.
- `chunked_upload`: resumable offset-based chunks with final size and SHA-256 validation.
- `streaming_upload`: uploads append-only records while later layers are still being scanned.

The streaming format flushes after `delta_stream_group_size` transformer
layers. A value of zero flushes every record.

## Correctness model

Delta artifacts store logical Hugging Face parameter names and global shapes.
Inference workers map each logical update to the local vLLM layout only when it
is applied. The dense layout adapter handles replicated, row-parallel,
column-parallel, vocabulary-parallel, fused QKV, fused gate/up, and grouped-query
attention layouts at any valid tensor-parallel size.

Changed biases are included. Values normally retain the target weight dtype;
when subtraction and addition in that dtype cannot reproduce the target bits,
the artifact stores wider values. The inference loader performs the addition
in the promoted dtype before converting the result to the parameter dtype.
This affects weight-transfer payloads only and does not change optimizer or
reduction precision.

Every transactional delta carries a `version` and `base_version`. The server
validates the base during both stage and commit, serializes commits, and makes
repeated commits idempotent. A failed worker application marks the server dirty;
generation cannot resume until the base model is reloaded.

## Fan-out, relay, and recovery

`admin_base_url` may contain multiple static inference endpoints. Stage and
commit fan out to all healthy endpoints. PrimeRL records per-endpoint results
and can retire a failed endpoint when leases are enabled.

With `retain_all_deltas = true`, a recovered endpoint reloads its original base
weights and replays the complete committed chain before it is marked healthy.
Recovery validates that every replay entry names the preceding version as its
base.

For region-local fan-out, configure one inference server as a relay seed:

```toml
[relay]
enabled = true
peers = [
  "http://region-peer-a:8100/v1",
  "http://region-peer-b:8100/v1",
]
fail_on_peer_error = true
stage_timeout_s = 3600.0
commit_timeout_s = 600.0
reload_timeout_s = 600.0
```

The orchestrator sends each artifact only to relay seeds; each seed forwards
stage, commit, and reload operations to its peers. Keep
`fail_on_peer_error = true` when peers serve rollouts so a partially updated
region cannot silently continue.

PrimeRL's default `sticky_least_loaded` vLLM router policy provides
session-affine, load-aware rollout routing across replicas. Auto-launched
routers probe `/weight_health`, which removes a worker from rollout routing
while a failed update has left its weights dirty. Lease state separately
controls the admin endpoints used for synchronization and replay. External
routers should use the same health endpoint.

## Supported models and limitations

The current adapter supports unquantized dense Qwen3/Llama-style models.
Pipeline parallelism, expert parallelism, MoE layouts, LoRA, quantized or packed
weights, and nonstandard fused-weight conventions are rejected. Delta mode also
rejects checkpoint resume unless inference has first been synchronized to the
same full base version.

Stage/commit requires one API server to own the version state for each
independent inference endpoint. Replica scaling should use multiple independent
endpoints or a relay seed, not multiple API server processes sharing one engine.

## Verification utilities

Verify exact reconstruction of an artifact:

```bash
uv run python scripts/verify_sparse_delta.py \
  --base /path/to/base/model.safetensors \
  --target /path/to/updated/model.safetensors \
  --delta /path/to/delta.safetensors
```

Benchmark extraction and compact index encoding:

```bash
uv run python scripts/benchmark_extract_delta.py \
  --base /path/to/base/model.safetensors \
  --target /path/to/updated/model.safetensors
```
