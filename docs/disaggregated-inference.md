# Disaggregated Prefill/Decode Inference

Run MoE models with separate prefill and decode node groups for higher throughput.

## Quick Start

See [`configs/glm5_disagg_inference/inference.toml`](../configs/glm5_disagg_inference/inference.toml) for an example config.

```bash
uv run inference @ configs/glm5_disagg_inference/inference.toml --output-dir /data/$USER/outputs
```

## Prefill/Decode Ratio

| Workload | Recommended ratio (P:D) | Why |
|---|---|---|
| Agentic (SWE, Lean) | **3:1** | Long growing contexts → prefill-heavy |
| Non-agentic (math, chat) | **1:2** | Short prompts, long generations → decode-heavy |

Monitor live queue depths:
```bash
curl -s http://<prefill_node>:8100/metrics | grep num_requests_waiting
curl -s http://<decode_node>:8200/metrics | grep num_requests_waiting
```

If prefill has queued requests and decode has zero, add more prefill nodes (and vice versa).

For historical averages (cumulative over the entire run), query the histogram metrics:
```bash
# Average queue time per request (seconds)
curl -s http://<node>:<port>/metrics | awk '
  /request_queue_time_seconds_sum\{/  { sum += $2 }
  /request_queue_time_seconds_count\{/ { count += $2 }
  END { if (count > 0) printf "avg queue: %.2fs (%d requests)\n", sum/count, count }
'

# Average prefill/decode compute time
curl -s http://<node>:<port>/metrics | awk '
  /request_prefill_time_seconds_sum\{/  { ps += $2 }
  /request_prefill_time_seconds_count\{/ { pc += $2 }
  /request_decode_time_seconds_sum\{/   { ds += $2 }
  /request_decode_time_seconds_count\{/  { dc += $2 }
  END {
    if (pc > 0) printf "avg prefill: %.2fs\n", ps/pc
    if (dc > 0) printf "avg decode:  %.2fs\n", ds/dc
  }
'
```

Other useful metrics on the `/metrics` endpoint:
- `vllm:e2e_request_latency_seconds` — end-to-end latency
- `vllm:kv_cache_usage_perc` — KV cache memory pressure
- `vllm:nixl_xfer_time_seconds` — NIXL KV transfer duration
- `vllm:nixl_bytes_transferred` — bytes per KV transfer

## UCX 1.19

NVSHMEM requires UCX >= 1.19 for multi-GPU CUDA support. Most clusters ship UCX 1.17 (via HPC-X), which causes `cuStreamCreate: invalid device context` errors during DeepEP internode dispatch.

**Check your version:**
```bash
/opt/hpcx/ucx/bin/ucx_info -v | head -1
# If < 1.19, you need to build from source
```

**Build UCX 1.19 (run once on a GPU node):**
```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
```

This installs UCX 1.19 to `prime-rl/third_party/ucx/`. The sbatch template automatically adds it to `LD_LIBRARY_PATH`, overriding the system version.

## Troubleshooting

### `DeepEP error: timeout (dispatch CPU)`
NVSHMEM internode communication failing. Check:
1. UCX version >= 1.19? (`third_party/ucx/bin/ucx_info -v`)
2. NVSHMEM libs reachable at `/tmp/deepep_build/nvshmem/lib/`? If not:
   ```bash
   ssh <node> 'mkdir -p /tmp/deepep_build/nvshmem && \
       ln -sfn <venv>/lib/python3.12/site-packages/nvidia/nvshmem/lib \
       /tmp/deepep_build/nvshmem/lib'
   ```
3. IBGDA driver enabled? `ssh <node> 'cat /proc/driver/nvidia/params | grep EnableStreamMemOPs'` should show `1`.

### Router healthy but requests hang
NIXL side channel not running on prefill. Check:
```bash
ssh <prefill_node> 'ss -tlnp sport ge :5600 sport le :5608 | grep -c LISTEN'
# Should show 8 (one per DP rank). If 0, check logs for UCX/NVSHMEM errors.
```
