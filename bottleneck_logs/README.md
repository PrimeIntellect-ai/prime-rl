# Bottleneck Investigation - Model Reload Performance

## Overview
This investigation aims to identify bottlenecks in the reload_model/update_weights operations on the dual A100 machine (ubuntu@216.81.248.153).

## Instrumentation Added

### 1. Orchestrator Client (`src/prime_rl/orchestrator/client.py`)
- Added dedicated `_admin_client()` with short-lived connections to avoid queuing behind streaming requests
- Added `_trace_id()` for request correlation
- Instrumented `update_weights()` and `reload_weights()` with:
  - `[weights][{tid}] client.send` - Client send timestamp
  - `[weights][{tid}] client.done` - Client completion with metrics:
    - `wall_ms`: Total client-side time
    - `rpc_ms`: Server-reported RPC time (from server response)
    - `queue_ms`: Queuing delay (wall_ms - rpc_ms)

### 2. Inference Server (`src/prime_rl/inference/vllm/server.py`)
- Added `FirstByteMiddleware` to log when server receives requests:
  - `[weights][{tid}] server.recv` - First byte received timestamp
- Instrumented custom endpoints:
  - `/update_weights`:
    - `[weights][{tid}] rpc.start op=update_weights` - Engine RPC start
    - `[weights][{tid}] rpc.done op=update_weights rpc_ms={X}` - Engine RPC completion
  - `/reload_weights`:
    - `[weights][{tid}] rpc.start op=reload_weights` - Engine RPC start
    - `[weights][{tid}] rpc.done op=reload_weights rpc_ms={X}` - Engine RPC completion

### 3. Trainer (`src/prime_rl/trainer/rl/train.py`)
- Added checkpoint write timing:
  - `[ckpt] write.done step={X} write_ms={Y}` - Checkpoint write completion

### 4. Orchestrator (`src/prime_rl/orchestrator/orchestrator.py`)
- Added checkpoint wait timing:
  - `[ckpt] wait.start target_step={X}` - Start waiting for checkpoint
  - `[ckpt] wait.done target_step={X} wait_ms={Y}` - Checkpoint available
- Added generation batch timing:
  - `[gen] batch.start inflight={X} target_batch={Y}` - Batch generation start
  - `[gen] batch.done completions={X} dur_ms={Y} inflight={Z}` - Batch completion
- Added rollout metrics:
  - `[rollout] trunc_pct={X} current_step={Y} ckpt_step={Z} staleness={W}` - Truncation and staleness stats

## Expected Bottleneck Signatures

### Front-end/Accept Queue (Main Suspect)
```
[weights][abc123] client.send url=... t=1234.567890
[long gap - several seconds]
[weights][abc123] server.recv path=/update_weights t=1237.890123  ← Delayed receive
[weights][abc123] rpc.start op=update_weights t=1237.890234
[weights][abc123] rpc.done op=update_weights t=1238.123456 rpc_ms=233.2  ← Fast RPC
[weights][abc123] client.done wall_ms=3556.1 rpc_ms=233.2 queue_ms=3322.9  ← Large queue_ms
```
**Diagnosis**: `queue_ms >> rpc_ms` indicates the server was busy accepting/processing other requests (likely streaming inference).

### Engine Reload Slow
```
[weights][abc123] client.send url=... t=1234.567890
[weights][abc123] server.recv path=/update_weights t=1234.567901  ← Immediate receive
[weights][abc123] rpc.start op=update_weights t=1234.568012
[long gap]
[weights][abc123] rpc.done op=update_weights t=1245.678123 rpc_ms=11110.1  ← Slow RPC
[weights][abc123] client.done wall_ms=11110.5 rpc_ms=11110.1 queue_ms=0.4
```
**Diagnosis**: `rpc_ms >> 2000ms` indicates the weight loading operation itself is slow.

### Checkpoint I/O Slow
```
[ckpt] wait.start target_step=42
[long gap]
[ckpt] wait.done target_step=42 wait_ms=5432.1  ← Long wait
```
**Diagnosis**: Trainer is slow to write or orchestrator is polling wrong path.

## Metrics to Track

### Critical Path (Weight Update)
1. **Queue Delay** (`queue_ms`): Should be < 300ms, ideally < 100ms
2. **RPC Time** (`rpc_ms`): Should be < 2000ms depending on model size
3. **Wall Time** (`wall_ms`): Total client-side time

### Checkpoint Pipeline
1. **Write Time** (`write_ms`): Trainer checkpoint write duration
2. **Wait Time** (`wait_ms`): Orchestrator waiting for checkpoint

### Generation Pressure
1. **Batch Duration** (`dur_ms`): Time to complete batch
2. **Inflight Tasks**: Number of concurrent streaming requests

### RL Hygiene
1. **Truncation %** (`trunc_pct`): Should be near 0 if max_seq_len is sufficient
2. **Staleness**: `current_step - ckpt_step` (should be ≤ async_level)

## Testing Protocol

### 1. Deploy Instrumented Code
```bash
# From local machine
cd /Users/ameenp/prime-development/PRIMERL-127/prime-rl
git add -A
git commit -m "[ENG-XXX] Add comprehensive bottleneck logging"
git push

# On remote machine
ssh ubuntu@216.81.248.153
cd ~/prime-rl  # or wherever repo is
git pull
```

### 2. Start Services
```bash
# Terminal 1: Start vLLM inference server
uv run inference --model <model_name> --host 0.0.0.0 --port 8000

# Terminal 2: Start trainer
uv run trainer --config configs/trainer.yaml

# Terminal 3: Start orchestrator
uv run orchestrator --config configs/orchestrator.yaml
```

### 3. Monitor Logs
```bash
# Watch orchestrator logs for [weights] tags
tail -f <output_dir>/logs/orchestrator.log | grep -E '\[weights\]|\[ckpt\]|\[gen\]|\[rollout\]'

# Watch inference server logs
# (vLLM logs to stdout by default)

# Watch trainer logs
tail -f <output_dir>/logs/trainer/rank_0.log | grep '\[ckpt\]'
```

### 4. Trigger Weight Update
Weight updates happen automatically when `progress.step - ckpt_step > async_level`.
Look for the sequence in orchestrator logs.

## Expected Results (Baseline)
- **Small models (≤2B)**: `wall_ms` < 1000ms, `queue_ms` < 100ms, `rpc_ms` < 500ms
- **Medium models (7B)**: `wall_ms` < 3000ms, `queue_ms` < 300ms, `rpc_ms` < 2000ms
- **Large models (70B+)**: `wall_ms` varies, but `queue_ms` should still be < 500ms

## Fix Validation
After applying fixes (e.g., dedicated admin client, separate API server):
1. Run same workload
2. Compare `queue_ms` values
3. `queue_ms` should drop from ~30000ms to < 300ms
4. `rpc_ms` should remain stable
5. Generation throughput should not be affected

## Remote Machine Status (as of investigation start)
- **Host**: 0038-dsm-gba100-prxmx30196
- **GPUs**: 2x NVIDIA A100 80GB PCIe (0% utilization, 0 MiB used)
- **Status**: No active processes found
- **Next Steps**: Deploy and start services to reproduce bottleneck

## Log Analysis Commands
```bash
# Extract all weight update operations with trace IDs
grep '\[weights\]' orchestrator.log | grep -E 'client\.(send|done)|server\.recv|rpc\.(start|done)'

# Calculate queue_ms statistics
grep 'queue_ms=' orchestrator.log | sed -n 's/.*queue_ms=\([0-9.]*\).*/\1/p' | \
  awk '{sum+=$1; sumsq+=$1*$1; n++} END {print "Mean:", sum/n, "StdDev:", sqrt(sumsq/n - (sum/n)^2), "Max:", max}'

# Find slowest updates
grep 'wall_ms=' orchestrator.log | sort -t= -k2 -n | tail -10

# Correlate with generation activity
grep -E '\[gen\] batch\.(start|done)|\[weights\]' orchestrator.log
```

## File Locations
- **Modified Files**:
  - `src/prime_rl/orchestrator/client.py`
  - `src/prime_rl/inference/vllm/server.py`
  - `src/prime_rl/trainer/rl/train.py`
  - `src/prime_rl/orchestrator/orchestrator.py`
- **Log Output**: This directory (`bottleneck_logs/`)
