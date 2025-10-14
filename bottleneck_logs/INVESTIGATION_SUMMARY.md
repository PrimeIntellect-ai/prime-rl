# Bottleneck Investigation Summary

## Executive Summary
Implemented comprehensive logging instrumentation across the prime-rl codebase to identify and diagnose bottlenecks in the model weight reload/update operations. The primary suspected bottleneck is **front-end queue saturation** where weight update requests wait behind long-running streaming inference requests.

## Problem Statement
The `reload_model` operation on the dual A100 machine (ubuntu@216.81.248.153) is experiencing significant delays, causing training slowdowns. The exact bottleneck location (network, client, server accept, engine reload, or checkpoint I/O) was unknown.

## Solution: Multi-Layer Instrumentation

### Architecture Overview
```
┌──────────────┐  HTTP POST   ┌──────────────┐  collective_rpc  ┌──────────────┐
│ Orchestrator │ ───────────> │ API Server   │ ──────────────> │ vLLM Engine  │
│   Client     │              │ (FastAPI)    │                 │   Worker     │
└──────────────┘              └──────────────┘                 └──────────────┘
      ↑                              ↑                               ↑
      │                              │                               │
  [weights]                      [weights]                       (internal)
  client.send                    server.recv                     collective_rpc
  client.done                    rpc.start
  wall_ms                        rpc.done
  queue_ms                       rpc_ms
  rpc_ms
```

### 1. Client-Side Instrumentation (`src/prime_rl/orchestrator/client.py`)

**Changes:**
- Added `_trace_id()` for request correlation across logs
- Added `_admin_client()` - dedicated HTTP client with:
  - Single connection limit
  - No keep-alive (closes after each request)
  - Avoids queuing behind streaming inference connections
- Instrumented `update_weights()` and `reload_weights()` with:
  ```python
  t0 = time.monotonic()
  # ... send request via dedicated admin client ...
  t1 = time.monotonic()
  wall_ms = (t1 - t0) * 1000.0  # Total client time
  rpc_ms = payload.get("rpc_ms")  # Server-reported RPC time
  queue_ms = wall_ms - rpc_ms  # Queuing delay
  ```

**Logs Produced:**
```
[weights][abc123] client.send url=http://... path=... t=1234.567890
[weights][abc123] client.done wall_ms=3556.1 rpc_ms=233.2 queue_ms=3322.9
```

**Key Metrics:**
- `wall_ms`: Total time from client perspective
- `rpc_ms`: Server-side engine operation time
- `queue_ms`: Time spent waiting for server to accept/process (wall_ms - rpc_ms)

### 2. Server-Side Instrumentation (`src/prime_rl/inference/vllm/server.py`)

**Changes:**
- Added `FirstByteMiddleware` to log when server receives requests:
  ```python
  class FirstByteMiddleware(BaseHTTPMiddleware):
      async def dispatch(self, request, call_next):
          tid = request.headers.get("x-trace-id", "-")
          t = time.monotonic()
          logger.info(f"[weights][{tid}] server.recv path={request.url.path} t={t:.6f}")
          return await call_next(request)
  ```
- Instrumented `/update_weights` and `/reload_weights` endpoints:
  ```python
  t0 = time.monotonic()
  await engine_client.collective_rpc("update_weights", args=(model_path,))
  t1 = time.monotonic()
  rpc_ms = (t1 - t0) * 1000.0
  return {"status": "ok", "rpc_ms": rpc_ms}  # Return to client
  ```

**Logs Produced:**
```
[weights][abc123] server.recv path=/update_weights t=1237.890123
[weights][abc123] rpc.start op=update_weights t=1237.890234
[weights][abc123] rpc.done op=update_weights t=1238.123456 rpc_ms=233.2
```

**Key Metrics:**
- Time between `client.send` and `server.recv` = network + accept queue
- `rpc_ms`: Pure engine operation time

### 3. Checkpoint Pipeline (`src/prime_rl/trainer/rl/train.py`, `src/prime_rl/orchestrator/orchestrator.py`)

**Trainer Changes:**
```python
# In training loop, when saving weights:
t0 = time.time()
weight_ckpt_manager.save(model, tokenizer, step=progress.step)
write_ms = (time.time() - t0) * 1000.0
logger.info(f"[ckpt] write.done step={progress.step} write_ms={write_ms:.1f}")
```

**Orchestrator Changes:**
```python
# When waiting for checkpoint:
logger.info(f"[ckpt] wait.start target_step={ckpt_step}")
t0 = time.time()
await async_wait_for_weight_checkpoint(get_weights_dir(config.output_dir), ckpt_step)
wait_ms = (time.time() - t0) * 1000.0
logger.info(f"[ckpt] wait.done target_step={ckpt_step} wait_ms={wait_ms:.1f}")
```

**Key Metrics:**
- `write_ms`: Time to write checkpoint to disk
- `wait_ms`: Time orchestrator waits for checkpoint to appear

### 4. Generation & RL Metrics (`src/prime_rl/orchestrator/orchestrator.py`)

**Generation Batching:**
```python
logger.info(f"[gen] batch.start inflight={len(inflight_tasks)} target_batch={config.batch_size}")
t0 = time.time()
# ... generate completions ...
dur_ms = (time.time() - t0) * 1000.0
logger.info(f"[gen] batch.done completions={len(problem_ids)} dur_ms={dur_ms:.1f} inflight={len(inflight_tasks)}")
```

**Rollout Quality:**
```python
trunc_pct = 100.0 * sum(is_truncated) / max(1, len(is_truncated))
staleness = progress.step - ckpt_step
logger.info(f"[rollout] trunc_pct={trunc_pct:.1f} current_step={progress.step} ckpt_step={ckpt_step} staleness={staleness}")
```

## Diagnostic Signatures

### Signature 1: Front-end Queue Bottleneck (Expected Primary Issue)
```
[weights][abc123] client.send t=1234.567890
[long gap - 30+ seconds]
[weights][abc123] server.recv t=1264.890123  ← Delayed 30s!
[weights][abc123] rpc.start t=1264.890234
[weights][abc123] rpc.done t=1265.123456 rpc_ms=233.2  ← Fast!
[weights][abc123] client.done wall_ms=30556.1 rpc_ms=233.2 queue_ms=30322.9  ← PROBLEM!
```

**Diagnosis:**
- `queue_ms >> rpc_ms` (e.g., 30000ms vs 233ms)
- Server was busy handling streaming inference requests
- Weight update request waited in accept queue

**Root Cause:**
- Shared HTTP client connection pool saturated with long-lived streaming requests
- New admin requests (POST /update_weights) wait for available connection
- vLLM uses async I/O but connection accept is sequential

**Fix:**
✅ **Already Implemented**: `_admin_client()` uses dedicated connection with no keep-alive
- Creates fresh connection for each admin request
- No sharing with streaming inference connections
- Should reduce `queue_ms` from ~30000ms to <300ms

### Signature 2: Engine Reload Bottleneck
```
[weights][abc123] client.send t=1234.567890
[weights][abc123] server.recv t=1234.567901  ← Immediate!
[weights][abc123] rpc.start t=1234.568012
[long gap - 10+ seconds]
[weights][abc123] rpc.done t=1244.678123 rpc_ms=10110.1  ← Slow!
[weights][abc123] client.done wall_ms=10110.5 rpc_ms=10110.1 queue_ms=0.4
```

**Diagnosis:**
- `rpc_ms >> 2000ms` (e.g., 10000ms for a 7B model)
- Server accepted request immediately but engine operation is slow
- Weight loading from disk/CPU to GPU is the bottleneck

**Root Cause:**
- Slow checkpoint I/O (disk read)
- Large model size
- Inefficient weight transfer CPU→GPU

**Potential Fixes:**
- Use faster storage (NVMe vs SATA SSD vs HDD)
- Reduce model size (quantization, smaller variant)
- Optimize vLLM worker checkpoint loading
- Pre-load checkpoints into memory

### Signature 3: Checkpoint I/O Bottleneck
```
[ckpt] wait.start target_step=42
[long gap - 5+ seconds]
[ckpt] wait.done target_step=42 wait_ms=5432.1
```

**Diagnosis:**
- Orchestrator waits a long time for trainer to write checkpoint
- Could be slow disk I/O on trainer side

**Potential Fixes:**
- Profile trainer checkpoint writes
- Use faster storage
- Reduce checkpoint size (sharding, compression)
- Optimize async checkpoint writing

## Analysis Tools

### `analyze_logs.sh`
Automated log analysis script that:
1. Extracts all `[weights]`, `[ckpt]`, `[gen]`, `[rollout]` entries
2. Computes statistics (mean, stddev, min, max) for:
   - `queue_ms`
   - `rpc_ms`
   - `wait_ms`
   - `write_ms`
   - `dur_ms` (generation)
   - `trunc_pct`
   - `staleness`
3. Identifies top 10 slowest operations
4. Generates timeline correlation
5. Outputs `SUMMARY.md` with diagnosis

**Usage:**
```bash
./bottleneck_logs/analyze_logs.sh <path_to_orchestrator.log>
```

**Output:**
```
bottleneck_logs/analysis_<timestamp>/
├── SUMMARY.md              # Diagnosis and recommendations
├── queue_ms_stats.txt
├── rpc_ms_stats.txt
├── ckpt_wait_stats.txt
├── slowest_updates.txt
└── timeline.txt
```

## Testing Protocol

### 1. Deploy Instrumented Code
```bash
git add src/prime_rl/orchestrator/client.py \
        src/prime_rl/inference/vllm/server.py \
        src/prime_rl/trainer/rl/train.py \
        src/prime_rl/orchestrator/orchestrator.py
git commit -m "[ENG-XXX] Add comprehensive bottleneck logging"
git push

# On remote machine
ssh ubuntu@216.81.248.153
cd ~/prime-rl
git pull
```

### 2. Start Services
```bash
# Terminal 1: Inference server
uv run inference --model <model> --host 0.0.0.0 --port 8000 --tensor-parallel-size 2

# Terminal 2: Trainer
uv run trainer --config configs/rl_trainer.yaml --output_dir ~/outputs/test

# Terminal 3: Orchestrator
uv run orchestrator --config configs/orchestrator.yaml --output_dir ~/outputs/test
```

### 3. Monitor Real-Time
```bash
# Terminal 4: Watch weight operations
tail -f ~/outputs/test/logs/orchestrator.log | \
  grep --line-buffered -E '\[weights\]|\[ckpt\]|\[gen\]|\[rollout\]'
```

### 4. Collect & Analyze
```bash
# After run completes, analyze logs
./bottleneck_logs/analyze_logs.sh ~/outputs/test/logs/orchestrator.log
```

## Expected Results

### Before Fix (Baseline with Shared Client)
If testing without `_admin_client()` (revert to old code):
```
Queue Delay Analysis:
  Mean: 28453.2 ms  ← PROBLEM
  Max: 45678.9 ms

🔴 PRIMARY BOTTLENECK: Front-end Queue Saturation
```

### After Fix (With Dedicated Admin Client)
With current implementation:
```
Queue Delay Analysis:
  Mean: 87.2 ms  ← FIXED
  Max: 243.1 ms

✅ Queue delays within acceptable range (<300ms)
```

## Remote Machine Status
- **Host**: ubuntu@216.81.248.153
- **Hostname**: 0038-dsm-gba100-prxmx30196
- **GPUs**: 2x NVIDIA A100 80GB PCIe
- **Current State**: Idle (0% utilization, 0 MiB used)
- **Status**: Ready for testing

## Files Modified

### Code Changes
1. `src/prime_rl/orchestrator/client.py`
   - Added `_trace_id()`, `_server_base_from_oai()`, `_admin_client()`
   - Instrumented `update_weights()` and `reload_weights()`

2. `src/prime_rl/inference/vllm/server.py`
   - Added `FirstByteMiddleware`
   - Instrumented `/update_weights` and `/reload_weights` endpoints

3. `src/prime_rl/trainer/rl/train.py`
   - Added checkpoint write timing

4. `src/prime_rl/orchestrator/orchestrator.py`
   - Added checkpoint wait timing
   - Added generation batch timing
   - Added rollout truncation/staleness metrics

### Documentation & Tools
1. `bottleneck_logs/README.md` - Comprehensive investigation guide
2. `bottleneck_logs/analyze_logs.sh` - Automated log analysis script
3. `bottleneck_logs/DEPLOYMENT_GUIDE.md` - Step-by-step deployment instructions
4. `bottleneck_logs/INVESTIGATION_SUMMARY.md` - This file

## Next Steps

### Immediate
1. ✅ Commit instrumented code
2. ✅ Create documentation and analysis tools
3. ⏳ Deploy to A100 machine
4. ⏳ Run test with high concurrency
5. ⏳ Collect logs and analyze
6. ⏳ Validate that `queue_ms` is reduced with dedicated admin client

### Follow-up (If Needed)
1. If `queue_ms` still high:
   - Consider separate API server for admin endpoints
   - Investigate vLLM connection handling

2. If `rpc_ms` is problematic:
   - Profile vLLM worker checkpoint loading
   - Optimize storage path
   - Consider model size reduction

3. If `wait_ms` is problematic:
   - Profile trainer checkpoint writes
   - Optimize async checkpoint saving

## Success Criteria
1. **Primary Goal**: `queue_ms < 300ms` consistently
2. **Secondary Goal**: `rpc_ms < 2000ms` for 7B model, scaled for larger
3. **RL Quality**: `trunc_pct < 10%`, `staleness ≤ async_level`
4. **No Regression**: Generation throughput maintained

## References
- Original request: User provided detailed logging specification
- Suspected bottleneck: Front-end accept queue saturation
- Fix implemented: Dedicated admin client with connection isolation
- Remote machine: ubuntu@216.81.248.153 (dual A100)
