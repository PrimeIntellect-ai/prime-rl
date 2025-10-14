# Log Patterns Quick Reference

## Weight Update Operations

### Normal Flow (Fast)
```
[weights][abc123] client.send url=http://192.168.1.100:8000 path=/models/.../step_42 t=1234.567890
[weights][abc123] server.recv path=/update_weights t=1234.567901  [~11ms gap]
[weights][abc123] rpc.start op=update_weights t=1234.568012
[weights][abc123] rpc.done op=update_weights t=1235.823456 rpc_ms=1255.4
[weights][abc123] client.done wall_ms=1255.6 rpc_ms=1255.4 queue_ms=0.2

✅ Good: queue_ms ~0ms, rpc_ms reasonable for model size
```

### Bottleneck Pattern (Front-end Queue)
```
[weights][def456] client.send url=http://192.168.1.100:8000 path=/models/.../step_43 t=1240.123456
[weights][def456] server.recv path=/update_weights t=1270.456789  [~30 SECOND GAP!]
[weights][def456] rpc.start op=update_weights t=1270.456890
[weights][def456] rpc.done op=update_weights t=1271.678901 rpc_ms=1222.0
[weights][def456] client.done wall_ms=31555.4 rpc_ms=1222.0 queue_ms=30333.4

⚠️  Problem: queue_ms = 30s! Server was busy accepting/handling other requests.
```

### Bottleneck Pattern (Slow Engine)
```
[weights][ghi789] client.send url=http://192.168.1.100:8000 path=/models/.../step_44 t=1280.234567
[weights][ghi789] server.recv path=/update_weights t=1280.234578  [~11ms gap - fast accept]
[weights][ghi789] rpc.start op=update_weights t=1280.234689
[weights][ghi789] rpc.done op=update_weights t=1295.345678 rpc_ms=15111.0  [~15s RPC!]
[weights][ghi789] client.done wall_ms=15111.1 rpc_ms=15111.0 queue_ms=0.1

⚠️  Problem: rpc_ms = 15s! Weight loading itself is slow (I/O or GPU transfer).
```

## Checkpoint Operations

### Normal Flow
```
[ckpt] write.done step=42 write_ms=823.4
[ckpt] wait.start target_step=42
[ckpt] wait.done target_step=42 wait_ms=12.3

✅ Good: Checkpoint written and immediately available
```

### Slow Checkpoint Write
```
[ckpt] write.done step=43 write_ms=8234.5

⚠️  Problem: Taking 8+ seconds to write checkpoint (slow disk or large model)
```

### Long Checkpoint Wait
```
[ckpt] wait.start target_step=43
[ckpt] wait.done target_step=43 wait_ms=5432.1

⚠️  Problem: Orchestrator waited 5+ seconds for checkpoint (trainer slow or polling issue)
```

## Generation Operations

### Normal Flow
```
[gen] batch.start inflight=96 target_batch=64
[gen] batch.done completions=64 dur_ms=2345.6 inflight=96

✅ Good: Batch completed in reasonable time
```

### Spike During Weight Update
```
[gen] batch.start inflight=128 target_batch=64
[weights][xyz] client.send ...
[weights][xyz] client.done wall_ms=30000 queue_ms=29000 ...
[gen] batch.done completions=64 dur_ms=45678.9 inflight=140

⚠️  Problem: Generation batch spiked to 45s coinciding with weight update queue delay
         Indicates shared resource contention.
```

## Rollout Operations

### Good Quality
```
[rollout] trunc_pct=2.3 current_step=10 ckpt_step=9 staleness=1

✅ Good: Low truncation, staleness within async_level
```

### High Truncation
```
[rollout] trunc_pct=78.5 current_step=10 ckpt_step=9 staleness=1

⚠️  Problem: 78% truncation! max_seq_len too small or problems too long.
```

### High Staleness (Barrier Hit)
```
[rollout] trunc_pct=3.2 current_step=15 ckpt_step=10 staleness=5

⚠️  Note: Staleness = async_level (5), orchestrator hit async barrier.
         Check why checkpoint wasn't ready sooner.
```

## Grep Commands

### Extract All Weight Operations
```bash
grep '\[weights\]' orchestrator.log
```

### Extract Specific Trace ID
```bash
grep '\[abc123\]' orchestrator.log
```

### Find Slow Updates (queue_ms > 1000)
```bash
grep 'queue_ms=' orchestrator.log | awk -F'queue_ms=' '{print $2}' | awk '{if ($1 > 1000) print $0}'
```

### Correlate Weight Updates with Generation
```bash
grep -E '\[weights\]|\[gen\] batch\.(start|done)' orchestrator.log | tail -100
```

### Timeline of Single Weight Update
```bash
TID="abc123"
grep "\[$TID\]" orchestrator.log
grep "\[$TID\]" inference_server.log  # If inference logs available
```

## Timing Breakdown

### Client Perspective (`orchestrator.log`)
```
client.send (t0)
    ↓
    [network transit + server queue]
    ↓
server.recv (from inference log, if available)
    ↓
    [server processing + engine RPC]
    ↓
client.done (t1)

wall_ms = t1 - t0  (total client-side time)
rpc_ms = from server response payload
queue_ms = wall_ms - rpc_ms  (time before server started RPC)
```

### Server Perspective (`inference_server.log`)
```
server.recv (middleware logs this)
    ↓
rpc.start (endpoint logs this)
    ↓
    [engine collective_rpc("update_weights")]
    ↓
rpc.done (endpoint logs this)

rpc_ms = time from rpc.start to rpc.done
```

## Expected Ranges (7B Model, 2x A100)

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| `queue_ms` | <100ms | 100-1000ms | >1000ms |
| `rpc_ms` (update_weights) | <2000ms | 2000-5000ms | >5000ms |
| `rpc_ms` (reload_weights) | <500ms | 500-2000ms | >2000ms |
| `write_ms` | <1000ms | 1000-5000ms | >5000ms |
| `wait_ms` | <500ms | 500-2000ms | >2000ms |
| `dur_ms` (batch) | 1000-5000ms | 5000-10000ms | >10000ms |
| `trunc_pct` | <10% | 10-30% | >30% |
| `staleness` | ≤async_level | >async_level | >async_level+2 |

## Common Issues & Signatures

### Issue 1: Shared Connection Pool Saturation
**Signature:**
- High `queue_ms` (>10s)
- Low `rpc_ms` (<2s)
- Coincides with high `inflight` generation tasks

**Fix:**
- Use dedicated admin client ✅ (implemented)

### Issue 2: Slow Weight Loading
**Signature:**
- Low `queue_ms` (<100ms)
- High `rpc_ms` (>5s)
- Happens even with no concurrent generation

**Fix:**
- Faster storage
- Smaller model
- Profile vLLM worker loading

### Issue 3: Checkpoint Pipeline Stall
**Signature:**
- High `wait_ms` (>2s)
- Orchestrator stuck at async barrier
- Trainer logs show slow `write_ms`

**Fix:**
- Optimize checkpoint writing
- Check filesystem performance

### Issue 4: Poor RL Quality
**Signature:**
- High `trunc_pct` (>30%)
- Large `staleness` spikes

**Fix:**
- Increase `max_seq_len`
- Tune `async_level`
- Check advantage computation

## Log Analysis Workflow

1. **Run experiment** with instrumented code
2. **Collect logs** from all components
3. **Run analysis script**:
   ```bash
   ./analyze_logs.sh orchestrator.log
   ```
4. **Check SUMMARY.md** for diagnosis
5. **Examine raw extracts** in `analysis_<timestamp>/` directory
6. **Correlate timelines** if needed:
   ```bash
   grep -E '\[weights\]|\[gen\]' orchestrator.log > timeline.txt
   ```
7. **Compare before/after** if testing fixes
