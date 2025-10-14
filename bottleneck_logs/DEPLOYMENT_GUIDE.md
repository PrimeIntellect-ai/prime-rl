# Deployment Guide for Bottleneck Investigation

## Quick Start

### 1. Commit and Push Changes
```bash
# From local machine (current directory)
cd /Users/ameenp/prime-development/PRIMERL-127/prime-rl
git add src/prime_rl/orchestrator/client.py \
        src/prime_rl/inference/vllm/server.py \
        src/prime_rl/trainer/rl/train.py \
        src/prime_rl/orchestrator/orchestrator.py \
        bottleneck_logs/

git commit -m "[ENG-XXX] Add comprehensive bottleneck logging

- Added trace ID tracking and timing for update_weights/reload_weights
- Instrumented with dedicated admin client to avoid queuing
- Added FirstByteMiddleware to track server receive time
- Added checkpoint write/wait timing
- Added generation batch timing
- Added rollout truncation/staleness metrics
- Created analysis scripts and documentation

Addresses slow reload_model operations by identifying queue vs RPC bottlenecks."

git push origin ameen/feat-multi-step-rollout
```

### 2. Deploy to A100 Machine
```bash
# SSH to remote machine
ssh ubuntu@216.81.248.153

# Navigate to prime-rl repo (adjust path as needed)
cd ~/prime-rl

# Pull latest changes
git fetch
git checkout ameen/feat-multi-step-rollout
git pull

# Install dependencies if needed
uv sync
```

### 3. Start Services (Example - Adjust configs as needed)

#### Terminal 1: Start vLLM Inference Server
```bash
ssh ubuntu@216.81.248.153

# Example command (adjust model, tensor parallel, etc.)
uv run inference \
  --model "meta-llama/Llama-2-7b-hf" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  2>&1 | tee ~/logs/inference_$(date +%Y%m%d_%H%M%S).log
```

#### Terminal 2: Start Trainer
```bash
ssh ubuntu@216.81.248.153

# Example command (adjust config path)
uv run trainer \
  --config configs/rl_trainer.yaml \
  --output_dir ~/outputs/bottleneck_test \
  --max_steps 10

# Logs will be in ~/outputs/bottleneck_test/logs/trainer/rank_0.log
```

#### Terminal 3: Start Orchestrator
```bash
ssh ubuntu@216.81.248.153

# Example command (adjust config path)
uv run orchestrator \
  --config configs/orchestrator.yaml \
  --output_dir ~/outputs/bottleneck_test \
  --max_steps 10 \
  --batch_size 64

# Logs will be in ~/outputs/bottleneck_test/logs/orchestrator.log
```

### 4. Monitor in Real-Time

#### Terminal 4: Watch Weight Update Logs
```bash
ssh ubuntu@216.81.248.153

# Watch orchestrator for weight operations
tail -f ~/outputs/bottleneck_test/logs/orchestrator.log | \
  grep --line-buffered -E '\[weights\]|\[ckpt\]|\[gen\]|\[rollout\]'
```

Look for patterns like:
```
[weights][abc123] client.send url=... t=1234.567890
[weights][abc123] client.done wall_ms=3556.1 rpc_ms=233.2 queue_ms=3322.9
```

If `queue_ms` is large (> 1000ms), you've reproduced the bottleneck!

### 5. Collect Logs After Run

```bash
# From remote machine
cd ~/outputs/bottleneck_test/logs

# Create tarball
tar -czf bottleneck_logs_$(date +%Y%m%d_%H%M%S).tar.gz \
  orchestrator.log \
  trainer/rank_*.log \
  ~/logs/inference_*.log

# Copy to local machine
# From local machine:
scp ubuntu@216.81.248.153:~/outputs/bottleneck_test/logs/bottleneck_logs_*.tar.gz \
  /Users/ameenp/prime-development/PRIMERL-127/prime-rl/bottleneck_logs/
```

### 6. Analyze Logs Locally

```bash
# From local machine
cd /Users/ameenp/prime-development/PRIMERL-127/prime-rl

# Extract tarball
cd bottleneck_logs
tar -xzf bottleneck_logs_*.tar.gz

# Run analysis script
./analyze_logs.sh orchestrator.log
```

This will generate:
- `bottleneck_logs/analysis_<timestamp>/`
  - `SUMMARY.md` - Key findings and diagnosis
  - `queue_ms_stats.txt` - Queue delay statistics
  - `rpc_ms_stats.txt` - RPC time statistics
  - `ckpt_wait_stats.txt` - Checkpoint wait statistics
  - `slowest_updates.txt` - Top 10 slowest operations
  - `timeline.txt` - Correlation between weight updates and generation

## Expected Outcomes

### Before Fix (Using Shared HTTP Client)
```
Queue Delay Statistics:
  Mean: 32445.3 ms  ← PROBLEM!
  Max: 45678.9 ms

🔴 PRIMARY BOTTLENECK: Front-end Queue Saturation
   - Weight update requests wait behind streaming inference
```

### After Fix (Using Dedicated Admin Client)
```
Queue Delay Statistics:
  Mean: 87.2 ms  ← FIXED!
  Max: 243.1 ms

✅ Queue delays within acceptable range
```

## Troubleshooting

### No [weights] logs appearing
- Check that `async_level` is set correctly
- Ensure trainer is actually writing checkpoints (`progress.step > 0`)
- Verify orchestrator hits the async barrier (`progress.step - ckpt_step > async_level`)

### No bottleneck observed
- Increase concurrent generation load:
  - Increase `batch_size`
  - Decrease `rollouts_per_example` to generate more problems
  - Increase `MAX_INFLIGHT_PROBLEMS` multiplier
- Use a faster model (e.g., 2B) to increase request rate
- Lower `max_tokens` to speed up individual requests

### Logs missing timestamp correlation
- Ensure clocks are synchronized (NTP)
- Use monotonic timestamps (already implemented via `time.monotonic()`)
- Focus on relative timings within same log file

## Configuration Tips

### To Stress Test Front-end
```yaml
# configs/orchestrator.yaml
batch_size: 128
rollouts_per_example: 4
async_level: 1  # Frequent weight updates

sampling:
  max_tokens: 512
  temperature: 0.8
```

### To Test Small Model (Fast Iteration)
```yaml
# configs/orchestrator.yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  # Or any fast 1-2B model
```

### To Test Large Model (Real Workload)
```yaml
# configs/orchestrator.yaml
model:
  name: "meta-llama/Llama-2-70b-hf"  # Or your actual production model
```

## Next Steps After Diagnosis

### If queue_ms is the bottleneck (most likely):
✅ **Already implemented**: Dedicated `_admin_client()` in `client.py`
- Verify by comparing queue_ms before/after this code change

### If rpc_ms is the bottleneck:
1. **Profile weight loading**:
   - Add profiling to vLLM worker's `update_weights` method
   - Check if it's loading from disk vs CPU vs GPU memory
2. **Optimize storage**:
   - Use faster NVMe storage for checkpoints
   - Consider in-memory checkpoint sharing (shm, mmap)
3. **Reduce model size**:
   - Use smaller model variants
   - Apply quantization (int8, int4)

### If ckpt wait_ms is the bottleneck:
1. **Profile trainer checkpoint writes**:
   - Check if sharded save is working correctly
   - Verify filesystem is fast (not NFS, networked storage)
2. **Optimize I/O**:
   - Use local SSD/NVMe
   - Tune `async_level` in trainer config

## Commands Reference

### SSH
```bash
ssh ubuntu@216.81.248.153
```

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live updates
```

### Check Running Processes
```bash
ps aux | grep -E 'python|vllm|orchestrator|trainer'
tmux ls  # Check for tmux sessions
```

### Kill Stuck Processes
```bash
# Find PIDs
ps aux | grep vllm
ps aux | grep orchestrator
ps aux | grep trainer

# Kill gracefully
kill <PID>

# Force kill if needed
kill -9 <PID>
```

### Check Disk Usage
```bash
df -h
du -sh ~/outputs/bottleneck_test
```

### Check Network (if using remote inference)
```bash
ping 216.81.248.153
telnet 216.81.248.153 8000  # Check if port is open
curl http://216.81.248.153:8000/health  # vLLM health check
```
