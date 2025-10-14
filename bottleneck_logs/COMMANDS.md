# Commands Cheat Sheet

## SSH & Connection

```bash
# SSH to A100 machine
ssh ubuntu@216.81.248.153

# SSH with port forwarding (if needed)
ssh -L 8000:localhost:8000 ubuntu@216.81.248.153

# Copy file to remote
scp myfile.txt ubuntu@216.81.248.153:~/

# Copy file from remote
scp ubuntu@216.81.248.153:~/logs/orchestrator.log ./bottleneck_logs/

# Copy directory recursively
scp -r ubuntu@216.81.248.153:~/outputs/test/logs/ ./bottleneck_logs/test_run/
```

## GPU Monitoring

```bash
# Check GPU status (one-time)
nvidia-smi

# Watch GPU status (live, updates every 1s)
watch -n 1 nvidia-smi

# Query specific metrics
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv

# Check CUDA version
nvcc --version
nvidia-smi | grep "CUDA Version"

# Check GPU topology
nvidia-smi topo -m
```

## Process Management

```bash
# Find prime-rl processes
ps aux | grep -E 'python|vllm|orchestrator|trainer' | grep -v grep

# Find processes by port
lsof -i :8000  # vLLM inference server
netstat -tlnp | grep :8000

# Kill by PID
kill <PID>

# Force kill
kill -9 <PID>

# Kill all vllm processes
pkill -f vllm

# Check CPU/memory usage
htop
top
```

## Tmux Session Management

```bash
# List tmux sessions
tmux ls

# Create new session
tmux new -s inference
tmux new -s trainer
tmux new -s orchestrator

# Attach to session
tmux attach -t inference

# Detach from session (inside tmux)
Ctrl+b, then d

# Kill session
tmux kill-session -t inference

# Kill all sessions
tmux kill-server
```

## Log Management

```bash
# Tail orchestrator log
tail -f ~/outputs/test/logs/orchestrator.log

# Tail with filtering
tail -f ~/outputs/test/logs/orchestrator.log | grep --line-buffered -E '\[weights\]|\[ckpt\]'

# Search in logs
grep '\[weights\]' ~/outputs/test/logs/orchestrator.log

# Count occurrences
grep -c 'queue_ms=' ~/outputs/test/logs/orchestrator.log

# Extract specific field
grep 'queue_ms=' orchestrator.log | sed -n 's/.*queue_ms=\([0-9.]*\).*/\1/p'

# Compress logs
tar -czf bottleneck_logs_$(date +%Y%m%d_%H%M%S).tar.gz \
  ~/outputs/test/logs/orchestrator.log \
  ~/outputs/test/logs/trainer/rank_*.log \
  ~/logs/inference_*.log

# Decompress
tar -xzf bottleneck_logs_20250314_123456.tar.gz
```

## Disk Management

```bash
# Check disk space
df -h

# Check directory size
du -sh ~/outputs/
du -sh ~/outputs/test/logs/

# Find large files
du -ah ~/outputs/ | sort -rh | head -20

# Delete old checkpoints
rm -rf ~/outputs/old_experiment/

# Clean up tmp files
find /tmp -name "*.tmp" -mtime +1 -delete
```

## Git Operations (On Remote)

```bash
# Clone repo
git clone https://github.com/PrimeIntellect-ai/prime-rl.git ~/prime-rl

# Update repo
cd ~/prime-rl
git fetch
git pull

# Switch branch
git checkout ameen/feat-multi-step-rollout

# Check status
git status
git log --oneline -10

# View changes
git diff HEAD~1
```

## Python/UV Environment

```bash
# Install dependencies
uv sync

# Run commands
uv run inference --help
uv run orchestrator --help
uv run trainer --help

# Check Python version
python --version
uv python --version

# List installed packages
uv pip list

# Check for specific package
uv pip show vllm
```

## Starting Services

### Inference Server
```bash
# Basic
uv run inference \
  --model "meta-llama/Llama-2-7b-hf" \
  --host 0.0.0.0 \
  --port 8000

# With tensor parallelism (2 GPUs)
uv run inference \
  --model "meta-llama/Llama-2-7b-hf" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9

# With logging to file
uv run inference \
  --model "meta-llama/Llama-2-7b-hf" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  2>&1 | tee ~/logs/inference_$(date +%Y%m%d_%H%M%S).log
```

### Trainer
```bash
# Basic
uv run trainer \
  --config configs/rl_trainer.yaml \
  --output_dir ~/outputs/bottleneck_test

# With max steps
uv run trainer \
  --config configs/rl_trainer.yaml \
  --output_dir ~/outputs/bottleneck_test \
  --max_steps 10

# Check logs
tail -f ~/outputs/bottleneck_test/logs/trainer/rank_0.log
```

### Orchestrator
```bash
# Basic
uv run orchestrator \
  --config configs/orchestrator.yaml \
  --output_dir ~/outputs/bottleneck_test

# With specific settings
uv run orchestrator \
  --config configs/orchestrator.yaml \
  --output_dir ~/outputs/bottleneck_test \
  --max_steps 10 \
  --batch_size 64 \
  --async_level 2

# Check logs
tail -f ~/outputs/bottleneck_test/logs/orchestrator.log | \
  grep --line-buffered -E '\[weights\]|\[ckpt\]|\[gen\]|\[rollout\]'
```

## Analysis Commands

### Quick Stats
```bash
# Count weight updates
grep -c '\[weights\].*client.done' orchestrator.log

# Average queue_ms
grep 'queue_ms=' orchestrator.log | \
  sed -n 's/.*queue_ms=\([0-9.]*\).*/\1/p' | \
  awk '{sum+=$1; n++} END {print sum/n}'

# Max queue_ms
grep 'queue_ms=' orchestrator.log | \
  sed -n 's/.*queue_ms=\([0-9.]*\).*/\1/p' | \
  sort -n | tail -1

# Find slow updates (queue_ms > 5000)
grep 'queue_ms=' orchestrator.log | \
  awk -F'queue_ms=' '{if ($2 > 5000) print $0}'
```

### Extract Specific Metrics
```bash
# Extract all queue_ms values
grep 'queue_ms=' orchestrator.log | \
  sed -n 's/.*queue_ms=\([0-9.]*\).*/\1/p' > queue_ms.txt

# Extract all rpc_ms values
grep 'rpc_ms=' orchestrator.log | \
  sed -n 's/.*rpc_ms=\([0-9.]*\).*/\1/p' > rpc_ms.txt

# Extract all truncation percentages
grep 'trunc_pct=' orchestrator.log | \
  sed -n 's/.*trunc_pct=\([0-9.]*\).*/\1/p' > trunc_pct.txt
```

### Run Full Analysis
```bash
# From local machine after copying logs
cd /Users/ameenp/prime-development/PRIMERL-127/prime-rl
./bottleneck_logs/analyze_logs.sh bottleneck_logs/orchestrator.log
```

### Compare Before/After
```bash
# Analyze baseline (before fix)
./analyze_logs.sh bottleneck_logs/baseline/orchestrator.log
mv bottleneck_logs/analysis_* bottleneck_logs/analysis_baseline/

# Analyze after fix
./analyze_logs.sh bottleneck_logs/after_fix/orchestrator.log
mv bottleneck_logs/analysis_* bottleneck_logs/analysis_after_fix/

# Compare
diff bottleneck_logs/analysis_baseline/SUMMARY.md \
     bottleneck_logs/analysis_after_fix/SUMMARY.md
```

## Debugging

### Check vLLM Server Health
```bash
# From remote or local
curl http://216.81.248.153:8000/health
curl http://216.81.248.153:8000/v1/models

# Test inference
curl http://216.81.248.153:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

### Check Network
```bash
# Ping
ping 216.81.248.153

# Check port
telnet 216.81.248.153 8000
nc -zv 216.81.248.153 8000

# Check firewall
sudo ufw status
```

### Check File System
```bash
# Check if checkpoint exists
ls -lh ~/outputs/test/weights/step_42/

# Watch for new checkpoints
watch -n 1 "ls -lth ~/outputs/test/weights/ | head -10"

# Check inode usage
df -i
```

### Python Debugging
```bash
# Print Python traceback
python -c "import traceback; traceback.print_exc()"

# Check imports
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

# Run with debug logging
VLLM_LOGGING_LEVEL=DEBUG uv run inference ...
```

## Performance Monitoring

### System Metrics
```bash
# CPU usage
mpstat 1

# Memory usage
vmstat 1
free -h

# I/O stats
iostat -x 1

# Network stats
iftop
nethogs
```

### GPU Profiling
```bash
# Profile GPU utilization
nvidia-smi dmon -i 0,1 -s pucvmet -c 10

# Profile GPU processes
nvidia-smi pmon -i 0,1

# Check GPU clocks
nvidia-smi -q -d CLOCK

# Reset GPU
sudo nvidia-smi -r  # Requires root
```

## Cleanup

```bash
# Kill all prime-rl processes
pkill -f 'uv run'
pkill -f vllm
pkill -f orchestrator
pkill -f trainer

# Clean up outputs
rm -rf ~/outputs/test/

# Clean up logs
rm -f ~/logs/inference_*.log

# Clean up temp files
rm -rf /tmp/vllm_*
```

## One-Liner Recipes

```bash
# Copy logs and analyze in one go
scp ubuntu@216.81.248.153:~/outputs/test/logs/orchestrator.log \
  ./bottleneck_logs/test_run.log && \
  ./bottleneck_logs/analyze_logs.sh ./bottleneck_logs/test_run.log

# Watch for weight updates live
ssh ubuntu@216.81.248.153 \
  "tail -f ~/outputs/test/logs/orchestrator.log" | \
  grep --line-buffered '\[weights\]'

# Count steps completed
ssh ubuntu@216.81.248.153 \
  "grep -c 'Step.*success' ~/outputs/test/logs/orchestrator.log"

# Get latest queue_ms
ssh ubuntu@216.81.248.153 \
  "grep 'queue_ms=' ~/outputs/test/logs/orchestrator.log | tail -1"
```
