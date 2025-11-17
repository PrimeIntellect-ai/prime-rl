# RL Rollout Viewer

Real-time viewer for RL training runs integrated with the Prime platform.

## Quick Start

### 1. Enable Platform Upload

Add platform configuration to your training config TOML file:

```toml
[monitor.platform]
enabled = true
api_url = "https://api.primeintellect.ai"
upload_rollouts = true
upload_interval = 1  # Upload metrics every step
max_rollouts_per_step = 10  # Limit rollouts to upload per step
```

### 2. Set Environment Variables

Before running your training, set:

```bash
export PRIME_API_KEY="your-api-key"
export PRIME_USER_ID="your-user-id"
export PRIME_ENV_ID="environment-id"  # Optional
export PRIME_ENV_NAME="Environment Name"  # Optional
export PRIME_TEAM_ID="team-id"  # Optional, if training under a team
```

### 3. Start Training

```bash
uv run rl configs/your-config.toml
```

### 4. View in Dashboard

Navigate to https://app.primeintellect.ai/dashboard/rl to see your run streaming in real-time.

## Remote Training with Local Backend

If you're training on a remote server but want to upload to your local backend:

### Set Up SSH Reverse Tunnel

From your **local machine**, create a reverse SSH tunnel to expose your local backend to the remote server:

```bash
# Forward local port 82 to remote server's localhost:82
ssh -R 82:localhost:82 -p <SSH_PORT> <USER>@<REMOTE_SERVER> -N

# Example:
ssh -R 82:localhost:82 -p 40121 root@213.192.2.68 -N
```

Keep this terminal running while training.

### Run in Background

```bash
# Run tunnel in background
ssh -R 82:localhost:82 -p 40121 root@213.192.2.68 -N -f

# To kill later:
ps aux | grep "ssh -R 82"
kill <PID>
```

### Verify Connection

On the **remote training server**, test that the tunnel works:

```bash
curl http://localhost:82/healthcheck
# Should return: {"status":"ok"}
```

### Complete Workflow

**Terminal 1 (Local):** Start your backend
```bash
cd /path/to/pi-backend
# Start backend on port 82
uvicorn app.main:app --port 82
```

**Terminal 2 (Local):** Create reverse tunnel
```bash
ssh -R 82:localhost:82 -p 40121 root@213.192.2.68 -N
```

**Terminal 3 (Remote):** Run training
```bash
ssh -p 40121 root@213.192.2.68
cd ~/Dev/prime-rl
uv run rl configs/your-config.toml
```

The training server will connect to `http://localhost:82` which tunnels to your local backend!


### Full Platform Config

```toml
[monitor.platform]
enabled = true                    # Enable platform upload
api_key = "your-key"             # Optional, reads from PRIME_API_KEY env var
api_url = "https://api.primeintellect.ai"  # Platform API URL
upload_rollouts = true           # Upload sample rollouts
upload_interval = 1              # Upload frequency (every N steps)
max_rollouts_per_step = 10       # Max rollouts per step
```

### Minimal Config

```toml
[monitor.platform]
enabled = true
```

All other settings will use defaults and read from environment variables.