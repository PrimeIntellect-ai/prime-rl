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