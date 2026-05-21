#!/bin/bash
# Sweep launcher: provisions pods, sets them up, assigns experiment configs, and runs them.
# Usage:
#   ./launcher.sh                    # Dry run: show plan but don't provision
#   ./launcher.sh --run              # Provision pods and run all experiments
#   ./launcher.sh --run --test       # Provision ONE pod and run ONE config (validation)
#   ./launcher.sh --resume           # Resume on existing pods (skip provisioning)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIME="${PRIME:-/home/ubuntu/.local/bin/prime}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30"

# --- Configuration ---
GPU_ID="3aa391"                   # H100 80GB x2, $4.70/hr
POD_NAME_PREFIX="sweep-v2"
CONFIGS_PER_POD=12                # 12 configs per pod (runs sequentially)
BRANCH="worktree-sweep-eval-splits"

# --- Collect all configs (from the matrix/ directory for the full sweep) ---
ALL_CONFIGS=()
for f in "$SCRIPT_DIR"/matrix/*.toml; do
    ALL_CONFIGS+=("configs/sweep-v2/matrix/$(basename "$f")")
done
TOTAL_CONFIGS=${#ALL_CONFIGS[@]}

echo "=== Sweep Launcher ==="
echo "Configs found: $TOTAL_CONFIGS"
echo "GPU selection: $GPU_ID"
echo ""

# --- Parse args ---
RUN=false
TEST=false
RESUME=false
for arg in "$@"; do
    case "$arg" in
        --run) RUN=true ;;
        --test) TEST=true ;;
        --resume) RESUME=true; RUN=true ;;
    esac
done

if $TEST; then
    echo "*** TEST MODE: will provision 1 pod with 1 config ***"
    ALL_CONFIGS=("${ALL_CONFIGS[0]}")
    TOTAL_CONFIGS=1
fi

# Calculate number of pods needed
NUM_PODS=$(( (TOTAL_CONFIGS + CONFIGS_PER_POD - 1) / CONFIGS_PER_POD ))
echo "Pods needed: $NUM_PODS"
echo "Configs per pod: $CONFIGS_PER_POD"
echo ""

# --- Plan assignment ---
echo "=== Assignment Plan ==="
for ((i=0; i<NUM_PODS; i++)); do
    start=$((i * CONFIGS_PER_POD))
    end=$((start + CONFIGS_PER_POD))
    if [ $end -gt $TOTAL_CONFIGS ]; then end=$TOTAL_CONFIGS; fi
    pod_configs=("${ALL_CONFIGS[@]:$start:$((end - start))}")
    echo "Pod $i (${#pod_configs[@]} configs): $(echo "${pod_configs[*]}" | tr ' ' '\n' | sed 's|.*/||;s|\.toml||' | tr '\n' ' ')"
done
echo ""

if ! $RUN; then
    echo "(Dry run. Pass --run to provision and execute.)"
    exit 0
fi

# --- Provision pods ---
POD_IDS=()

if $RESUME; then
    echo "=== Resuming: finding existing pods ==="
    existing=$($PRIME pods list --output json --plain 2>/dev/null)
    while IFS= read -r pod_id; do
        POD_IDS+=("$pod_id")
    done < <(echo "$existing" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for p in data.get('pods', []):
    if p['status'] == 'ACTIVE' and p['name'].startswith('$POD_NAME_PREFIX'):
        print(p['id'])
")
    echo "Found ${#POD_IDS[@]} existing pods"
else
    echo "=== Provisioning $NUM_PODS pods ==="
    for ((i=0; i<NUM_PODS; i++)); do
        pod_name="${POD_NAME_PREFIX}-${i}"
        echo "Creating pod: $pod_name (GPU: $GPU_ID)..."
        output=$($PRIME pods create \
            --id "$GPU_ID" \
            --name "$pod_name" \
            --disk-size 200 \
            -y \
            --plain 2>&1) || true
        echo "$output"

        # Extract pod ID from creation output
        pod_id=$(echo "$output" | grep -oP '[a-f0-9]{32}' | head -1 || true)
        if [ -z "$pod_id" ]; then
            echo "ERROR: Failed to extract pod ID for $pod_name"
            echo "Output was: $output"
            continue
        fi
        POD_IDS+=("$pod_id")
        echo "  Pod ID: $pod_id"
    done
fi

echo ""
echo "=== Waiting for pods to be ACTIVE ==="
for pod_id in "${POD_IDS[@]}"; do
    echo -n "Waiting for $pod_id..."
    for attempt in $(seq 1 120); do
        pod_json=$($PRIME pods status "$pod_id" --output json --plain 2>/dev/null || echo '{}')
        status=$(echo "$pod_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
        ssh_addr=$(echo "$pod_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh','N/A'))" 2>/dev/null || echo "N/A")
        if [ "$status" = "ACTIVE" ] && [ "$ssh_addr" != "N/A" ]; then
            echo " READY ($ssh_addr)"
            break
        fi
        echo -n "."
        sleep 10
    done
done

# --- Set up and launch on each pod ---
echo ""
echo "=== Setting up pods and launching experiments ==="
for ((i=0; i<${#POD_IDS[@]}; i++)); do
    pod_id="${POD_IDS[$i]}"
    ssh_info=$($PRIME pods status "$pod_id" --output json --plain 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('ssh', ''))
")
    echo "Pod $i ($pod_id): $ssh_info"

    if [ -z "$ssh_info" ]; then
        echo "  ERROR: No SSH info for pod $pod_id, skipping"
        continue
    fi

    # Ensure fish config dir exists (uv installer needs it)
    ssh $SSH_OPTS "$ssh_info" "sudo mkdir -p /home/ubuntu/.config/fish/conf.d 2>/dev/null; sudo chown -R ubuntu:ubuntu /home/ubuntu/.config 2>/dev/null" || true

    # Copy setup and run scripts
    echo "  Copying scripts..."
    scp $SSH_OPTS \
        "$SCRIPT_DIR/pod-setup.sh" \
        "$SCRIPT_DIR/pod-run.sh" \
        "${ssh_info}:" 2>&1 || { echo "  SCP failed, skipping pod"; continue; }

    # Run setup
    echo "  Running setup (this takes a few minutes)..."
    ssh $SSH_OPTS "$ssh_info" \
        "chmod +x pod-setup.sh pod-run.sh && bash pod-setup.sh" 2>&1 | tail -10

    # Assign configs for this pod
    start=$((i * CONFIGS_PER_POD))
    end=$((start + CONFIGS_PER_POD))
    if [ $end -gt $TOTAL_CONFIGS ]; then end=$TOTAL_CONFIGS; fi
    pod_configs=("${ALL_CONFIGS[@]:$start:$((end - start))}")

    echo "  Launching ${#pod_configs[@]} experiments..."
    config_args="${pod_configs[*]}"
    # Use -f to background SSH properly: tells SSH to go to background just before command execution
    ssh -f $SSH_OPTS "$ssh_info" \
        "cd ~/prime-rl && nohup bash configs/sweep-v2/pod-run.sh $config_args > ~/sweep-run.log 2>&1 &"
    echo "  Experiments launched in background on pod $pod_id"
done

echo ""
echo "=========================================="
echo "LAUNCHER COMPLETE"
echo "=========================================="
echo "Pods provisioned: ${#POD_IDS[@]}"
echo "Total configs assigned: $TOTAL_CONFIGS"
echo ""
echo "Monitor progress:"
for ((i=0; i<${#POD_IDS[@]}; i++)); do
    pod_id="${POD_IDS[$i]}"
    echo "  prime pods ssh $pod_id    # then: tail -f ~/sweep-run.log"
done
echo ""
echo "Collect results:"
for ((i=0; i<${#POD_IDS[@]}; i++)); do
    pod_id="${POD_IDS[$i]}"
    echo "  ssh $SSH_OPTS <pod-ssh-addr> 'cat ~/prime-rl/configs/sweep-v2/logs/*.log'"
done
echo ""
echo "Terminate all pods:"
echo "  for id in ${POD_IDS[*]}; do prime pods terminate \$id -y --plain; done"
