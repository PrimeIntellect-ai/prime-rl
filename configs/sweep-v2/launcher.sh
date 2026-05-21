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

# --- Configuration ---
GPU_ID="8b33a6"                   # A6000 48GB x2, $1.08/hr (cheapest 2-GPU)
# GPU_ID="3aa391"                 # H100 80GB x2, $4.70/hr (fastest)
POD_NAME_PREFIX="sweep-v2"
CONFIGS_PER_POD=12                # All 12 configs per pod (runs sequentially ~50 steps each)
BRANCH="worktree-sweep-eval-splits"

# --- Collect all configs ---
ALL_CONFIGS=()
for f in "$SCRIPT_DIR"/*.toml; do
    ALL_CONFIGS+=("configs/sweep-v2/$(basename "$f")")
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
    echo "Pod $i: ${pod_configs[*]}"
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
echo "=== Waiting for pods to be ready ==="
for pod_id in "${POD_IDS[@]}"; do
    echo -n "Waiting for $pod_id..."
    for attempt in $(seq 1 120); do
        status=$($PRIME pods status "$pod_id" --output json --plain 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('installation_status', d.get('status', 'UNKNOWN')))
" 2>/dev/null || echo "UNKNOWN")
        if [ "$status" = "FINISHED" ]; then
            echo " READY"
            break
        fi
        echo -n "."
        sleep 10
    done
done

# --- Get SSH info for each pod ---
echo ""
echo "=== Setting up pods ==="
for ((i=0; i<${#POD_IDS[@]}; i++)); do
    pod_id="${POD_IDS[$i]}"
    ssh_info=$($PRIME pods status "$pod_id" --output json --plain 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('ssh', ''))
")
    echo "Pod $i ($pod_id): $ssh_info"

    if [ -z "$ssh_info" ]; then
        echo "  ERROR: No SSH info for pod $pod_id"
        continue
    fi

    # Copy setup script to pod
    echo "  Copying setup script..."
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
        "$SCRIPT_DIR/pod-setup.sh" \
        "$SCRIPT_DIR/pod-run.sh" \
        "${ssh_info}:" 2>&1 || { echo "  SCP failed"; continue; }

    # Run setup
    echo "  Running setup (this takes a few minutes)..."
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$ssh_info" \
        "chmod +x pod-setup.sh pod-run.sh && bash pod-setup.sh" 2>&1 | tail -20

    # Copy pod-run.sh into repo
    ssh -o StrictHostKeyChecking=no "$ssh_info" \
        "cp pod-run.sh ~/prime-rl/configs/sweep-v2/pod-run.sh && chmod +x ~/prime-rl/configs/sweep-v2/pod-run.sh"

    # Assign and run configs
    start=$((i * CONFIGS_PER_POD))
    end=$((start + CONFIGS_PER_POD))
    if [ $end -gt $TOTAL_CONFIGS ]; then end=$TOTAL_CONFIGS; fi
    pod_configs=("${ALL_CONFIGS[@]:$start:$((end - start))}")

    echo "  Starting experiments: ${pod_configs[*]}"
    # Run in background via nohup so it survives SSH disconnect
    config_args="${pod_configs[*]}"
    ssh -o StrictHostKeyChecking=no "$ssh_info" \
        "cd ~/prime-rl && nohup bash configs/sweep-v2/pod-run.sh $config_args > ~/sweep-run.log 2>&1 &" &
    echo "  Experiments launched in background on pod $pod_id"
done

wait

echo ""
echo "=== LAUNCHER COMPLETE ==="
echo "Pods provisioned: ${#POD_IDS[@]}"
echo "Total configs assigned: $TOTAL_CONFIGS"
echo ""
echo "Monitor progress:"
for ((i=0; i<${#POD_IDS[@]}; i++)); do
    pod_id="${POD_IDS[$i]}"
    echo "  prime pods ssh $pod_id -- tail -f ~/sweep-run.log"
done
echo ""
echo "Collect results:"
for ((i=0; i<${#POD_IDS[@]}; i++)); do
    pod_id="${POD_IDS[$i]}"
    echo "  prime pods ssh $pod_id -- cat ~/prime-rl/configs/sweep-v2/logs/*.log"
done
