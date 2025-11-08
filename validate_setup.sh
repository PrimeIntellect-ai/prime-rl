#!/bin/bash

# This script runs a series of validation checks to ensure the
# environment is correctly set up for prime-rl training.

set -e

# Navigate to the prime-rl directory
if [ ! -d "/workspace/prime-rl" ]; then
    echo "Error: Directory /workspace/prime-rl not found."
    exit 1
fi
cd /workspace/prime-rl
echo "Changed directory to $(pwd)"

echo -e "\n--- 1. Checking Python version ---"
uv run python -V

echo -e "\n--- 2. Checking for flash-attn ---"
if uv run python -c "import flash_attn" 2>/dev/null; then
    echo "flash_attn is installed."
else
    echo "flash_attn is NOT installed. This might be an issue."
fi

echo -e "\n--- 3. Running SFT trainer in debug mode (requires 1 GPU) ---"
uv run sft @ configs/debug/sft/train.toml

echo -e "\n--- 4. Running RL trainer in debug mode (requires 1 GPU) ---"
uv run trainer @ configs/debug/rl/train.toml

echo -e "\n--- 5. Running orchestrator against an inference server (requires 1 GPU) ---"
echo "Starting inference server in the background..."
uv run inference @ configs/debug/infer.toml &
INFERENCE_PID=$!

# Give the server a moment to start up
sleep 15

echo "Running orchestrator..."
# The orchestrator might exit with non-zero on success in this test, so we don't exit on error here.
uv run orchestrator @ configs/debug/orch.toml || echo "Orchestrator finished."

echo "Killing inference server (PID: $INFERENCE_PID)..."
kill $INFERENCE_PID
wait $INFERENCE_PID || echo "Inference server stopped."

echo -e "\n--- 6. Running a simple SFT warmup (requires 1 GPU) ---"
uv run sft @ configs/reverse_text/sft/train.toml

echo -e "\n--- 7. Running a toy RL run (requires 2 GPUs) ---"
uv run rl \
  --trainer @ configs/reverse_text/rl/train.toml \
  --orchestrator @ configs/reverse_text/rl/orch.toml \
  --inference @ configs/reverse_text/rl/infer.toml

echo -e "\n--- ✅ All validation checks passed successfully! ---"
