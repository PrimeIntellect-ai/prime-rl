"""Baseten job: Test block masking overlay works in vLLM.

Adds 4 block masking tokens to Qwen3-0.6B, installs the overlay, and runs
inference tests to verify the block masking system initializes and processes
correctly. No SFT needed — tests the infrastructure, not the model's ability
to generate block markers.

Usage:
    uvx truss train push memento_inference_test.py \
        --job-name memento-infer-test-v2-r1
"""

import os
import shlex

from truss.base import truss_config
from truss_train import definitions

BASE_IMAGE = os.environ.get("BASE_IMAGE", "primeintellect/prime-rl:main")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "prime-rl-memento-sft")

START_COMMAND = r"""
set -euxo pipefail

WORKSPACE_DIR="${BASETEN_WORKSPACE_DIR:-/b10/workspace}"
IMAGE_APP_DIR="${PRIME_RL_APP_DIR:-/app}"

export HOME="${HOME:-${WORKSPACE_DIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${UV_CACHE_DIR}"

if [ -n "${PRIME_RL_PROJECT_DIR:-}" ]; then
  PROJECT_DIR="${PRIME_RL_PROJECT_DIR}"
elif [ -f "${WORKSPACE_DIR}/pyproject.toml" ]; then
  PROJECT_DIR="${WORKSPACE_DIR}"
else
  PROJECT_DIR="${IMAGE_APP_DIR}"
fi

if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  if [ -d "${IMAGE_APP_DIR}/.venv" ]; then
    export UV_PROJECT_ENVIRONMENT="${IMAGE_APP_DIR}/.venv"
  else
    export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
  fi
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/packages/prime-rl-configs/src:${PYTHONPATH:-}"

echo "=== Launch manifest ==="
echo "workspace_dir=${WORKSPACE_DIR}"
echo "project_dir=${PROJECT_DIR}"
echo "image_app_dir=${IMAGE_APP_DIR}"
echo "uv_project_environment=${UV_PROJECT_ENVIRONMENT}"
echo "pythonpath=${PYTHONPATH}"
echo "base_image=${BASE_IMAGE:-unknown}"
echo "block_masking_async_mode=${BLOCK_MASKING_ASYNC_MODE:-safe_sync}"
echo "memento_inference_extra_args=${MEMENTO_INFERENCE_EXTRA_ARGS:-}"
if command -v sha256sum >/dev/null 2>&1 && [ -f uv.lock ]; then
  echo "uv_lock_sha256=$(sha256sum uv.lock | awk '{print $1}')"
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git_sha=$(git rev-parse HEAD)"
  echo "git_status_short_begin"
  git status --short || true
  echo "git_status_short_end"
fi

for path in \
  pyproject.toml \
  uv.lock \
  src/prime_rl/inference/patches.py \
  src/prime_rl/inference/block_masking/__init__.py \
  src/prime_rl/inference/block_masking/config.py \
  src/prime_rl/inference/vllm/padded_input_scrub.py \
  packages/prime-rl-configs/pyproject.toml \
  overlays/vllm/v1/core/sched/output.py \
  overlays/vllm/v1/core/sched/scheduler.py \
  scripts/memento_runtime_preflight.py \
  scripts/install_block_masking_overlay.sh \
  scripts/test_block_masking_inference.py \
  scripts/prepare_memento_sft.py
do
  if [ ! -e "$path" ]; then
    echo "ERROR: Missing required launch artifact: $path"
    exit 1
  fi
done

if [ "${SYNC_PROJECT_DEPS:-0}" = "1" ]; then
  echo "=== Syncing project dependencies from bundled lockfile ==="
  UV_SYNC_ARGS="${UV_SYNC_ARGS:---locked --inexact}"
  uv sync ${UV_SYNC_ARGS}
elif [ "${INSTALL_VLLM_WHEEL:-1}" = "1" ]; then
  echo "=== Installing target vLLM wheel ==="
  uv pip install \
    --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
    --reinstall \
    --no-deps \
    "${VLLM_WHEEL_URL}"
fi
UV_RUN_EXTRA="--no-sync"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PRIME_DISABLE_VERSION_CHECK="${PRIME_DISABLE_VERSION_CHECK:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
: "${VLLM_WHEEL_URL:=https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_x86_64.whl}"

echo "=== Runtime preflight ==="
uv run ${UV_RUN_EXTRA} python scripts/memento_runtime_preflight.py

# --- Step 1: Install block masking overlay ---
echo "=== Installing block masking overlay ==="
bash scripts/install_block_masking_overlay.sh "${UV_PROJECT_ENVIRONMENT}"

# --- Step 2: Prepare model (add block masking tokens, no dataset needed) ---
echo "=== Preparing model (adding block masking tokens) ==="
uv run ${UV_RUN_EXTRA} python scripts/prepare_memento_sft.py \
  --output-dir /tmp/memento-test-prep \
  --model-only

# --- Step 3: Run inference test ---
echo "=== Running block masking inference test ==="
INFERENCE_TEST_LOG="${WORKSPACE_DIR}/block_masking_inference_test.log"
BLOCK_MASKING_EVENT_LOG="${WORKSPACE_DIR}/block_masking_events.log"
rm -f "${BLOCK_MASKING_EVENT_LOG}"
touch "${BLOCK_MASKING_EVENT_LOG}"
export PRIME_RL_BLOCK_MASKING_EVENT_LOG="${BLOCK_MASKING_EVENT_LOG}"
set +e
uv run ${UV_RUN_EXTRA} python scripts/test_block_masking_inference.py \
  --model /tmp/memento-test-prep/model \
  --max-tokens 256 \
  --max-model-len 4096 \
  --block-masking-async-mode "${BLOCK_MASKING_ASYNC_MODE:-safe_sync}" \
  ${MEMENTO_INFERENCE_EXTRA_ARGS:-} \
  2>&1 | tee "${INFERENCE_TEST_LOG}"
test_status="${PIPESTATUS[0]}"
set -e
if [ "${test_status}" != "0" ]; then
  exit "${test_status}"
fi

compaction_count="$(grep -Ec "Block masking (prompt|generated|deferred) compaction" "${INFERENCE_TEST_LOG}" || true)"
span_success_count="$(grep -c "SpanRemovalResult(success=True" "${INFERENCE_TEST_LOG}" || true)"
kv_copy_count="$({ grep -h "Block masking KV copy ops executed" "${INFERENCE_TEST_LOG}" "${BLOCK_MASKING_EVENT_LOG}" 2>/dev/null || true; } | wc -l | tr -d ' ')"
block_truncation_count="$({ grep -h "Block masking block table truncations applied" "${INFERENCE_TEST_LOG}" "${BLOCK_MASKING_EVENT_LOG}" 2>/dev/null || true; } | wc -l | tr -d ' ')"
echo "memento_compaction_count=${compaction_count}"
echo "memento_span_success_count=${span_success_count}"
echo "memento_kv_copy_count=${kv_copy_count}"
echo "memento_block_truncation_count=${block_truncation_count}"
if [ -s "${BLOCK_MASKING_EVENT_LOG}" ]; then
  echo "=== Block masking GPU event log tail ==="
  tail -40 "${BLOCK_MASKING_EVENT_LOG}"
fi
if grep -q "SpanRemovalResult(success=False" "${INFERENCE_TEST_LOG}"; then
  echo "ERROR: failed SpanRemovalResult detected"
  grep -n "SpanRemovalResult(success=False" "${INFERENCE_TEST_LOG}" || true
  exit 1
fi
if grep "error_message=" "${INFERENCE_TEST_LOG}" | grep -v "error_message=None" >/dev/null; then
  echo "ERROR: non-empty compaction error_message detected"
  grep -n "error_message=" "${INFERENCE_TEST_LOG}" | grep -v "error_message=None" || true
  exit 1
fi
if [ "${compaction_count}" -lt "${MIN_MEMENTO_COMPACTIONS:-1}" ]; then
  echo "ERROR: expected at least ${MIN_MEMENTO_COMPACTIONS:-1} compaction log(s)"
  exit 1
fi
if [ "${span_success_count}" -lt "${MIN_MEMENTO_COMPACTIONS:-1}" ]; then
  echo "ERROR: expected at least ${MIN_MEMENTO_COMPACTIONS:-1} successful SpanRemovalResult log(s)"
  exit 1
fi
if [ "${kv_copy_count}" -lt "${MIN_MEMENTO_KV_COPIES:-1}" ]; then
  echo "ERROR: expected at least ${MIN_MEMENTO_KV_COPIES:-1} KV-copy execution log(s)"
  exit 1
fi
if [ "${block_truncation_count}" -lt "${MIN_MEMENTO_BLOCK_TRUNCATIONS:-1}" ]; then
  echo "ERROR: expected at least ${MIN_MEMENTO_BLOCK_TRUNCATIONS:-1} block-table truncation log(s)"
  exit 1
fi

echo "=== Inference test complete ==="
""".strip()

training_runtime = definitions.Runtime(
    start_commands=[f"/bin/bash -lc {shlex.quote(START_COMMAND)}"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name=os.environ.get("HF_SECRET_NAME", "hf_access_token")
        ),
        "BASETEN_WORKSPACE_DIR": os.environ.get(
            "BASETEN_WORKSPACE_DIR", "/b10/workspace"
        ),
        "PRIME_RL_APP_DIR": os.environ.get("PRIME_RL_APP_DIR", "/app"),
        "BASE_IMAGE": BASE_IMAGE,
        "PRIME_DISABLE_VERSION_CHECK": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "UV_LINK_MODE": os.environ.get("UV_LINK_MODE", "copy"),
        "SYNC_PROJECT_DEPS": os.environ.get("SYNC_PROJECT_DEPS", "0"),
        "UV_SYNC_ARGS": os.environ.get("UV_SYNC_ARGS", "--locked --inexact"),
        "INSTALL_VLLM_WHEEL": os.environ.get("INSTALL_VLLM_WHEEL", "1"),
        "VLLM_WHEEL_URL": os.environ.get(
            "VLLM_WHEEL_URL",
            "https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_x86_64.whl",
        ),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_MOE_USE_DEEP_GEMM": "0",
        "BLOCK_MASKING_ASYNC_MODE": os.environ.get(
            "BLOCK_MASKING_ASYNC_MODE", "safe_sync"
        ),
        "MEMENTO_INFERENCE_EXTRA_ARGS": os.environ.get(
            "MEMENTO_INFERENCE_EXTRA_ARGS", ""
        ),
        "MIN_MEMENTO_COMPACTIONS": os.environ.get("MIN_MEMENTO_COMPACTIONS", "1"),
        "MIN_MEMENTO_KV_COPIES": os.environ.get("MIN_MEMENTO_KV_COPIES", "1"),
        "MIN_MEMENTO_BLOCK_TRUNCATIONS": os.environ.get(
            "MIN_MEMENTO_BLOCK_TRUNCATIONS", "1"
        ),
        "EXPECTED_VLLM_VERSION_PREFIX": os.environ.get(
            "EXPECTED_VLLM_VERSION_PREFIX", "0.21."
        ),
        "HOME": os.environ.get("BASETEN_HOME", "/b10/workspace"),
        "XDG_CACHE_HOME": os.environ.get(
            "BASETEN_XDG_CACHE_HOME", "/b10/workspace/.cache"
        ),
        "HF_HOME": os.environ.get(
            "BASETEN_HF_HOME", "/b10/workspace/.cache/huggingface"
        ),
        "UV_CACHE_DIR": os.environ.get(
            "BASETEN_UV_CACHE_DIR", "/b10/workspace/.cache/uv"
        ),
    },
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
    ),
    node_count=1,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name=PROJECT_NAME,
    job=training_job,
)
