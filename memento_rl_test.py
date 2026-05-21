"""Baseten job: full Prime RL prod-loop smoke with Memento block masking.

This runs the real ``rl`` entrypoint with one inference H100 and one trainer
H100. It prepares a Qwen3-0.6B checkpoint with Memento boundary tokens, installs
the vLLM block-masking overlay, then validates that the production async RL loop
starts inference, generates rollouts, and runs trainer backward with block
masking enabled. Deterministic compaction is covered by the standalone inference
smoke, and policy-update handoff should be covered by a longer/stricter async
smoke.

Usage:
    env BLOCK_MASKING_ASYNC_MODE=async_barrier uvx truss train push \
        memento_rl_test.py --job-name memento-vllm021-rl-prod-r1
"""

import os
import shlex

from truss.base import truss_config
from truss_train import definitions

BASE_IMAGE = os.environ.get("BASE_IMAGE", "primeintellect/prime-rl:main")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "prime-rl-memento-prod-rl")

START_COMMAND = r"""
set -euxo pipefail

WORKSPACE_DIR="${BASETEN_WORKSPACE_DIR:-/b10/workspace}"
IMAGE_APP_DIR="${PRIME_RL_APP_DIR:-/app}"
RUN_OUTPUT_DIR="${MEMENTO_RL_OUTPUT_DIR:-/b10/workspace/outputs/memento-rl-prod-async}"
RUN_CONFIG="${MEMENTO_RL_CONFIG:-configs/ci/integration/memento_async_rl.toml}"

dump_prime_logs() {
  local status="$1"
  if [ "${status}" = "0" ]; then
    return 0
  fi

  echo "=== Prime RL smoke failed with status ${status}; dumping child logs ==="
  for file in \
    "${RUN_OUTPUT_DIR}/logs/orchestrator.log" \
    "${RUN_OUTPUT_DIR}/logs/trainer.log" \
    "${RUN_OUTPUT_DIR}/logs/inference.log"
  do
    if [ -s "${file}" ]; then
      echo "=== tail -200 ${file} ==="
      tail -200 "${file}" || true
    else
      echo "=== missing or empty ${file} ==="
    fi
  done

  if [ -d "${RUN_OUTPUT_DIR}/logs/trainer/torchrun" ]; then
    echo "=== trainer torchrun log tails ==="
    find "${RUN_OUTPUT_DIR}/logs/trainer/torchrun" \
      -type f \( -name stdout.log -o -name stderr.log \) \
      -print \
      -exec tail -120 {} \; || true
  fi
}

trap 'status=$?; dump_prime_logs "$status"; exit "$status"' EXIT

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
echo "run_output_dir=${RUN_OUTPUT_DIR}"
echo "run_config=${RUN_CONFIG}"
echo "uv_project_environment=${UV_PROJECT_ENVIRONMENT}"
echo "pythonpath=${PYTHONPATH}"
echo "base_image=${BASE_IMAGE:-unknown}"
echo "prime_rl_disable_quack_rmsnorm=${PRIME_RL_DISABLE_QUACK_RMSNORM:-1}"
echo "expect_memento_enabled=${EXPECT_MEMENTO_ENABLED:-1}"
echo "require_memento_compaction=${REQUIRE_MEMENTO_COMPACTION:-0}"
echo "expected_final_step=${EXPECTED_FINAL_STEP:-1}"
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
  packages/prime-rl-configs/src/prime_rl/configs/inference.py \
  packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py \
  src/prime_rl/inference/patches.py \
  src/prime_rl/inference/block_masking/__init__.py \
  src/prime_rl/inference/block_masking/config.py \
  src/prime_rl/inference/vllm/padded_input_scrub.py \
  src/prime_rl/orchestrator/envs.py \
  overlays/vllm/v1/core/sched/output.py \
  overlays/vllm/v1/core/sched/scheduler.py \
  scripts/memento_runtime_preflight.py \
  scripts/install_block_masking_overlay.sh \
  scripts/prepare_memento_sft.py \
  "${RUN_CONFIG}"
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

if [ "${INSTALL_MEMENTO_BOUNDARY_ENV:-0}" = "1" ]; then
  echo "=== Enabling Memento boundary fixture env ==="
  export PYTHONPATH="${PROJECT_DIR}/tests/fixtures/memento_boundary_env/src:${PYTHONPATH}"
  uv run ${UV_RUN_EXTRA} python -c "import memento_boundary_env; print('memento_boundary_env import ok')"
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PRIME_DISABLE_VERSION_CHECK="${PRIME_DISABLE_VERSION_CHECK:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export PRIME_RL_DISABLE_QUACK_RMSNORM="${PRIME_RL_DISABLE_QUACK_RMSNORM:-1}"
: "${VLLM_WHEEL_URL:=https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_x86_64.whl}"

echo "=== Runtime preflight ==="
uv run ${UV_RUN_EXTRA} python scripts/memento_runtime_preflight.py

echo "=== Installing block masking overlay ==="
bash scripts/install_block_masking_overlay.sh "${UV_PROJECT_ENVIRONMENT}"

echo "=== Preparing model (adding block masking tokens) ==="
uv run ${UV_RUN_EXTRA} python scripts/prepare_memento_sft.py \
  --output-dir /tmp/memento-test-prep \
  --model-only

echo "=== Running full Prime RL Memento async smoke ==="
PATCHED_RUN_CONFIG="${WORKSPACE_DIR}/memento_rl_effective_config.toml"
sed "0,/^output_dir = /s|^output_dir = .*|output_dir = \"${RUN_OUTPUT_DIR}\"|" \
  "${RUN_CONFIG}" > "${PATCHED_RUN_CONFIG}"
echo "effective_run_config=${PATCHED_RUN_CONFIG}"
grep -n '^output_dir = ' "${PATCHED_RUN_CONFIG}"
rm -rf "${RUN_OUTPUT_DIR}"
mkdir -p "${RUN_OUTPUT_DIR}/logs"
BLOCK_MASKING_EVENT_LOG="${RUN_OUTPUT_DIR}/logs/block_masking_events.log"
rm -f "${BLOCK_MASKING_EVENT_LOG}"
touch "${BLOCK_MASKING_EVENT_LOG}"
export PRIME_RL_BLOCK_MASKING_EVENT_LOG="${BLOCK_MASKING_EVENT_LOG}"
uv run ${UV_RUN_EXTRA} rl @ "${PATCHED_RUN_CONFIG}"

echo "=== Verifying prod-loop Memento signals ==="
INFERENCE_LOG="${RUN_OUTPUT_DIR}/logs/inference.log"
ORCH_LOG="${RUN_OUTPUT_DIR}/logs/orchestrator.log"
TRAINER_LOG="${RUN_OUTPUT_DIR}/logs/trainer.log"
EVENT_LOG="${RUN_OUTPUT_DIR}/logs/block_masking_events.log"

for file in "${INFERENCE_LOG}" "${ORCH_LOG}" "${TRAINER_LOG}"; do
  if [ ! -s "${file}" ]; then
    echo "ERROR: Missing or empty log file: ${file}"
    exit 1
  fi
done

grep -q "Asynchronous scheduling is enabled" "${INFERENCE_LOG}"
if [ "${EXPECT_MEMENTO_ENABLED:-1}" = "1" ]; then
  grep -q "Block masking async barrier engine path active" "${INFERENCE_LOG}"
fi
grep -q "Starting orchestrator loop" "${ORCH_LOG}"
grep -q "Step ${EXPECTED_FINAL_STEP:-1}" "${ORCH_LOG}"
grep -q "Auto-selected implementation: custom" "${TRAINER_LOG}"
grep -q "Starting training loop" "${TRAINER_LOG}"
grep -q "Step ${EXPECTED_FINAL_STEP:-1}" "${TRAINER_LOG}"
grep -q "RL trainer finished" "${TRAINER_LOG}"

if grep -Eiq "(Loss:|Grad\\. Norm:|Entropy:|Mismatch KL:)[^[:cntrl:]]*(nan|inf)" "${TRAINER_LOG}"; then
  echo "ERROR: non-finite trainer metric detected"
  grep -Ein "(Loss:|Grad\\. Norm:|Entropy:|Mismatch KL:)[^[:cntrl:]]*(nan|inf)" "${TRAINER_LOG}" || true
  exit 1
fi

compaction_count="$(grep -Ec "Block masking (prompt|generated|deferred) compaction" "${INFERENCE_LOG}" || true)"
span_success_count="$(grep -c "SpanRemovalResult(success=True" "${INFERENCE_LOG}" || true)"
kv_copy_count="$({ grep -h "Block masking KV copy ops executed" "${INFERENCE_LOG}" "${EVENT_LOG}" 2>/dev/null || true; } | wc -l | tr -d ' ')"
block_truncation_count="$({ grep -h "Block masking block table truncations applied" "${INFERENCE_LOG}" "${EVENT_LOG}" 2>/dev/null || true; } | wc -l | tr -d ' ')"
echo "memento_compaction_count=${compaction_count}"
echo "memento_span_success_count=${span_success_count}"
echo "memento_kv_copy_count=${kv_copy_count}"
echo "memento_block_truncation_count=${block_truncation_count}"
max_observed_off_policy_level="$({ grep -oE "Max\\. Off-Policy Level: [0-9]+" "${ORCH_LOG}" || true; } | awk '{print $4}' | sort -n | tail -1)"
max_observed_off_policy_level="${max_observed_off_policy_level:-0}"
echo "max_observed_off_policy_level=${max_observed_off_policy_level}"
if [ -s "${EVENT_LOG}" ]; then
  echo "=== Block masking GPU event log tail ==="
  tail -40 "${EVENT_LOG}"
fi

if grep -q "SpanRemovalResult(success=False" "${INFERENCE_LOG}"; then
  echo "ERROR: failed SpanRemovalResult detected"
  grep -n "SpanRemovalResult(success=False" "${INFERENCE_LOG}" || true
  exit 1
fi
if grep "error_message=" "${INFERENCE_LOG}" | grep -v "error_message=None" >/dev/null; then
  echo "ERROR: non-empty compaction error_message detected"
  grep -n "error_message=" "${INFERENCE_LOG}" | grep -v "error_message=None" || true
  exit 1
fi

if [ "${REQUIRE_MEMENTO_COMPACTION:-0}" = "1" ]; then
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
fi
if [ "${max_observed_off_policy_level}" -lt "${MIN_MAX_OFF_POLICY_LEVEL:-0}" ]; then
  echo "ERROR: expected max observed off-policy level to be at least ${MIN_MAX_OFF_POLICY_LEVEL:-0}"
  exit 1
fi

stable_count=0
if [ -d "${RUN_OUTPUT_DIR}" ]; then
  stable_count="$(find "${RUN_OUTPUT_DIR}" -path '*/broadcasts/step_*/STABLE' -type f | wc -l | tr -d ' ')"
fi
echo "broadcast_stable_count=${stable_count}"
find "${RUN_OUTPUT_DIR}" -path '*/broadcasts/step_*/STABLE' -type f | sort | tail -20 || true
if [ "${stable_count}" -lt "${MIN_BROADCAST_STABLE:-1}" ]; then
  echo "ERROR: expected at least ${MIN_BROADCAST_STABLE:-1} broadcast STABLE marker(s)"
  exit 1
fi

echo "=== Key signal excerpts ==="
grep -n "Asynchronous scheduling is enabled\|Block masking async barrier" "${INFERENCE_LOG}" | tail -40
grep -n "Block masking .*compaction\|SpanRemovalResult(success=True\|Block masking KV copy ops executed\|Block masking block table truncations applied" "${INFERENCE_LOG}" | tail -80 || true
grep -n "Starting orchestrator loop\|Step " "${ORCH_LOG}" | tail -40
grep -n "Auto-selected implementation\|Starting training loop\|Step \|RL trainer finished" "${TRAINER_LOG}" | tail -40

echo "=== Full Prime RL Memento async smoke complete ==="
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
        "PRIME_RL_DISABLE_QUACK_RMSNORM": os.environ.get(
            "PRIME_RL_DISABLE_QUACK_RMSNORM", "1"
        ),
        "BLOCK_MASKING_ASYNC_MODE": os.environ.get(
            "BLOCK_MASKING_ASYNC_MODE", "async_barrier"
        ),
        "INSTALL_MEMENTO_BOUNDARY_ENV": os.environ.get(
            "INSTALL_MEMENTO_BOUNDARY_ENV", "0"
        ),
        "REQUIRE_MEMENTO_COMPACTION": os.environ.get(
            "REQUIRE_MEMENTO_COMPACTION", "0"
        ),
        "EXPECT_MEMENTO_ENABLED": os.environ.get("EXPECT_MEMENTO_ENABLED", "1"),
        "EXPECTED_FINAL_STEP": os.environ.get("EXPECTED_FINAL_STEP", "1"),
        "MIN_BROADCAST_STABLE": os.environ.get("MIN_BROADCAST_STABLE", "1"),
        "MIN_MEMENTO_COMPACTIONS": os.environ.get("MIN_MEMENTO_COMPACTIONS", "1"),
        "MIN_MEMENTO_KV_COPIES": os.environ.get("MIN_MEMENTO_KV_COPIES", "1"),
        "MIN_MEMENTO_BLOCK_TRUNCATIONS": os.environ.get(
            "MIN_MEMENTO_BLOCK_TRUNCATIONS", "1"
        ),
        "MIN_MAX_OFF_POLICY_LEVEL": os.environ.get("MIN_MAX_OFF_POLICY_LEVEL", "0"),
        "EXPECTED_VLLM_VERSION_PREFIX": os.environ.get(
            "EXPECTED_VLLM_VERSION_PREFIX", "0.21."
        ),
        "MEMENTO_RL_CONFIG": os.environ.get(
            "MEMENTO_RL_CONFIG", "configs/ci/integration/memento_async_rl.toml"
        ),
        "MEMENTO_RL_OUTPUT_DIR": os.environ.get(
            "MEMENTO_RL_OUTPUT_DIR", "/b10/workspace/outputs/memento-rl-prod-async"
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
        count=2,
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
