#!/usr/bin/env bash
set -euo pipefail

# Pin the process environment explicitly. The Helm chart also supplies generic
# cache defaults; these versioned paths are the authoritative values for this
# exact runtime and avoid relying on duplicate-env ordering.
export UV_CACHE_DIR=/inference-cache/uv
export UV_LINK_MODE=copy
export FLASHINFER_WORKSPACE_BASE=/inference-cache
export VLLM_CACHE_ROOT=/inference-cache/vllm
export TORCH_EXTENSIONS_DIR=/inference-cache/torch-extensions
export TRITON_CACHE_DIR=/inference-cache/triton
export CUDA_CACHE_PATH=/inference-cache/cuda

# Verify that this is the immutable experiment image, including the approved
# writable-scale fix. Nothing is installed or modified in the pod.
/app/.venv/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path

import vllm

expected = {
    "vllm": "0.23.1rc1.dev1392+g910cc8543",
    "flashinfer-python": "0.6.15.post1",
    "flashinfer-cubin": "0.6.15.post1",
}
actual = {package: version(package) for package in expected}
if actual != expected:
    raise RuntimeError(f"Unexpected NVFP4 runtime: {actual!r} != {expected!r}")

path = (
    Path(vllm.__file__).parent
    / "model_executor/layers/quantization/utils/flashinfer_fp4_moe.py"
)
source = path.read_text()
for scale in ("a13_scale", "a2_scale"):
    old = f"{scale}.max().to(torch.float32).expand(num_experts)"
    new = f"{scale}.max().to(torch.float32).repeat(num_experts)"
    if source.count(old) != 0 or source.count(new) != 2:
        raise RuntimeError(f"Writable NVFP4 scale fix is absent or unexpected in {path}")
print(f"Verified pinned NVFP4 runtime and writable-scale fix in {path}")
PY

export NVFP4_CACHE_HIT=0
if [[ -f "$NVFP4_CACHE_READY_MARKER" ]] && \
   [[ "$(<"$NVFP4_CACHE_READY_MARKER")" == "$NVFP4_CACHE_KEY" ]]; then
  # Wheel extraction gives byte-identical CUDA sources fresh mtimes on every
  # pod. Refresh cached build outputs so Ninja does not rebuild them solely
  # because the source inode was recreated.
  find "$FLASHINFER_WORKSPACE_BASE" \
    -type f \( -name '*.so' -o -name '*.o' -o -name '*.a' \) \
    -exec touch {} +
  export NVFP4_CACHE_HIT=1
  echo "Reusing persistent NVFP4 JIT cache: $NVFP4_CACHE_KEY"
else
  # Mark the versioned cache ready only after the API health endpoint proves
  # that model load, online quantization, and lazy kernel setup all succeeded.
  (
    for _ in $(seq 1 2400); do
      if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        marker_tmp="${NVFP4_CACHE_READY_MARKER}.$$"
        printf '%s\n' "$NVFP4_CACHE_KEY" >"$marker_tmp"
        mv "$marker_tmp" "$NVFP4_CACHE_READY_MARKER"
        echo "Marked persistent NVFP4 JIT cache ready: $NVFP4_CACHE_KEY"
        exit 0
      fi
      sleep 5
    done
    echo "Inference did not become healthy; leaving cache unmarked" >&2
  ) &
fi

cd /app
exec uv run --active --no-sync inference @ /etc/prime-rl/inference.toml "$@"
