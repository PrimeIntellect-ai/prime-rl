#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR=${ARTIFACT_DIR:-$PWD/dist/dynamo}
DYNAMO_ENV=${DYNAMO_ENV:-$PWD/.venv-dynamo}
mapfile -t vllm_wheels < <(find "$ARTIFACT_DIR" -maxdepth 1 -type f -name 'vllm-*.whl' -print)
mapfile -t dynamo_wheels < <(find "$ARTIFACT_DIR" -maxdepth 1 -type f   \( -name 'ai_dynamo-*.whl' -o -name 'ai_dynamo_runtime-*.whl' \) -print)

[[ ${#vllm_wheels[@]} -eq 1 ]] || { echo "Expected one vLLM wheel in $ARTIFACT_DIR" >&2; exit 1; }
[[ ${#dynamo_wheels[@]} -eq 2 ]] || { echo "Expected ai-dynamo and ai-dynamo-runtime wheels in $ARTIFACT_DIR" >&2; exit 1; }
[[ -x "$ARTIFACT_DIR/dynamo-vllm-sidecar" ]] || { echo "Missing dynamo-vllm-sidecar" >&2; exit 1; }

# Keep the custom inference stack isolated: its Torch and Pydantic constraints
# differ from Prime-RL's trainer environment.
uv venv --clear --python 3.12 "$DYNAMO_ENV"
uv pip install --python "$DYNAMO_ENV/bin/python" "${vllm_wheels[0]}" "${dynamo_wheels[@]}"

"$DYNAMO_ENV/bin/vllm" --version
"$ARTIFACT_DIR/dynamo-vllm-sidecar" --help >/dev/null
