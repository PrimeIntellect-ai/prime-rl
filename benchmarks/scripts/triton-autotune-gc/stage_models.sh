#!/bin/bash
# Usage: stage_models.sh MODELS_DIR. Downloads config and tokenizer files only (no weights); runs use random init.
set -euo pipefail
OUT=$1
for repo in zai-org/GLM-5.3-Flash-BF16 Qwen/Qwen3.8-Flash-Next; do
  hf download "$repo" --local-dir "$OUT/${repo/\//--}" \
    --include "*.json" --include "*.txt" --include "*.model" --include "*.jinja" --exclude "*.safetensors.index.json"
done
python3 - "$OUT/Qwen--Qwen3.8-Flash-Next/config.json" <<'PY'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
config.get("text_config", config)["ngram_vocab_size_base"] = 1_000_000
json.dump(config, open(path, "w"), indent=2)
PY
