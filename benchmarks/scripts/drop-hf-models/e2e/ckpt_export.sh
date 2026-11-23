#!/usr/bin/env bash
set -uo pipefail

HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HARNESS/env.sh"
source "$HARNESS/lib.sh"

[ $# -eq 2 ] || { echo "usage: ckpt_export.sh <arm> <ckpt-dense|ckpt-moe>" >&2; exit 2; }
arm=$1
ckpt=$2
export CUDA_VISIBLE_DEVICES=""
export_dir=$E2E_OUT/$arm/export
mkdir -p "$export_dir"

export_and_load() {
  local worktree=$1 step_dir=$2 name=$3
  local out=$export_dir/$name
  setup_uv "$worktree"
  rm -rf "$out" "$out.hfload"
  (cd "$worktree" && $UV_RUN python tools/convert_dcp_to_bf16.py "$step_dir" "$out" --workers 4) >"$out.export.log" 2>&1 ||
    { echo "FAIL: export, see $out.export.log" >"$out.hfload"; return 1; }
  (cd "$worktree" && $UV_RUN python -c \
    'import sys; from pathlib import Path; from tests.utils import check_hf_load; check_hf_load(Path(sys.argv[1]))' "$out") \
    >"$out.hfload.log" 2>&1 && echo PASS >"$out.hfload" || echo "FAIL: see $out.hfload.log" >"$out.hfload"
}

record_source_generation_config() {
  local step_dir=$1 dest=$E2E_OUT/source/$ckpt
  [ -d "$dest" ] && return
  mkdir -p "$dest"
  setup_uv "$E2E_WT_MAIN"
  "$PY" - "$step_dir" "$dest" <<'EOF'
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

step_dir, dest = Path(sys.argv[1]), Path(sys.argv[2])
config = json.loads((step_dir.parent.parent / "configs" / "latest" / "resolved" / "sft.json").read_text())
name = config["model"]["name"]
(dest / "model_name").write_text(name + "\n")
try:
    shutil.copyfile(hf_hub_download(name, "generation_config.json"), dest / "generation_config.json")
except EntryNotFoundError:
    (dest / "generation_config.absent").touch()
EOF
}

main_step10=$E2E_OUT/main1/$ckpt-write/checkpoints/step_10
own_step20=$E2E_OUT/$arm/$ckpt-resume/checkpoints/step_20
worktree=$(arm_worktree "$arm")

record_source_generation_config "$main_step10"
export_and_load "$worktree" "$main_step10" "$ckpt-main-step10"
[ -d "$own_step20" ] && export_and_load "$E2E_WT_MAIN" "$own_step20" "$ckpt-own-step20-by-main"
cat "$export_dir"/$ckpt-*.hfload
