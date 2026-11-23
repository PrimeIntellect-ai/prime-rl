ALL_CELLS=(
  rl-dense-nccl ckpt-dense-write ckpt-dense-resume sft-glm rl-moe-nccl
  sft-dense sft-vlm sft-glm-cp2 rl-dense-fs rl-moe-fs ckpt-moe-write ckpt-moe-resume
  sft-qwen-moe
)
TIER1=(rl-dense-nccl ckpt-dense-write ckpt-dense-resume sft-glm rl-moe-nccl)
TIER2=(sft-dense sft-vlm sft-glm-cp2 rl-dense-fs rl-moe-fs ckpt-moe-write ckpt-moe-resume)
TIER3=(sft-qwen-moe)
NOISE=(rl-dense-nccl ckpt-dense-resume sft-glm sft-dense rl-moe-nccl)

arm_worktree() {
  case $1 in
    main*) echo "$E2E_WT_MAIN" ;;
    pr1*) echo "$E2E_WT_PR1" ;;
    pr2*) echo "$E2E_WT_PR2" ;;
    *) return 1 ;;
  esac
}

setup_uv() {
  local worktree=$1
  if [ "$E2E_NODE_LOCAL_VENV" = 1 ]; then
    export VIRTUAL_ENV=$E2E_SCRATCH/venv-$(basename "$worktree")
    UV_RUN="uv run --active --no-sync"
  else
    unset VIRTUAL_ENV
    UV_RUN="uv run --no-sync"
    PY=$worktree/.venv/bin/python
    return
  fi
  PY=$VIRTUAL_ENV/bin/python
}
