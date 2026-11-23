export E2E_REPO_PARENT=/home/garrett/github/PrimeIntellect-ai
export E2E_WT_MAIN=$E2E_REPO_PARENT/prime-rl-run-drophf-main
export E2E_WT_PR1=$E2E_REPO_PARENT/prime-rl-run-drophf-pr1
export E2E_WT_PR2=$E2E_REPO_PARENT/prime-rl-run-drophf-pr2

export PRL_OUTPUT_DIR=/home/garrett/prl_output_dir
export E2E_OUT=$PRL_OUTPUT_DIR/drophf-e2e
export E2E_DATA=$E2E_OUT/data
export E2E_SCRATCH=/tmp/$USER/drophf-e2e

export HF_HOME=/home/huggingface/
export TRITON_CACHE_DIR=/tmp/$USER/triton-E

E2E_NODE_LOCAL_VENV=${E2E_NODE_LOCAL_VENV:-1}
