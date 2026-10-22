#!/bin/bash
# usage: launch.sh <run> <extra env...> -- <sft args...>
set -u
RUN=$1; shift
ENVS=()
while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
C=/tmp/$USER/$RUN

CMD="rm -rf $C && mkdir -p $C && cd ~/github/PrimeIntellect-ai/prime-rl-feat-dist-moe && env TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUTE_DSL_CACHE_DIR=$C/cute ${ENVS[*]:-} uv run --no-sync sft $* --output-dir $HOME/tmp/dist-moe-sft --run.name $RUN"
echo "srun --jobid=${JOB:-2761} --overlap bash -lc '$CMD'" >> ~/tmp/dist-moe-sft/commands.sh
srun --jobid=${JOB:-2761} --overlap bash -lc "$CMD" < /dev/null > ~/tmp/dist-moe-sft/$RUN.out 2>&1
echo "exit=$?" >> ~/tmp/dist-moe-sft/$RUN.out
