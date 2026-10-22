#!/bin/bash
# usage: launch.sh <main|exp|cp|stack> <run> <extra env...> -- <sft args...>
set -u
case $1 in main) R=~/github/PrimeIntellect-ai/prime-rl-main;; exp) R=~/github/PrimeIntellect-ai/prime-rl-exp-ep-a2a-on-main;; cp) R=~/github/PrimeIntellect-ai/prime-rl-fix-cp-attn-params-as-args;; stack) R=~/github/PrimeIntellect-ai/prime-rl-exp-cp-params-plus-ep-a2a;; esac; shift
RUN=$1; shift
ENVS=()
while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
C=/tmp/$USER/pr4086/$RUN
CMD="rm -rf $C && mkdir -p $C && cd $R && echo HEAD=\$(git rev-parse --short HEAD) && env TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor ${ENVS[*]:-} uv run --no-sync sft $* --output-dir $HOME/tmp/pr4086-ab/runs --run.name $RUN"
echo "srun --jobid=${JOB:-2816} --overlap bash -lc '$CMD'" >> ~/tmp/pr4086-ab/commands.sh
srun --jobid=${JOB:-2816} --overlap bash -lc "$CMD" < /dev/null > ~/tmp/pr4086-ab/logs/$RUN.out 2>&1
echo "exit=$?" >> ~/tmp/pr4086-ab/logs/$RUN.out
