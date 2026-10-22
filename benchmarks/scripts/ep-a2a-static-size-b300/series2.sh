#!/bin/bash
# Same-node arms for both PR descriptions: main reruns, #4086 alone (exp), rebased #4068 alone (cp), both stacked (stack).
L=~/tmp/pr4086-ab/launch.sh
C=~/tmp/pr4086-ab/cfg
TL="TORCH_LOGS=graph_breaks,recompiles"
go() { echo "start $2 $(date -u +%T)"; $L "$@"; echo "done $2 $(date -u +%T) $(tail -1 ~/tmp/pr4086-ab/logs/$2.out)"; }
G=$C/glm-4.5-air-8l.toml
go stack glm-air-cp2-stack $TL -- @ $G @ $C/cp2.toml
go cp glm-air-cp2-4068 $TL -- @ $G @ $C/cp2.toml
go main glm-air-cp2-main $TL -- @ $G @ $C/cp2.toml
go exp glm-air-cp1-exp $TL -- @ $G
go main glm-air-cp1-main $TL -- @ $G
go main qwen-bf16-main $TL -- @ $C/qwen3-30b-a3b-12l.toml
go main mini-glm-bf16-main $TL -- @ $C/mini-glm-moe.toml
