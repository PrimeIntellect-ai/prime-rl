#!/bin/bash
# PR 4086 (cherry-picked onto main as exp/ep-a2a-on-main) vs main: bf16 sanity, MXFP8 transport and compute, cp=2 real data.
L=~/tmp/pr4086-ab/launch.sh
C=~/tmp/pr4086-ab/cfg
TL="TORCH_LOGS=graph_breaks,recompiles"
go() { echo "start $2 $(date -u +%T)"; $L "$@"; echo "done $2 $(date -u +%T) $(tail -1 ~/tmp/pr4086-ab/logs/$2.out)"; }
Q=$C/qwen3-30b-a3b-12l.toml
go exp mxt-exp $TL -- @ $Q @ $C/mxfp8-transport.toml
go main mxt-main $TL -- @ $Q @ $C/mxfp8-transport.toml
go main mxt-main-fg $TL -- @ $Q @ $C/mxfp8-transport.toml @ $C/compile-fullgraph.toml
go exp mxct-exp $TL -- @ $Q @ $C/mxfp8-compute-transport.toml
go main mxct-main $TL -- @ $Q @ $C/mxfp8-compute-transport.toml
go main mxct-main-fg $TL -- @ $Q @ $C/mxfp8-compute-transport.toml @ $C/compile-fullgraph.toml
go exp glm-air-cp2-exp $TL -- @ $C/glm-4.5-air-8l.toml @ $C/cp2.toml
go exp qwen-bf16-exp $TL -- @ $Q
go exp mini-glm-bf16-exp $TL -- @ $C/mini-glm-moe.toml
