#!/bin/bash
# usage: mklist.sh <out list> <harvest dirs...>   one kernel file per unique basename, only kernel files with a benchmark harness
out=$1; shift
for d in "$@"; do grep -rl --include='*.py' 'def get_args' $d/ind | xargs grep -L 'async_compile.triton(' ; done | awk -F/ '!seen[$NF]++' > $out
wc -l < $out
