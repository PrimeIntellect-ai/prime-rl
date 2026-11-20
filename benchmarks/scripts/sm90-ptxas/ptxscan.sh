#!/bin/bash
# usage: ptxscan.sh <triton cache dirs...>  prints count of packed 16-bit integer ops per PTX file
for d in "$@"; do find $d -name '*.ptx'; done | while read p; do
  n=$(grep -cE '\.(s|u)16x2' $p)
  echo "$n $(basename $p .ptx) $p"
done
