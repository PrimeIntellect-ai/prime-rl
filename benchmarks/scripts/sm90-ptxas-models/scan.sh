#!/bin/bash
# usage: scan.sh <gpu pair csv> <shape>   scan every harvested kernel of a shape under p129 and p133, then dump and count differing configs
pair=$1; shape=$2
M=/home/garrett/tmp/compile-tests-orch/ptxas-scan/models
S=/home/garrett/tmp/compile-tests-orch/ptxas-scan
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-compile-test-resume
h=/tmp/qkvm-harvest-$shape; o=/tmp/qkvm-scan-$shape; rm -rf $o; mkdir -p $o
start=$(date +%s)
grep -rl --include='*.py' 'def get_args' $h/ind | xargs grep -L 'async_compile.triton(' | awk -F/ '!seen[$NF]++' > $M/list-$shape.txt
$M/runscan.sh $M/list-$shape.txt $o 4 $pair p129 p133
cd $WT
uv run --no-sync python $S/compare.py $o > $M/cmp-$shape.txt 2>&1
uv run --no-sync python - $M/cmp-$shape.txt $M/dump-$shape.json <<'PY'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.startswith(("DIFF", "nondet"))]
json.dump([[r[1], r[3]] for r in rows if len(r) > 3], open(sys.argv[2], "w"))
PY
awk -F'\t' '/^(DIFF|nondet)/ {print $2}' $M/cmp-$shape.txt | sort -u > $M/difflist-$shape.txt
if [ -s $M/difflist-$shape.txt ]; then
  DUMP=$M/dump-$shape.json $M/runscan.sh $M/difflist-$shape.txt $o-dump 1 $pair p129 p133
  uv run --no-sync python $S/count.py $o-dump > $M/count-$shape.txt 2>&1
fi
for arm in p129 p133; do echo "$arm $(find $o/cache-$arm-* -name '*.ptx' | xargs grep -h '^\.version' | sort | uniq -c | tr '\n' ' ')"; done > $M/ptxver-$shape.txt
echo "scan-$shape secs=$(($(date +%s)-start)) kernels=$(wc -l < $M/list-$shape.txt) $(head -1 $M/cmp-$shape.txt)" >> $M/scan-summary.txt
