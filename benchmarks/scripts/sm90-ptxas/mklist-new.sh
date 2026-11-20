#!/bin/bash
# usage: mklist-new.sh <out list> <exclude list files (comma sep)> <harvest dirs...>   like mklist.sh, dropping basenames already in the exclude lists
out=$1; excl=$2; shift 2
tmp=$out.all
/home/garrett/tmp/compile-tests-orch/ptxas-scan/mklist.sh $tmp "$@" > /dev/null
cat ${excl//,/ } | awk -F/ '{print $NF}' | sort -u > $out.excl
awk -F/ 'NR==FNR {skip[$1]=1; next} !skip[$NF]' $out.excl $tmp > $out
wc -l < $out
