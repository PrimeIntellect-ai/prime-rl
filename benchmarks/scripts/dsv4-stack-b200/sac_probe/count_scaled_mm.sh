#!/usr/bin/env bash
D=/tmp/$USER/dsv4-perf-proxy/$1/inductor
for f in $(grep -rl "extern_kernels._scaled_mm\|torch.ops.aten._scaled_mm\|aten._scaled_mm" $D --include=*.py); do
    kind=fwd
    grep -q "tangents_" $f && kind=bwd
    n=$(grep -cE "_scaled_mm\(|_scaled_mm\.default\(" $f)
    echo "$kind scaled_mm_calls=$n"
done | sort | uniq -c
