#!/usr/bin/env bash
# The launches behind the reported results of the Dion Muon EP local-expert A/B, in order. A record, not one
# script to rerun: each line ran inside Slurm allocation 3409. Arm A pins Dion 6f9242d, arm B the fix; each
# arm runs from its own worktree detached at the arm's commit, with its own venv.
set -uo pipefail

H=benchmarks/scripts/dion-muon-ep
A=$HOME/github/PrimeIntellect-ai/prime-rl-dion-ab-a
B=$HOME/github/PrimeIntellect-ai/prime-rl-dion-ab-b
N=prime-nebius-puku-h200-gpu
P1=$N-062
P2=$N-064,$N-043
F8=$N-022,$N-024,$N-025,$N-027,$N-043,$N-048,$N-062,$N-064
TRACES=$HOME/tmp/profiling/dion-muon-ep/traces

# P1: 1 node, 6 layers. Arm A runs out of memory at step 1.
$A/$H/run_cell.sh $A p1 dion-ab-p1-a-1 $P1
$B/$H/run_cell.sh $B p1 dion-ab-p1-b-1 $P1

# P2: 2 nodes, 6 layers, untraced timing then a 5-step trace per arm
$A/$H/run_cell.sh $A p2 dion-ab-p2-a-1 $P2
$B/$H/run_cell.sh $B p2 dion-ab-p2-b-1 $P2
$A/$H/run_cell.sh $A p2 dion-ab-p2-a-trace3 $P2 --max-steps 5 --trace-path $TRACES/p2-a3
$B/$H/run_cell.sh $B p2 dion-ab-p2-b-trace $P2 --max-steps 5 --trace-path $TRACES/p2-b

# F8: 8 nodes, full model. Arm A runs out of memory at step 1.
$A/$H/run_cell.sh $A f8 dion-ab-f8-a-2 $F8 --dist-timeout-seconds 1800
$B/$H/run_cell.sh $B f8 dion-ab-f8-b-1 $F8 --dist-timeout-seconds 1800

# F8 truncated to 20 layers, so the baseline fits
$A/$H/run_cell.sh $A f8 dion-ab-f8l20-a-1 $F8 --model.debug.num-layers 20 --dist-timeout-seconds 1800
$B/$H/run_cell.sh $B f8 dion-ab-f8l20-b-1 $F8 --model.debug.num-layers 20 --dist-timeout-seconds 1800
