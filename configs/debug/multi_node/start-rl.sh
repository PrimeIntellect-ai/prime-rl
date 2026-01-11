#!/bin/bash

export NUM_INFER_NODES=1
export NUM_TRAIN_NODES=1
export NUM_TOTAL_NODES=$((NUM_TRAIN_NODES + NUM_INFER_NODES))


sbatch -N $NUM_TOTAL_NODES --job-name=repro configs/debug/multi_node/rl.sh

