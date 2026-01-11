export NUM_INFER_NODES=1
export NUM_TRAIN_NODES=1
export NUM_TOTAL_NODES=$((NUM_TRAIN_NODES + NUM_INFER_NODES))



export 

sbatch -N $NUM_TOTAL_NODES --job-name=repro debug/multi_node/rl.sh

