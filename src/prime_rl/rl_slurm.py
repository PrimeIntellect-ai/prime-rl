from pathlib import Path
from typing import Annotated

import tomli_w
from jinja2 import Template
from pydantic import Field

from prime_rl.config import BaseRLLauncherConfig
from prime_rl.utils.pydantic_config import BaseConfig, parse_argv

SLURM_TEMPLATE = """#!/bin/bash

#SBATCH --job-name={{ job_name }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output={{ slurm_log_dir }}/job_%j.log
#SBATCH --error={{ slurm_log_dir }}/job_%j.log

# Configs

export NUM_TRAIN_NODES={{ num_train_nodes }}
export NUM_INFER_NODES={{ num_infer_nodes }}

if [ $((NUM_TRAIN_NODES + NUM_INFER_NODES)) != $SLURM_JOB_NUM_NODES ]; then
    echo "NUM_TRAIN_NODES + NUM_INFER_NODES must equal SLURM_JOB_NUM_NODES"
    exit 1
fi

# Paths
export BASE_DIR="{{ base_dir }}"
export OUTPUT_DIR="{{ output_dir }}/$SLURM_JOB_NAME"


mkdir -p $OUTPUT_DIR/slurm

# Clear previous weights and rollouts
# WARN: This is potentially dangerous and could lead to data loss
# rm -rf $OUTPUT_DIR/weights $OUTPUT_DIR/rollouts
rm -rf $OUTPUT_DIR/rollouts

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Networking
export HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export TRAIN_HOSTS=${HOSTNAMES[@]:0:$NUM_TRAIN_NODES}
export INFER_HOSTS=${HOSTNAMES[@]:$NUM_TRAIN_NODES:$SLURM_JOB_NUM_NODES}

INFER_URLS=""
for host in ${INFER_HOSTS[@]}; do
    if [ -z "$INFER_URLS" ]; then
        INFER_URLS="http://$host:8000/v1"
    else
        INFER_URLS="$INFER_URLS,http://$host:8000/v1"
    fi
done
export INFER_URLS
echo "HOSTNAMES=${HOSTNAMES[@]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[@]}"
echo "INFER_HOSTS=${INFER_HOSTS[@]}"
echo "INFER_URLS=${INFER_URLS}"

export MASTER_ADDR="${HOSTNAMES[0]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Cleanup
srun bash -c 'pkill -9 -f torchrun || true && fuser -k ${MASTER_PORT}/tcp || true'
srun bash -c 'pkill -9 -f prime-rl || true'
srun bash -c 'pkill -9 -f VLLM || true && fuser -k 8000/tcp || true'

# Install environment
cd $BASE_DIR
source .env
source .venv/bin/activate

# Install environment as local package
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Run RL
srun bash -c '
    # Setup environment
    cd $BASE_DIR
    source .env
    source .venv/bin/activate


    # Higher ulimit
    ulimit -n 65536
    export GIT_LFS_SKIP_SMUDGE=1

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[[:blank:]]' |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA

    if [ $SLURM_PROCID -ge $NUM_TRAIN_NODES ]; then
        # This is required for vLLM graph compile to work
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
        INFER_NODE_RANK=$((SLURM_PROCID - NUM_TRAIN_NODES))
        uv run inference \
        @ $OUTPUT_DIR/configs/infer.toml 2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${INFER_NODE_RANK}_infer_node_rank_${INFER_NODE_RANK}.log
    else

        if [ "$SLURM_PROCID" -eq 0 ]; then
            uv run orchestrator \
            --client.base-url $INFER_URLS \
            @ $OUTPUT_DIR/configs/orch.toml 2>&1 | tee $OUTPUT_DIR/slurm/latest_orchestrator.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_orchestrator.log & disown
        fi

        # This is required for compilation to work correctly
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

        echo $HOSTNAMES | tee  $OUTPUT_DIR/slurm/latest_train_node_rank_${SLURM_PROCID}.log
        uv run torchrun \
        --nnodes=$NUM_TRAIN_NODES \
        --nproc-per-node=8 \
        --node-rank=$SLURM_PROCID \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv-id=job_$SLURM_JOB_ID \
        --log-dir=$OUTPUT_DIR/torchrun \
        --tee=3 \
        --redirects=3 \
        --local-ranks-filter=0 \
        src/prime_rl/trainer/rl/train.py \
        @ $OUTPUT_DIR/configs/train.toml 2>&1 | tee -a $OUTPUT_DIR/slurm/latest_train_node_rank_${SLURM_PROCID}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_train_node_rank_${SLURM_PROCID}.log
    fi
'
"""


class SlurmConfig(BaseConfig):
    job_name: Annotated[str, Field(description="The name of the job.")] = "prime-rl"
    log_dir: Annotated[Path, Field(description="The directory to store the slrum logs.")] = Path("outputs/logs")


class RLSLURMConfig(BaseRLLauncherConfig):
    """Configures an RL training run using SLURM."""

    num_training_nodes: Annotated[int, Field(description="The number of training nodes to use.")]
    num_inference_nodes: Annotated[int, Field(description="The number of inference nodes to use.")]

    base_dir: Annotated[
        Path | None,
        Field(description="The base directory of the project. If None, will use the current working directory."),
    ] = None

    output_dir: Annotated[
        Path,
        Field(description="The directory to store the outputs. Should typically be set to an experiment identifier."),
    ] = Path("outputs")

    slurm: SlurmConfig = SlurmConfig()


def rl_slurm(config: RLSLURMConfig):
    if config.weight_broadcast and config.weight_broadcast.type == "nccl":
        raise NotImplementedError("NCCL weight broadcast is not supported for SLURM.")

    template = Template(SLURM_TEMPLATE)
    base_dir = config.base_dir or Path.cwd()
    slurm_script = template.render(
        job_name=config.slurm.job_name,
        num_nodes=config.num_training_nodes + config.num_inference_nodes,
        num_train_nodes=config.num_training_nodes,
        num_infer_nodes=config.num_inference_nodes,
        slurm_log_dir=str(config.slurm.log_dir),
        base_dir=str(base_dir),
        output_dir=str(config.output_dir),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(config.output_dir / "configs/inference.toml", "wb") as f:
        tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)
    with open(config.output_dir / "configs/orchestrator.toml", "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)
    with open(config.output_dir / "configs/trainer.toml", "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(config.output_dir / "slurm.sh", "w") as f:
        f.write(slurm_script)

    print(f"Slurm script written to {config.output_dir / 'slurm.sh'}")
    print(f"run with: sbatch {config.output_dir / 'slurm.sh'}")

    print(f"to view trainer logs: tail -f {config.output_dir / 'slurm/latest_train_node_rank_0.log'}")
    print(f"to view orchestrator logs: tail -f {config.output_dir / 'slurm/latest_orchestrator.log'}")
    print(f"to view inference logs: tail -f {config.output_dir / 'slurm/latest_infer_node_rank_0.log'}")


def main():
    rl_slurm(parse_argv(RLSLURMConfig))


if __name__ == "__main__":
    main()
