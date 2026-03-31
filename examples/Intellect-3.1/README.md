# Intellect-3.1

This example guides you through reproducing the [INTELLECT-3.1](https://huggingface.co/PrimeIntellect/INTELLECT-3.1) RL training on top of [GLM-4.5-Air-Base](https://huggingface.co/zai-org/GLM-4.5-Air-Base).

## Requirements

You need access to a Slurm cluster with at least 16 nodes to run this example. Each node must have a shared filesystem. In this guide we assume the NFS is mounted at `/shared`; you can change it to your own path.

If you don't have access to that many nodes, you can downscale inference and training nodes to lower values and scale down the batch size accordingly.

For training, going below 4 nodes might cause OOM.

You also need to have prime-rl cloned on your cluster into the shared filesystem.

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
uv sync --all-extras
```

You might also want to create a `.env` file inside the prime-rl directory to store environment variables used during training like W&B and Hugging Face tokens. The `.env` file is automatically sourced during training.

```bash
touch .env
```

```bash
echo "WANDB_API_KEY=your_wandb_api_key" >> .env
echo "HUGGINGFACE_TOKEN=your_huggingface_token" >> .env
```

### sandbox

The [mini-swe-agent-plus](https://github.com/PrimeIntellect-ai/sandbox-mini-swe-agent-plus) environment is configured to use Prime Intellect Sandboxes. You can find more information about the sandboxes [here](https://docs.primeintellect.ai/sandboxes/overview).

You will need to create a sandbox account and add the credentials to the `.env` file.

Alternatively, you can adapt the code of the environment to use your own sandbox implementation.

## Tmux session

We recommend using the tmux helper to start the run and look at the logs.

From your Slurm head node:

```bash
bash scripts/slurm_tmux.sh intellect-3.1 /shared/outputs/intellect-3.1
```

You can then attach to it by doing `tmux attach -t intellect-3.1`.

## Start the run

Run the following command to start the RL training:

PS: If using the tmux helper, you can run the command in the `Terminal` (window 0) pane and look at the logs in the `Logs` (window 1) pane.

```bash
uv run rl @ examples/Intellect-3.1/rl.toml --output-dir /shared/outputs/intellect-3.1
```

Output of the command:
```
XXX:XX:XX    INFO Wrote subconfigs to /shared/outputs/intellect-3.1/configs [rl.py::515]
XXX:XX:XX    INFO Wrote SLURM script to /shared/outputs/intellect-3.1/rl.sbatch [rl.py::534]
XXX:XX:XX    INFO Submitting: sbatch /shared/outputs/intellect-3.1/rl.sbatch [rl.py::540]
XXX:XX:XX SUCCESS Submitted batch job YYYY

Logs:
  Trainer:          tail -F /shared/outputs/intellect-3.1/slurm/latest_train_node_rank_0.log
  Orchestrator:     tail -F /shared/outputs/intellect-3.1/slurm/latest_orchestrator.log
  Inference:        tail -F /shared/outputs/intellect-3.1/slurm/latest_infer_node_rank_0.log
  Envs:             tail -F /shared/outputs/intellect-3.1/logs/envs/*/*/*.log
   Train:           tail -F /shared/outputs/intellect-3.1/logs/envs/train/*/*.log
    swe:           tail -F /shared/outputs/intellect-3.1/logs/envs/train/swe/*.log 
```


