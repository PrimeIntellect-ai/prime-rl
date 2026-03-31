# GLM-5 PD Disaggregation

This example guides you through deploying a highly efficient GLM-5 RL training with PD disaggregation and FP8 inference on agentic SWE tasks.

## Requirements

You need access to a Slurm cluster with at least 32 nodes to run this example. Each node must have a shared filesystem. In this guide we assume the NFS is mounted at `/shared`; you can change it to your own path.

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
bash scripts/tmux.sh glm5-pd-disag /shared/outputs/glm5-pd-disag
```

You can then attach to it by doing `tmux attach -t glm5-pd-disag`.

## Start the run

Run the following command to start the RL training:

PS: If using the tmux helper, you can run the command in the `Terminal` (window 0) pane and look at the logs in the `Logs` (window 1) pane.

```bash
uv run rl @ examples/glm5_pd_disag/rl.toml --output-dir /shared/outputs/glm5-pd-disag
```

Output of the command:
```
XXX:XX:XX    INFO Wrote subconfigs to /shared/outputs/glm5-pd-disag/configs [rl.py::515]
XXX:XX:XX    INFO Wrote SLURM script to /shared/outputs/glm5-pd-disag/rl.sbatch [rl.py::534]
XXX:XX:XX    INFO Submitting: sbatch /shared/outputs/glm5-pd-disag/rl.sbatch [rl.py::540]
XXX:XX:XX SUCCESS Submitted batch job YYYY

Logs:
  Trainer:          tail -F /shared/outputs/glm5-pd-disag/logs/trainer.log
  Orchestrator:     tail -F /shared/outputs/glm5-pd-disag/logs/orchestrator.log
  Inference:        tail -F /shared/outputs/glm5-pd-disag/logs/inference.log
  Envs:             tail -F /shared/outputs/glm5-pd-disag/logs/envs/*/*/*.log
   Train:           tail -F /shared/outputs/glm5-pd-disag/logs/envs/train/*/*.log
    swe:           tail -F /shared/outputs/glm5-pd-disag/logs/envs/train/swe/*.log 
```


