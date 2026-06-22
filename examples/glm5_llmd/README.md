# High-throughput GLM-5 with llm-d

This example trains `GLM-5.1` with RL on agentic SWE tasks using P/D disaggregation, FP8 inference, and a Mooncake distributed CPU KV store — fronted by the upstream [**llm-d**](https://llm-d.ai) router (Endpoint Picker + Envoy).

**llm-d is the faster way to run GLM-5.** It is the recommended router backend for large GLM-5 runs: its `active-request-scorer` is an in-flight load balancer that spreads requests across ranks immediately, instead of relying on metrics that are scraped on a delay (as the default `vllm-router` does). Combined with prefix-cache affinity for grouped rollouts, this keeps prefill and decode ranks evenly loaded under the bursty request pattern of RL. If you want the simpler default router instead, see the [GLM-5 PD disaggregation example](../glm5_pd_disag/README.md).

## Requirements

You need access to a Slurm cluster with at least **32 nodes** (16 trainer + 16 inference, 8 GPUs each) and a shared filesystem. In this guide we assume the NFS is mounted at `/shared`; you can change it to your own path.

You also need prime-rl cloned on your cluster into the shared filesystem.

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
uv sync --all-extras
```

### Install llm-d

The llm-d router ships as vendored binaries (`epp`, `envoy`, `pd-sidecar`). Build them once into `third_party/llmd/bin`:

```bash
bash scripts/install_llmd.sh
```

You might also want to create a `.env` file inside the prime-rl directory to store environment variables used during training like W&B and Hugging Face tokens. The `.env` file is automatically sourced during training.

```bash
touch .env
echo "WANDB_API_KEY=your_wandb_api_key" >> .env
echo "HUGGINGFACE_TOKEN=your_huggingface_token" >> .env
```

### sandbox

The SWE environment is configured to use Prime Intellect Sandboxes. You can find more information about the sandboxes [here](https://docs.primeintellect.ai/sandboxes/overview). You will need to create a sandbox account and add the credentials to the `.env` file. Alternatively, you can adapt the code of the environment to use your own sandbox implementation.

## Tweak before launching

This config is tuned for our cluster — a few values **must** be adapted to yours before you run. Search the config for these markers:

- **`output_dir`** (`# FILL IN`) — point it at your shared filesystem.
- **`[slurm] partition`** (`# FILL IN`) — set it to your cluster's Slurm partition.
- **`[inference.kv_cache_offload] device_name`** (`# do it yourself`) — the list of RDMA NICs for Mooncake. Auto-detection is unreliable, so set it by hand from `nvidia-smi topo -m` on your nodes.

A few more knobs you may want to tune for your hardware:

- **Trainer parallelism** — this config assumes **16 trainer nodes** at **131K** sequence length with `cp = 4` (context parallel) and the `adamw` optimizer. If you have fewer trainer nodes, drop `seq_len`, lower `num_train_nodes`, and reduce `cp` accordingly.
- **Scorer weights** (`[inference.deployment.router]`) — the `scorers` / `prefill_scorer_overrides` / `decode_scorer_overrides` weights follow the upstream llm-d P/D guide and are a good starting point; tune them if your prefill/decode balance drifts.
- **Mooncake CPU pool** (`[inference.kv_cache_offload.cpu] num_bytes`) — defaults to 1TB/node; lower it if your nodes have less RAM.

## Tmux session

We recommend using the tmux helper to start the run and look at the logs.

From your Slurm head node:

```bash
bash scripts/tmux.sh glm5-llmd /shared/outputs/glm5-llmd
```

You can then attach to it by doing `tmux attach -t glm5-llmd`.

## Start the run

Run the following command to start the RL training:

PS: If using the tmux helper, you can run the command in the `Terminal` (window 0) pane and look at the logs in the `Logs` (window 1) pane.

```bash
uv run rl @ examples/glm5_llmd/rl.toml --output-dir /shared/outputs/glm5-llmd
```

Output of the command:
```
XXX:XX:XX    INFO Wrote subconfigs to /shared/outputs/glm5-llmd/configs [rl.py::515]
XXX:XX:XX    INFO Wrote SLURM script to /shared/outputs/glm5-llmd/rl.sbatch [rl.py::534]
XXX:XX:XX    INFO Submitting: sbatch /shared/outputs/glm5-llmd/rl.sbatch [rl.py::540]
XXX:XX:XX SUCCESS Submitted batch job YYYY

Logs:
  Trainer:          tail -F /shared/outputs/glm5-llmd/logs/trainer.log
  Orchestrator:     tail -F /shared/outputs/glm5-llmd/logs/orchestrator.log
  Inference:        tail -F /shared/outputs/glm5-llmd/logs/inference.log
  Envs:             tail -F /shared/outputs/glm5-llmd/logs/envs/*/*/*.log
```
