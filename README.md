<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
PRIME-RL: Decentralized RL Training at Scale
</h3>

---

<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yaml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yaml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/cpu_tests.yaml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/cpu_tests.yaml/badge.svg" alt="Test" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/gpu_tests.yaml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/gpu_tests.yaml/badge.svg" alt="Test" />
  </a>
</p>

## Overview

PRIME-RL is a framework for large-scale reinforcement learning. It is designed to be easy-to-use and hackable, yet capable of scaling to 1000+ GPUs. Beyond that, here is why we think you might like it:

1. Integrates natively with [`verifiers`](https://github.com/willccbb/verifiers) environments for training and evaluation via the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars)
2. Supports end-to-end post-training workflows, including SFT and RL
3. Rayless multi-node deployment with [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training and [vLLM](https://github.com/vllm-project/vllm) inference backend
4. Designed for asynchronous training for performance and stability in decentralized settings
5. Hackable, modular and extensible by nature

## Setup

> *We develop and test on NVIDIA A100, H100, H200, and B200; likely compatible with other Ampere, Hopper and Blackwell-class GPUs. If setup fails, please create an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

**Quick Setup (Recommended)**

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>
Manual Setup
</summary>
<br>

1. Clone the repository

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install dependencies from the lock file

```bash
uv sync && uv sync --all-extras
```

</details>

<details>
<summary>
Validate your environment setup
</summary>
<br>

1. Check that the environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run SFT trainer in debug model (*this requires 1 GPU*)

```bash
uv run sft @ examples/debug/sft.toml
```

4. Check that you can run the RL trainer debug mode (*this requires 1 GPU*)

```bash
uv run trainer @ examples/debug/rl/train.toml
```

5. Check that you can run the orchestrator against an inference server (*this requires 1 GPU*)

```bash
uv run inference @ examples/debug/rl/infer.toml
```

```bash
uv run orchestrator @ examples/debug/rl/orch.toml
```

6. Check that you can run evals against an inference server (*this requires 1 GPU*)

*Your inference should still be running from step 5. If not, start it again with `uv run inference @ examples/debug/rl/infer.toml`.*

```bash
uv run eval @ examples/debug/eval.toml
```

</details>

## Additional Setup

1. If you want to log your runs to W&B (`wandb`), log in

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

2. If you require gated/ private models or datasets from HuggingFace, log in

```bash
uv run huggingface-cli login
# Or set `export HF_TOKEN=...`
```

3. You may want to increase the maximum number of open files to prevent errors like `Too many open files`.

```bash
ulimit -n 32000
```

## RL

The main usecase of PRIME-RL is RL training. Three main abstractions facilitate RL training: the **orchestrator**, the **trainer**, and the **inference** service.

![Architecture](assets/architecture.png)

We demonstrate how to setup an RL training in the toy [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment. We train a small SFT-warmed up (see [SFT](#sft)) model ([`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT)) to learn to reverse a small chunk of text. Training is extremely quick (~5min on 2x4090) because we allow a maximum context of 128 tokens. We use this run for development and in CI.

To check all available configuration options, run `uv run rl --help`.

### Single-Node Training

First, start a pre-layouted `tmux` session to view the logs from all submodules.

```bash
bash scripts/tmux.sh
```

Then, start the training with the `rl` entrypoint 

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --inference @ examples/reverse_text/rl/infer.toml
```

By default, this command will spin up and tear down the inference server with each invocation. For development purposes it is often useful to start the inference server once and keep it alive across experiments to avoid suffering the vLLM startup time repeatedly.

```bash
# Run this in the `Inference` pane
uv run inference @ examples/reverse_text/rl/infer.toml
```

Then, you can repeatedly restart the trainer and orchestrator in the `Trainer` pane.

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml
```

You can also choose to start each submodule manually. To do so, use the `inference`, `orchestrator` and `trainer` entrypoints.

```bash
# Run this in the `Inference` pane
uv run inference @ examples/reverse_text/rl/infer.toml
```

```bash
# Run this in the `Orchestrator` pane
uv run orchestrator @ examples/reverse_text/rl/orch.toml
```

```bash
# Run this in the `Trainer` pane
uv run trainer @ examples/reverse_text/rl/train.toml
```

### Multi-Node Training

> PRIME-RL is fully compatible with SLURM job scheduling which is convenient for multi-node deployments. We will soon add docs for this.

> We currently require shared file system for multi-node RL training.

**Non-Colocated Trainer and Inference**

On all nodes, export the path to the shared file system (`df -h`), the public IP address (`curl ipinfo.io/ip`) and an API key as environment variables.

```bash
# Export this on all nodes
export OUTPUT_DIR=... # Absolute path to a shared directory
export INFERENCE_SERVER_IP=... # Public IP address of the inference server node
export API_KEY=... # API key for the inference server
```

```bash
# Run inference on one node
uv run inference \
  @ examples/reverse_text/rl/infer.toml \
  --api-key $API_KEY
```

```bash
# Run this on another node
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --orchestrator.client.base-url http://$INFERENCE_SERVER_IP:8000 \
  --orchestrator.client.api-key-var API_KEY \
  --output-dir $OUTPUT_DIR
```

**Multi-Node Trainer**

We rely on `torch.distributed` for multi-node trainer deployments ([docs](https://docs.pytorch.org/docs/stable/elastic/run.html)).

The `torchrun` entrypoint can be used in multi-node distributed training. It will set up the correct number processes on each node and set up inter-node communication. 

For this to work, you need to decide which node will be the master node. On this node, find the private IP with `ip a | grep 10.` or `ip a | grep 192.`.

Then, on each node run 

```bash
export RDZV_ENDPOINT=10.15.42.1:1234
```

*Replace `10.15.42.1` with the private IP address of your master node and `1234` with any open port on the master node.*

Then,  to start the training a training across two full nodes, run the following commands

```bash
# Node 0
uv run  torchrun \
    --nnodes=2 \
    --nproc_per_node 8 \
    --node-rank 0 \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    src/prime_rl/trainer/rl/train.py
```

```bash
# Node 1
uv run  torchrun \
    --nnodes=2 \
    --nproc_per_node 8 \
    --node-rank 1 \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    src/prime_rl/trainer/rl/train.py
```

**Multi-Node Inference**

We rely on vLLM's internal load balancing for data parallel deployment ([docs](https://docs.vllm.ai/en/v0.10.0/serving/data_parallel_deployment.html)).

First, ensure that your nodes are in the same private network and can reach each other. If not, a simple solution is to set up a VPN using [Tailscale](https://tailscale.com). Follow their documentation to setup a VPN on each node. Then, configure the GLOO and NCCL network interface

```bash
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
```

*For example, if you have colocated nodes this is often an Ethernet interface `eth0`. If you use Tailscale VPN, it typically installs a new network interface `tailscale0`.*

Choose one of your nodes to be the head node. On this node, find the private IP with `ip a | grep 10.` or `ip a | grep 192.`

```bash
export DATA_PARALLEL_ADDRESS=10.15.42.1
```

*Replace `10.15.42.1` with the private IP address of your head node.*

Then, to run TP=4 and DP=4 with DP ranks 0 and 1 on the head node and DP ranks 2 and 3 on the second node

```bash
# Node 0  (With IP <reachable-ip>)
uv run inference \
	--data-parallel-size 4 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port 13345
```

```bash
# Node 1
uv run inference \
	--data-parallel-size 4 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port 13345 \
	--data-parallel-start-rank 2 \
	--headless
```

*We have found that restarting the server might require cleaning the RPC port with `fuser -k 13345/tcp` used for communication between the head node and the headless engine cores.*

## SFT

We demonstrate how to setup an SFT warmup using the toy [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment. We have generated a small dataset ([PrimeIntellect/Reverse-Text-SFT](https://huggingface.co/PrimeIntellect/Reverse-Text-SFT)) of examples where the prompt is a small chunk of text and the completion is the reverse of that chunk. We will fine-tune `PrimeIntellect/Qwen3-0.6B` (`Qwen/Qwen3-0.6B` but with Qwen-2.5 chat template) on this dataset. Again, because of the small context, training should be extremely quick.

To check all available configuration options, run `uv run sft --help`.


### Single-Node Training

On a single GPU, start the training with the `sft` entrypoint

```bash
uv run sft @ examples/reverse_text/sft.toml
```

If you have access to multiple GPUs, use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nproc-per-node` to start the training. 

```bash
uv run torchrun \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft.toml
```

### Multi-Node Training

On multiple nodes (potentially with multiple GPUs), use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nnodes` and `--nproc-per-node` to start the training. You need to set up the rendezvous endpoint to allow the nodes to find each other. This should be the private IP address of your master node that needs to be reachable from all other nodes. For more details, see the [PyTorch documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).

```bash
export RDZV_ENDPOINT=...
```

```bash
uv run torchrun \
  --nnodes=2 \
  --nproc-per-node 8 \
  --rdzv-endpoint=$RDZV_ENDPOINT \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft.toml
```

## Evals

You can use PRIME-RL to eval [verifiers](https://github.com/willccbb/verifiers) environments hosted on the [Environment Hub](https://hub.primeintellect.ai) against both API models, local models and checkpoints from an SFT or RL training using the `eval` entrypoint.

> You can also use the `vf-eval` entrypoint for evaluating a *single* environment against API models or local models. However, if want to evaluate multiple environments in parallel and/ or evaluate a training checkpoint, the PRIME-RL `eval` entrypoint is likely more convenient.

We demonstrate evals by evaluating two common benchmarks [`gpqa`](https://app.primeintellect.ai/dashboard/environments/primeintellect/gpqa) and [`math500`](https://app.primeintellect.ai/dashboard/environments/primeintellect/math500).

To check all available configuration options, run `uv run eval --help`.

### API Models

To evaluate API models, you need to set the API base URL and API key. We will exemplify this with the OpenAI API, but the same principles apply to other inference providers.

First, set the API key as an environment variable.

```bash
export OPENAI_API_KEY=...
```

Then, start evaluation by setting the base URL, the name of the environment variable containing the API key, and the model identifier that is exposed by the API.

```bash
uv run eval \
  --client.base-url https://api.openai.com/v1 \
  --client.api-key-var OPENAI_API_KEY \
  --model.name gpt-5-nano \
  --environment-ids math500,aime2025
```

### Local Models

To evaluate any HF model, start an inference server with the desired model before running the `eval` command. For example, to evaluate `Qwen/Qwen3-0.6B` on the `math500` and `aime2025` environments, run the following commands:

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
```

```bash
uv run eval \
  --model.name Qwen/Qwen3-0.6B \
  --environment-ids math500,aime2025
```

### Checkpoints

To evaluate a SFT or RL checkpoint, start an inference server for your base model and specify the directory containing the checkpoints with `--weights-dir`. 

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
```

```bash
uv run eval \
  --model.name Qwen/Qwen3-0.6B \
  --weights-dir outputs/weights
```

By default, this will evaluate the base model and all step checkpoints found in the weights directory. To skip evaling the base model, set `--no-eval-base` and to evaluate only specific steps, set `--steps` as a comma-separated list of integers representing the steps to evaluate.

## Advanced Usage

### Single-GPU Training

If you only have access to a single GPU, you may still be able to run small experiments. To do so, configure your inference server to use only a fraction of the available memory. By default, the `rl` entrypoint assumes fully disjoint training and inference, so you will have to start your single-GPU memory manually as follows:

```bash
# Run this in the `Inference` pane
uv run inference @ examples/reverse_text/rl/infer.toml --gpu-memory-utilization 0.5
```

*Tune this value such that you have enough GPU memory for the RL trainer*

```bash
# Run this in the `Orchestrator` pane
uv run orchestrator @ examples/reverse_text/rl/orch.toml
```

```bash
# Run this in the `Trainer` pane
uv run trainer examples/reverse_text/rl/train.toml
```

### Parallel Experiments

For quick ablations, it can be more efficient to parallelize experiments within a node (e.g. split your GPUs to run two experiments in parallel). Because the trainer communicates with the orchestrator via shared file system, and the orchestrator communicates with the inference engine via an OAI-compatible API, the connection points have to be uniquely set. For example, if you have access to 4 GPUs you can run two 2 GPU training runs in parallel as follows:

Start the first experiment in a tmux session `exp1` with outputs directory `outputs1`. Specify it both in the tmux script, as well as in the start command (*will use the first 2 GPUs*)

```bash
bash scripts/tmux.sh -s exp1 -o outputs1
```

```bash
# Start the first experiment
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --output-dir outputs1
```

For the second experiment, start a second tmux session named `exp2` with outputs directory `outputs2`. In addition, specify a new server port for the inference engine and orchestrator (*will use the last 2 GPUs*)

```bash
bash scripts/tmux.sh -s exp-2 -o outputs2
```

```bash
# Start the second experiment
CUDA_VISIBLE_DEVICES=2,3 uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --inference.server.port http://localhost:8001 \
  --orchestrator.client.base-url http://localhost:8001 \
  --output-dir outputs2
```

## Developer

> For now, development is only possible on CUDA-enabled devices.

### Setup

1. Install [pre-commit](https://pre-commit.com) hooks

```bash
uv run pre-commit install
```

### Configs

**Sources**

We support the following sources for configuration, in this order of precedence:

1. **Command-line arguments**: You can pass (nested) arguments as `--key.subkey value` to the script. For example, to set the model name you can run `--model.name`

2. **Config files**: You can pass `.toml` config files (defined in the `configs` directory) using the `@` prefix. For example, to use the `debug.toml` config file, you can run `uv run inference @ configs/debug/infer.toml`. (*If you leave a space between the `@` and the config file, you will get shell path auto-completions.*)

3. **Environment variables**: You can set environment variables to override the config values. All environment variables must be prefixed with `PRIME_` and use the `__` delimiter to nest the keys. For example, to set the model name you can run `export PRIME_MODEL__NAME=Qwen/Qwen3-0.6B`.

4. **Defaults**: For almost all config arguments, we have a default value which will be used if no other source is provided.

In general we recommend setting configurations via config files to define reproducible experiments and use command-line arguments to override the config values to run variants of the same experiment. Environment variables are usually only used in production settings to communicate with the [Prime Protocol](https://github.com/PrimeIntellect-ai/protocol) worker. In most cases, you should not need to use environment variables.

The precedence order will be important if multiple sources try to configure the same argument. For example, in the following command, all sources will define a model name

```toml
# qwen8b.toml
[model]
name = "Qwen/Qwen3-8B"
```

```toml
# qwen14b.toml
[model]
name = "Qwen/Qwen-14B"
```

```bash
PRIME_MODEL__NAME=Qwen/Qwen3-4B uv run inference @qwen8b.toml @qwen14b.toml --model.name Qwen/Qwen3-32B
```

In this example, the CLI argument `--model.name Qwen/Qwen3-32B` will take precendence and the script will use `Qwen/Qwen3-32B` as the model name. If the CLI argument wasn't set, then the second config file would take precedence and the script would use `Qwen/Qwen-14B` as the model name. If the second config file wasn't set, then the first config file would take precedence and the script would use `Qwen/Qwen3-8B` as the model name. Finally, if the first config file wasn't set, then the environment variable would take precedence and the script would use `Qwen/Qwen-4B` as the model name. If the environment variable wasn't set, then the default value would be used and the script would use `Qwen/Qwen3-0.6B` as the model name.

### Environments

`prime-rl` supports Environment modules built with `verifiers` ([repo](https://github.com/willccbb/verifiers)) for training tasks. All of our current research environments live in a separate [Prime Environments](https://github.com/PrimeIntellect-ai/prime-environments) repository. 

To add a new training or evaluation environment, please follow the instructions in the [Prime Environments](https://github.com/PrimeIntellect-ai/prime-environments) repository.

To then use it as part of `prime-rl`, install the newly pushed environment via the Environment Hub. 

To install your Environment module temporarily within `prime-rl`, do:
```bash
uv run prime env install custom-environment
```

To persist your Environment module installation in the package-wide `pyproject.toml`, do:
```bash
uv add --optional vf "custom-environment @ https://hub.primeintellect.ai/your-username/custom-environment/@latest/custom-environment-0.1.3-py2.py3-none-any.whl"
```

For quick API-based testing post-installation, do:
```bash
uv run vf-eval custom-environment # -h for config options; defaults to gpt-4.1-mini, 5 prompts, 3 rollouts each
```

For training, create `trainer`/`inference`/`orchestrator` config files following the aforementioned examples, then set `id = custom-environment` in the `[environment]` section of your `orchestrator` config (along with any desired Environment-level args in `[environment.args]`).

### W&B

For any serious run we recommend logging to W&B. Since it is disabled by default, you have to set up W&B. First, make sure that you are logged in.

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

Both the trainer and orchestrator can log to W&B as separate runs using the `--monitor.wandb` subconfig. You can set the project (`--monitor.wandb.project`, defaults to `prime-rl`), run name (`--monitor.wandb.name`, defaults to `None` which will make W&B generate a name randomly), run ID (`--monitor.wandb.id`, defaults to `None`), the log directory (`--monitor.wandb.dir`, defaults to `logs`) and whether or not to run in offline mode (`--monitor.wandb.offline`, defaults to `False`). 

First, start your inference server

```bash
uv run inference @ configs/reverse_text/infer.toml
```

Then, start the trainer and orchestrator with logging enabled.

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml --monitor.wandb.project example-project --monitor.wandb.name trainer
```

```bash
uv run orchestrator @ configs/reverse_text/orch.toml --monitor.wandb.project example-project --monitor.wandb.name orchestrator
```

Usually it will be more convenient to use the `rl` entrypoint. To setup W&B concisely, you can specify shared configs using the `--wandb` subconfig, e.g. the project (`--wandb.project`), run name (`--wandb.name`), directory (`--wandb.dir`) and offline mode (`--wandb.offline`). It will automatically share these configs to the trainer and orchestrator. For the run name, it will automatically suffix the specified name with `-trainer` and `-orchestrator` to clearly distinguish those runs.

```bash
uv run rl   \
  --trainer @ configs/reverse_text/train.toml  \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --wandb.project example-project \
  --wandb.name example-run
```

We support logging samples (e.g. prompt, completion, reward, advantage for selected rollouts) and distributions (e.g. reward, advantage, entropy distributions) as W&B tables using the `monitor.wandb.log-extras` subconfig. On the orchestrator you can log activate logging samples (`--monitor.wandb.log-extras.samples`) and distributions (`--monitor.wandb.log-extras.samples`). On the trainer you can only log distributions (`--monitor.wandb.log-extras.distributions`). On both, you can specify the logging step interval using `--monitor.wandb.log-extras.interval`. To log all extras on trainer and orchestrator every 10 steps, 

```bash
uv run rl   \
  --trainer @ configs/reverse_text/train.toml  \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --wandb.project example-project \
  --wandb.name example-run \
  --trainer.monitor.wandb.log-extras.distributions \
  --trainer.monitor.wandb.log-extras.interval 10 \
  --orchestrator.monitor.wandb.log-extras.samples \
  --orchestrator.monitor.wandb.log-extras.distributions \
  --orchestrator.monitor.wandb.log-extras.interval 10
```

### Checkpointing

Our codebase supports checkpointing. Because of the trainer/ orchestrator design, as well as the natural asynchrony checkpointing is non-standard.

- Trainer (`src/prime_rl/trainer/ckpt.py`): Checkpoints FSDP model shard, optimizer state and progress (training step, total samples, total tokens)
- Orchestrator (`src/prime_rl/orchestrator/ckpt.py`): Checkpoints orchestrator progress

*NB: Each run with asynchrony level `async_level` and some checkpoint step `x`, requires weight checkpoints in the step range `[x-async_level, x]`. Currently we do not duplicate weight checkpoints into the `checkpoints` directory but simply keep them around in `weights`, by keeping the trainer from cleaning up weight checkpoints that are required for resuming training. This way, the orchestrator only needs to checkpoint its progress (read: step) to load the correct weights into the inference engine upon resuming.*

The default checkpoint directory is `checkpoints` and each checkpoint step will live in a subdirectory enumerated by the step, i.e. `checkpoints/step_{step}`. The trainer checkpoint is called `trainer.pt` for single GPU workloads, else `trainer_{local_rank}.pt`. The orchestrator checkpoint is called `orchestrator.pt`. Thus, this is a typical directory structure:

```bash
checkpoints
├── step_10
│   ├── orchestrator.pt
│   └── trainer.pt
├── step_25
│   ├── orchestrator.pt
│   └── trainer.pt
└── step_30
    ├── orchestrator.pt
    └── trainer.pt
```

Checkpointing is configured by the `CheckpointConfig`, with the config key `--ckpt`. One can specify the interval (`--ckpt.interval`, defaults to `50`), whether to save checkpoints asynchronoously  (`--ckpt.save-async`, defaults to `False`), and how many recent step checkpoints to keep on disk (`--ckpt.keep`, defaults to `None` which means no cleanup).

By default, runs do no write checkpoints to save disk space. To checkpoint every 10 steps on our debug RL run, run the following command

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml --ckpt.interval 10 
```

To resume a run use the `--ckpt.resume-step` flag. To resume from the checkpoint step 10 from the previous command, run the following command

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml --ckpt.resume-step 10
```

Because we save progress information, resuming from a checkpoint is fully W&B compatible. By default, resuming from a checkpoint, will simply create a new run. To resume the same W&B run, you'd have to pass the same W&B run ID for both the trainer and the orchestrator, e.g.

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml \
  --monitor.wandb.project <project> \
  --ckpt.resume-step 10 \
  --monitor.wandb.id <trainer-run-id> \
```

You also need to restart the orchestrator from a checkpoint, the api is the same as the trainer, e.g.

```bash
uv run orchestrator @ configs/reverse_text/orch.toml \
  --monitor.wandb.project <project> \
  --ckpt.resume-step 10 \
  --monitor.wandb.id <orchestrator-run-id>
```

If you started your run using `rl.py`, you can resume the same run by passing the same W&B run ID for both the trainer and the orchestrator, e.g.

```bash
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --ckpt.resume-step 10 \
  --trainer.monitor.wandb.id <trainer-run-id> \
  --orchestrator.monitor.wandb.id <orchestrator-run-id> 
```

You don't need to restart the inference server if started manually, the orchestrator will automatically send the right checkpoint to the inference server when resuming.

### Benchmarking

We provide a convenient way to benchmark the performance of the inference engine and trainer using the `--bench` flag. It will run each module in isolation for a few steps and log performance statistics to the console and, optionally, W&B.

**Inference**

To benchmark inference, first spin up the inference server with an experiment configuration

```bash
uv run inference @ configs/reverse_text/infer.toml
```

Then, start the orchestrator with the matching configuration file in benchmark mode

```bash
uv run orchestrator @ configs/reverse_text/orch.toml --bench
```

**Trainer**

To benchmark the RL trainer, simply run the trainer against a fake data loader with batch certain specifications.

```bash
uv run trainer @ configs/reverse_text/train.toml --bench --data.fake.micro_batch_size 8 --data.fake.batch_size 128 --data.fake.seq_len 128
```

**RL**

You can benchmark both the RL trainer and inference at the same time with the `rl.py` entrypoint. Note, that the benchmarking is still decoupled.

```bash
uv run rl   \
  --trainer @ configs/reverse_text/train.toml  \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --bench
```

**SFT**

Benchmark the SFT trainer against `fixed` or `variable` length fake data by specifyin `--data.fake.type`

```bash
uv run sft --bench --data.fake.type fixed --data.micro-batch-size 8 --data.batch-size 8 --data.seq-len 128
```

### Tests

Run the full test suite 

```bash
uv run pytest -v
```

To run unit tests, run

```bash
uv run pytest tests/unit -v
```

To run integration tests, run

```bash
uv run pytest tests/integration -v
```

To run CPU-only tests, use the inverse of the `gpu` marker:

```bash
uv run pytest -v -m "not gpu"
```

To run fast tests, use the inverse of the `slow` marker:

```bash
uv run pytest -v -m "not slow"
```

### Step Definition

At each step `n`, all artifacts (e.g., checkpoint, rollout, gradient) are tagged with the same step number.
- Step 0:
  - Uses checkpoint 0 on rollout 0 to compute gradient 0.
  - Then computes checkpoint 1 as: `ckpt 1 = ckpt 0 - grad 0`

In general, the model used for generating rollouts at step `n` is from `ckpt[n - async_level]`.

- When async_level = 0, the rollout and gradient are based on the same model version.
  This is equivalent to synchronous on-policy training.

## License

This project is licensed under the Apache 2.0 license, as found in the [License](LICENSE) file.

## Citation

If you find our work useful, feel free to cite it using

```tex
@misc{primeintellect2025prime-rl,
  author = {Prime Intellect},
  title = {PRIME-RL},
  url = {https://github.com/PrimeIntellect-ai/prime-rl},
  year = {2025}
}
```
