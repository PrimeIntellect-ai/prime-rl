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

## Installation

> *`prime-rl` is developed and tested on NVIDIA A100, H100, H200, and B200; likely compatible with other Ampere, Hopper and Blackwell-class GPUs. If your installation fails, please create an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

**Quick Installation (Recommended)**

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>
Manual Installation
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

3. Synchronize the environment

```bash
uv sync && uv sync --all-extras
```

</details>


<details>
<summary>
Validate your environment setup
</summary>
<br>

1. Check that environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run SFT trainer in debug model (*this requires 1 GPU)

```bash
uv run sft @ configs/debug/sft.toml
```

4. Check that you can run the RL trainer debug mode (*this requires 1 GPU*)

```bash
uv run trainer @ configs/debug/train.toml
```

5. Check that you can run the orchestrator against an inference server (*this requires 1 GPU*)

```bash
uv run inference @ configs/debug/infer.toml
```

```bash
uv run orchestrator @ configs/debug/orch.toml
```

4. Check that you can run a simple SFT warmup (*this requires 1 GPU*)

```bash
uv run sft @ configs/reverse_text/sft.toml
```

5. Check that you can run a toy RL run (*this requires 2 GPUs*)

```bash
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml
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

3. You may want to increase the number of open files to prevent errors like `Too many open files`.

```bash
ulimit -n 32000
```

4.  We provide a convenient tmux layout script to start a run and view logs. To start the session simply run

```bash
bash scripts/tmux.sh
```

The default session name is `prime-rl` and outputs directory is `outputs`. They are configurable, as described in `bash scripts/tmux.sh -h`.

## RL

We provide a convenience endpoint `rl` for single-node RL experiments. It configures and starts the trainer, orchestrator and, optionally, an inference server. It allows setting shared configs conveniently (e.g. the model name or async level should be the same across all modules) and dispatches and monitors subprocesses. To stream the logs from each module we use file logging. By default, only the trainer logs will be displayed on the main process (*use the tmux layout script to conveniently view all logs*).

To check all available configuration options, run `uv run rl --help`.

**Reverse Text**

Train a tiny model (`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`) that has undergone SFT warmup (see below) to learn to reverse a small chunk of text. Training is extremely quick because we allow a maximum context of 128 tokens. We use this run in CI.

```bash
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml
```

*With two small GPUs (e.g. RTX 3090/4090), this experiment should finish in less than 5 minutes.*

**Hendrycks Math**

Train a small model (`willcb/DeepSeek-R1-Distill-Qwen-1.5B`) on high-school level math questions. It is recommended to have at least 2xA100-80GB GPUs or more for this experiment.

On eight GPUs, run the following command to run the experiment.

```bash
uv run rl \
  --trainer @ configs/hendrycks_math/1b/train.toml \
  --orchestrator @ configs/hendrycks_math/1b/orch.toml \
  --inference @ configs/hendrycks_math/1b/infer.toml \
  --trainer-gpus 2 --inference-gpus 6
```

*NB: This setup requires 8 GPUs - 2 are used for the FSDP trainer, 6 are used for inference.*

**INTELLECT-2 Math**

Train a small model (`willcb/DeepSeek-R1-Distill-Qwen-1.5B`) on complex math questions from the INTELLECT-2 dataset.

```bash
uv run rl \
  --trainer @ configs/intellect_math/1b/train.toml \
  --orchestrator @ configs/intellect_math/1b/orch.toml \
  --inference @ configs/intellect_math/1b/infer.toml \
  --trainer-gpus 2 --inference-gpus 6
```

*NB: This setup requires 8 GPUs - 2 are used for the FSDP trainer, 6 are used for inference.*

### Multi-Node Training

The trainer and inference can be decoupled naively as they communicate via HTTP and shared file system. Simply spin up your inference server on one node, and your trainer on another node. Make sure that the orchestrator (colocated with the trainer) points to the public IP address of the node hosting the inference server and that they have access to a shared path used for outputs.

If you want to go even bigger, you can run the trainer and/ or inference on multi-node.

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

*It will automatically set up the local and global world information correctly.*

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

### Multiple Experiments per Node

For small models/ quick ablations, it can be more efficient to parallelize experiments within a node (e.g. split your GPUs to run two experiments in parallel). Because the trainer communicates with the orchestrator via a shared file system, and the orchestrator communicates with the inference engine via an OAI-compatible API, the connection points have to be uniquely set. For example, if you have access to 4 GPUs you can run two 2 GPU training runs in parallel as follows:

Start the first experiment in a tmux session `exp-1` with outputs directory `outputs`. Specify it both in the tmux script, as well as in the start command (*will use the first 2 GPUs*)

```bash
bash scripts/tmux.sh -s exp-1 -o outputs1
```

```bash
# Start the first experiment
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --outputs-dir outputs1
```

For the second experiment, start a second tmux session named `exp-2` with outputs directory `outputs2`. In addition, specify a new server port for the inference engine and orchestrator (*will use the first 2 GPUs*)

```bash
bash scripts/tmux.sh -s exp-2 -o outputs2
```

```bash
# Start the second experiment
CUDA_VISIBLE_DEVICES=2,3 uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --inference.server.port 8001 \
  --orchestrator.client.port 8001 \
  --outputs-dir outputs2
```

## SFT

We have built-in support for SFT using the `sft` entrypoint. SFT often proves useful prior to RL training to get the model in-distribution and have higher initial reward that it can hillclimb from. 

To check all available configuration options, run `uv run sft --help`.

**Reverse Text**

Finetune `PrimeIntellect/Qwen3-0.6B` (`Qwen/Qwen3-0.6B` but with Qwen-2.5 chat template) to reverse a tiny chunk of text, used as a warmup model train in `reverse-text` environment. We use this run in CI.

```bash
uv run sft @ configs/reverse_text/sft.toml
```

If you want to train on multiple GPUs or nodes, use `torchrun`. For example, to do the same run as above on 4 colocated GPUs.

```bash
uv run torchrun --nproc-per-node 4 src/prime_rl/trainer/sft/train.py @ configs/reverse_text/sft.toml
```

## Evals

We provide a convenience endpoint for running a full evaluation suite of common benchmarks such as AIME, MATH-500 or LiveCodeBench against your model using the `eval` entrypoint.

```bash
uv run inference --model.name Qwen/Qwen3-0.6B --max-model-len 2048
```

```bash
uv run eval --model.name Qwen/Qwen3-0.6B --environment-ids math500,aime2024,aime2025
```

To check all available configuration options, run `uv run eval --help`.

## Multi-Node Training

*TBD*

## Developer

*For now, development is only possible on CUDA-enabled devices.*

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

### Persistent Inference Server

For development purposes it is useful start the inference server once and keep it alive across experiments to avoid suffering the vLLM startup time repeatedly. The recommended workflow is as follows:

1. Start the preconfigured tmux session using the tmux script

```bash
bash scripts/tmux.sh
```

2. Start the inference server in the `Inference` pane.

```bash
uv run inference @ configs/reverse_text/infer.toml
```

3. Start the trainer and orchestrator in the `Trainer` pane.

```bash
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml
```

To kill the tmux session when you're done:

```bash
tmux kill-session -t prime-rl
```

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
