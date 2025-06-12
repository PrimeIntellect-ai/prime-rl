# prime-rl - decentralized RL training at scale

prime-rl is a codebase for decentralized RL training at scale.



## install
quick install

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/install.sh | bash
```


## Dev


1. Clone: 

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Set up the environment (will default to Python 3.11)

```bash
uv sync && uv sync --extra fa
```

You can check that `flash_attn` is installed correctly by running `uv run python -c "import flash_attn"` and ensure no error is thrown.

4. Precommit install

```bash
uv run pre-commit install
```

6. debug run 

training

```bash
uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/debug.toml
```

inference
```bash
uv run python src/zeroband/infer.py @ configs/inference/debug.toml
```


## Simple Math Run

This debug run trains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on the `justus27/math-hendrycks-genesys-format` dataset using separate inference and training processes.
Depending on the number of available GPUs, we have to adjust the number of generated samples on the inference workers to match the batch size of the training process.

Training samples per step: `batch_size * step_per_rollout`
Inference samples per step: `batch_size * dp`

If you have 2 GPUs, run the following commands:

```bash
# Start inference worker
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/simple_math.toml --parallel.dp 1 --max-batch-size 512
```

```bash
# Start trainer
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=1
uv  run torchrun src/zeroband/train.py @ configs/training/simple_math.toml
```

If you have 4 GPUs, run the following commands:

```bash
# Start inference workers
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/simple_math.toml --parallel.dp 2 --max-batch-size 256
```

```bash
# Start trainer
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=2
uv  run torchrun src/zeroband/train.py @ configs/training/simple_math.toml
```

If you have 8 GPUs, run the following commands:

```bash
# Start inference workers
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/simple_math.toml
```

```bash
# Start trainer
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/simple_math.toml --data.num_workers 2
```


## 2k seq length run

on two different terminal do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/deepscaler.toml
```

then start the trainer

```bash
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/deepscaler.toml
```

if running on h100 node instead of H200 you should add ` --train.micro_bs 4`

## Distributed inference

Inference supports running in multi-node multi-GPU setups supporting DP, TP and PP, and sensible combinations of these.
Below are examples of how to run inference for different parallelization strategies.

Single Node (DP=1, TP=1, PP=1, *requires 1 GPU*)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/zeroband/infer.py @ configs/inference/debug.toml
```

Only TP (TP=2, PP=1, DP=1, *requires 2 GPUs*)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --parallel.tp 2
```

Only DP (DP=2, TP=1, PP=1, *requires 2 GPUs*)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --parallel.dp 2
```

Only PP (DP=1, TP=1, PP=2, *requires 2 GPUs*)

```bash
# Node 1
CUDA_VISIBLE_DEVICES=0 uv run python src/zeroband/infer.py @ configs/inference/debug.toml \
	--parallel.pp.rank 0 \
	--parallel.pp.world-size 2 \
	--seed 69
```

```bash
# Node 2
CUDA_VISIBLE_DEVICES=1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml \
	--parallel.pp.rank 1 \
	--parallel.pp.world-size 2 \
	--seed 69
```

*Note: Setting the seed here is important to ensure model shards work on the same data shards.*

DP+TP (DP=2, TP=2, PP=1, *requires 4 GPUs*)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --parallel.dp 2 --parallel.tp auto
```

PP+TP (DP=1, TP=2, PP=2, *requires 4 GPUs*)

```bash
# Node 1
CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml \
	--parallel.tp auto \
	--parallel.pp.rank 0 \
	--parallel.pp.world-size 2 \
	--seed 69
```

```bash
# Node 2
CUDA_VISIBLE_DEVICES=2,3 uv run python src/zeroband/infer.py @ configs/inference/debug.toml \
	--parallel.tp auto \
	--parallel.pp.rank 1 \
	--parallel.pp.world-size 2 \
	--seed 69
```

*Note: To check the logs of `prime-iroh` (used for connecting PP nodes), you can add the `RUST_LOG=prime_iroh=info` environment variable.*

We don't support DP+PP and so that configuration will raise an exception.

## Tests

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

## Configs

We use `pydantic-settings` to configure `prime-rl`. To get an overview of the available configurations, run the following command:

```bash
uv run python src/zeroband/train.py --help
```

```bash
uv run python src/zeroband/infer.py --help
```

### Sources

We support the following sources for configuration, in this order of precedence:

1. **Command-line arguments**: You can pass (nested) arguments as `--key.subkey value` to the script. For example, to set the model name you can run `--model.name`

2. **Config files**: You can pass `.toml` config files (defined in the `configs` directory) using the `@` prefix. For example, to use the `debug.toml` config file, you can run `uv run python src/zeroband/infer.py @ configs/inference/debug.toml`. (*If you leave a space between the `@` and the config file, you will get shell path auto-completions.*)

3. **Environment variables**: You can set environment variables to override the config values. All environment variables must be prefixed with `PRIME_` and use the `__` delimiter to nest the keys. For example, to set the model name you can run `export PRIME_MODEL__NAME=Qwen/Qwen3-0.6B`.

4. **Defaults**: For almost all config arguments, we have a default value which will be used if no other source is provided.

In general we recommend setting configurations via config files to define reproducible experiments and use command-line arguments to override the config values to run variants of the same experiment. Environment variables are usually only used in production settings to communicate with the [Prime Protocol](https://github.com/PrimeIntellect-ai/protocol) worker. In most cases, you should not need to use environment variables.

The precendence order will be important if multiple sources try to configure the same argument. For example, in the following command, all sources will define a model name

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
PRIME_MODEL__NAME=Qwen/Qwen3-4B uv run src/zeroband/infer.py @qwen8b.toml @qwen14b.toml --model.name Qwen/Qwen3-32B
```

In this example, the CLI argument `--model.name Qwen/Qwen3-32B` will take precendence and the script will use `Qwen/Qwen3-32B` as the model name. If the CLI argument wasn't set, then the second config file would take precedence and the script would use `Qwen/Qwen-14B` as the model name. If the second config file wasn't set, then the first config file would take precedence and the script would use `Qwen/Qwen3-8B` as the model name. Finally, if the first config file wasn't set, then the environment variable would take precedence and the script would use `Qwen/Qwen-4B` as the model name. If the environment variable wasn't set, then the default value would be used and the script would use `Qwen/Qwen3-0.6B` as the model name.


## Citation

If you find `prime-rl` useful, feel free to cite our work:

```bash
@misc{primeintellectteam2025intellect2reasoningmodeltrained,
      title={INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning}, 
      author={Prime Intellect Team and Sami Jaghouar and Justus Mattern and Jack Min Ong and Jannik Straube and Manveer Basra and Aaron Pazdera and Kushal Thaman and Matthew Di Ferrante and Felix Gabriel and Fares Obeid and Kemal Erdem and Michael Keiblinger and Johannes Hagemann},
      year={2025},
      eprint={2505.07291},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.07291}, 
}
```
