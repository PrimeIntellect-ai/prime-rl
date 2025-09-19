# Usage

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