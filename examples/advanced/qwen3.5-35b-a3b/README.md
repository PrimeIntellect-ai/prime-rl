# Qwen3.5-35B-A3B

RL on [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) across three domains: math, SWE, and agentic tool use. The model is a hybrid MoE (256 experts, top-8 plus a shared expert) that mixes Gated DeltaNet linear attention with full attention, and one checkpoint covers both thinking and non-thinking modes. Math and SWE train with thinking on (the chat template default); the tool config turns it off with `[orchestrator.renderer] enable_thinking = false`, and the tools run colocated on [Modal](https://modal.com). The checkpoint is a vision-language model; these configs train text-only, so they leave `[model.vlm]` unset. All configs use the custom MoE trainer implementation, the AdamW optimizer, NCCL weight broadcast, and W&B logging; `math.toml` and `swe.toml` add expert parallelism (`ep = 8`). `swe.toml` adds `cp = 2` context parallelism for its 131k context; the other two run at 32k. Context parallelism uses `cp_style = "ulysses"`, the only style the DeltaNet layers support.

| Config | Trains on | Evals on | Runtime | Topology |
|---|---|---|---|---|
| [`math.toml`](math.toml) | `i3_math` | `aime2025` (every 25 steps) | `subprocess` — runs locally, no sandbox | 2 train + 2 infer nodes |
| [`swe.toml`](swe.toml) | `r2e-gym` (`bash` harness) | `swebench-verified` (every 25 steps) | sandbox (Prime Intellect by default) | 2 train + 2 infer nodes |
| [`tool.toml`](tool.toml) | `general-agent` (colocated tools) | — | `modal` | 1 train + 1 infer node |
| [`sft/h200/`](sft/h200) | SFT on `PrimeIntellect/INTELLECT-3-SFT-10K` (math) | — | — | 2 train nodes |

`tool.toml` is the odd one out: 400 steps, `group_size = 16`, checkpoints every 50 steps with `keep_last = 1`, and inference at `dp = 2` / `tp = 4`. It also trains on a single node: FP32 weights, gradients, and AdamW state come to about 16 bytes per parameter, roughly 70 GB per GPU for the ~35B parameters before activations, so plan for 141 GB (H200-class) GPUs. `math.toml` and `swe.toml` run 512-task batches with checkpoints every 100 steps.

## Requirements

- A Slurm cluster with 8-GPU nodes and a shared filesystem. This guide assumes the shared filesystem is mounted at `/shared` — adjust to your own path.
- **Sandboxes** (`swe.toml` only) — the SWE agents run in sandboxes, wired for [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) by default. If you use those, install the `prime` CLI separately and log in:

```bash
uv tool install prime
prime login   # or: prime config set-api-key <your-key>
```

  To run on your own infrastructure instead, swap `env.agent.runtime` on the source for a runtime your environments support (e.g. a local Docker backend).
- **Modal** (`tool.toml` only) — the tool agents run on Modal (`env.agent.runtime.type = "modal"`). Authenticate once:

```bash
uv run modal setup   # or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
```

- Environment variables, exported in the shell you launch from — the launcher passes its environment to every component:

  - `WANDB_API_KEY` — every config logs to W&B (`[monitors.wandb]`).
  - `HF_TOKEN` — optional; the Qwen checkpoint is public, but a token avoids rate limits.

`math.toml` needs none of the above beyond W&B: its agents run as local subprocesses on the env server.

## Setup

Clone prime-rl onto the shared filesystem and install everything, including the environments — they are opt-in uv workspace members, so `--all-packages` is required:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras --all-packages
```

## Tweak before launching

- None of the configs set a `[slurm] partition` — sbatch picks your cluster default. Override with `--slurm.partition <your-partition>`.
- `--output-dir` defaults to `outputs/` next to the checkout; on a cluster, point it at the shared filesystem.
- Validate the full config without submitting a job by appending `--dry-run`: it writes `<run_dir>/launcher/rl.sbatch` plus the resolved per-process configs, and exits.

## Start the run

The `rl` entrypoint submits an sbatch job whenever the config has a `[slurm]` table — there is no separate launcher. From the shared checkout:

```bash
uv run rl @ examples/advanced/qwen3.5-35b-a3b/math.toml \
  --output-dir /shared/outputs/qwen35b \
  --run.name qwen35b-math
```

Swap `math.toml` for `swe.toml` or `tool.toml` to run the other domains. Pass `--run.name`: the run directory is `<output_dir>/<run_name>` and you need a stable name to resume later (unset, it auto-generates as `<envs>--<model>--<short-id>`).

## Monitor with the dashboard

Start the local run dashboard on the head node — it only reads the run directories, so it is safe to point at a live run while the job is training:

```bash
uv run dashboard /shared/outputs/qwen35b   # serves http://localhost:7788
```

If the head node is remote, forward the port from your laptop and open `http://localhost:7788` in a browser:

```bash
ssh -L 7788:localhost:7788 <head-node>
```

Pick the run and you get five views:

- **Metrics** — the W&B-style overview, read from the run's `metrics.jsonl`. Watch `reward/{all,env}/mean` trend upward over steps, and `seq_len/*` + `is_truncated/*` for rollout health.
- **Configs** — the launch TOML next to the merged, resolved per-process configs the run actually started with.
- **Trace** — a per-episode rollout viewer with per-token overlays (advantage, entropy, sampling mismatch, loss/content masks), showing the transcript, a wall-clock timeline of the rollout, a terminal replay of model and tool activity, and a semantic graph of the model-call chain.
- **Logs** — the merged component logs (trainer, orchestrator, inference), also on disk under `<run_dir>/logs/`.
- **Reports** — markdown reports written to `<run_dir>/reports/`, if any tooling produces them.

Pass several output directories to track parallel experiments side by side (`uv run dashboard /shared/outputs/a /shared/outputs/b`); a taken port automatically bumps to the next free one. SLURM stdout/stderr and the generated sbatch script land in `<run_dir>/launcher/`.

## Checkpoint, resume, export

Checkpoints land in `<run_dir>/checkpoints/step_<N>` — every 100 steps for `math.toml`/`swe.toml`, every 50 steps (`keep_last = 1`) for `tool.toml`. To resume, re-run the same command with `--resume` (latest checkpoint) or `--resume.step <N>`, the same `--run.name` / `--output-dir`, and a `--max-steps` at least the target final step:

```bash
uv run rl @ examples/advanced/qwen3.5-35b-a3b/math.toml \
  --output-dir /shared/outputs/qwen35b \
  --run.name qwen35b-math \
  --resume --max-steps 1000
```

Trainer checkpoints are DCP-sharded; export HF-format weights with:

```bash
uv run python tools/convert_dcp_to_bf16.py /shared/outputs/qwen35b/qwen35b-math/checkpoints/step_100
```

See [Training](../../../docs/training.md) for the full knobs and metrics reference, and [Scaling](../../../docs/scaling.md) for SLURM and multi-node details.

## SFT

The SFT configs live under [`sft/h200/`](sft/h200). They are tuned for two 8-GPU H200 nodes and train only (no inference nodes). Combine the base config with a data overlay:

```bash
uv run sft @ examples/advanced/qwen3.5-35b-a3b/sft/h200/base.toml @ examples/advanced/qwen3.5-35b-a3b/sft/h200/math-10k.toml \
  --output-dir /shared/outputs/qwen35b \
  --run.name qwen35b-sft-math
```

This starts an SFT run with the following setup:

- The model is `Qwen/Qwen3.5-35B-A3B`, trained with the custom MoE implementation, expert parallelism (`ep = 8`), full activation checkpointing with offloading, and AdamW.
- The data is `PrimeIntellect/INTELLECT-3-SFT-10K` (math split), at a 32k sequence length with a batch of 16 samples per step.

For a fake-data dry run, add [`fake.toml`](sft/h200/fake.toml) instead of the data overlay. `base.toml` sets `[slurm] partition = "all"` and `HF_HOME = "/home/huggingface"`. Change both to match your cluster. For longer contexts, raise `data.seq_len` and set `model.cp` (with `model.cp_style = "ulysses"`) to split each sequence across GPUs. You can monitor the SFT run with the same dashboard.
