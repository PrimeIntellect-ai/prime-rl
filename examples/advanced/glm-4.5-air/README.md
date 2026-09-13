# GLM-4.5-Air

RL on [`zai-org/GLM-4.5-Air`](https://huggingface.co/zai-org/GLM-4.5-Air) — a 100B MoE — at 131k context, across three agentic domains: web search, SWE, and terminal. Rollouts run in sandboxes ([Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) by default — see [Requirements](#requirements) for using your own), driven either by the `rlm` agent harness (with its `search` skill, or ipython-only) or a plain `bash` harness. All three configs share the same recipe: GRPO with a linear length penalty on input tokens (and turns, for SWE/terminal), the custom MoE trainer implementation with `cp = 4` (ulysses) context parallelism, router replay, NCCL weight broadcast, checkpoints every 50 steps (`keep_last = 1`), and evals every 20 steps (including step 0).

| Config | Trains on | Evals on | Topology |
|---|---|---|---|
| [`search.toml`](search.toml) | `openseeker` + `redsearcher` (search QA, `rlm` harness + `search` skill) | `browsecomp` (500 examples) | 2 train + 4 infer nodes |
| [`swe.toml`](swe.toml) | `scaleswe` (`bash` harness) | `swebench-verified` | 1 train + 3 infer nodes |
| [`terminal.toml`](terminal.toml) | `tmax` (per-task Docker image, `rlm` ipython harness) | `swebench-verified` + `terminal-bench-2` (avg@4) | 2 train + 4 infer nodes |

Inference serves the bf16 checkpoint as-is (no quantization), with `tensor_parallel_size = 8`. `search.toml` and `terminal.toml` train with Muon on 2 nodes; `swe.toml` fits the trainer on a single node with full optimizer offload and the stateless sign-SGD optimizer. Search answers are scored by a reference judge (`Qwen/Qwen3-235B-A22B-Instruct-2507`).

## Requirements

- A Slurm cluster with 8-GPU nodes and a shared filesystem. This guide assumes the shared filesystem is mounted at `/shared` — adjust to your own path.
- **Sandboxes.** Rollout and eval agents run in sandboxes. These configs are wired for [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) by default — if you use those, log the `prime` CLI in (it ships with prime-rl's dependencies), and each config's `slurm.pre_run_command` will clean up the run's orphaned sandboxes by label before launching:

```bash
uv run prime login   # or: uv run prime config set-api-key <your-key>
```

  To run on your own infrastructure instead, swap `env.agent.runtime` on each source for a runtime your environments support (e.g. a local Docker backend) and drop the `slurm.pre_run_command` cleanup line.

- Environment variables, exported in the shell you launch from — the launcher passes its environment to every component:

  - `HF_TOKEN` — `zai-org/GLM-4.5-Air` is a gated model.
  - `WANDB_API_KEY` — every config logs to W&B (`[monitors.wandb]`).
  - `SERPER_API_KEY` — search envs only; forwarded into the agent sandbox for the `search` skill (`harness.forward_env`).

## Setup

Clone prime-rl onto the shared filesystem and install everything, including the environments — they are opt-in uv workspace members, so `--all-packages` is required:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras --all-packages
```

## Tweak before launching

- `[slurm] partition` is set to `all` — override with `--slurm.partition <your-partition>`, or edit the config.
- `--output-dir` defaults to `outputs/` next to the checkout; on a cluster, point it at the shared filesystem.
- Validate the full config without submitting a job by appending `--dry-run`: it writes `<run_dir>/launcher/rl.sbatch` plus the resolved per-process configs, and exits.

## Start the run

The `rl` entrypoint submits an sbatch job whenever the config has a `[slurm]` table — there is no separate launcher. From the shared checkout:

```bash
uv run rl @ examples/advanced/glm-4.5-air/swe.toml \
  --output-dir /shared/outputs/glm45air \
  --run.name glm45air-swe
```

Swap `swe.toml` for `search.toml` or `terminal.toml` to run the other domains. Pass `--run.name`: the run directory is `<output_dir>/<run_name>` and you need a stable name to resume later (unset, it auto-generates as `<envs>--<model>--<short-id>`).

## Monitor with the dashboard

Start the local run dashboard on the head node — it only reads the run directories, so it is safe to point at a live run while the job is training:

```bash
uv run dashboard /shared/outputs/glm45air   # serves http://localhost:7788
```

If the head node is remote, forward the port from your laptop and open `http://localhost:7788` in a browser:

```bash
ssh -L 7788:localhost:7788 <head-node>
```

Pick the run (`<run_name>` from the launch) and you get five views:

- **Metrics** — the W&B-style overview, read from the run's `metrics.jsonl`. Watch `reward/{all,env}/mean` trend upward over steps, and `seq_len/*` + `is_truncated/*` for rollout health.
- **Configs** — the launch TOML next to the merged, resolved per-process configs the run actually started with.
- **Trace** — a per-episode rollout viewer with per-token overlays (advantage, entropy, sampling mismatch, loss/content masks), showing the transcript, a wall-clock timeline of the rollout, a terminal replay of model and tool activity, and a semantic graph of the model-call chain.
- **Logs** — the merged component logs (trainer, orchestrator, inference), also on disk under `<run_dir>/logs/`.
- **Reports** — markdown reports written to `<run_dir>/reports/`, if any tooling produces them.

Pass several output directories to track parallel experiments side by side (`uv run dashboard /shared/outputs/a /shared/outputs/b`); a taken port automatically bumps to the next free one. SLURM stdout/stderr and the generated sbatch script land in `<run_dir>/launcher/`.

## Checkpoint, resume, export

Checkpoints land in `<run_dir>/checkpoints/step_<N>` every 50 steps (`keep_last = 1`). To resume, re-run the same command with `--resume` (latest checkpoint) or `--resume.step <N>`, the same `--run.name` / `--output-dir`, and a `--max-steps` at least the target final step:

```bash
uv run rl @ examples/advanced/glm-4.5-air/swe.toml \
  --output-dir /shared/outputs/glm45air \
  --run.name glm45air-swe \
  --resume --max-steps 1000
```

Trainer checkpoints are DCP-sharded; export HF-format weights with:

```bash
uv run python tools/convert_dcp_to_bf16.py /shared/outputs/glm45air/glm45air-swe/checkpoints/step_100
```

See [Training](../../../docs/training.md) for the full knobs and metrics reference, and [Scaling](../../../docs/scaling.md) for SLURM and multi-node details.
