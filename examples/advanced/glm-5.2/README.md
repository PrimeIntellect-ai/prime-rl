# GLM-5 family at scale

Large-scale RL and serving for the GLM-5 family — `zai-org/GLM-5`, `GLM-5.1`, and `GLM-5.2-FP8` — at 131k context: 16 trainer nodes on the custom MoE implementation (expert parallelism; the llm-d variant adds context parallelism) and P/D-disaggregated FP8 inference. The `-llmd` variants front the inference plane with the [**llm-d**](https://llm-d.ai) router (Endpoint Picker + Envoy): its `active-request-scorer` load-balances in flight — instead of reacting to delayed metrics like the default `vllm-router` — which, combined with prefix-cache affinity for grouped rollouts, keeps prefill and decode ranks evenly loaded under RL's bursty request pattern. They also offload KV to a Mooncake distributed CPU pool (1TB per node by default).

| Config | What it runs | Topology |
|---|---|---|
| [`swe.toml`](swe.toml) | GLM-5 RL plane: 16 train nodes, eval on `swebench-verified` every 20 steps, default vLLM router over 2 P/D replicas (4 prefill + 4 decode nodes each). Declares **no train source** — compose yours on top with an `[[orchestrator.train.source]]` overlay. | 16 train + 16 infer nodes |
| [`swe-llmd.toml`](swe-llmd.toml) | Full-stack GLM-5.1 RL: trains on `r2e-gym` (`rlm` harness), FP8-quantized trainer (`deepgemm_fp8` MoE compute), llm-d router + Mooncake KV offload. | 16 train + 16 infer nodes |
| [`infer/pd.toml`](infer/pd.toml) | Inference-only pre-flight: P/D-disaggregated `GLM-5-FP8`. | 6 infer nodes |
| [`infer/pd-llmd.toml`](infer/pd-llmd.toml) | Inference-only pre-flight: `GLM-5.2-FP8` with llm-d + Mooncake. | 16 infer nodes |

## Requirements

- A Slurm cluster with 8-GPU nodes, a shared filesystem, and at least **32 nodes** (16 trainer + 16 inference) for the RL configs. This guide assumes the shared filesystem is mounted at `/shared` — adjust to your own path. If you have fewer nodes, drop `seq_len`, lower `num_train_nodes`, and reduce `cp` (llm-d variant) accordingly.
- InfiniBand/RDMA NICs for the Mooncake KV pool (llm-d variants).
- **Sandboxes.** Rollout and eval agents run in sandboxes, wired for [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) by default. If you use those, log the `prime` CLI in — it ships with prime-rl's dependencies:

```bash
uv run prime login   # or: uv run prime config set-api-key <your-key>
```

  To run on your own infrastructure instead, swap `env.agent.runtime` on each source for a runtime your environments support (e.g. a local Docker backend).

- Environment variables, exported in the shell you launch from — the launcher passes its environment to every component:

  - `HF_TOKEN` — the GLM-5 family checkpoints are gated models.
  - `WANDB_API_KEY` — both RL configs log to W&B (`[monitors.wandb]`).

### Install llm-d (llmd variants only)

The llm-d router ships as vendored binaries (`epp`, `envoy`, `pd-sidecar`). Build them once into `third_party/llmd/bin`:

```bash
bash scripts/install_llmd.sh
```

## Setup

Clone prime-rl onto the shared filesystem and install everything, including the environments — they are opt-in uv workspace members, so `--all-packages` is required:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras --all-packages
```

## Tweak before launching

- `swe-llmd.toml` marks two `# FILL IN` values: `output_dir` (point it at your shared filesystem) and `[slurm] partition`. `swe.toml` needs `--slurm.partition <your-partition>` or `--output-dir` overrides on the command line.
- `[inference.kv_cache_offload] device_name` — the RDMA NIC list for Mooncake. Auto-detection is unreliable; set it by hand from `nvidia-smi topo -m` on your nodes.
- `[inference.kv_cache_offload.cpu] num_bytes` — 1TB per node by default; lower it if your nodes have less RAM.
- `swe-llmd.toml` ships without `[ckpt]` — add a checkpoint overlay if you want resume support.

## Start the run

The `rl` entrypoint submits an sbatch job whenever the config has a `[slurm]` table — there is no separate launcher. From the shared checkout:

```bash
# GLM-5.1 RL with llm-d + Mooncake (recommended)
uv run rl @ examples/advanced/glm-5.2/swe-llmd.toml

# GLM-5 base plane — compose your train source onto it
uv run rl @ examples/advanced/glm-5.2/swe.toml @ my-train-source.toml
```

The inference configs are standalone pre-flights: they serve the FP8 checkpoint through the same entrypoint the trainer uses (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster` included — never call `vllm serve` directly), and are a fast way to check that this cluster can serve the model at all before committing it to a run:

```bash
uv run inference @ examples/advanced/glm-5.2/infer/pd.toml
uv run inference @ examples/advanced/glm-5.2/infer/pd-llmd.toml
```

## Monitor with the dashboard

Start the local run dashboard on the head node — it only reads the run directories, so it is safe to point at a live run while the job is training:

```bash
uv run dashboard /shared/outputs   # serves http://localhost:7788
```

If the head node is remote, forward the port from your laptop and open `http://localhost:7788` in a browser:

```bash
ssh -L 7788:localhost:7788 <head-node>
```

Pick the run and you get five views:

- **Metrics** — the W&B-style overview, read from the run's `metrics.jsonl`. Watch `reward/{all,env}/mean` trend upward over steps, and `seq_len/*` + `is_truncated/*` for rollout health.
- **Configs** — the launch TOML next to the merged, resolved per-process configs the run actually started with.
- **Trace** — a per-episode rollout viewer with per-token overlays (advantage, entropy, sampling mismatch, loss/content masks), showing the transcript, a wall-clock timeline, a terminal replay of model and tool activity, and a semantic graph of the model-call chain.
- **Logs** — the merged component logs (trainer, orchestrator, inference), also on disk under `<run_dir>/logs/`.
- **Reports** — markdown reports written to `<run_dir>/reports/`, if any tooling produces them.

Pass several output directories to track parallel experiments side by side (`uv run dashboard /shared/outputs/a /shared/outputs/b`); a taken port automatically bumps to the next free one. SLURM stdout/stderr and the generated sbatch script land in `<run_dir>/launcher/`.

## Checkpoint and resume

`swe.toml` checkpoints every 100 steps (`[ckpt] interval = 100`). To resume, re-run the same command with `--resume` (latest checkpoint) or `--resume.step <N>`, a stable `--run.name`, and a `--max-steps` at least the target final step. Trainer checkpoints are DCP-sharded; export HF-format weights with `tools/convert_dcp_to_bf16.py`. See [Training](../../../docs/training.md) for the full resume and export reference.

See [Scaling](../../../docs/scaling.md) for SLURM details and [Inference](../../../docs/inference.md) for the disaggregated-inference and router reference.
