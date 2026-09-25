# GLM-5 family at scale

Large-scale RL and serving for the GLM-5 family — `zai-org/GLM-5.3-BF16` for training and serving alike, quantized to per-block FP8 online on the vLLM side — at 131k context: 16 trainer nodes on the custom MoE implementation (expert + context parallelism) and P/D-disaggregated FP8 inference. The [`swe-llmd.toml`](swe-llmd.toml) append fronts the same inference plane with the [**llm-d**](https://llm-d.ai) router (Endpoint Picker + Envoy): its `active-request-scorer` load-balances in flight — instead of reacting to delayed metrics like the default `vllm-router` — which, combined with prefix-cache affinity for grouped rollouts, keeps prefill and decode ranks evenly loaded under RL's bursty request pattern. It also offloads KV to a Mooncake distributed CPU pool (1TB per node by default).

| Config | What it runs | Topology |
|---|---|---|
| [`swe.toml`](swe.toml) | Base GLM-5.3 RL: trains on `r2e-gym` (`rlm` harness), FP8-quantized trainer (`deepgemm_fp8` MoE compute), serves `GLM-5.3-BF16` with online per-block FP8 quantization behind the default `vllm-router`. | 16 train + 16 infer nodes |
| [`swe-llmd.toml`](swe-llmd.toml) | Append for `swe.toml`: swaps in the llm-d router (EPP scorer tuning) + Mooncake KV offload — launch as `rl @ swe.toml @ swe-llmd.toml`. | same |
| [`infer/pd.toml`](infer/pd.toml) | Inference-only pre-flight: P/D-disaggregated `GLM-5-FP8`. | 6 infer nodes |
| [`infer/pd-llmd.toml`](infer/pd-llmd.toml) | Inference-only pre-flight: `GLM-5.2-FP8` with llm-d + Mooncake. | 16 infer nodes |

## Requirements

- A Slurm cluster with 8-GPU nodes, a shared filesystem, and at least **32 nodes** (16 trainer + 16 inference) for the RL configs. This guide assumes the shared filesystem is mounted at `/shared` — adjust to your own path. If you have fewer nodes, drop `seq_len`, lower `num_train_nodes`, and reduce `cp` accordingly.
- InfiniBand/RDMA NICs for the Mooncake KV pool (llm-d append).
- **Sandboxes.** Rollout and eval agents run in sandboxes, wired for [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) by default. If you use those, install the `prime` CLI separately and log in:

```bash
uv tool install prime
prime login   # or: prime config set-api-key <your-key>
```

  To run on your own infrastructure instead, swap `env.agent.runtime` on each source for a runtime your environments support (e.g. a local Docker backend).

- Environment variables, exported in the shell you launch from — the launcher passes its environment to every component:

  - `HF_TOKEN` — the GLM-5 family checkpoints are gated models.
  - `WANDB_API_KEY` — both RL variants log to W&B (`[monitors.wandb]`).

### Install llm-d (llm-d append only)

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

- `swe.toml` marks two `# FILL IN` values: `output_dir` (point it at your shared filesystem) and `[slurm] partition`.
- `swe-llmd.toml` `[inference.kv_cache_offload] device_name` — the RDMA NIC list for Mooncake. Auto-detection is unreliable; set it by hand from `nvidia-smi topo -m` on your nodes.
- `swe-llmd.toml` `[inference.kv_cache_offload.cpu] num_bytes` — 1TB per node by default; lower it if your nodes have less RAM.

## Start the run

The `rl` entrypoint submits an sbatch job whenever the config has a `[slurm]` table — there is no separate launcher. From the shared checkout:

```bash
# GLM-5.3 RL behind the default vllm-router
uv run rl @ examples/advanced/glm-5.3/swe.toml

# with the llm-d router + Mooncake KV offload appended
uv run rl @ examples/advanced/glm-5.3/swe.toml @ examples/advanced/glm-5.3/swe-llmd.toml
```

The inference configs are standalone pre-flights: they serve the FP8 checkpoint through the same entrypoint the trainer uses (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster` included — never call `vllm serve` directly), and are a fast way to check that this cluster can serve the model at all before committing it to a run:

```bash
uv run inference @ examples/advanced/glm-5.3/infer/pd.toml
uv run inference @ examples/advanced/glm-5.3/infer/pd-llmd.toml
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

Both RL configs ship without `[ckpt]` — add a `[ckpt]` overlay (e.g. `interval = 100`) for periodic checkpoints. To resume, re-run the same command with `--resume` (latest checkpoint) or `--resume.step <N>`, a stable `--run.name`, and a `--max-steps` at least the target final step. Trainer checkpoints are DCP-sharded; export HF-format weights with `tools/convert_dcp_to_bf16.py`. See [Training](../../../docs/training.md) for the full resume and export reference.

See [Scaling](../../../docs/scaling.md) for SLURM details and [Inference](../../../docs/inference.md) for the disaggregated-inference and router reference.


# SFT

You can run SFT with the following command:

```bash
uv run sft @ examples/advanced/glm-5.3/sft.toml @ examples/advanced/glm-5.3/sft-math-10k.toml
```

This will start a SFT run with the following configuration:

- The model is `zai-org/GLM-5.3-BF16`
- The data is `PrimeIntellect/INTELLECT-3-SFT-10K`

you can use the same dashboard to monitor the SFT run.