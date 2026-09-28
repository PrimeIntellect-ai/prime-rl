# Hosted SFT

Run `prime-rl` supervised fine-tuning on Prime Intellect's hosted GPU clusters — no cluster credentials, no `kubectl`, no SLURM. You create a storage volume with the `prime` CLI, stage a public Hugging Face dataset onto it, dispatch a `prime-rl` SFT config with `prime train`, and watch the loss curve and logs on the platform dashboard. When the run finishes, teardown is automatic and your checkpoints are on the volume.

This walkthrough trains a small Qwen3 model to reverse text (the [`reverse-text`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/reverse-text) example) end-to-end on a single GPU. Every output shown is real, captured from live runs of this exact flow, so you know what to expect at each step. Two sessions are spliced: the staging transcript is from a live platform-API staging run of this dataset (on a shared staging volume, hence `sft-datasets` in its output), and the training transcript is from a run trained on this same dataset after it was staged on that run's volume. Run IDs, volume names, and timestamps differ per invocation. For SFT on your own infrastructure, see [Training](training.md).

## Prerequisites

- The `prime` CLI, installed and logged in:

```bash
uv tool install prime
prime login
```

  **Version note — this flow is not in a released CLI yet.** `prime volumes`, `prime volumes stage`, and hosted-SFT dispatch ship in the upcoming CLI release that bundles volumes, SFT, and dataset staging ([prime-cli PR #935](https://github.com/PrimeIntellect-ai/prime-cli/pull/935), [prime-cli PR #964](https://github.com/PrimeIntellect-ai/prime-cli/pull/964)). On a released build (v0.7.7 and earlier) `prime volumes stage` does not exist, and dispatching an SFT config misroutes it against the RL schema with confusing field errors. This page assumes that release.

- A Prime account with hosted training access.
- A model **cached on your cluster** — hosted SFT boots models from the cluster model cache, not from the Hub. List what is available:

```bash
prime train models
```

- A **public** Hugging Face dataset. Staging is public-only for now; private and gated datasets are not supported yet.

## 1. Create a volume

A volume is persistent storage on your training cluster. Hosted SFT uses it twice: your staged dataset is mounted read-only at `/datasets` inside the trainer, and run outputs (checkpoints, metrics) are written back under `runs/<runId>/` on the same volume.

```bash
prime volumes create reverse-text-sft-e2e --size 100Gi
```

```text
Creating volume reverse-text-sft-e2e (100Gi).
Use it with: prime train config.toml --volume reverse-text-sft-e2e
```

The volume reaches `RUNNING` about ten seconds after the cluster accepts it:

```bash
prime volumes list
```

```text
Name                  Size   Status   Namespace            Created
reverse-text-sft-e2e  100Gi  RUNNING  prime-user-cmug944…  2026-09-28T20:04:33…
```

`prime volumes resize <name> --size <size>` grows a volume in place, and `prime volumes delete -y <name>` deletes a volume and everything on it.

## 2. Stage the dataset

This is the step that puts your dataset **on the volume** — you never download anything to your laptop, and the trainer never downloads from the Hub. `prime volumes stage` pulls the dataset on the cluster itself, on a CPU-only job, and verifies it before publishing it under `datasets/<name>` on the volume:

```bash
prime volumes stage willcb/R1-reverse-wikipedia-paragraphs-v1-1000 \
  --volume reverse-text-sft-e2e \
  --path reverse-text
```

- `--path` names the directory under `datasets/` on the volume. It defaults to the dataset's repository basename. Here the dataset lands at `/datasets/reverse-text`, which is exactly what `data.name` in the config below points at.
- `--revision` pins a branch, tag, or commit (default `main`); the staging job resolves it to one immutable commit before downloading.
- Re-staging the same dataset at the same revision is a no-op — you get `already_staged` and nothing is overwritten. A different revision at the same path fails instead of silently replacing your data.
- `--kube-context` falls back to running the staging job through your own kubeconfig (operator path); without it, everything goes through the platform API.

The CLI admits the staging job and streams progress until the dataset is verified. The output below is verbatim from a live staging run of this same dataset on a shared staging volume — `sft-datasets` in the output is that session's volume, not the walkthrough's:

```text
Volume sft-datasets · cluster lfxxf6afiwriu03czkmjwa9u · API staging (no kubectl required)
Stage e4effe27-9702-4d9f-9bcb-95cff52bc48e admitted · status PENDING
Status RUNNING (2s elapsed)
Status RUNNING (4s elapsed)
Status RUNNING (7s elapsed)
Status RUNNING (10s elapsed)
```

On success you get the verified inventory — revision, byte count, file count, and the splits/columns the trainer will see:

```json
{
  "status": "staged",
  "source": "willcb/R1-reverse-wikipedia-paragraphs-v1-1000",
  "revision": "4a9f4237b1858fef1d3da428fb47f73de3ae7d67",
  "requestedRevision": "main",
  "volume": "sft-datasets",
  "dataName": "/datasets/reverse-text",
  "bytes": 2706811,
  "files": 3,
  "configs": {
    "default": {
      "splits": {
        "train": {
          "rows": 1000,
          "columns": ["prompt", "completion", "answer", "reward"]
        }
      }
    }
  },
  "elapsedSeconds": 2.7
}
```

Before publishing, the staging job re-loads the dataset in a fresh process with the Hub disabled — if it cannot be loaded offline, staging fails and nothing lands on the volume. A `.prime-stage-manifest` file inside the dataset directory records the source repo, revision, and inventory for later re-runs.

## 3. Write the config

The config is a `prime-rl` SFT config — the same schema as [`examples/basic/reverse-text/sft.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/basic/reverse-text/sft.toml), with four adaptations for the hosted path:

1. **`data.name` points at the staged volume path** (`/datasets/reverse-text`), not a Hub ID.
2. **No `[eval]` or `[inference]` blocks** — hosted SFT is trainer-only; online evals and an inference plane are rejected at dispatch (see [What the errors mean](#what-the-errors-mean)).
3. **No `output_dir`** — the platform injects it and routes outputs to your volume.
4. Keep the first run tiny (`max_steps = 10`, one GPU) so a smoke test costs minutes, not hours.

```toml
# Hosted SFT: reverse-text walkthrough
# Adapted from examples/basic/reverse-text/sft.toml ([eval]/[inference] stripped,
# data.name -> staged volume path, output_dir omitted: the platform injects it).
max_steps = 10        # total optimizer steps — enough to see a healthy loss curve

[ckpt]               # checkpoint at the end of training, saved to the volume

[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"  # must be in the cluster model cache

[data]
name = "/datasets/reverse-text"  # staged path on the volume (step 2) — Hub IDs are rejected
type = "sft"                    # supervised fine-tuning; "fake" runs with synthetic data
seq_len = 2048                  # max sequence length per example
batch_size = 8                  # sequences per optimizer step

[optim]
lr = 2e-5                       # constant learning rate for this smoke run

[deployment]
type = "single_node"            # one trainer node
num_train_gpus = 1              # training GPUs
gpus_per_node = 1               # GPUs per node

[renderer]
name = "prime-qwen3"            # chat-template renderer for the Qwen3 family
```

`[data]` accepts the same fields as local SFT — `splits = ["train"]` restricts training to specific splits of a multi-split dataset.

## 4. Dispatch

```bash
prime train sft.toml --volume reverse-text-sft-e2e
```

```text
Creating Hosted Training run...
Dispatched hosted run kqeggj5dl7k85bc0mk1i4y5i

Monitor run at:
  http://localhost:3000/dashboard/training/kqeggj5dl7k85bc0mk1i4y5i
```

The CLI prints a link straight to the run page in the dashboard (this walkthrough ran against a local stack, hence `localhost` — on the hosted platform the same run page is served at `https://app.primeintellect.ai/dashboard/training/<runId>`). The run moves through `PENDING` → `CREATING` → `RUNNING` → `COMPLETED`; this one took about 3 minutes wall-clock on one H200.

If your cluster runs an unpublished runtime image, pin it explicitly with `--image-tag <tag>` (this walkthrough pinned the `prime-rl` image `commit-6f4ab3b73`). Do not set `HF_HOME` or other Hugging Face cache variables in the config — the platform owns the HF cache inside the run.

## 5. Watch it run

The dashboard page for the run shows the status timeline, the per-step loss curve, and the live trainer logs. The run below dispatched this walkthrough's config against its own volume, with the same dataset staged under its repository basename (`/datasets/willcb-r1-reverse-wikipedia-paragraphs-v1-1000` — in your run, the staged path is whatever `--path` you chose in step 2). The same data is available from the CLI:

```bash
prime train get kqeggj5dl7k85bc0mk1i4y5i
```

```text
Run kqeggj5dl7k85bc0mk1i4y5i

  Status: COMPLETED
  Model: PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT
  Environments:
  Max Steps: 10
  Batch Size: 8
  Rollouts per Example: 1
  Created: 2026-09-28 20:33
  Started: 2026-09-28 20:34
  Completed: 2026-09-28 20:37
```

Per-step metrics (the loss curve the dashboard renders):

```bash
prime train metrics kqeggj5dl7k85bc0mk1i4y5i
```

```text
step: [1, 2, 3, 4, 5, 6, 7, 8, 9]
loss/mean:     [1.108, 1.1788, 0.9996, 0.9468, 1.1179, 1.044, 1.0353, 1.0038, 0.9944]
loss/perplexity: [3.028, 3.251, 2.717, 2.577, 3.058, 2.841, 2.816, 2.729, 2.703]
```

Ten steps with a constant learning rate on a 0.6B model will not descend cleanly — the point is that the curve is visible, healthy, and NaN-free.

The pretty trainer log:

```bash
prime train logs kqeggj5dl7k85bc0mk1i4y5i
```

```text
20:35:14 WARNING No checkpoints found in
/data/outputs/fft-kqeggj5dl7k85bc0mk1i4y5i/checkpoints. Starting from scratch.
20:35:34 [WARNING] Model uses tied word embeddings, so skipping the last-layer
no-reshard optimization.
20:35:38 [INFO] Initializing scheduler with 10 steps (type='constant')
Generating train split: 100%|████████████████████| 1000/1000 [00:00<00:00, 23213.48
examples/s]
20:35:47 [SUCCESS] Step 1 |    8.5s | Loss 1.1080 | Grad. Norm 12.9495 | LR
2.00e-05 | Throughput 0 tokens/s | MFU 0.0% | Peak Mem. 10.7/139.8 GiB (7.7%)
20:35:52 [SUCCESS] Step 3 |    2.4s | Loss 0.9996 | Grad. Norm 7.4464 | LR
2.00e-05 | Throughput 6704 tokens/s | MFU 2.9% | Peak Mem. 10.7/139.8 GiB (7.7%)
```

The `Generating train split: 100%|…| 1000/1000` line is the proof the trainer read your **staged** dataset from the volume (1000 examples = the staged parquet) — there is no Hub download during training.

When the run completes, teardown is automatic: trainer pods are removed and the GPUs are released immediately. Your outputs stay on the volume:

```text
/volume/runs/kqeggj5dl7k85bc0mk1i4y5i/outputs/fft-kqeggj5dl7k85bc0mk1i4y5i/checkpoints/step_10/trainer
/volume/runs/kqeggj5dl7k85bc0mk1i4y5i/outputs/fft-kqeggj5dl7k85bc0mk1i4y5i/monitors/file/metrics.jsonl
```

The staged `datasets/` tree is read-only for the run and is left untouched. With `[ckpt]` set, the final checkpoint is at `runs/<runId>/outputs/fft-<runId>/checkpoints/step_<max_steps>/trainer` on the volume — `prime train checkpoints` does not list SFT checkpoints yet, so use the volume path.

## What the errors mean

The dispatch-time rejections are guardrails, not failures of your setup. Each one tells you exactly what to change:

**`[eval]` / `[inference]` blocks in the config.** Hosted SFT is trainer-only. If you dispatch the stock public SFT example unchanged, it fails fast before any GPUs are allocated:

```text
Error: hosted SFT runs are trainer-only — [eval], [inference] (online evals) are
not supported on the dedicated path. Remove them, or run locally with prime-rl's
SFT launcher.
```

Remove those blocks for hosted runs, or use `uv run sft` locally ([Training](training.md)).

**Model not in the cluster cache.** Hosted SFT boots from the cluster model cache; a Hub ID the cluster has not cached is rejected before dispatch:

```text
Error: HTTP 400: Model 'PrimeIntellect/Qwen3-0.6B' is not available on this
cluster. Reach out to Prime support to get it added.
```

Check `prime train models` first and pick a cached model, or ask Prime support to cache the one you need.

**Hub ID in `data.name`.** Hosted SFT does not download datasets during training — data must be staged on a volume first. A Hub ID (with or without `--volume`) gets the same actionable 400:

```text
Error: HTTP 400: data.name: Hosted SFT requires staged local data: run `prime
volumes create <volume>`, CPU-stage an HF-loadable dataset directory under
`datasets/<name>` on that volume, set `data.name = "/datasets/<name>"`, and
launch `prime train sft.toml --volume <volume>`. Hub IDs and remote URLs are not
supported for hosted SFT.
```

The recipe is steps 1–4 of this guide. `data.type = "fake"` without `--volume` is the one no-volume path: it trains on synthetic data and is a fine connectivity smoke test, just not real data.

## Limitations

- **Not in a released CLI yet.** Volumes, staging, and hosted-SFT dispatch ship together in the upcoming CLI release — see the version note in [Prerequisites](#prerequisites). A build without SFT support validates the config against the RL schema and rejects it with confusing field errors; if `prime volumes stage` does not exist, upgrade the CLI.
- **Public datasets only.** `prime volumes stage` accepts public HF dataset repositories; private and gated datasets (HF tokens) are not supported yet.
- **Cluster-cached models only.** You cannot add models to the cache yourself — `prime train models` lists what is available, and Prime support can add others.
- **Trainer-only.** `[eval]` and `[inference]` blocks are rejected; online evals during hosted SFT are not available.
- **SFT checkpoints are on the volume but not yet listed** by `prime train checkpoints`.
- **Usage accounting shows 0 tokens / $0.00 for SFT** — token accounting is an RL-rollout concept, and SFT runs report zero. The `prime train components` view labels the single trainer row `orchestrator`; that label is cosmetic.

## Scaling up

The same flow — create volume, stage dataset, adapt config, dispatch with `--volume` — carries to multi-node runs. The config changes only: switch `[deployment]` to `multi_node`, add node/GPU counts, and add the model parallelism knobs your model needs. A real GLM-5.3 run through this flow (`zai-org/GLM-5.3-BF16`, 8 nodes × 8 H200, staged `INTELLECT-3-SFT-10K` math split) completed 4 steps in about 5.5 minutes wall-clock:

```toml
max_steps = 4

[model]
name = "zai-org/GLM-5.3-BF16"
impl = "custom"          # PrimeRL custom MoE implementation
cp = 8                   # context parallelism
ep = 8                   # expert parallelism
optim_cpu_offload = true

[optim]
type = "sign_sgd"
lr = 5e-5
weight_decay = 0.01

[renderer]
name = "glm-5.3"

[deployment]
type = "multi_node"
num_train_nodes = 8
gpus_per_node = 8

[data]
type = "sft"
name = "/datasets/intellect-3-sft-10k"
splits = ["math"]
batch_size = 8
micro_batch_size = 1
seq_len = 4096
```

See [Scaling](scaling.md) for the parallelism knobs and [Algorithms](algorithms.md) for the model config details.
