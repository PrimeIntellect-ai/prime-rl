# Plan: repeat the H200 MoE throughput runs on one 8x B300 node

Goal: rerun on B300 the exact trainer-only proxy runs measured on 8x H200, for DeepSeek V4 Flash (DSv4), GLM-4.5-Air and GLM-5.3 (DeepSeek sparse attention, `glm_moe_dsa`), and report the same metrics side by side.

## Execution scope for this B300 pass (supersedes "Run matrix" below)

The question: does the roofline prediction that B300's MFU is about 0.83x H200's for DSv4 with the same software hold, and do model-specific roofline predictions hold for GLM-4.5-Air and GLM-5.3? Measured B300/H200 MFU ratios are compared with roofline predictions per model. Comparing at the same layer count and config as an H200 run is enough for the ratio; no layer-count extrapolation is needed in this pass. Do the steps in order and report after each.

### Allocation

Use Slurm job 3183 (`garrett-b300-repro`, 1 node, exclusive, 3 h), queued on 2026-10-30. It waits behind another job that will be cancelled when its work is done; check with `squeue --me` and start GPU work only once 3183 is `RUNNING`. Wait with a loop on `squeue -j 3183 -h -o %T` reaching `RUNNING`, not on process-name matching. Use only that job (`srun --jobid=3183 --overlap ...`, which `run_arm.sh` does when given 3183). Never allocate more nodes, never use sudo or Slurm reservations, never touch other users' jobs. Steps 1 and the report need no GPU and can start right away; never run pytest, model builds or heavy compute on the login node.

### Environment

- Worktree: this branch's worktree at `~/github/PrimeIntellect-ai/prime-rl-exp-dsv4-b300-repro` (already set up: submodules as recorded, venv from `uv sync --all-extras --all-packages`, `--check` clean). If you pull new commits, rerun `git submodule update --init --recursive` and `uv sync --all-extras --all-packages`. Always `uv run --no-sync`.
- Shared HF cache: `run_arm.sh` does not set it, and the default `~/.cache/huggingface` is wrong on this cluster. In the shell that launches `run_arm.sh`, `export HF_HUB_CACHE=/home/huggingface/hub HF_XET_CACHE=/home/huggingface/xet HF_HUB_OFFLINE=1` (`srun` inherits them). All three checkpoints are cached there and already converted (`<snapshot>/prime/.prime-v1` exists): `PrimeIntellect/DeepSeek-V4-Flash-0731-bf16`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-5.3-BF16`.
- `CUDA_HOME=/usr/local/cuda-13.0` in `run_arm.sh` is correct here.
- Before the first launch of each config, dump the fully resolved trainer config (`prime_rl.configs.trainer.TrainerConfig` via `prime_rl.utils.config.cli` with the same `@ proxy.toml` and overrides) and check that it differs from `proxy.toml` only as intended: no activation or optimizer offloading, `force_balanced_routing = true`, and the precision fields you set. Also import `prime_rl.trainer.rl.train` once on the node to catch import errors before a 10 minute compile.
- Blockwise FP8 should be accepted on B300: the gate in `src/prime_rl/trainer/model.py` (~line 778) allows compute capability major 9 or 10, and B300 is SM103. If it still refuses at runtime, report MXFP8 only and say so.

### Step 1: roofline predictions (no GPU)

Tool: `~/.claude/skills/profiling/scripts/roofline.py` (read its module docstring; run with `uv run --script`). Existing B300 ops files: `~/tmp/profiling/dsv4-perf-proxy/inputs/{dsv4_flash,glm53,glm45air}_b300_ops.toml` (per-op FLOPs and compulsory HBM bytes per micro batch, full activation checkpointing, B300 peaks from NVIDIA's HGX B300 table; each file's `assumptions` list documents its model). Do not modify the skill script; put new input files in `~/tmp/profiling/dsv4-perf-proxy/inputs/` and outputs in `.../derived/`, and append every command to `~/tmp/profiling/dsv4-perf-proxy/commands.sh`.
1. Make an H200 copy of each ops file that changes only the device section: dense BF16, FP8 tensor and FP32 peaks and HBM bandwidth for one H200 GPU, each sourced from NVIDIA's published H200 spec (record the URL; dense, not sparse figures; mark anything unsourced UNVERIFIED). H200 has no MXFP8, so H200 runs map to the FP8 peak.
2. Set each file's workload to match the runs below with `--set` (e.g. `--set n_layers=6 doc_len=... T=32768`); short documents average `doc_len = 6144` (uniform 4096 to 8192), long context `doc_len = 245760`. Check how each file names these parameters before setting them.
3. For each run config below, compute the ideal MFU (both the `prime-rl perf/mfu` and `causal attention` definitions the files report) on H200 and on B300, and the predicted ratio B300/H200 for `perf/mfu` and for `mfu_6dn` (6N only; derive it from the files' parameters). Table it.

### Step 2: MXFP8 on B300 (3 runs, ~45 min)

Common overrides: `--model.debug.force-balanced-routing --model.ep 8 --model.quantization.type mxfp8 --model.quantization.recipe mxfp8_rceil_wgrad_with_hp --model.moe.compute.type mxfp8` (production's recipes; the MoE recipe defaults to `mxfp8_rceil`). Confirm the field names against `packages/prime-rl-configs/src/prime_rl/configs/trainer.py`.

Setting overrides:
- Long context (`long`): `--model.seq-len 262144 --model.cp 8 --data.fake.batch-size 8 --data.fake.doc-len-range '[229376, 262144]'`
- Short documents (`short`): `--model.seq-len 32768 --model.cp 1 --data.fake.batch-size 64 --data.fake.doc-len-range '[4096, 8192]'`

Model overrides:
- DSv4 Flash: none (`proxy.toml` model).
- GLM-4.5-Air: `--model.name zai-org/GLM-4.5-Air`
- GLM-5.3: `--model.name zai-org/GLM-5.3-BF16 --model.debug.random-init --model.conversion-dir <private scratch dir>`

| run name | model | setting | layers | H200 reference (FP8): s/step, tok/s/GPU, perf/mfu, mfu_6dn |
|---|---|---|---|---|
| `b300-dsv4-long-6L-mxfp8` | DSv4 Flash | long | 6 | 14.5 s, 17.0k, 30.8%, 23.6% |
| `b300-air-short-12L-mxfp8` | GLM-4.5-Air | short | 12 | 19.8 s, 11.9k, 40.9%, 27.4% |
| `b300-glm53-short-4L-mxfp8` | GLM-5.3 | short | 4 | 12.0 s, 19.7k, 35.4%, 31.9% |

The H200 references are fwd+bwd seconds per step, so compare against B300's `time/forward_backward`. Pass layers with `--model.debug.num-layers <n>`. For GLM-5.3, random init plus a private conversion dir is required even though the shared snapshot is converted. If GLM-4.5-Air's snapshot lacks a usable conversion, also give it a private `--model.conversion-dir`. Steps: keep `max_steps = 20` from `proxy.toml` (or `--max-steps 12` with steady state 4 to 12 if time is short; say which).

### Step 3: FP8 blockwise on B300 (3 runs, ~45 min)

The same three runs without the MXFP8 overrides (`proxy.toml` defaults: `quantization.type = "fp8"`, `moe.compute.type = "deepgemm_fp8"`), named `...-fp8`. These give the clean same-software ratio against H200; Step 2 against Step 3 gives the MXFP8-vs-FP8 difference on B300.

### Measurement and report

- Medians over steps 4 to the last step from `<run dir>/monitors/file/metrics.jsonl` (merge records by `step`): `time/forward_backward`, `perf/throughput_per_gpu`, `perf/mfu`, `perf/mfu_6dn`, `perf/peak_memory`. Prefer the profiling skill's `run_metrics.py --summarize` (`~/.claude/skills/profiling/scripts/`) over ad hoc parsing.
- Recompiles: all layer-graph recompiles should be in step 1. Use `compile_events.py <run dir>` (its defaults match this launcher's `logs/attempt_1/` layout); if it prints a "no records" note, the logs were not found, which is not evidence of zero recompiles. Report any late recompile with its guard.
- Report one table per precision: model, config, B300 vs H200 for each metric, measured B300/H200 ratios (`perf/mfu`, `mfu_6dn`, tok/s/GPU), and the Step 1 predicted ratios beside them. State what was measured where (truncated layer counts, fake data, balanced routing) and any failure or fallback.
- Commit result summaries (not run directories) and any script changes to this branch under `benchmarks/scripts/dsv4-perf-proxy/`. Do not push. Cancel job 3183 (`scancel 3183`) when all steps are done.

## Code state

This branch, `exp/dsv4-b300-repro`, is prime-rl `main` at f98f9937a (includes #4330 indexer CP balancing and #4333 DeepGEMM mHC pre-pass projection) plus:
- `ee35cdf13`: `data.fake.doc_len_range` (fake RL rows of variable-length documents, padded to the micro-batch multiple).
- `7cdc1bd99`: `data.fake.pad_to_seq_len` (pads fake rows to `seq_len`).
- The harness under `benchmarks/scripts/dsv4-perf-proxy/`: `proxy.toml` (base trainer config), `run_arm.sh` (bare torchrun launcher), `extrapolate.py`, `bucket_kernels.py`, `four_buckets.py`, and microbenchmarks.

Setup: check out this branch, `git submodule update --init --recursive`, `uv sync --all-extras --all-packages`. Always `uv run --no-sync`.

## Harness

`benchmarks/scripts/dsv4-perf-proxy/run_arm.sh <slurm job id> <run name> [trainer CLI overrides...]` runs `torchrun --nproc-per-node 8 -m prime_rl.trainer.rl.train @ benchmarks/scripts/dsv4-perf-proxy/proxy.toml <overrides>` inside an existing allocation via `srun --jobid=<id> --overlap`, with fresh per-run compile caches under `/tmp/$USER/dsv4-perf-proxy/<run>/`, `TORCH_LOGS=recompiles,graph_breaks`, and the run dir at `${OUT_ROOT:-$HOME/prl_output_dir/dsv4-perf-proxy}/<run>` (it refuses an existing name). It exports `CUDA_HOME=/usr/local/cuda-13.0`. Adjust that line and set `OUT_ROOT` if the B300 cluster differs. If you are not inside SLURM, run the inner torchrun command directly.

`proxy.toml` settings shared by every run: `seq_len` and `cp` set per run, `ep = 8`, FP8 blockwise (`quantization.type = "fp8"` with `extra_ignore_patterns = ["o_a_proj", 'compressor\.', 'indexer\.']`, `moe.compute.type = "deepgemm_fp8"`), torch MoE dispatch, full activation checkpointing (freq 1), `compile.fullgraph = true`, `offload = "none"`, `ac_offloading = None`, Muon (lr 1e-6, weight decay 0, max norm 1.0), icepop loss, `matmul_precision = "highest"`, `dist_timeout_seconds = 14400`, 20 steps, `data.fake.generate_samples = true`. Model default: `PrimeIntellect/DeepSeek-V4-Flash-0731-bf16` (real weights). Any DSv4 Flash bf16 checkpoint with the same architecture works if that one is not cached.

Precision: run every timing run in both FP8 blockwise (the `proxy.toml` default, matching the H200 runs) and MXFP8 (production precision on Blackwell), if both are available on B300. prime-rl's blockwise FP8 path requires SM90 or SM100. If it refuses this GPU, say so and report MXFP8 only. MXFP8 overrides: `--model.quantization.type mxfp8 --model.quantization.recipe mxfp8_rceil_wgrad_with_hp --model.moe.compute.type mxfp8` (check the exact CLI against `packages/prime-rl-configs/src/prime_rl/configs/trainer.py`).

## Run matrix (balanced routing everywhere)

Common overrides: `--model.debug.force-balanced-routing --model.ep 8`. Two settings:
- Short documents: `--model.seq-len 32768 --model.cp 1 --data.fake.batch-size 64 --data.fake.doc-len-range '[4096, 8192]'`
- Long context: `--model.seq-len 262144 --model.cp 8 --data.fake.batch-size 8 --data.fake.doc-len-range '[229376, 262144]'`

Both give 32,768 tokens per GPU and 8 micro batches per rank per step.

| model | extra overrides | layer counts | full depth |
|---|---|---|---|
| DSv4 Flash | none (`proxy.toml` model) | 4 and 6 | 43 |
| GLM-4.5-Air | `--model.name zai-org/GLM-4.5-Air` | 6 and 12 | 46 |
| GLM-5.3 DSA | `--model.name zai-org/GLM-5.3-BF16 --model.debug.random-init --model.conversion-dir <private scratch dir>` | 4 and 8 | 78 (3 dense + 75 MoE) |

Use `--model.debug.num-layers <n>` for the layer count. For GLM-5.3, always pass `--model.debug.random-init` and a private `--model.conversion-dir`: a conversion run with a truncated model writes an incomplete cache and still marks it done, which would poison a shared snapshot. With balanced routing, random weights do not change the compute.

That is 24 timing runs (3 models x 2 settings x 2 layer counts x 2 precisions). With B300's larger memory, also try 12 layers for DSv4 and 12 or 16 for GLM-5.3 if time allows (better slopes).

Extra arms if time allows:
1. Padding check: DSv4 at 6 layers in both settings with `--data.fake.pad-to-seq-len` (pads rows to `seq_len`).
2. One trace per model in the short-document setting at the higher layer count: `--max-steps 4 --data.fake.batch-size 16 --trace-path <dir>` (2 micro batches per rank).

## Measurements

From each run's `<run dir>/monitors/file/metrics.jsonl` (rank 0, several records per step, merge by `step`), take medians over steps 4 to 20 of `time/forward_backward`, `perf/throughput_per_gpu` (tok/s/GPU), `perf/mfu`, `perf/mfu_6dn` and `perf/peak_memory`. Confirm all recompiles happen in step 1 (`grep -c "Recompiling function" <run dir>/logs/attempt_1/trainer.log`). Extrapolate to full depth with `uv run --no-sync python benchmarks/scripts/dsv4-perf-proxy/extrapolate.py --low <dir> <L1> --high <dir> <L2> --target-layers <N> --peak 2.25e15 --world-size 8 --steady-start 4` (2.25e15 is the B300 dense BF16 peak that `perf.py` uses). `perf/mfu` counts dense attention without a causal discount and divides FP8 work by the BF16 peak, so compare models on tok/s/GPU and `mfu_6dn`.

## H200 reference results (8x H200, same branch state, balanced routing, medians over steps 4 to 20)

Short documents (32k, cp 1):

| model | layers | fwd+bwd s per step | tok/s/GPU | perf/mfu | mfu_6dn |
|---|---|---|---|---|---|
| DSv4 Flash | 4 / 6 / 43 extrapolated | 9.6 / 12.9 / 74.6 | 24.6k / 18.3k / 3.18k | 26.3 / 26.8 / 27.9% | 25.2 / 25.3 / 25.6% |
| GLM-4.5-Air | 6 / 12 / 46 extrapolated | 11.4 / 19.8 / 67.6 | 20.8k / 11.9k / 3.50k | 39.3 / 40.9 / 42.4% | 27.6 / 27.4 / 27.2% |
| GLM-5.3 DSA | 4 / 78 extrapolated | 12.0 / n/a | 19.7k / 1.11k | 35.4 / 30.7% | 31.9 / 27.1% |

Long context (262k, cp 8):

| model | layers | fwd+bwd s per step | tok/s/GPU | perf/mfu | mfu_6dn |
|---|---|---|---|---|---|
| DSv4 Flash | 4 / 6 / 43 extrapolated | 10.3 / 14.5 / 92.2 | 23.9k / 17.0k / 2.67k | 29.8 / 30.8 / 32.9% | 24.6 / 23.6 / 21.5% |
| GLM-4.5-Air | 6 / 12 / 46 extrapolated | 68.2 / 135 / 517 | 3.61k / 1.82k / 476 | 84.2 / 84.7 / 85.2% | 4.8 / 4.2 / 3.7% |
| GLM-5.3 DSA | 78 extrapolated | n/a | 1.10k | 35.1% | 26.8% |

Other H200 facts for comparison:
- Per added MoE layer and micro batch, GLM-5.3 costs 341 ms against 208 ms for DSv4.
- Going from 32k to 262k, DSv4's per-layer cost rises 26% (sparse attention and the indexer grow with position, which leaves the CP ranks unbalanced). GLM-5.3's rises 4.5%.
- FP8 GEMMs run at 68 to 76% of the FP8 peak in all three models.
- Padding rows to `seq_len` cost +3.2% (short documents, 9.7% padding) and +5.6% (long context, 5.8% padding), with no compile savings.
- By roofline arithmetic, B300's higher compute-to-bandwidth ratio should put DSv4's MFU at roughly 0.83x its H200 value with the same software, because memory-bound work scales with bandwidth (~1.67x) rather than compute (2.27x).

## Report

For each model and setting: B300 tok/s/GPU, `perf/mfu`, `mfu_6dn`, fwd+bwd seconds and peak memory, per layer count and extrapolated, next to the H200 numbers above. Include the B300/H200 speedup ratios, which arm (FP8 or MXFP8) each number comes from, recompile checks, failures and fallbacks. Commit any scripts or result files to this branch under `benchmarks/scripts/dsv4-perf-proxy/` and do not push without approval.
