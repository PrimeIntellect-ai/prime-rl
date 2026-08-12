# SFT profiling campaign

Per-step torch profiler flamegraphs (perfetto-loadable chrome traces) for SFT runs, swept
over models, dtypes and sequence lengths, to find optimization targets in prime-rl.

## Matrix

- **Models**: `Qwen/Qwen3-30B-A3B` (real weights, 12/48 layers) and `zai-org/GLM-4.5`
  (random init + forced balanced routing — the ~700GB checkpoint doesn't fit local disk —
  12/92 layers, keeping the 3 leading dense layers).
- **Dtypes**: `bf16` (default mixed precision) and `mxfp8` (`model.quantization.type=mxfp8`).
- **Sequence lengths**: 8192 → 131072 in steps of 8192 (16 values).
- **Fixed**: 8× B200, EP=8, flash attention (auto → FA4 on SM100), real SFT data
  (`allenai/tulu-3-sft-mixture`, packed), micro batch size 1, global batch 8, torch.compile
  + full activation checkpointing (trainer defaults).

Each run does 9 steps: 2 skipped (compile/warmup) + 1 profiler warmup + 6 recorded with
`ProfilerStep#` frames, Python stacks (`with_stack`) and CUDA kernels.

## Usage

```bash
uv run python benchmarks/profiling/run_campaign.py            # full 64-run sweep, resume-safe
uv run python benchmarks/profiling/run_campaign.py --models qwen3-30b --dtypes bf16 --seq-lens 8192
uv run python benchmarks/profiling/run_campaign.py --runs-dir benchmarks/profiling/runs-no-offload \
    --extra-args "--model.optim-cpu-offload false"             # variant sweep into a separate dir
uv run python benchmarks/profiling/run_campaign.py --dtypes bf16 fp8   # Hopper: fp8 instead of mxfp8
uv run python benchmarks/profiling/trace_tools.py --trace <trace_0.json.gz> --out-dir <dir>  # re-process
```

On Hopper (H100/H200), `attn = "auto"` resolves to FA3 and mxfp8 is unavailable (SM100-only) —
use `--dtypes bf16 fp8` for the blockwise-FP8 equivalent sweep.

## Output layout

```
runs/
├── manifest.jsonl                  # one record per run: status, duration, avg step time
├── rollup.csv                      # model,dtype,seq_len → avg step/GPU-busy/forward/backward ms
└── <model>/<dtype>/seq_<len>/
    ├── launcher.log                # full launcher + trainer stdout
    ├── output/                     # resolved configs, trainer logs
    └── processed/
        ├── steps/step_<n>.json.gz  # one perfetto trace per recorded training step
        ├── average.json.gz         # synthetic "average step": CPU+Python call tree with
        │                           # durations averaged across steps + avg GPU kernel track
        └── summary.json            # per-step wall/GPU-busy time, top kernels, phase times
```

Open any `.json.gz` at https://ui.perfetto.dev (or `chrome://tracing`). In the per-step
traces the Python track gives the Python-side flamegraph and the `cpu_op`/`kernel` tracks
the C++/CUDA side; `average.json.gz` shows the same call tree with slice widths equal to
the mean time per step, so one look shows where an average step goes.
