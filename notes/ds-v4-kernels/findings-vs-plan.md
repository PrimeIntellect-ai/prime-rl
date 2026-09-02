# Where the handoff plan and the repository disagree

`PLAN.md` was written in an earlier session. Every claim in it was re-checked against the
worktree before any code was written. Most held. The ones that did not are recorded here, with
what the correct fact is and what it changes.

## 1. There are eight local H200s. This is not a SLURM login node.

`PLAN.md` opens with "There are no local GPUs: this is a SLURM login node" and instructs the
implementer to allocate with `srun --partition=all --nodes=1 --gres=gpu:8 --pty bash`.

That is no longer true. `nvidia-smi -L` lists 8 NVIDIA H200s, each 143771 MiB, all at 0 MiB used,
and `torch.cuda.device_count()` returns 8 with `get_device_capability(0) == (9, 0)`. SLURM is also
installed and `sinfo` reports the partition, so both statements were probably true of the machine
the plan was written on; this box is a compute node.

**Effect:** the sweep runs directly, no `srun`, no `sbatch`. The "eight configurations
concurrently, one per GPU" design is unchanged, it just does not need a job allocation.

## 2. This worktree had no virtualenv, and `uv run` alone does not create a usable one.

There was no `.venv/` here at all. `torch` lives in the `gpu` optional-dependency group
(`pyproject.toml`), so a bare `uv run python ...` resolves only the base dependency set, builds a
196-package environment with no torch in it, and fails with `ModuleNotFoundError: No module named
'torch'`. The fix is the command `AGENTS.md` already documents, `uv sync --all-extras`, which is
what produced the working environment.

**Effect:** none on the measurements, but it costs a cycle and roughly ten minutes of wheel
downloads. Any later session picking this worktree up should run `uv sync --all-extras` first.

## 3. The real model has two sliding layers. The bare-default config has none, and the plan's
   layer mix for it is wrong.

`PLAN.md` says the 43-layer default gives "23 HCA and 20 CSA and zero sliding" with the
representative HCA layer at index 4. The zero-sliding part is right, the counts are not: the
default branch in `configuration_deepseek_v4.py:200-206` emits two `heavily_compressed_attention`
bootstrap layers then alternates CSA/HCA over the remaining 41, giving 22 HCA and 21 CSA.

More importantly, the default is the wrong config to measure. The real checkpoint is cached at
`/home/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b.../config.json`
and ships the legacy schema, `compress_ratios = [0, 0, 4, 128, 4, 128, ..., 4, 0, 0, 0]`, 46
entries of which `DeepseekV4Config` keeps the first `num_hidden_layers = 43`. That resolves to

| layer type | indices | count |
|---|---|---|
| `sliding_attention` | 0, 1 | 2 |
| `compressed_sparse_attention` | 2, 4, 6, ..., 42 | 21 |
| `heavily_compressed_attention` | 3, 5, 7, ..., 41 | 20 |

The trailing three `0`s in the file belong to the MTP head (`num_nextn_predict_layers = 1`) and
are discarded by the truncation, which is why the file has 46 ratios for 43 layers.

**Effect:** the harness builds its config from that `config.json` rather than from bare defaults,
and the representative layers are **sliding = 0, CSA = 2, HCA = 3**, not the plan's 0/3/4. The
aggregate 43-layer memory weighting is 2 / 21 / 20, not 0 / 20 / 23.

## 4. CSA entry count is `t/4`, not `t/2`.

`PLAN.md` writes `e ≈ t/2` for CSA layers, reasoning from "two overlapping series at
`compress_rate 4`", while explicitly flagging the factor for confirmation. Confirmed: it is `t/4`.

`CompressionLayout.build` (`attention.py:207`) gives document `doc` exactly
`L_doc // compress_rate` entries, independent of `n_series`. `n_series = 2` does not add entries;
it widens each entry's *pooling window* from `compress_rate` slots to `2 * compress_rate`, by
having entry `e` pool the `Ca` half of entry `e-1`'s tokens alongside the `Cb` half of its own
(`_overlap_with_previous_window`, `attention.py:388-406`). So

```
e = t / 4     for CSA   (compress_rate 4)
e = t / 128   for HCA   (compress_rate 128)
```

**Effect:** halves every `e`-dependent term in the analytic memory model. The CSA attention score
tensor is `1.25 t^2` wide, not `1.5 t^2`, and each fp32 indexer score copy is `64 t^2` bytes
rather than `128 t^2`. The ranking is not obviously changed, because the indexer materializes
about three of those copies against the attention core's one, but the numbers are.

## 5. Assorted stale paths and line numbers

- `deps/prime-kernels/rmsnorm/run_bench.sh` is really
  `deps/prime-kernels/prime_kernels/rmsnorm/run_bench.sh` (one directory level deeper).
- The test helpers define `_CSA_LAYER, _HCA_LAYER = 1, 2` at `test_deepseek_v4.py:78`. There is no
  `_SLIDING_LAYER` constant; the plan lists three.
- `DeepseekV4IndexerScorer.forward` is `attention.py:462-467`, the plan says `:461-466`.
- FA4's head-dim gate is `_validate_head_dims` in `flash_attn/cute/interface.py:112`, the plan
  says `:114-126`. The claim itself is exactly right: `is_deepseek_mla_absorbed_shape`
  (`head_dim_v == 512`) is only reachable for `compute_capability in [10, 11]`, and the
  `compute_capability == 9` branch asserts `8 <= head_dim <= 256`. H200 is sm90, so `head_dim 512`
  is rejected outright.
- "39-iteration Sinkhorn loop" is loose but arithmetically right. `hc_sinkhorn_iters` defaults to
  20 and the real config sets 20; `hyperconnections.py:60-63` does one column normalization before
  the loop and two per iteration for `iters - 1 = 19` iterations, so 39 normalization steps, each
  a separate pair of tiny kernel launches.

## Claims re-checked and confirmed unchanged

`eager_attention_with_sinks` at `attention.py:145-166` and its dense `(b, h, t, t+e)` score
tensor; the `_supports_flash_attn = False` block at `modeling_deepseek_v4.py:107-112`; the CP
rejection at `:241-245`; `PackedContext.build` called once per forward at `:254`; the mHC stream
mixing at `:65-67` and `:73-75`; the fp32 flatten and Sinkhorn loop in `hyperconnections.py`;
`MemoryProfiler` at `utils.py:325-345` and its `get_world().rank` dependency;
`ActivationCheckpointConfig.mode` defaulting to `"full"` at `configs/trainer.py:32`;
`aten::bmm` / `aten::mm` in `DEFAULT_SELECTIVE_TARGETS`; `*.jsonl` and `outputs/` at
`.gitignore:6,10`; Megatron-LM `dev` at `/home/garrett/github/NVIDIA/Megatron-LM` HEAD
`24fc94d27`; triton 3.7.1 installed.
