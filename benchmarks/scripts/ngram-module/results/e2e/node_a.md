# PR #4293 e2e verification, node A (job 7048, prime-nebius-puku-h200-gpu-020)

Summary: the branch passes every unit test that was run. With full AC, ep 8 and reshard on (cell 5), the branch matches main bit for bit in loss over 20 steps, and its grad norm is within 8.7e-8 relative. With reshard off (cell 6), the branch trains all 20 steps, while current main still crashes in the first backward with `Autograd not support dtype: Long`. A branch-only checkpoint round trip (save at step 10, resume a second run from it) reproduces steps 11 to 20 bit for bit, and both the DCP checkpoint and its HF export name the n-gram tables as expected.

## Arms

| arm    | commit      | worktree (under `/home/garrett/github/PrimeIntellect-ai/`) | venv                             |
| ------ | ----------- | ---------------------------------------------------------- | -------------------------------- |
| main   | `522bdc015` | `prime-rl-run-ngram-main-522bdc0`                          | `/tmp/garrett/ngram-e2e2-main`   |
| branch | `4766d44a9` | `prime-rl-run-ngram-branch-4766d44`                        | `/tmp/garrett/ngram-e2e2-branch` |

Node-local venvs built with `uv venv --python 3.12 <venv>` and `VIRTUAL_ENV=<venv> uv sync --active --frozen --all-extras --all-packages`. Both report torch `2.13.0+cu130`, CUDA available, 8 devices, and import `prime_rl` from their own worktree.

## Unit tests (branch arm)

Driver: `harness-e2e2/run_unit.sh <worktree> <venv> <out-dir> <cpu|gpu>`, logs and JUnit XML in `unit/branch/`.

| suite                                                                     | step    | passed | failed | skipped |
| ------------------------------------------------------------------------- | ------- | -----: | -----: | ------: |
| `pytest tests/unit -m "not gpu"` (207 deselected)                         | no GPUs |    997 |      0 |       1 |
| `torchrun --nproc-per-node=4 -m pytest .../test_ngram_lookup.py`          | 8 GPUs  |  9 x 4 |      0 |       0 |
| `pytest tests/unit/train/models/qwen3_8_flash_next` (incl. `test_vlm.py`) | 8 GPUs  |     26 |      0 |       0 |
| `pytest -m gpu tests/unit/train/models tests/unit/train/test_model.py`    | 8 GPUs  |    143 |      0 |       5 |

The last row ignores `qwen3_8_flash_next` (covered by the row above). Skips: `test_qwen3_vl_e2e.py` (HF snapshot not cached), `test_gpt_oss.py` (needs `torchrun --nproc-per-node=2`), and the 4 GPU cases of `test_ngram_lookup.py` (need torchrun, covered by the torchrun row). The torchrun row is 9 tests passing on each of the 4 ranks. No failures, so no reruns on main were needed.

## Checkpoint with vision weights

The original synthetic checkpoint `ckpt/qwen38-synth` no longer loads on either arm. Since the earlier A/B (main at `1810d5b`), main gained #3916 (`7f528dfc6`, Qwen3.8 multimodal support): `get_model` in `trainer/model.py` now builds the registered VLM class for any config with a `vision_config`, and that class always has `model.visual`. The checkpoint's `config.json` carries a `vision_config`, but its weights were built before #3916 and hold no `model.visual.*` tensors. A first round of the four cells (run dirs under `runs/`, conversion caches under `conv/`) failed in both arms at weight load with `RuntimeError: Missing key in checkpoint state_dict: model.visual.blocks.0.attn.proj.bias.`, before any step.

`harness-e2e2/add_vision_weights.py` wrote `ckpt/qwen38-synth-vl`, run with the main arm's venv and code (`522bdc015`):

- It builds main's VLM class (`Qwen3_8FlashNextForCausalLM`, chosen by `get_custom_vlm_cls`) on CPU after `torch.manual_seed(0)`. The constructor's `post_init` applies the transformers default init: Linear, Conv and Embedding weights normal with std 0.02 (the config's `initializer_range`), biases zero, LayerNorm weights one.
- It converts that model's state dict to HF names and asserts that the source keys are a subset with matching shapes and that every added key starts with `model.visual.`.
- It writes the 471 source tensors unchanged (checked bit-exact after writing) plus the 333 vision tensors (441,851,632 values) cast to bf16. `config.json` and the tokenizer files are copied unchanged (`cmp` identical).

`harness-e2e2/check_checkpoint_keys.py` builds each arm's model on the meta device, converts the checkpoint with the arm's own `convert_to_prime` (the trainer's load path), and compares against the model's state dict. On both arms: the checkpoint is detected as HF and the model as PrimeRL, 558 model keys against 558 converted keys, 0 missing, 0 unexpected, 0 shape mismatches. `READY` was then created in the checkpoint dir.

The vision weights enter the forward identically in both arms. Without images, the VLM forward runs the vision tower on a zero dummy patch and adds `image_embeds.sum() * 0.0` to the token embeddings (both arms share this code from #3916). So the vision tower contributes exactly zero to activations and exactly zero gradient. Consistent with that, main cell 5 on `qwen38-synth-vl` matches the earlier A/B's main cell 5 (text-only checkpoint, main at `1810d5b`) bit for bit in loss and grad norm at all 20 steps.

## SFT parity

Driver: `harness-e2e2/run_parity.sh <main-worktree> <main-venv> <branch-worktree> <branch-venv> <root> <checkpoint>`, which calls `run_ab.py` for main cell 5, branch cell 5, branch cell 6 and main cell 6, serially on the same node. Each run uses cwd the arm's worktree and env `PRL_OUTPUT_DIR=<run dir>`, `TORCHINDUCTOR_CACHE_DIR=/tmp/garrett/ngram-module-cache/<arm>/<cell>/r0-*/inductor` (fresh per run), `TORCH_LOGS=recompiles`, `VIRTUAL_ENV=<arm venv>`:

```
uv run --active --no-sync sft @ /home/garrett/tmp/ngram-module/harness-e2e2/configs/ab.toml \
  --model.conversion-dir /home/garrett/tmp/ngram-module/e2e2/node-a/conv-qwen38-synth-vl/<arm> \
  --deployment.num-train-gpus 8 --model.ac.mode full --model.reshard-after-forward <true|false> \
  --model.ep 8 --model.cp 1 \
  --env-vars '{"TRITON_CACHE_DIR": "/tmp/garrett/ngram-module-cache/<arm>/<cell>/r0-*/triton"}' \
  --model.name /home/garrett/tmp/ngram-module/ckpt/qwen38-synth-vl
```

`ab.toml` is unchanged from the earlier A/B: fake random variable-length data, seq_len 4096, batch 8, lr 1e-3, 20 steps, `model.offload = "none"`, `model.ac_offloading = None`, `fullgraph = false`, no checkpoints, no W&B. Each arm converts into its own fresh cache dir under `conv-qwen38-synth-vl/`. Run dirs are under `runs-qwen38-synth-vl/`. `parity-vl/parity.md` has the full report from `collect.py` (step time, startup, recompiles).

| cell | reshard | arm    | exit | steps | peak GiB | median step (s) |
| ---- | ------- | ------ | ---: | ----: | -------: | --------------: |
| 5    | true    | main   |    0 |    20 |     16.6 |           0.414 |
| 5    | true    | branch |    0 |    20 |     16.7 |           0.432 |
| 6    | false   | branch |    0 |    20 |     17.6 |           0.406 |
| 6    | false   | main   |    1 |     0 |        - |               - |

Main cell 6 failed in the first backward on all ranks with `RuntimeError: Autograd not support dtype: Long`. Peak memory is rank 0's `perf/peak_memory`. Median step is `time/step` over steps 2 to 20 (after the compile step 1, with no later recompiles). Lower is better for both. Step times come from single runs on a toy model and support no speed claim.

Parity, from `loss/mean` and `optim/grad_norm` in each run's `monitors/file/metrics.jsonl` (rank 0). Loss gap: abs(loss x - loss y). Grad gap: abs(grad norm x - grad norm y) / grad norm y. Lower is closer for both.

| comparison                     | steps | max loss gap | steps with loss gap | max grad gap | steps with grad gap |
| ------------------------------ | ----: | -----------: | ------------------: | -----------: | ------------------: |
| branch cell 5 vs main cell 5   |    20 |            0 |                   0 |      8.7e-08 |          1 (step 2) |
| branch cell 6 vs branch cell 5 |    20 |            0 |                   0 |            0 |                   0 |

The single grad norm difference is at step 2: 0.6816244125366211 (branch) against 0.6816244721412659 (main), one float32 ulp apart, with the loss at every step, including later steps, bit-identical. This matches the expectation (bit-identical loss, grad norm within about 1e-7 relative), so no noise-floor repeats were run.

Per-step values, cell 5 (branch cell 6 is identical to branch cell 5 at every step):

| step | loss main | loss branch | grad norm main | grad norm branch |
| ---: | --------: | ----------: | -------------: | ---------------: |
|    1 | 12.473270 |   12.473270 |       0.733467 |         0.733467 |
|    2 | 12.469576 |   12.469576 |       0.681624 |         0.681624 |
|    3 | 12.464398 |   12.464398 |       0.678155 |         0.678155 |
|    4 | 12.465224 |   12.465224 |       0.679082 |         0.679082 |
|    5 | 12.469022 |   12.469022 |       0.686259 |         0.686259 |
|    6 | 12.468101 |   12.468101 |       0.615731 |         0.615731 |
|    7 | 12.465182 |   12.465182 |       0.688537 |         0.688537 |
|    8 | 12.467583 |   12.467583 |       0.642970 |         0.642970 |
|    9 | 12.466207 |   12.466207 |       0.641272 |         0.641272 |
|   10 | 12.464683 |   12.464683 |       0.653806 |         0.653806 |
|   11 | 12.467341 |   12.467341 |       0.653079 |         0.653079 |
|   12 | 12.464930 |   12.464930 |       0.631533 |         0.631533 |
|   13 | 12.461618 |   12.461618 |       0.676690 |         0.676690 |
|   14 | 12.465039 |   12.465039 |       0.683034 |         0.683034 |
|   15 | 12.465069 |   12.465069 |       0.589686 |         0.589686 |
|   16 | 12.464437 |   12.464437 |       0.633046 |         0.633046 |
|   17 | 12.468750 |   12.468750 |       0.683644 |         0.683644 |
|   18 | 12.465524 |   12.465524 |       0.620410 |         0.620410 |
|   19 | 12.465643 |   12.465643 |       0.638816 |         0.638816 |
|   20 | 12.462943 |   12.462943 |       0.610530 |         0.610530 |

## Checkpoint round trip (branch arm only)

Driver: `harness-e2e2/run_ckpt.sh <worktree> <venv> <root> <checkpoint>`. Both runs use the cell 5 setup above (8 GPUs, full AC, reshard on, ep 8, `qwen38-synth-vl`, the branch's conversion cache) with one extra override each:

- Run A (`runs-qwen38-synth-vl/branch/cell5-ckpt/r0`): `--ckpt.interval 10`, 20 steps. It saved `checkpoints/step_10` (in 9.6 s) and, at the end of training, `checkpoints/step_20`, 18 GB each.
- Run B (`runs-qwen38-synth-vl/branch/cell5-resume/r0`): `--resume.dir <run A>/ab/checkpoints/step_10`, no `[ckpt]`, so it saves nothing. The trainer logged `Resuming from step 10 (total_tokens=294912, total_samples=85, ...)` and ran steps 11 to 20.

The SFT `[ckpt]` config (`output_dir`, `interval`, `keep_last`, `keep_interval`, `skip_scheduler`, `skip_dataloader`, `skip_optimizer`) has no HF weight export option. The trainer writes only DCP checkpoints, and HF export is the offline tool `tools/convert_dcp_to_bf16.py`, which was run on run A's step 10 with `torchrun --nproc-per-node=8` (log `logs/ckpt-export.log`).

| comparison                       | steps    | max abs loss diff | max abs grad norm diff |
| -------------------------------- | -------- | ----------------: | ---------------------: |
| run B vs run A                   | 11 to 20 |                 0 |                      0 |
| run A vs parity branch cell 5    | 1 to 20  |                 0 |                      0 |

Names, checked by `harness-e2e2/check_ngram_ckpt_names.py` (log `logs/ckpt-names.log`):

- DCP step 10 (1239 keys): the tables are `app.model.model.language_model.ngram.tables.{1,5}.weight`, with the three layout buffers `layer_multipliers`, `ngram_heads_offsets` and `ngram_heads_vocab_sizes` under the same prefix. The optimizer state is `app.optimizers.state.model.language_model.ngram.tables.{1,5}.weight.{exp_avg,exp_avg_sq,step}`. No other key contains `.ngram.`.
- HF export of step 10: 804 keys, exactly the key set of `qwen38-synth-vl`. For layers 1 and 5, all 131 expected keys are present: `model.language_model.layers.{1,5}.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` plus the three layout buffers, and no `shard_128`. The six exported layout buffers are int64 and equal to the source checkpoint's. The exported layer 1 `shard_0` differs from the source by up to 8.7e-3, so the export holds the trained weights.

All checkpoints (step 10 with its HF export, and step 20, about 40 GB) were deleted after recording these results.
