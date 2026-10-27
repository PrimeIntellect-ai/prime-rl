# PR #4293 end-to-end verification on node B (job 7049, prime-nebius-puku-h200-gpu-052)

Branch `feat/ngram-module` moves the Qwen3.8 n-gram tables into a model-level module and, since `4766d44a9`, sends them with the non-layer weights over NCCL and NIXL. This file covers trainer-side export identity, RL runs with vLLM over NCCL and NIXL (main vs branch), and a positive control that makes the inference-side tables stale on purpose.

## Arms and setup

| Arm    | Commit      | Worktree under `/home/garrett/github/PrimeIntellect-ai/` | Node-local venv                  |
| ------ | ----------- | -------------------------------------------------------- | -------------------------------- |
| main   | `522bdc015` | `prime-rl-run-ngram-main-522bdc0`                        | `/tmp/garrett/ngram-e2e2-main`   |
| branch | `4766d44a9` | `prime-rl-run-ngram-branch-4766d44`                      | `/tmp/garrett/ngram-e2e2-branch` |

Both worktrees are detached and frozen, and every export record reports `dirty=False`. Each run used fresh node-local compile caches under `/tmp/garrett/ngram-e2e2-cache/<run>` and a per-arm conversion cache under this directory. Paths below are relative to `/home/garrett/tmp/ngram-module/`.

Checkpoint: `ckpt/qwen38-synth-vl` (node A's copy of `qwen38-synth` with seeded random vision weights). Main `#3916` builds the vision tower for any config with `vision_config`, so the original `qwen38-synth` fails on both arms with `Missing key in checkpoint state_dict: model.visual.blocks.0.attn.proj.bias`. That failure is not a branch problem.

vLLM cannot load that checkpoint as is, on main, for two reasons unrelated to the branch:

- `AssertionError: unsupported output_gate_type=None` in `QwenGatedDeltaNetAttention.__init__` (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:568`). The trainer's config class pops `text_config.output_gate_type` when the synthetic checkpoint is saved, and the trainer hardcodes `output_gate_activation="sigmoid"`. The real Qwen3.8 config has `"output_gate_type": "sigmoid"`.
- `OSError: Can't load image processor ... preprocessor_config.json` in the vLLM API server, because the synthetic checkpoint ships no processor configs.

The RL runs therefore use `e2e2/node-b/ckpt/qwen38-synth-vl-vllm`, built by `harness-e2e2/vllm_overlay_checkpoint.py`: the same `model.safetensors` (symlinked), the same config plus `text_config.output_gate_type = "sigmoid"` from the real snapshot, and the snapshot's `preprocessor_config.json` and `video_preprocessor_config.json`. No weight differs from `qwen38-synth-vl`. With the overlay, vLLM loads the checkpoint on both arms (`Loading weights took 0.80 seconds` on main and 0.97 seconds on branch, 2.22 GiB per worker). The export drivers load `qwen38-synth-vl` directly.

## Task 1: export identity (trainer side)

`bash harness-e2e2/drivers_matrix.sh` runs `export_driver.py` per arm and layout through `drivers_launch.sh`: `torchrun` with `@ configs/smoke.toml --model.name ckpt/qwen38-synth-vl`, plus `--model.dp-replicate 2` on GPUs 0,1 for dp2 or `--model.ep 8` on 8 GPUs for ep8. Neither arm has `get_consuming_layers` anymore, so the driver uses main's `partition_weights` and `build_transfer_group_index` signatures, and it ran on both arms. Comparison: `uv run --no-project --python 3.12 harness-e2e2/compare_export.py <main.json> <branch.json> --out export_<layout>.md`. Full reports: `e2e2/node-b/export_ep8.md` and `e2e2/node-b/export_dp2.md`.

| Check                                     | ep8     | dp2     | Detail (ep8)                                   |
| ----------------------------------------- | ------- | ------- | ---------------------------------------------- |
| NCCL integer                              | pass    | pass    | 265 vs 265                                     |
| NCCL packet count                         | pass    | pass    | 265 vs 265                                     |
| NCCL dtype, shape, sha256 per key         | pass    | pass    |                                                |
| NCCL send phase, non-table keys           | pass    | pass    |                                                |
| NCCL send phase, n-gram table keys        | pass    | pass    | 262 moved layers 1,5 -> non-layer, 0 unchanged |
| NCCL chunk packet sequence                | pass    | pass    |                                                |
| NCCL intra-packet key order (info)        | same    | same    |                                                |
| Filesystem key set                        | pass    | pass    |                                                |
| Filesystem dtype, shape, sha256 per key   | pass    | pass    |                                                |
| Filesystem shard file count (info)        | same    | same    | 8 vs 8 files                                   |
| NIXL group names                          | pass    | pass    | ['non_layer', 'layer.0', 'layer.1', 'layer.2', |
| NIXL group, non-table keys                | pass    | pass    |                                                |
| NIXL group, n-gram table keys             | pass    | pass    | 2 moved layers 1,5 -> non-layer, 6 unchanged   |
| Save partition owner, non-table keys      | pass    | pass    | same grouping, relabeled {2: 1, 1: 2}          |
| Save partition owner, n-gram table keys   | pass    | pass    | 8 moved layers 1,5 -> non-layer, 0 unchanged   |

What the rows show:

- Both layouts send 265 NCCL packets with 804 keys, and every key has the same dtype, shape and sha256 in the NCCL payload and in the filesystem save (804 keys, in 8 files for ep8 and 2 for dp2).
- `compare_export.py` now labels each NCCL packet with its send phase (the non-layer phase, then one phase per layer, each followed by its row-chunk packets) instead of comparing packets by index. Only the 262 n-gram table keys of layers 1 and 5 change phase (2 x 128 row chunks and 2 x 3 hash buffers), and each moves from its layer's phase to the non-layer phase, as designed. Branch: packet 0 is non-layer (344 keys), packets 1-256 are the tables' row chunks, then one packet per layer. Main: packet 0 is non-layer (338 keys), and each table's chunks follow its layer's packet.
- NIXL: the two table weights move from groups `layer.1` and `layer.5` to `non_layer`. The hash buffers are integer tensors, which NIXL does not transfer on either arm.
- Save partition: the table keys move to the owner of the non-layer unit. For ep8 every other key keeps its grouping, but owner ranks 1 and 2 swap: `partition_weights` assigns units greedily by byte size, and without their tables, layers 1 and 5 have equal sizes and tie-break by name. The check accepts a one-to-one relabeling of owner ranks and fails on any regrouping.
- Trainer-side table sharding is unchanged on both arms: `_S(0, 8), S(0)` for ep8 and `R, S(0)` for dp2.

Negative self-test of the adapted checker (`e2e2/node-b/selftest/`, mutated copies of the branch ep8 record): moving one `linear_attn.norm.weight` into the non-layer packet fails `NCCL send phase, non-table keys`, and moving table 1's 128 row chunks behind layer 3's packet fails `NCCL send phase, n-gram table keys` and `NCCL chunk packet sequence`.

## Task 2: RL with vLLM, NCCL weight broadcast

Config `harness-e2e2/configs/rl_nccl.toml`: one node, 4 trainer GPUs (`ep = 4`, no offloading, compile with `fullgraph = false`) and 4 inference GPUs (vLLM data parallel 4, default `EngramConfig(cpu_offload=True)`, `mamba_cache_dtype = "float32"`), reverse-text environment, batch of 128 rollouts with up to 128 completion tokens, AdamW with `lr = 1e-3`, 20 steps, no checkpoints, no W&B. The dry run resolved `trainer.weight_broadcast.inference_world_size = 4`. Two settings matter for this test:

- `max_staleness = 0`: every trained token was sampled by the exact policy version the trainer computes logprobs with, so `mismatch_kl/all/mean` measures trainer-vs-inference disagreement under the same weights, not policy drift.
- `[orchestrator.train.algo] type = "debug"`: a constant advantage of 1 on every sampled token. A random model scores 0 on every reverse-text rollout, so GRPO gives zero advantages, and the first attempt never produced a trainable batch (`No admitted train payload after 8704 finalized units`). The debug algorithm exists for this infra case and gives every step a real weight update.

Why `lr = 1e-3`: the synthetic tables are initialized with std 0.020. Adam moves every element that receives a gradient by about `lr` per step, so in 20 steps the rows that every sample touches (the shared system prompt and chat template n-grams) can drift by up to about 0.02, the size of the initialization. Tables left stale on the inference side would then differ from the trainer's by about their own scale. The positive control below shows that this is visible in the metric.

Command per arm (`bash harness-e2e2/rl_matrix.sh rl_launch.sh harness-e2e2/configs/rl_nccl.toml rl-nccl main branch`):

```
uv run --active --no-sync rl @ harness-e2e2/configs/rl_nccl.toml --output-dir e2e2/node-b/rl/<arm> \
  --run.name rl-nccl-<arm> --trainer.model.conversion-dir e2e2/node-b/rl/conv/<arm> --inference.vllm.data-parallel-size 4
```

Positive control (`stale`): the branch arm with `configs/rl_nccl_stale_tables.toml`, identical except that the inference processes get `PYTHONPATH=harness-e2e2/stale_tables_control`. Its `sitecustomize.py` makes each vLLM worker drop every `...ngram_embedding.shard_*` key it receives over NCCL, so the inference tables keep their checkpoint values while the trainer updates its own. A branch bug in receiving the tables would look like this.

Per-step trainer metrics from `monitors/file/metrics.jsonl` (producer `trainer`): `mismatch_kl/all/mean` (lower is better) and `loss/mean`.

| step   | main mismatch KL | branch mismatch KL | stale mismatch KL | main loss | branch loss | stale loss |
| ------ | ---------------- | ------------------ | ----------------- | --------- | ----------- | ---------- |
|      1 |         0.003553 |           0.003594 |          0.003504 |  -0.06252 |    -0.06252 |   -0.06254 |
|      2 |         0.003201 |           0.003665 |          0.003847 |   -0.0833 |    -0.08343 |   -0.08334 |
|      3 |         0.003108 |           0.003444 |          0.004338 |  -0.08329 |     -0.0625 |   -0.06258 |
|      4 |         0.003088 |            0.00309 |          0.004737 |  -0.08342 |    -0.08331 |   -0.08331 |
|      5 |         0.003032 |           0.003126 |          0.005352 |  -0.06257 |    -0.08333 |   -0.08327 |
|      6 |         0.002837 |           0.002998 |          0.005702 |   -0.0625 |    -0.06239 |   -0.06239 |
|      7 |          0.00287 |              0.003 |          0.005985 |  -0.08328 |    -0.06244 |    -0.0624 |
|      8 |         0.002914 |           0.002889 |          0.006275 |   -0.0625 |     -0.0625 |   -0.08333 |
|      9 |         0.003576 |           0.002954 |          0.006685 |  -0.08328 |    -0.08332 |   -0.08337 |
|     10 |         0.002894 |           0.002907 |          0.006625 |  -0.08323 |    -0.08336 |   -0.08341 |
|     11 |         0.002852 |           0.002951 |          0.007276 |  -0.08327 |    -0.08335 |   -0.08322 |
|     12 |         0.002805 |           0.002975 |          0.007545 |  -0.08335 |    -0.08328 |    -0.0834 |
|     13 |           0.0028 |           0.002846 |          0.007486 |  -0.06245 |    -0.08326 |   -0.08334 |
|     14 |         0.002947 |           0.003023 |          0.008055 |  -0.08336 |    -0.08333 |   -0.08335 |
|     15 |          0.00292 |             0.0027 |          0.008385 |  -0.08333 |     -0.0833 |   -0.08342 |
|     16 |           0.0028 |           0.002884 |          0.008368 |   -0.0833 |    -0.08335 |    -0.0625 |
|     17 |         0.002766 |           0.002861 |          0.008823 |  -0.08341 |    -0.08335 |   -0.06263 |
|     18 |         0.002748 |           0.002718 |          0.008927 |  -0.08333 |    -0.08337 |   -0.08326 |
|     19 |         0.002899 |           0.002884 |          0.009252 |   -0.0625 |    -0.08333 |   -0.08326 |
|     20 |         0.002845 |           0.003206 |          0.009568 |  -0.08337 |    -0.08337 |   -0.08343 |

| Run    | Mismatch KL step 1 | Step 20  | Max (step)     | Mean steps 1-5 | Mean steps 16-20 |
| ------ | ------------------ | -------- | -------------- | -------------- | ---------------- |
| main   | 0.003553           | 0.002845 | 0.003576 (9)   | 0.003196       | 0.002812         |
| branch | 0.003594           | 0.003206 | 0.003665 (2)   | 0.003384       | 0.002911         |
| stale  | 0.003504           | 0.009568 | 0.009568 (20)  | 0.004356       | 0.008988         |

vLLM layerwise-reload warnings (`harness-e2e2/rl_inference_checks.py`), per inference worker and weight reload, over 4 workers and 21 reloads each (startup plus 20 steps):

| Failed-load warnings per worker and reload | main       | branch     | stale      |
| ------------------------------------------ | ---------- | ---------- | ---------- |
| Reloads per worker (workers)               | 21-21 (4)  | 21-21 (4)  | 21-21 (4)  |
| FusedMMInputNorm                           | 1          | 1          | 1          |
| Qwen4ExpNGramEmbedding                     | 2          | 2          | 2          |
| Qwen4ExpPLEPinnedHostEmbedding             | 0          | 0          | 2          |
| RotaryEmbedding                            | 1          | 1          | 1          |

The warnings that main and branch share are pre-existing and harmless to the tables. `Qwen4ExpNGramEmbedding` owns the 3 hash buffers of each table and copies them with `buffer.copy_`, which the layerwise tracker does not count, so it reports a failed load and puts back the old buffers, whose values never change. `RotaryEmbedding` and `FusedMMInputNorm` warn the same way on both arms and in the control. The table weight lives in the child module `Qwen4ExpPLEPinnedHostEmbedding`, loaded through its `weight_loader`, and that module warns only in the stale-table control, twice per reload (one per table).

### Same NCCL A/B launched through the generated SLURM script

To match the NIXL launch path, both arms were also run through `rl_sbatch_launch.sh` (`bash harness-e2e2/rl_matrix.sh rl_sbatch_launch.sh harness-e2e2/configs/rl_nccl.toml rl-nccl-sbatch main branch`), which renders `single_node_rl.sbatch.j2` and runs it inside job 7049. The result holds: mismatch KL goes from 0.0037 to 0.0029 on main and from 0.0036 to 0.0029 on branch, with maxima of 0.003659 and 0.003616 at step 1, and the reload warnings again match.

| step   | main mismatch KL | branch mismatch KL | main loss | branch loss |
| ------ | ---------------- | ------------------ | --------- | ----------- |
|      1 |         0.003659 |           0.003616 |  -0.06255 |    -0.06251 |
|      2 |         0.003609 |           0.003158 |  -0.08331 |    -0.08336 |
|      3 |         0.003499 |           0.003518 |  -0.08328 |    -0.08342 |
|      4 |         0.003009 |           0.002828 |  -0.08329 |    -0.08332 |
|      5 |         0.003215 |            0.00291 |  -0.08345 |    -0.08328 |
|      6 |         0.002841 |           0.002842 |  -0.08338 |    -0.06248 |
|      7 |         0.002752 |            0.00299 |  -0.08328 |    -0.06247 |
|      8 |         0.002777 |           0.002896 |  -0.08334 |    -0.08338 |
|      9 |         0.002879 |            0.00289 |  -0.08333 |    -0.08329 |
|     10 |         0.002909 |            0.00282 |  -0.08342 |    -0.06253 |
|     11 |         0.002741 |           0.002918 |  -0.08342 |    -0.08335 |
|     12 |         0.002834 |           0.002998 |  -0.08334 |    -0.06253 |
|     13 |         0.002937 |           0.002808 |  -0.08328 |    -0.08334 |
|     14 |         0.002779 |           0.002752 |  -0.08343 |    -0.08335 |
|     15 |         0.002918 |           0.002652 |  -0.08335 |    -0.08321 |
|     16 |         0.002585 |           0.003052 |  -0.08336 |    -0.08344 |
|     17 |         0.002841 |           0.002912 |  -0.08344 |    -0.08343 |
|     18 |         0.002858 |           0.002993 |  -0.08339 |    -0.08331 |
|     19 |         0.002812 |           0.002834 |   -0.0833 |    -0.08336 |
|     20 |         0.002927 |            0.00291 |  -0.06253 |    -0.06255 |

| Failed-load warnings per worker and reload | main       | branch     |
| ------------------------------------------ | ---------- | ---------- |
| Reloads per worker (workers)               | 21-21 (4)  | 21-21 (4)  |
| FusedMMInputNorm                           | 1          | 1          |
| Qwen4ExpNGramEmbedding                     | 2          | 2          |
| RotaryEmbedding                            | 1          | 1          |

## Task 3: RL with vLLM, NIXL weight broadcast

Blocked on both arms by a pre-existing limitation of main's NIXL transfer-plan tracing with Qwen3.8's linear-attention layers. The failure is identical on main and branch and happens before any table is transferred.

Setup (all in gitignored paths of the main worktree, symlinked into the branch worktree, via `harness-e2e2/build_nixl_deps.sh`):

- UCX 1.19.1 with `cuda_copy`, `cuda_ipc`, `rc_verbs` and `rc_mlx5`: copied from `prime-rl-feat-ds-v4-nixl/third_party/ucx`. Building it here was not possible because the node lacks the rdma-core headers and autotools, and there is no sudo.
- NIXL 0.10.1 built from source as `nixl_cu13` against that UCX and installed into both venvs. The node has no `pkg-config`, so the first build silently skipped the UCX plugin (`NIXL WARNING ... Skipping backend registration UCX due to the missing plugin`). Passing `-Ducx_path` fixed it, and `Backend UCX was instantiated` then shows on the trainer and in a standalone probe.
- ModelExpress v0.3.0 and redis 7.4.2 built with `scripts/install_modelexpress.sh`, using cargo from `/home/garrett/.cargo-x86_64/bin` and protoc 28.3 from the official release zip.

Launch: `bash harness-e2e2/rl_matrix.sh rl_sbatch_launch.sh harness-e2e2/configs/rl_nixl.toml rl-nixl main branch`. `rl_sbatch_launch.sh` dry-runs `rl` with `--slurm.project-dir <worktree>` to render `single_node_rl.sbatch.j2`, swaps `source .venv/bin/activate` and `uv sync` for the node-local venv, and runs the script with `bash` inside job 7049. Config `configs/rl_nixl.toml` is `rl_nccl.toml` with `weight_broadcast.type = "nixl"`. The resolved trainer config has `inference_world_size = 4`. The script also moves `MX_STATE_DIR` out of the run directory, because the template's default `<run_dir>/modelexpress` makes the launched `rl` refuse the run directory with `FileExistsError: Run directory ... already contains artifacts from a previous run`. That is a separate pre-existing issue of the single-node NIXL template. The NIXL attempts ran an earlier copy of this launcher, `rl_nixl_launch.sh`, which differed only in also requiring the `MX_STATE_DIR` line.

Failure on both arms, during the startup broadcast (v0), in `NIXLWeightUpdateWorker.initialize_transfer` -> `trace_weight_loads` -> vLLM `mamba_mixer2.py:253` (`.view_as(target_slice)`), on all 4 inference workers of each run:

```
prime_rl.transports.weights.nixl.graph.UnsupportedOpError: unsupported operation aten.view.default on
'model.language_model.layers.0.linear_attn.conv1d.weight', recorded chain=(TensorOperation(name='__getitem__',
args=(slice(0, 1024, None),), kwargs={}),)
```

The NIXL tracer does not support the `view` that vLLM's GDN `conv1d` weight loader applies to a sliced weight, so no Qwen3.8 model with linear attention can build a NIXL transfer plan at these commits. The orchestrator then fails and the launcher stops the run (`Orchestrator failed with exit code 1`). Making NIXL work for this model needs a change on main, for example `aten.view` support in `transports/weights/nixl/graph.py`, which is outside this PR. The export check above still covers the branch's NIXL grouping change: the table weights move to the `non_layer` transfer group.

## Conclusions

- Export identity holds on both layouts (ep8 and dp2): equal NCCL integer and packet count (265), and every key's dtype, shape and sha256 equal in the NCCL payload (804 keys) and the filesystem save (804 keys). The only placement changes are the designed ones: the 262 table keys move from layers 1 and 5 to the non-layer NCCL phase, the two table weights move to the `non_layer` NIXL group, and the table keys move to the non-layer save unit (with a by-design swap of owner ranks 1 and 2 for ep8).
- vLLM loads the synthetic checkpoint once `output_gate_type` and the processor configs are restored. Without them it fails on main with the errors above, which are checkpoint-builder issues, not branch issues.
- NCCL RL: the branch receives the tables correctly in the non-layer packet. Mismatch KL stays flat on both arms (0.0036 to 0.0028 on main, 0.0036 to 0.0032 on branch, never above 0.0037 on either), every inference worker reloads 21 times on both arms, and the vLLM reload warnings are identical on both arms. The same holds when both arms are launched through the generated SLURM script. The positive control, with the inference tables deliberately left stale, grows monotonically from 0.0035 to 0.0096 (2.7x) and adds a `Qwen4ExpPLEPinnedHostEmbedding: Failed to load weights` warning per table and reload. Neither signature appears on the branch.
- No branch-only problem was found. NIXL RL could not run on either arm because of a pre-existing NIXL tracer limitation with Qwen3.8's GDN `conv1d` loader (`UnsupportedOpError: unsupported operation aten.view.default`), so the branch's NIXL receive path is only covered by the export grouping check.

## Harness files (under `harness-e2e2/`)

Created or changed by this node (all other files in `harness-e2e2/` belong to node A):

- `compare_export.py` (changed): NCCL packets are compared by send phase, and placement checks accept only n-gram table keys moving from layer `i` to the non-layer packet, group or unit. The save-partition check accepts a one-to-one relabeling of owner ranks.
- `drivers_launch.sh`, `drivers_matrix.sh` (changed): job 7049, `harness-e2e2/export_driver.py`, outputs under `e2e2/node-b/drivers`, fresh conversion root, checkpoint `ckpt/qwen38-synth-vl`, export cells only.
- `vllm_overlay_checkpoint.py` (new): vLLM overlay of a synthetic checkpoint (`output_gate_type` and processor configs).
- `configs/rl_nccl.toml`, `configs/rl_nixl.toml`, `configs/rl_nccl_stale_tables.toml` (new): RL configs.
- `rl_launch.sh`, `rl_sbatch_launch.sh`, `rl_matrix.sh` (new): RL launchers for plain `uv run rl` and for the generated single-node SLURM script inside job 7049.
- `stale_tables_control/sitecustomize.py` (new): positive control that drops table chunks on the vLLM side.
- `rl_metrics.py`, `rl_inference_checks.py` (new): per-step metric table and vLLM reload-warning table.
- `build_nixl_deps.sh` (new): UCX copy, NIXL wheel build, ModelExpress build, branch symlinks.

`export_driver.py` is unchanged: its fallback to main's signatures already covers both arms.
