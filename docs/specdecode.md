# Speculative decoding and draft training

`prime-rl` can serve a draft through vLLM, train it with upstream [Speculators](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/), or optimize it alongside an RL policy. Install the GPU and optional training dependencies on your compute environment:

```bash
uv sync --all-extras
```

For a workspace installation that also needs every bundled environment, use `uv sync --all-extras --all-packages`.

## Inference and evaluation

The example pins Qwen3-4B and its DSpark draft. Run these commands inside a GPU allocation:

```bash
uv run inference @ examples/specdecode/inference.toml @ examples/specdecode/dspark.toml
uv run eval @ examples/specdecode/aime25.toml --run.name dspark-aime25
```

Omit the DSpark overlay to measure the target alone. The evaluation uses all 30 AIME 2025 questions with eight completions per question. Keep sampling, context length, concurrency and hardware fixed when comparing throughput and accuracy.

The DSpark and joint RL examples set `vllm.compilation_config.pass_config.fuse_allreduce_rms = false` while retaining compilation, CUDA graphs, and vLLM's default backend and AOT settings. This pass combines tensor-parallel all-reduce with RMSNorm; TP1 has no tensor-parallel all-reduce to fuse. Disabling the pass is not a complete workaround for observed vLLM 0.30.0 TP2 × DP2 startup failures: ordinary all-reduce can still dispatch to FlashInfer independently. Validate startup and cached restart for the deployment configuration.

### Accuracy and reproducibility

Standard speculative decoding preserves the target distribution through rejection sampling: a proposal from distribution `q` is accepted with probability `min(1, p/q)`, where `p` is the target probability. A rejection draws from normalized `max(p-q, 0)`; greedy decoding accepts only target-argmax matches. A poor draft reduces acceptance and may slow inference, but does not change this mathematical guarantee. See the [speculative decoding algorithm](https://arxiv.org/abs/2211.17192).

Use `rejection_sample_method = "standard"` for quality comparisons. Synthetic acceptance is a benchmarking option that bypasses this guarantee. A fixed request seed does not ensure identical sampled text across speculative and ordinary decoding because their random draws differ. Kernel and batch shapes can also cause small floating-point differences in target logits. Compare repeated samples with matched settings and per-question uncertainty, and use greedy token comparisons and weight fingerprints to investigate discrepancies. Avg@16 is mean correctness over sixteen completions per question; pass@16 measures whether any of those completions is correct.

## Standalone draft training

`uv run specdecode` validates upstream training options, records the resolved configuration, and launches local `torchrun` workers through `prime_rl.specdecode.worker`. Each worker explicitly compiles the upstream FlexAttention function before calling the upstream Speculators trainer. This prevents activation-checkpoint recomputation from falling back to eager attention outside the compiled model. Its `train` tables follow the upstream configuration schema. `num_gpus` controls the number of workers; `output_dir` holds configuration snapshots and checkpoints. Use `--dry-run` to validate a recipe before loading models.

First prepare tokenized target responses with upstream `uv run speculators prepare-data`. The saved dataset must include `input_ids`, `loss_mask`, and `seq_len`. Configure `train.data.data_path` to point to that dataset.

For a target without a compatible draft, omit `train.draft.from_pretrained` and configure `num_layers`, `draft_arch`, and `target_layer_ids` to initialize a new draft through upstream Speculators. Generate responses with the intended target, preserve its actual token IDs and assistant loss masks, and keep validation questions separate from training questions. Match the intended domain and context lengths; exclude downstream evaluation questions from draft training. Check held-out loss and actual serving acceptance before using the checkpoint for RL.

For custom DSpark data with `sample_from_anchor = true`, `loss_mask[i]` selects the prediction of token `i + 1`. Shift an assistant-token mask left by one and clear its final position before saving the training dataset. Exclude each record's last `block_size` anchor positions so a sampled block cannot cross a packed-document boundary. Keep the original token IDs and raw masks for provenance. The joint RL adapter performs this alignment on rollout batches.

Online training requests missing target features from a running extraction server:

```bash
uv run inference @ examples/specdecode/extract-hidden-states.toml
uv run specdecode @ examples/specdecode/qwen3_train_dspark.toml @ examples/specdecode/online.toml
```

Use separate allocated GPUs for inference and training. The extraction layers must match the draft checkpoint's requested target layers, plus the target's final pre-normalization layer. The example uses layers 1, 9, 17, 25, 33 and 36. Both processes must access the hidden-state connector's shared storage.

For offline training, generate all features first:

```bash
uv run speculators generate-offline-data \
  --preprocessed-data outputs/specdecode/data \
  --endpoint http://localhost:18300/v1 \
  --validate-outputs --fail-on-error
uv run specdecode @ examples/specdecode/qwen3_train_dspark.toml
```

The default file backend reads `<data_path>/hidden_states`. To select another directory, set `hidden_states_path` under `[train.backend]`. Offline mode uses `train.generation.on_missing = "raise"`; online mode sets it to `"generate"`. Upstream manages standalone checkpoint saving and resume within `train.trainer.save_path`.

## Joint RL training

The six-GPU smoke example uses two context-parallel trainer ranks and TP2 × DP2 inference:

```bash
uv run rl @ examples/specdecode/rl.toml --run.name joint-dspark
```

The optional trainer section controls draft training:

```toml
[trainer.speculator]
name = "RedHatAI/Qwen3-4B-speculator.dspark"
lr = 0.00001
loss_weight = 1.0
freeze_backbone = false

[trainer.speculator.training]
max_anchors = 16
loss_fn = '{"ce": 0.1, "tv": 0.9}'
```

The RL config supplies the draft checkpoint to inference and rejects mismatched checkpoint names or revisions. vLLM's `speculative_config` controls the decoding method and number of speculative tokens. The implementation delegates model loading and loss computation to upstream Speculators; DSpark is the validated example, while another algorithm may require adapting its feature alignment.

The draft joins the policy's FSDP mesh, optimizer and checkpoint. `lr` overrides its learning rate while retaining the policy's scheduler; omitting it shares the policy rate. Draft loss uses detached policy features and frozen verifier projections refreshed from the current policy each step. It does not backpropagate through the policy. `freeze_backbone = true` trains only the draft. Context-parallel target features are gathered before draft training, so each CP rank processes the full draft sequence. Packed-document boundaries and the rollout loss mask restrict eligible anchors.

Joint training supports filesystem and NCCL weight broadcasts. LoRA policy training, multimodal token batches and NIXL draft routing are rejected during configuration validation.

## Checkpoints and weight updates

The usual RL checkpoint and `--resume` options save and restore both models and their optimizer state. Keep the optimizer offload mode unchanged when resuming. Export a joint checkpoint with:

```bash
uv run python tools/convert_dcp_to_bf16.py outputs/joint-dspark/checkpoints/step_3
```

The output contains policy weights and a `speculator/` directory with the draft weights and configuration. Set the inference model and speculative model paths to those directories. The FP8 converter also writes the draft separately in BF16.

Regular broadcasts carry draft weights under `speculator.` and update both models before inference resumes. A filesystem worker also accepts a draft-only checkpoint via `POST /update_weights?target=draft` with `{"weight_dir": "/path/to/draft"}`. Pause requests, perform the update and then resume; the draft-only update invalidates the prefix cache. Shared parameters remain owned by the target and are preserved during draft-only updates. The RL admin plane handles the pause/update/resume lifecycle for training broadcasts, and rollout requests use a policy-version cache salt. Keep `VLLM_SERVER_DEV_MODE` disabled because vLLM's development update route conflicts with this endpoint.

For validation, `GET /weight_fingerprint?target=draft` returns per-worker tensor hashes, nonzero counts and addresses, including derived GPU weight attributes. This is an expensive diagnostic operation intended for idle workers. Reloads preserve the draft's captured GPU tensor addresses.

## Metrics

File monitors and W&B receive upstream draft losses and acceptance estimates under `speculator/`. Standard GPU optimizer training also records `speculator/grad_norm` before clipping. Inference counters produce these pooled interval metrics:

- `inference/agg/spec_decode_acceptance_rate/pooled`: accepted draft tokens divided by proposed draft tokens.
- `inference/agg/spec_decode_acceptance_length/pooled`: one plus accepted draft tokens divided by draft attempts, including the target's bonus token.
- `inference/agg/generation_tokens_total:rate/sum`: generated tokens per second across inference engines.

Ratios use summed counter deltas across engines, exclude reset or empty intervals, and are omitted until two valid scrapes exist. Training acceptance estimates and inference acceptance measure different workloads and should be interpreted separately.
