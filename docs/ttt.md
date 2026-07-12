# Test-Time Training (TTT)

TTT trains a per-rollout LoRA adapter at context-compaction boundaries. The compacting
harness asks the model for a handoff summary, replaces the old conversation with the summary,
and continues sampling through an adapter trained on the abandoned branch. For RL, each
branch carries the adapter checkpoint that sampled it, and the trainer replays that frozen
adapter while updating only the policy's base weights.

`ttt-implementation.md` records the detailed design. `ttt-plan.md` is the historical proposal.
Experiment configs are under [`configs/ttt/`](../configs/ttt/README.md).

## Architecture

```text
compacting harness       verifiers interception hook          TTT service
rewrite conversation  -> detect the new trace branch       -> POST /update
next model request     <- select versioned adapter          <- save PEFT checkpoint
                       <- salt the prefix cache             <- load it into every vLLM replica
```

The supported path has one important invariant: every sampled branch uses one TTT adapter
version. The hook updates only when the prepared conversation stops extending the current
leaf, then stamps subsequently committed nodes with the new version. `Branch.ttt_version`
raises if sampled nodes from multiple versions appear on one branch. The supplied compacting
harness rebuilds the prompt as a fresh system-plus-summary branch and satisfies this invariant;
arbitrary partial-history rewrites are not a supported TTT trigger contract.

Updates are synchronous from the rollout's perspective. The next turn is not sampled until
training, checkpoint export, and inference-adapter loading have completed. Sampling uses a
version-specific cache salt so vLLM cannot reuse KV state computed under an older adapter.

TTT requires the renderer/train client because raw updates use the exact token IDs seen by
inference. TTT-enabled eval environments are routed through that client too. Eval adapters are
released and their checkpoints deleted after each rollout; they are never replayed by the RL
trainer.

## Configuration

An environment opts in through `verifiers.v1.TTTConfig`:

```toml
[[orchestrator.train.env]]
taskset = { id = "deepdive-v1" }
harness = { id = "compacting", compact_at_tokens = 8192 }
ttt = { base_url = "http://localhost:8092" }
```

The service is configured with `TTTServiceConfig` and launched with
`uv run ttt @ config.toml`. Important shared fields are the base model, tokenizer used for
Q&A rendering, LoRA rank/targets, optimizer, output directory, adapter prefix, every vLLM
admin URL, and the per-replica admin timeout. Checkpoints are written as
`<output_dir>/ttt/<rollout_id>/v<version>/`.

Two service engines are available:

- `peft` is the default single-process Hugging Face/PEFT engine. It swaps one rollout's LoRA
  tensors and optimizer state into a shared model for each serialized update. Use it for
  small models and standalone evaluation; do not launch it under multi-process `torchrun`.
  This legacy path assumes the verifier resends an identical payload on retry, trusts its
  generated rollout/adapter identity, and performs sequential best-effort replica admin
  operations. It is not the hardened multi-replica service described below.
- `fsdp` uses Prime-RL's custom model stack, FSDP2, and resident MultiLoRA slots. Updates for
  different rollouts can share a packed forward while retaining separate slots, losses,
  optimizers, and checkpoints. It requires flash attention and a chunked fused LM head.
  Context parallelism is rejected at service startup because packed sequences are currently
  replicated on every rank. VLM inputs and grouped-expert or output-head LoRA targets are not
  supported. For MoE deployments such as GLM-4.5-Air, use attention projections only.

The FSDP engine validates token IDs against the loaded vocabulary before GPU execution,
uses the DTensor-aware gradient-clipping path, and checks loss, gradients, parameters, and
optimizer state for non-finite values before publishing a checkpoint. Its frozen output-head
backward computes only the activation gradient, avoiding a vocabulary-sized gradient for the
frozen weight.

### Launcher-managed service

For multi-node SLURM runs, add a top-level `[ttt]` block and set
`deployment.num_ttt_nodes`. The launcher allocates the TTT node after inference and trainer
nodes, starts the FSDP service under one `torchrun`, fills its model, tokenizer, output path,
matmul precision, and vLLM admin roots, and waits for `/health` before starting the
orchestrator. Environment URLs equal to `"auto"` are replaced with the managed service URL;
explicit URLs are preserved for intentionally external services.

Managed launch requires:

- the FSDP engine;
- at least one launcher-owned inference replica, so the template has vLLM admin roots to
  wire and a role alongside which to start the TTT service;
- one adapter prefix shared by the managed service and participating envs;
- `keep_checkpoints = true` when train environments use TTT;
- service slots and vLLM LoRA capacity large enough for train concurrency;
- the service model and Q&A tokenizer to match the policy configuration;
- full-weight policy training, since vLLM cannot compose a trainable policy LoRA with the
  per-rollout TTT LoRA;
- non-fullgraph compilation, because replay changes an eager forward hook per microbatch.

The ScaleSWE configs set the relevant capacities explicitly. Prime-RL's global inference
defaults are otherwise left unchanged by this feature.

Only envs whose TTT URL is `"auto"` participate in those managed-service contracts. Envs with
an explicit URL remain external and may use their own adapter prefix. A top-level managed
service with no `"auto"` env is rejected as unused configuration.

## Q&A at compaction

With `ttt.qa = {}`, the model generates question-answer items while the abandoned context is
still visible. Prompts emphasize self-contained facts and trigger-phrased lessons. Items are
parsed from tagged blocks, optionally verified, and trained standalone with the rollout's
system prompt and tool schemas as masked context. `qa.also_train_rollout` includes the raw
branch in the same adapter update.

The Q&A exchanges are also committed as `ttt_qa` trace branches. They have their own token
budget and do not inflate the rollout's normal turn/token limits, but their sampled tokens can
receive the rollout's RL signal.

Two mutually exclusive train-time variants route Q&A-derived examples to the policy's CE
loss in the same mixed batch as RL:

- `qa.recycle_to_policy` uses each rollout's own pairs.
- `qa.meta_lessons` compares the completed group's pairs and rewards, extracts general
  lessons, and attaches those lessons to the first post-filter survivor of that group. The
  group ownership is retained across batch splits so filtering the first member cannot drop
  the lessons.

## Frozen-adapter replay

`TrainingSample.ttt_adapter_path` identifies the checkpoint used by a branch. The packer does
not mix paths in one microbatch. `TTTReplayManager` loads the checkpoint, attaches frozen
LoRA deltas to supported `nn.Linear` modules, and casts them to the module's actual forward
dtype. Gradients flow through the delta into the base policy, not into the adapter.

Prime-RL `main` owns the graph-wide rule that a shared sampled node is trainable in only one
flattened branch sample. TTT relies on that existing trajectory behavior; it does not add a
second fork/rewrite detector or a TTT-specific deduplication policy.

Checkpoint cleanup is deliberately simple. `TTTCheckpointGC` defers directories referenced by
a shipped batch until the trainer publishes the corresponding policy version, deletes
conclusively errored/filtered rollout directories, and carries undecided overflow rollouts to
a later batch. Eval directories are deleted immediately. This state is in memory: there is no
persistent recovery manifest or explicit trainer-consumption marker. A final step with no
subsequent weight publication may therefore leave harmless checkpoint files for manual or
run-directory cleanup.

## Failure semantics and limits

Malformed or out-of-order requests fail the affected rollout. In the distributed FSDP engine,
rank 0 prepares Q&A token sequences once and broadcasts the exact materialized jobs. Validation
errors are isolated before slot mutation; unexpected failures after work starts terminate the
worker group rather than risk rank divergence.

FSDP adapter activation is transactional across inference replicas: loads fan out concurrently,
partial or ambiguous failures trigger an unload reconciliation, and a version is marked loaded
only after every replica succeeds. Per-rollout leases prevent a cancelled/timed-out request or
release from being overtaken by another operation on the same adapter. Release retries use a
bounded tombstone registry and repeat unload until every replica confirms the adapter is gone.

Those identity, exact-retry, lease, tombstone, and transactional-replica guarantees are FSDP
v2 guarantees. The standalone PEFT engine intentionally retains its simpler trusted-client,
single-endpoint semantics.

The implementation still needs a real Linux multi-GPU smoke test. The macOS unit suite cannot
exercise CUDA kernels, FSDP/DTensor collectives, live vLLM reload behavior, or the shared
cluster filesystem. The TTT service also starts from the policy base weights present at launch;
as the RL policy evolves, that base becomes stale. Frozen replay preserves the recorded adapter
delta, but it does not reconstruct the exact historical base model.
