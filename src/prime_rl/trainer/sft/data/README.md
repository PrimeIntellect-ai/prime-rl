# SFT data loading

`setup_dataloader` constructs the dataset and chooses the text SFT pipeline.
Fake data and multimodal data retain their local `CatDataset` path; distributed
multimodal transport is not supported. Text packing is always global.

## Stream and packing

`StridedShard` assigns global shuffled-stream positions modulo the trainer world
size, including CP ranks. Normal spawned `DataLoader` workers render each rank's
shard in order. Filtered examples yield empty records, not missing positions.
Shuffles are deterministic per epoch and keep their index arrays in memory.

Each rank gathers fixed-size CPU length/source-id metadata chunks. All ranks run
the same online best-fit algorithm on the reconstructed global order. It stops
at the first unplaceable example; there is no lookahead. A partially consumed
communication chunk does not change step membership. A training epoch without
any usable samples fails explicitly; finite validation can be empty.

For batch size `B`, micro-batch size `M`, sequence length `S`, DP size `D`:

- Pack the entire optimizer step into `B/M` rows of capacity `M*S`.
- Sort all rows by the sum of squared sample lengths, then assign consecutive
  groups of `D` rows to successive microsteps. Each rank gets `B/(D*M)` rows.
- Send input IDs, shifted targets, positions and masks as int64 CPU tensors with
  `all_to_all_single`. The shared schedule determines every split size; no
  object/pickle transport or extra size exchange is needed.
- Send the same logical rows to every CP peer. Only the trainer shards them for
  context parallelism. Padding has no loss and follows `CatDataset` boundaries.

Worker prefetch, the current step, and one communication chunk bound speculative
rendering. No rendered payload or pending-sample queue is checkpointed.

## Overlap and collective ordering

Each loader owns a dedicated **Gloo** process group, created by the main thread
in identical order on all ranks. A single producer thread issues blocking CPU
metadata gathers and tensor exchanges on that group. Rendering processes never
use distributed collectives. The producer performs no CUDA allocation, GPU copy,
CUDA stream operation, or collective on the model's groups. The main consumer
pins completed CPU rows before handing them to the trainer.

One complete step is prefetched while the model processes the preceding step.
Merely using a second NCCL communicator would not provide this independence:
concurrent NCCL groups require consistent cross-rank launch ordering, including
across threads. See [PyTorch's process-group guidance](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group).
Gloo isolation avoids that ordering dependency; it does **not** eliminate CPU,
memory, or network contention, which still requires measurement under load.

Validation owns a different Gloo group; a validation reset cannot interleave
collectives into the training prefetch stream. Construct/close loaders in the
same order on all ranks. All ranks must consume the same number of global steps.
`close()` drains the outstanding prefetch, joins the producer and rendering
workers, then destroys its group. Never cancel only one rank's pending future:
peers may already be waiting in the corresponding collective. Producer errors
propagate through the future; distributed failures also use the configured
process-group timeout and launcher's rank-failure handling.

## Checkpoint and finite tails

The consumer commits a progress snapshot only after delivering all local rows
of a global optimizer step. `state_dict()` rejects partial steps and never
waits for or snapshots the producer's speculative cursor. Its data state is
the first unconsumed global position; cumulative progress counters and a data
signature are retained for logging and validation. Resume re-renders from that
position. Worker count, communication chunk size and compatible DP/CP layouts
can change without dropping or replaying consumed samples.

Finite validation pads its final step, including entirely empty rows, so every
rank finishes together. Empty/filtered final source positions still advance the
committed cursor. Training loss normalization and the model's gradient
accumulation schedule are not changed by the loader.
