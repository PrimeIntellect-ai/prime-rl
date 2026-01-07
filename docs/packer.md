# Packer - Rollout Batching for Distributed RL Training

The **Packer** (`src/prime_rl/trainer/rl/packer.py`) is a critical component that bridges the Orchestrator (which generates rollouts) and the Trainer (which performs gradient updates). It runs on the master rank of the trainer.

## Architecture

```
                                 ┌─────────────────────────────────────────────────────────────┐
                                 │                      ORCHESTRATOR                           │
                                 │  (generates rollouts from inference, computes advantages)   │
                                 └─────────────────────────────┬───────────────────────────────┘
                                                               │
                                                               │ TrainingBatch
                                                               │ (examples, temperature, step, run_idx)
                                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            TRANSPORT LAYER                                                   │
│                                                                                                              │
│   ┌─────────────────────────────────┐                         ┌─────────────────────────────────┐            │
│   │  TrainingBatchReceiver          │                         │  MicroBatchSender               │            │
│   │  (filesystem or ZMQ)            │                         │  (filesystem or ZMQ)            │            │
│   │                                 │                         │                                 │            │
│   │  .receive() -> [TrainingBatch]  │                         │  .send(micro_batch_grid)        │            │
│   └────────────────┬────────────────┘                         └────────────────▲────────────────┘            │
│                    │                                                           │                             │
└────────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────┘
                     │                                                           │
                     ▼                                                           │
┌────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────┐
│                                              PACKER (master rank only)                                       │
│                                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  get_batch()                                                                                            │ │
│  │    - calls runs.check_for_changes()                                                                     │ │
│  │    - calls receiver.receive()                                                                           │ │
│  │    - filters out batches with run_idx=None                                                              │ │
│  │    - returns dict[run_idx -> TrainingBatch]                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                               │
│                                              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  has_enough_tokens(rollouts)                                                                            │ │
│  │    - threshold = seq_len * dp_world_size                                                                │ │
│  │    - estimates if next batch will exceed threshold                                                      │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                               │
│                                              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  pack()                                                                                                 │ │
│  │    - waits for enough tokens (with 10s timeout)                                                         │ │
│  │    - updates runs.progress[idx] (step, total_tokens, total_samples)                                     │ │
│  │    - sets runs.ready_to_update[idx] = True                                                              │ │
│  │    - calls prepare_batch() for packing logic                                                            │ │
│  │    - calls sender.send(micro_batch_grid)                                                                │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                              │
│  Dependencies:                                                                                               │
│    - self.runs      : Runs singleton (get_runs())                                                            │
│    - self.receiver  : TrainingBatchReceiver (setup_training_batch_receiver())                                │
│    - self.sender    : MicroBatchSender (setup_micro_batch_sender())                                          │
│    - self.tokenizer : PreTrainedTokenizer (passed in)                                                        │
│    - self.logger    : Logger (get_logger())                                                                  │
│                                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                     │
                     │ micro_batch_grid: list[list[MicroBatch]]
                     │ (one list per dp rank)
                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         TRAINER RANKS (dp_world_size)                                        │
│                                                                                                              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                         │
│   │    Rank 0       │  │    Rank 1       │  │    Rank 2       │  │    Rank 3       │  ...                    │
│   │                 │  │                 │  │                 │  │                 │                         │
│   │ receives        │  │ receives        │  │ receives        │  │ receives        │                         │
│   │ micro_batch_    │  │ micro_batch_    │  │ micro_batch_    │  │ micro_batch_    │                         │
│   │ grid[0]         │  │ grid[1]         │  │ grid[2]         │  │ grid[3]         │                         │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘                         │
│                                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           RUNS SINGLETON                                                     │
│                                                                                                              │
│   progress: dict[int, Progress]     # step, total_tokens, total_samples per run                              │
│   ready_to_update: list[bool]       # flags for weight sync                                                  │
│   max_runs: int                     # max concurrent LoRA runs                                               │
│   check_for_changes()               # check for new/removed runs                                             │
│                                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Core Responsibilities

### 1. Receiving Rollouts

The Packer receives `TrainingBatch` objects from the Orchestrator via the transport layer:

```python
def get_batch(self) -> dict[int, TrainingBatch]:
    batches = self.receiver.receive()
    return {batch.run_idx: batch for batch in batches if batch.run_idx is not None}
```

Returns a dictionary mapping `run_idx` → `TrainingBatch`, enabling multi-run (multi-LoRA) support.

### 2. Token Threshold Waiting

The Packer waits until enough tokens are accumulated before packing:

```python
threshold = seq_len * dp_world_size
```

This ensures efficient GPU utilization. It uses a rolling average to estimate if the next batch will exceed the threshold, with a 10-second timeout to prevent indefinite blocking.

### 3. Sequence Packing Algorithm

Uses **First-Fit Decreasing (FFD)** bin packing in `prepare_batch()`:

1. **Sample Preparation**: Each `TrainingSample` is converted to a `MicroBatch` with concatenated prompt + completion tokens
2. **Sorting**: Samples sorted by run_idx (stable), then by length descending
3. **Bin Packing**: Each sample is placed in the first micro-batch that has room, or a new one is created
4. **Padding**: Micro-batches are padded to `pad_to_multiple_of` alignment (for tensor cores)
5. **Distribution**: Micro-batches are distributed round-robin across data-parallel ranks

### 4. Multi-LoRA Support

Packing respects LoRA adapter boundaries:

```python
# From packer.py comment:
# We pack this way because MultiLoRAMoE currently does not support having different run_idx
# in a microbatch. So we need to pad at run_idx boundaries.
```

Each micro-batch tracks `lora_num_tokens` - token counts per LoRA adapter.

### 5. Progress Tracking

For each run, the Packer updates:
- `progress[idx].step` - Training steps completed
- `progress[idx].total_tokens` - Cumulative tokens processed
- `progress[idx].total_samples` - Cumulative samples processed
- `ready_to_update[idx]` - Flag for weight synchronization

## Data Structures

### TrainingSample (input from Orchestrator)
```python
class TrainingSample(msgspec.Struct):
    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    teacher_logprobs: list[float] | None
    advantage: float | None
```

### MicroBatch (output to Trainer)
```python
class MicroBatch(msgspec.Struct):
    input_ids: list[int]              # Packed token sequence(s)
    loss_mask: list[bool]             # Which tokens contribute to loss
    advantages: list[float]           # RL advantage estimates
    inference_logprobs: list[float]   # Policy logprobs during rollout
    position_ids: list[int]           # Token position embeddings
    temperature: float                # Sampling temperature
    teacher_logprobs: list[float] | None
    lora_num_tokens: list[int] | None # Tokens per LoRA adapter
```

## Transport Layer

Two backends supported:

### Filesystem (default)
- Master writes `rollouts/step_N/rank_X.bin` files
- Uses atomic `.tmp` rename for reliability
- Each rank reads its specific file

### ZMQ
- Uses PUSH/PULL sockets for network communication
- Buffers by step to handle out-of-order arrivals
- Better for multi-machine distributed training

## Integration with Training Loop

```python
# In DataLoader.wait_for_batch():
if self.world.is_master:
    self.packer.pack()      # Only master packs
self.receiver.wait()        # All ranks wait for their data
self.runs.sync_runs()       # Synchronize run state
```

## Key Design Features

| Feature | Purpose |
|---------|---------|
| Token threshold waiting | Batch efficient GPU utilization |
| First-Fit Decreasing | Minimize padding overhead |
| Run-based partitioning | Support multi-LoRA without conflicts |
| Atomic file writes | Reliable inter-process communication |
| Transport abstraction | Support local and distributed training |
