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
│  │    - buffers samples per run in self.buffers[run_idx]                                                   │ │
│  │    - stores (sample, temperature) tuples for later packing                                              │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                               │
│                                              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  has_enough_tokens()                                                                                    │ │
│  │    - threshold = seq_len * dp_world_size                                                                │ │
│  │    - checks internal buffers for token count                                                            │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                               │
│                                              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  _select_samples_round_robin(token_budget)                                                              │ │
│  │    - selects samples from buffers using round-robin fair scheduling                                     │ │
│  │    - takes samples evenly from runs with buffered work                                                  │ │
│  │    - skips runs with empty buffers                                                                      │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                               │
│                                              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  _update_run_progress(run_idx, num_samples, num_tokens)                                                 │ │
│  │    - tracks samples consumed per step via samples_consumed_this_step[run_idx]                           │ │
│  │    - increments progress[idx].step only when batch_size samples consumed                                │ │
│  │    - sets ready_to_update[idx] = True only on step completion                                           │ │
│  │    - reads batch_size from runs.config[run_idx].batch_size                                              │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                               │
│                                              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  pack()                                                                                                 │ │
│  │    - waits for enough tokens (with 10s timeout)                                                         │ │
│  │    - calls _select_samples_round_robin() for fair sample selection                                      │ │
│  │    - calls _update_run_progress() per run (step completion check)                                       │ │
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

### 1. Buffering Rollouts

The Packer receives `TrainingBatch` objects and buffers samples per run:

```python
def get_batch(self) -> None:
    batches = self.receiver.receive()
    for batch in batches:
        if batch.run_idx is None:
            continue
        for sample in batch.examples:
            self.buffers[batch.run_idx].append((sample, batch.temperature))
```

Each sample is stored with its temperature for later packing. This decouples receiving from packing, preventing one run from flooding the trainer.

### 2. Token Threshold Waiting

The Packer waits until buffered samples provide enough tokens:

```python
threshold = seq_len * dp_world_size
```

This ensures efficient GPU utilization. It checks internal buffers with a 10-second timeout to prevent indefinite blocking.

### 3. Fair Sample Selection (Round-Robin)

When packing, samples are selected fairly across runs:

```python
def _select_samples_round_robin(self, token_budget: int):
    # Takes samples evenly from runs with buffered work
    # Skips runs with empty buffers
    # Persists round-robin position across pack() calls
```

This prevents any single run from dominating trainer time.

### 4. Sequence Packing Algorithm

Uses **First-Fit Decreasing (FFD)** bin packing in `prepare_batch()`:

1. **Sample Preparation**: Each `TrainingSample` is converted to a `MicroBatch` with concatenated prompt + completion tokens
2. **Sorting**: Samples sorted by run_idx (stable), then by length descending
3. **Bin Packing**: Each sample is placed in the first micro-batch that has room, or a new one is created
4. **Padding**: Micro-batches are padded to `pad_to_multiple_of` alignment (for tensor cores)
5. **Distribution**: Micro-batches are distributed round-robin across data-parallel ranks

### 5. Multi-LoRA Support

Packing respects LoRA adapter boundaries:

```python
# From packer.py comment:
# We pack this way because MultiLoRAMoE currently does not support having different run_idx
# in a microbatch. So we need to pad at run_idx boundaries.
```

Each micro-batch tracks `lora_num_tokens` - token counts per LoRA adapter.

### 6. Step-Based Progress Tracking

Progress is tracked per-run, with step completion based on `batch_size`:

```python
def _update_run_progress(self, run_idx: int, num_samples: int, num_tokens: int) -> None:
    self.samples_consumed_this_step[run_idx] += num_samples
    batch_size = self.runs.config[run_idx].batch_size

    while self.samples_consumed_this_step[run_idx] >= batch_size:
        self.runs.progress[run_idx].step += 1
        self.runs.ready_to_update[run_idx] = True
        self.samples_consumed_this_step[run_idx] -= batch_size
```

- `progress[idx].step` - Incremented only when `batch_size` samples consumed
- `progress[idx].total_tokens` - Cumulative tokens processed
- `progress[idx].total_samples` - Cumulative samples processed
- `ready_to_update[idx]` - Set to True only on step completion
- `samples_consumed_this_step[idx]` - Tracks partial progress toward next step

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
| Per-run buffering | Decouple receiving from packing, prevent run flooding |
| Round-robin fair scheduling | Even work distribution across runs |
| Step-based progress | Increment step only when batch_size samples consumed |
| Token threshold waiting | Batch efficient GPU utilization |
| First-Fit Decreasing | Minimize padding overhead |
| Run-based partitioning | Support multi-LoRA without conflicts |
| Atomic file writes | Reliable inter-process communication |
| Transport abstraction | Support local and distributed training |
