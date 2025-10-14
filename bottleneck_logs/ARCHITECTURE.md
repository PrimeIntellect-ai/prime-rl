# Architecture Diagram - Bottleneck Investigation

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Prime-RL System                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐           ┌──────────────────┐           ┌──────────────┐
│   Orchestrator   │──────────>│  Inference API   │──────────>│  vLLM Engine │
│                  │           │     Server       │           │   (Worker)   │
│ - Sample         │ HTTP POST │ - FastAPI        │ RPC Call  │ - Model      │
│ - Generate       │           │ - Endpoints      │           │ - Weights    │
│ - Update weights │           │ - Middleware     │           │ - GPU Ops    │
└──────────────────┘           └──────────────────┘           └──────────────┘
         ↓                              ↑
         ↓                              │
    ┌────────┐                    ┌──────────┐
    │Trainer │                    │ Checkpts │
    │- Train │                    │ /weights/│
    │- Write │───────────────────>│  step_N/ │
    └────────┘  Write checkpoint  └──────────┘
```

## Weight Update Flow (Before Instrumentation)

```
Orchestrator                    API Server                     vLLM Engine
    │                               │                              │
    │  POST /update_weights         │                              │
    │──────────────────────────────>│                              │
    │         (queued...)           │                              │
    │                               │ collective_rpc()             │
    │                               │─────────────────────────────>│
    │                               │                              │ load_weights()
    │                               │                              │─────────────
    │                               │                              │
    │                               │<─────────────────────────────│
    │<──────────────────────────────│                              │
    │  200 OK                       │                              │
    │                               │                              │

Problem: "queued..." could be 30+ seconds when server is busy with streaming!
```

## Weight Update Flow (After Instrumentation)

```
Orchestrator                    API Server                     vLLM Engine
    │                               │                              │
    │ [weights][abc] client.send    │                              │
    │──────────────────────────────>│                              │
    │          t0                   │ [weights][abc] server.recv   │
    │                               │          t1                  │
    │                               │ [weights][abc] rpc.start     │
    │                               │          t2                  │
    │                               │ collective_rpc()             │
    │                               │─────────────────────────────>│
    │                               │                              │ load_weights()
    │                               │                              │─────────────
    │                               │                              │
    │                               │<─────────────────────────────│
    │                               │ [weights][abc] rpc.done      │
    │                               │ rpc_ms = t3 - t2             │
    │<──────────────────────────────│          t3                  │
    │ [weights][abc] client.done    │                              │
    │ wall_ms = t4 - t0             │                              │
    │ queue_ms = wall_ms - rpc_ms   │                              │
    │          t4                   │                              │

Now we can measure:
- Network + Accept time: t1 - t0
- Server processing time: t2 - t1
- Engine operation time: t3 - t2 (rpc_ms)
- Total client time: t4 - t0 (wall_ms)
- Queue delay: wall_ms - rpc_ms (queue_ms)
```

## Dedicated Admin Client Fix

### Before (Shared Connection Pool)
```
┌─────────────────────────┐
│  Orchestrator Client    │
│  (AsyncOpenAI)          │
│                         │
│  Connection Pool:       │
│  ┌───────────────────┐  │
│  │ Conn 1: Stream 1  │──┼─┐
│  │ Conn 2: Stream 2  │──┼─┤
│  │ Conn 3: Stream 3  │──┼─┤    ┌──────────────────┐
│  │ Conn 4: Stream 4  │──┼─┼───>│  API Server      │
│  │ ...                │──┼─┤    │  (Port 8000)     │
│  │ Conn N: Stream N  │──┼─┤    │                  │
│  │                   │  │ │    │  Accept Queue:   │
│  │ POST /update_wt   │──┼─┘    │  [Full! Wait...] │
│  └───────────────────┘  │       └──────────────────┘
│  max_connections=28000  │
│  max_keepalive=28000    │             ↑
└─────────────────────────┘             │
                                        │
Problem: New admin request waits for available connection or accept slot!
```

### After (Dedicated Admin Client)
```
┌─────────────────────────┐
│  Orchestrator Client    │
│                         │
│  Main Client:           │
│  ┌───────────────────┐  │
│  │ Conn 1: Stream 1  │──┼─┐
│  │ Conn 2: Stream 2  │──┼─┤
│  │ Conn 3: Stream 3  │──┼─┤    ┌──────────────────┐
│  │ ...               │──┼─┼───>│  API Server      │
│  └───────────────────┘  │ │    │  (Port 8000)     │
│                         │ │    │                  │
│  Admin Client:          │ │    │  Accept Queue:   │
│  ┌───────────────────┐  │ │    │  [Processing...] │
│  │ Fresh Connection  │──┼─┘    └──────────────────┘
│  │ No Keep-Alive     │  │             ↑
│  │ Closes After Use  │  │             │
│  └───────────────────┘  │             │
│  max_connections=1      │             │
│  max_keepalive=0        │             │
└─────────────────────────┘             │
                                        │
Solution: Dedicated connection bypasses streaming queue!
```

## Checkpoint Pipeline

```
Trainer                           Filesystem                    Orchestrator
   │                                  │                              │
   │ Train step N                     │                              │
   │──────────────                    │                              │
   │                                  │                              │
   │ [ckpt] write.done                │                              │
   │ weight_ckpt_manager.save()       │                              │
   │─────────────────────────────────>│                              │
   │          write_ms                │ /weights/step_N/             │
   │                                  │ - model.safetensors          │
   │                                  │ - config.json                │
   │                                  │                              │
   │                                  │          [ckpt] wait.start   │
   │                                  │<─────────────────────────────│
   │                                  │     polling for step_N       │
   │                                  │          ...                 │
   │                                  │          [ckpt] wait.done    │
   │                                  │─────────────────────────────>│
   │                                  │          wait_ms             │
   │                                  │                              │
   │                                  │  POST /update_weights        │
   │                                  │  path=/weights/step_N/       │
   │                                  │<─────────────────────────────│
```

## Generation & RL Pipeline

```
Orchestrator Loop:

   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Check Async Barrier                                      │
   │    if (progress.step - ckpt_step > async_level):            │
   │       [ckpt] wait.start → wait for checkpoint               │
   │       [weights] client.send → update weights                │
   ├─────────────────────────────────────────────────────────────┤
   │ 2. Generate Completions                                     │
   │    [gen] batch.start inflight=X target_batch=Y              │
   │    → vLLM inference (streaming)                             │
   │    [gen] batch.done completions=X dur_ms=Y                  │
   ├─────────────────────────────────────────────────────────────┤
   │ 3. Process Rollouts                                         │
   │    → compute advantages                                     │
   │    [rollout] trunc_pct=X staleness=Y                        │
   ├─────────────────────────────────────────────────────────────┤
   │ 4. Prepare Batch                                            │
   │    → serialize for trainer                                  │
   │    → write to /rollouts/step_N/rank_*.pt                    │
   ├─────────────────────────────────────────────────────────────┤
   │ 5. Trainer Consumes                                         │
   │    → load batch                                             │
   │    → forward/backward                                       │
   │    → update model                                           │
   │    [ckpt] write.done step=N+1                               │
   └─────────────────────────────────────────────────────────────┘
        │                                                      │
        └──────────────────────────────────────────────────────┘
                        Loop back to step 1
```

## Log Flow Timeline (Example)

```
Time    Component           Event
------  ------------------  ----------------------------------------------
0.000   Orchestrator        Step 5 begins
0.001   Orchestrator        progress.step=5, ckpt_step=3, async_level=2
0.002   Orchestrator        Hit async barrier (5-3 > 2)
0.003   Orchestrator        [ckpt] wait.start target_step=3
0.234   Trainer             [ckpt] write.done step=3 write_ms=234.5
0.235   Orchestrator        [ckpt] wait.done target_step=3 wait_ms=232.1
0.236   Orchestrator        [weights][abc123] client.send t=0.236
0.237   API Server          [weights][abc123] server.recv t=0.237
0.238   API Server          [weights][abc123] rpc.start t=0.238
1.456   vLLM Engine         (loads checkpoint from disk)
1.457   API Server          [weights][abc123] rpc.done t=1.457 rpc_ms=1219.0
1.458   Orchestrator        [weights][abc123] client.done wall_ms=1222.0 rpc_ms=1219.0 queue_ms=3.0
1.459   Orchestrator        [gen] batch.start inflight=96 target_batch=64
3.567   vLLM Engine         (generates 64 completions)
3.568   Orchestrator        [gen] batch.done completions=64 dur_ms=2109.0 inflight=96
3.569   Orchestrator        [rollout] trunc_pct=5.2 current_step=5 ckpt_step=3 staleness=2
3.570   Orchestrator        Step 5 complete (wall=3.570s)
```

## Metrics Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                         Metrics Collection                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Client-Side (Orchestrator):                                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ wall_ms   = Total time (client.send → client.done)         │  │
│  │ queue_ms  = Queuing delay (computed from wall_ms - rpc_ms) │  │
│  │ wait_ms   = Checkpoint wait time                           │  │
│  │ dur_ms    = Batch generation time                          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Server-Side (API Server):                                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ rpc_ms    = Engine operation time (rpc.start → rpc.done)   │  │
│  │ (Returned in HTTP response payload)                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Trainer-Side:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ write_ms  = Checkpoint write time                           │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Quality Metrics:                                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ trunc_pct = % truncated completions                         │  │
│  │ staleness = progress.step - ckpt_step                       │  │
│  │ inflight  = Concurrent streaming requests                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                      Analysis (analyze_logs.sh)                     │
├────────────────────────────────────────────────────────────────────┤
│  Computes: mean, stddev, min, max for each metric                  │
│  Identifies: bottleneck signatures                                 │
│  Produces:   SUMMARY.md with diagnosis                             │
└────────────────────────────────────────────────────────────────────┘
```

## File Organization

```
prime-rl/
├── src/prime_rl/
│   ├── orchestrator/
│   │   ├── client.py              ← Modified: +timing +admin_client
│   │   └── orchestrator.py        ← Modified: +gen/ckpt/rollout logs
│   ├── inference/
│   │   └── vllm/
│   │       └── server.py          ← Modified: +middleware +endpoints
│   └── trainer/
│       └── rl/
│           └── train.py           ← Modified: +ckpt write timing
│
└── bottleneck_logs/               ← New: Documentation & tools
    ├── INDEX.md                   → Navigation hub
    ├── INVESTIGATION_SUMMARY.md   → Technical overview
    ├── README.md                  → Comprehensive reference
    ├── DEPLOYMENT_GUIDE.md        → Testing instructions
    ├── LOG_PATTERNS.md            → Log examples
    ├── COMMANDS.md                → Command reference
    ├── ARCHITECTURE.md            → This file
    ├── CHANGES_SUMMARY.txt        → Change log
    ├── analyze_logs.sh            → Analysis script
    │
    └── analysis_<timestamp>/      ← Generated by analyze_logs.sh
        ├── SUMMARY.md
        ├── queue_ms_stats.txt
        ├── rpc_ms_stats.txt
        └── ...
```

## Data Flow Summary

```
┌─────────┐      ┌──────────┐      ┌─────────┐      ┌──────────┐
│ Trainer │─────>│Checkpoint│─────>│ Inference│─────>│Completion│
│         │      │  Files   │      │  Server │      │  Batch   │
└─────────┘      └──────────┘      └─────────┘      └──────────┘
     │                │                  │                │
     │[ckpt]          │[ckpt]           │[weights]       │[gen]
     │write.done      │wait.done        │client.done     │batch.done
     │                │                  │                │
     └────────────────┴──────────────────┴────────────────┘
                              │
                              ↓
                    ┌────────────────────┐
                    │  Orchestrator Logs │
                    │  → analyze_logs.sh │
                    │  → SUMMARY.md      │
                    └────────────────────┘
```
