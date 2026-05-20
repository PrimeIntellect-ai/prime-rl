# TTT Probes

Four small, single-purpose benchmarks the cluster session should run
**before** writing Phase A code, so the design decisions in the
implementation plan are validated against actual cluster hardware, not
my laptop guesses.

All probes are sized for Qwen3-4B-Instruct-2507 on a single H200, and
each should take under 10 minutes of compute. They can slot into a
gap between Forth runs.

**Run order matters** — probe 1's answer changes how probe 2 should be
shaped (in-memory vs disk paths).

---

## Probe 1: vLLM adapter snapshot churn

**Why**: TTT generates ~32 distinct adapter snapshots per rollout
× ~1000 in-flight rollouts = up to 16k unique adapters across a step.
If load/unload latency per adapter is anything like seconds, disk
transport is dead-on-arrival and we go to shm immediately (Phase C
becomes mandatory in Phase A).

**Target**: p50 load time, p99 load time, and how those scale with
the number of distinct adapters loaded so far (LRU cache cold vs warm).

**Setup**:
1. Boot a single vLLM server with Qwen3-4B-Instruct-2507, `enable_lora=true`,
   `max_lora_rank=16`, `max_loras=8` (small on purpose; we want to see
   LRU eviction behavior).
2. Pre-generate 200 random LoRA adapters of rank 8 against the same
   base, each written to a distinct path under
   `/beegfs/sebastian/ttt-probes/adapters/`. Each adapter file should
   be the standard PEFT `adapter_model.safetensors` shape.
3. For i in 0..200: send a 1-token generation request with
   `lora_request=LoRARequest(lora_name=f"a{i}", lora_int_id=i+1,
   lora_path=...)`. Time end-to-end and from server-side timestamps.

**Pass criteria**:
- p50 < 100 ms ⇒ disk transport is fine for Phase A.
- p50 100–500 ms ⇒ disk is workable but reaper must be aggressive.
- p50 > 500 ms ⇒ design needs shm transport before any smoke run.

**Probe**:
```python
# scripts/probe_1_adapter_churn.py — sketch, run on inference node
import time, json, requests, statistics
from pathlib import Path

BASE_URL = "http://localhost:8000"
ADAPTERS_DIR = Path("/beegfs/sebastian/ttt-probes/adapters")
N = 200

# Pre-generate adapters: see scripts/probe_make_adapters.py
adapters = sorted(ADAPTERS_DIR.iterdir())[:N]
assert len(adapters) == N

load_times, gen_times = [], []
for i, adir in enumerate(adapters):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/load_lora_adapter",
                      json={"lora_name": f"a{i}", "lora_path": str(adir)})
    r.raise_for_status()
    t1 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/v1/completions",
                      json={"model": f"a{i}", "prompt": "Hello",
                            "max_tokens": 1, "temperature": 0})
    r.raise_for_status()
    t2 = time.perf_counter()
    load_times.append((t1 - t0) * 1000)
    gen_times.append((t2 - t1) * 1000)

def stats(xs):
    xs = sorted(xs)
    return {"p50": xs[len(xs)//2],
            "p99": xs[int(len(xs)*0.99)],
            "max": xs[-1]}

print(json.dumps({"load_ms": stats(load_times),
                  "gen_first_token_ms": stats(gen_times),
                  "n_cold": min(8, N),
                  "n_after_eviction": max(0, N-8)}, indent=2))
```

---

## Probe 2: learner forward+backward latency at chunk size

**Why**: TTT learner trains 1024-token chunks. If forward+backward
per chunk takes longer than the between-chunk generation interval,
the learner is the bottleneck and we need batched multi-tenant
training (Phase D) sooner than I'd planned.

**Target**: median wall-clock for one `(forward, backward, optimizer
step)` cycle on a 1024-token chunk, with a rank-8 LoRA against
Qwen3-4B-Instruct-2507.

**Setup**:
1. Single H200, no FSDP needed (Qwen3-4B fits in bf16 + adam easily).
2. Load Qwen3-4B with `MultiLoRALinear` wrap, n_adapters=1, rank=8,
   alpha=16, target_modules = trainer defaults restricted to
   `[q,k,v,o,gate,up,down]_proj`.
3. Freeze base, train LoRA only.
4. Loop 50 times: random 1024-token input, forward, cross-entropy
   against shifted targets, backward, step, zero_grad. Time each
   iteration after a 5-iter warmup.

**Pass criteria**:
- Median < 200 ms ⇒ comfortable; chunk training fits in idle time.
- Median 200–800 ms ⇒ workable; learner concurrency matters earlier.
- Median > 800 ms ⇒ rethink chunk size or move to Phase D first.

For reference: at production train sampling
`max_completion_tokens=8192`, a typical chunk-aligned generation gap
is ~chunk_tokens / chunk_throughput. On H200 at moderate concurrency,
vLLM does maybe 5–15k tokens/s aggregate for Qwen3-4B; so the gap
between consecutive 1024-token chunks for one rollout is well under a
second. Learner needs to keep up.

---

## Probe 3: multi-LoRA forward overhead vs adapter count

**Why**: Trainer-side replay uses `MultiLoRALinear` per-token routing
to handle multiple chunk adapters in one forward. The cost of the
`_grouped_mm` path grows with `n_adapters` (stack cost on lora_A /
lora_B is the dominant term). This bounds how many distinct chunks
can sit in one replay microbatch.

**Target**: forward latency on a fixed-size batch (say 8192 tokens
flat) as a function of `n_adapters ∈ {1, 8, 64, 256, 1024}`. Both the
`use_grouped_mm=True` path and the for-loop fallback.

**Setup**:
1. Single H200.
2. Load Qwen3-4B with `MultiLoRALinear` wrap, n_adapters as the swept
   variable, rank=8.
3. Set `lora_num_tokens` such that ~equal tokens go to each adapter
   (or all to one for the "warm path" reference).
4. Time forward pass 50 times after warmup, median across runs.

**Pass criteria**:
- n_adapters=64 overhead < 1.5x vs n_adapters=1 ⇒ comfortable; can
  hold per-chunk adapters across a wide microbatch.
- n_adapters=64 overhead 1.5–3x ⇒ limit microbatch to ~16 distinct
  chunks; trainer groups microbatches by adapter set.
- n_adapters=64 overhead > 3x ⇒ replay must serialize chunks (one
  adapter at a time); plan changes.

**Probe**:
```python
# scripts/probe_3_multilora_forward.py — sketch
import torch, time, statistics
from prime_rl.trainer.models.layers.lora import MultiLoRALinear, set_lora_num_tokens, set_multilora_scaling

# build a dummy nn.Linear -> wrap it
base = torch.nn.Linear(4096, 4096, device="cuda", dtype=torch.bfloat16)
for n_adapters in [1, 8, 64, 256, 1024]:
    set_lora_num_tokens(None, reset_reference=True)
    set_multilora_scaling(None, reset_reference=True)
    set_lora_num_tokens(torch.zeros(n_adapters, dtype=torch.int32, device="cuda"),
                        reset_reference=True)
    set_multilora_scaling(torch.ones(n_adapters, dtype=torch.bfloat16, device="cuda"),
                          reset_reference=True)
    layer = MultiLoRALinear(base, rank=8, n_adapters=n_adapters,
                            alpha=16, dropout=0.0).cuda().bfloat16()
    # distribute 8192 tokens roughly equally across adapters
    per = 8192 // n_adapters
    counts = torch.full((n_adapters,), per, dtype=torch.int32, device="cuda")
    counts[0] += 8192 - per * n_adapters
    set_lora_num_tokens(counts)
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    # warmup
    for _ in range(5): layer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): layer(x)
    torch.cuda.synchronize()
    print(f"n_adapters={n_adapters}: {(time.perf_counter()-t0)*1000/50:.2f} ms/forward")
```

---

## Probe 4: adapter write throughput to /beegfs

**Why**: Probe 1 measures vLLM-side load. We also need to measure how
fast the learner can *write* a snapshot to `/beegfs`. If write
throughput is bad, we need shm even on the producer side.

**Target**: median time to write one rank-8 adapter state_dict
(safetensors, ~2 MB) to `/beegfs/sebastian/ttt-probes/adapters/N/`,
sustained across 100 writes.

**Setup**:
1. Build a representative state_dict in Python (Qwen3-4B-shaped LoRA
   tensors, rank 8, MLP+attention target_modules).
2. Loop 100 times: write to a fresh path under
   `/beegfs/sebastian/ttt-probes/adapters/`. Time each write.

**Pass criteria**:
- p50 < 20 ms ⇒ disk producer-side is fine.
- p50 20–100 ms ⇒ workable; might want to keep recent snapshots in shm
  and only spill to disk on eviction.
- p50 > 100 ms ⇒ shm needed for producer side too; `/beegfs` is too slow.

**Note**: `/beegfs` is shared with other users (83% full at the time of
writing). Throughput will vary with load.

---

## Probe results template

When each probe finishes, append to `docs/ttt-probe-results.md` (new
file) using this template:

```markdown
## Probe N (YYYY-MM-DD HH:MM UTC)

- Host: <hostname>
- Free GPUs: <number>
- /beegfs free: <GB>
- Result: <numbers>
- Verdict: <which pass criterion was hit>
- Implication for plan: <what changes>
```

Three numbers + one verdict per probe is enough. The plan and
cheat-sheet will be updated accordingly.

---

## What not to do

- **Don't run probes while production Forth runs are using the GPUs.**
  Wait for a maintenance window or a gap between checkpoint saves.
- **Don't probe against the live `feat/sft-on-tool-outputs` checkout.**
  The probes are self-contained Python scripts; they don't need any
  prime-rl branch state.
- **Don't pollute `/beegfs/huggingface/`.** Use
  `/beegfs/sebastian/ttt-probes/` for everything.
- **Don't `git checkout` anything in `/home/sebastian/prime-rl/`.**
  Submodules and uv environment are configured for the live runs.
- **Don't share an adapter `lora_int_id` across probes 1–4.** Even if
  they're the same path, using fresh ids matches the design invariant
  and avoids the HANDOVER-documented poisoning hypothesis.
