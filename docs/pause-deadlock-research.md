# vLLM DP Pause Deadlock Research

## Relevant Code Paths

- Prime-rl patch, now removed: `src/prime_rl/inference/patches.py::monkey_patch_dp_engine_core_pause_resume_deadlock`.
- Upstream two-phase pause: vLLM PR 39366, merged on 2026-04-29.
- Related upstream bug report: vLLM issue 36594.
- Related design RFC: vLLM issue 32103.

## Implementation Baseline

The vLLM implementation starts from the exact local source used to build the
current prime-rl upstream wheel:
`/home/matej/dev/vllm-0.21-revert42434-pr39568` on branch
`feat/v0.21.0-revert42434-pr39568` at commit
`a106aa65919e56afb7b01cf00c48f2d4ef75c6d9`.

That local branch is based on `vllm-project/vllm` `releases/v0.21.0` at
`ad7125a43`, then applies:

1. `584fcdef2` - revert routing replay device-cache/async-D2H behavior.
2. `a106aa659` - routed experts `ModelRunnerOutput` transfer and HTTP support.

The source wheel at
`/home/matej/dev/vllm-0.21-revert42434-pr39568/dist/vllm-0.21.0+cu129.r42434.pr39568.a106aa6-cp38-abi3-manylinux_2_34_x86_64.whl`
hashes to `0f0533a9122dfa738fc0f5532cb6795556f072b133f0058a22ff4be2d2f165f0`,
matching the current upstream prime-rl pin.

The DP pause fix is committed on top of that exact source as
`S1ro1/vllm` branch `fix/dp-pause-coordinator-r42434-pr39568` at commit
`550c363d5`. The prime-rl pin uses the precompiled local wheel:
`/home/matej/dev/vllm-0.21-revert42434-pr39568/dist/vllm-0.21.0+cu129.r42434.pr39568.a106aa6.pause550c363-cp38-abi3-manylinux_2_34_x86_64.whl`.

## Proof 1: Current Prime-rl Patch

The current patch bypasses upstream two-phase pause and gates `START_DP_WAVE`
while the local scheduler is paused:

```python
if not self.engines_running and self.scheduler.pause_state == PauseState.UNPAUSED:
    self.engines_running = True
```

This can deadlock:

1. Rank 0 is idle and handles `pause_keep`.
2. Because it is idle, pause returns with rank 0 in `PAUSED_ALL`,
   `engines_running=False`.
3. Rank 1 is not yet paused, or receives a late request, and starts a DP wave.
4. The coordinator broadcasts `START_DP_WAVE` to rank 0.
5. Rank 0 ignores it because its scheduler is paused.
6. Rank 1 enters a model/DP collective; rank 0 never enters.

The wait-for cycle is:

```text
rank1 waits for rank0 in model/DP collective
rank0 waits paused for resume/update completion and ignores START_DP_WAVE
```

This is the exact reason upstream PR 39366 rejected unconditional
`START_DP_WAVE` ignoring before global pause consensus.

## Proof 2: Upstream PR 39366 Two-phase Pause

PR 39366 fixes the stale wave issue by introducing `pending_pause`,
`ignore_start_dp_wave`, and `sync_dp_state`. However, pause admission is still
rank-local: each engine that receives `pause_scheduler` immediately sets:

```python
self.pending_pause = True
self.engines_running = True
```

The rank then executes dummy/model steps until the every-32-step
`sync_dp_state` observes that every DP rank has `pending_pause=True`.

This can deadlock when pause delivery is skewed:

1. In DP=32, ranks 0..30 process `pause_scheduler`.
2. Rank 31 has not processed its pause request yet.
3. Ranks 0..30 locally set `pending_pause=True` and `engines_running=True`.
4. Before reaching `sync_dp_state`, they enter a dummy model collective.
5. Rank 31 is idle in `input_queue.get()` and never enters that collective.
6. Ranks 0..30 are blocked before they can reach the pause consensus all-reduce.
7. Their pause futures never resolve, so the `/pause` caller also waits.

The wait-for cycle is:

```text
ranks 0..30 wait for rank31 in dummy model/EP/DP collective
rank31 waits for local pause input or remains idle
controller waits for all /pause futures
```

The larger the DP group, the more likely one rank is a tail. Wide EP makes the
pre-consensus dummy model step a high-risk blocking point.

The proof is executable:

```bash
uv run --no-project python scripts/research_pause_deadlock.py
```

## Proposed vLLM-side Fix

Pause should be a DP-coordinator-owned protocol, not N independent local
utility calls that happen to rendezvous in model-loop collectives.

Required invariants:

1. No rank enters a new pause-induced model/EP/DP collective unless every DP
   rank has acknowledged the same pause epoch, or the group was already in a
   running wave that must be drained.
2. `START_DP_WAVE` is ignored only after global pause consensus. Before that,
   paused-intent ranks must still participate in any wave needed to let peers
   reach the safe pause point.
3. `/pause` resolves only after the coordinator knows every DP rank is in the
   same paused epoch.
4. `/resume` is also coordinator-broadcast and acknowledged before any rank
   starts a new wave.

Concrete protocol:

1. Any API server receiving `/pause` sends `PAUSE_REQUEST(epoch, mode,
   clear_cache)` to `DPCoordinatorProc`.
2. The coordinator enters `PAUSING(epoch)`, suppresses or queues new
   `START_DP_WAVE` requests for that epoch, and broadcasts `PAUSE_DP` to all
   engines over the existing coordinator back channel.
3. Each engine sets the scheduler pause state and a pause epoch, then sends a
   `pause_ack(epoch, step_counter, engines_running)` control response.
4. If the coordinator's global engine state is idle, all acks complete pause
   immediately. Idle ranks must not start dummy model batches just to discover
   consensus.
5. If the group is mid-wave, the coordinator orders all ranks to drain to a
   shared safe point, preferably by forcing the next DP state sync immediately
   instead of waiting up to 32 dummy steps.
6. After all ranks report the safe paused state, the coordinator resolves the
   original pause utility future. Weight update collectives can start only
   after this point.
7. `/resume` broadcasts `RESUME_DP(epoch)`, clears pause flags on all ranks,
   waits for all resume acks, then starts exactly one DP wave if any rank has
   queued or unfinished work.

This preserves the current `keep` behavior: requests remain queued/frozen, KV
state is preserved when requested, and new requests are accepted but not
scheduled until resume. The change is only where pause consensus lives: in the
DP control plane instead of in best-effort local utility calls plus dummy model
collectives.
