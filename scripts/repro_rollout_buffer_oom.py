# Measure orchestrator rollout-buffer memory per buffered rollout.
# Run from the prime-rl repo root:
#   uv run python scripts/repro_rollout_buffer_oom.py full       # raw + samples retained (pre-fix behavior)
#   uv run python scripts/repro_rollout_buffer_oom.py samples    # samples only (simulates freeing `raw`)
#   uv run python scripts/repro_rollout_buffer_oom.py fixed      # what the sink holds today (compact + stripped)
#   uv run python scripts/repro_rollout_buffer_oom.py fixed --no-replay
# Each mode runs in its own process: RSS freed back to glibc arenas never returns
# to the kernel, so freeing inside one process measures nothing.
import base64
import gc
import os
import random
import sys

import numpy as np

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import RoutedExperts


def rss_gib():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024**2


N_ROLLOUTS = 20  # scale factor; incident had ~3,800 held rollouts
TOKENS = 120_000  # per trajectory (Scale-SWE rollouts ran 100-150k)
STEPS = 40  # agent turns -> per-step raw entries
MOE_LAYERS, TOPK = 75, 8  # GLM-5.1: 78 layers, 3 dense, top-k 8

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"  # "full" | "samples" | "fixed"
REPLAY = "--no-replay" not in sys.argv


def fake_rollout():
    per_step = TOKENS // STEPS
    # --- what `rollout.raw` holds after the env returns (never freed pre-fix) ---
    raw_traj = []
    for _ in range(STEPS):
        re_np = np.random.randint(0, 256, (per_step, MOE_LAYERS, TOPK), dtype=np.uint8)
        raw_traj.append(
            {
                "tokens": {
                    "prompt_ids": [random.randint(1000, 150000) for _ in range(per_step // 4)],
                    "completion_ids": [random.randint(1000, 150000) for _ in range(per_step)],
                    "completion_logprobs": [random.random() for _ in range(per_step)],
                    "routed_experts": {  # stays base64 in raw, exactly like the wire format
                        "data": base64.b64encode(re_np.tobytes()).decode(),
                        "shape": list(re_np.shape),
                        "dtype": "uint8",
                    },
                },
                "messages": [{"role": "assistant", "content": "x" * 2000}],
            }
        )
    # --- what interleave_rollout builds (boxed lists + decoded bytes) ---
    re_full = np.random.randint(0, 256, (TOKENS, MOE_LAYERS, TOPK), dtype=np.uint8)
    routed = RoutedExperts(data=re_full.tobytes(), shape=list(re_full.shape), dtype="uint8") if REPLAY else None
    sample = TrainingSample(
        prompt_ids=[random.randint(1000, 150000) for _ in range(TOKENS // 8)],
        prompt_mask=[True] * (TOKENS // 8),
        completion_ids=[random.randint(1000, 150000) for _ in range(TOKENS)],
        completion_mask=[True] * TOKENS,
        completion_logprobs=[random.random() for _ in range(TOKENS)],
        completion_temperatures=[],
        env_name="multiswe",
        routed_experts=routed,
    )
    if MODE == "samples":
        raw_traj = None  # simulate freeing `raw` after tokenize
    if MODE == "fixed":
        # What the sink holds after the fixes: per-token fields compacted to
        # numpy, per-step token payloads stripped to count stubs (messages
        # kept for sample logging), last step's ids stashed compactly,
        # temperatures fanned out as in process_group.
        from prime_rl.orchestrator.train_sink import compact_sample
        from prime_rl.orchestrator.trajectories import strip_trajectory_token_payloads

        compact_sample(sample)
        sample.completion_temperatures = np.full(len(sample.completion_ids), 0.8)
        raw = {"trajectory": raw_traj}
        strip_trajectory_token_payloads(raw)
        raw_traj = raw["trajectory"]
    return {"raw": raw_traj, "samples": [sample]}


gc.collect()
base = rss_gib()
buffer = [fake_rollout() for _ in range(N_ROLLOUTS)]
gc.collect()
full = rss_gib()

per = (full - base) / N_ROLLOUTS
label = MODE + ("" if REPLAY else "+noreplay")
print(f"{label:16s}: {per * 1024:7.1f} MiB/rollout -> {per * 3800:7.0f} GiB at 3,800 held rollouts (incident scale)")
