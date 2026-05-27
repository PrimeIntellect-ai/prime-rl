"""Deterministic state proofs for vLLM DP pause deadlock candidates.

This is intentionally a small control-flow model, not a vLLM import test. It
captures the relevant EngineCore state transitions from:

- prime-rl's current monkey patch: gate START_DP_WAVE while paused.
- vLLM PR 39366: two-phase pause with local pending_pause and dummy stepping
  until the existing DP sync observes all pending_pause flags.

Run with:
    uv run --no-project python scripts/research_pause_deadlock.py
"""

from dataclasses import dataclass


@dataclass
class Rank:
    rank: int
    pause_state: str = "UNPAUSED"
    engines_running: bool = False
    pending_pause: bool = False
    ignore_start_dp_wave: bool = False
    waiting_in: str | None = None
    pause_future_done: bool = False


def prove_current_patch_deadlock() -> list[str]:
    ranks = [Rank(0), Rank(1)]
    trace: list[str] = []

    ranks[0].pause_state = "PAUSED_ALL"
    ranks[0].pause_future_done = True
    trace.append("rank0 handles pause_keep while idle: PAUSED_ALL, engines_running=False, /pause returns")

    trace.append("rank1 is still unpaused or receives a late request and starts a DP wave")
    ranks[1].engines_running = True

    trace.append("coordinator broadcasts START_DP_WAVE to rank0")
    if ranks[0].pause_state == "UNPAUSED":
        ranks[0].engines_running = True
    else:
        trace.append("prime patch ignores START_DP_WAVE because rank0 is paused")

    ranks[1].waiting_in = "model_or_dp_collective"
    participants = {rank.rank for rank in ranks if rank.waiting_in == "model_or_dp_collective"}
    expected = {rank.rank for rank in ranks}

    trace.append(f"collective participants={sorted(participants)}, expected={sorted(expected)}")
    assert participants != expected
    trace.append("deadlock: rank1 waits for rank0 in the collective; rank0 is paused and will not enter")
    return trace


def prove_two_phase_subset_deadlock(dp_size: int = 32, delivered_pause_count: int = 31) -> list[str]:
    ranks = [Rank(rank) for rank in range(dp_size)]
    trace: list[str] = []

    for rank in ranks[:delivered_pause_count]:
        rank.pause_state = "PAUSED_ALL"
        rank.pending_pause = True
        rank.engines_running = True

    trace.append(
        f"upstream PR 39366 pause delivered to ranks 0..{delivered_pause_count - 1}; "
        f"rank{delivered_pause_count} has not processed pause"
    )
    trace.append("each delivered rank locally sets pending_pause=True and engines_running=True")

    for rank in ranks[:delivered_pause_count]:
        rank.waiting_in = "dummy_model_collective_before_pause_consensus"

    participants = {
        rank.rank for rank in ranks if rank.waiting_in == "dummy_model_collective_before_pause_consensus"
    }
    expected = {rank.rank for rank in ranks}

    trace.append(f"dummy collective participants={len(participants)} ranks, expected={len(expected)} ranks")
    assert participants != expected

    trace.append(
        "deadlock: delivered ranks are blocked before they can reach sync_dp_state; "
        "their pause futures cannot resolve"
    )
    trace.append(
        f"rank{delivered_pause_count} is still idle in input_queue.get(), so it never joins the dummy collective"
    )
    trace.append(
        "the HTTP/controller side waits for all /pause calls; the engine side waits for the missing rank"
    )
    return trace


def main() -> None:
    cases = {
        "current prime-rl patch": prove_current_patch_deadlock(),
        "upstream PR 39366 two-phase pause": prove_two_phase_subset_deadlock(),
    }
    for name, trace in cases.items():
        print(f"\n{name}")
        for step, line in enumerate(trace, start=1):
            print(f"{step}. {line}")


if __name__ == "__main__":
    main()
