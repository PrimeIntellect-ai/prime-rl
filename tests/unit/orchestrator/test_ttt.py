"""Unit tests for the TTT compaction-event detector.

The detector is the only self-contained piece of compaction-aligned
TTT that can be exercised without the rest of the (not-yet-built)
TTT machinery — learner service, transport types, trainer-side replay.
These tests pin the structural-signal contract so the parser stays
faithful to the RLM harness's own compaction detection.
"""

from prime_rl.orchestrator.ttt import (
    CompactionEvent,
    augment_rollouts_with_compaction_events,
    detect_compaction_events,
)


def _step(prompt: list[str], completion: list[str]) -> dict:
    """Synthetic trajectory step. Only the lengths matter for detection;
    using strings as message stand-ins keeps the fixtures readable."""
    return {"prompt": prompt, "completion": completion}


def test_empty_trajectory_yields_no_events():
    assert detect_compaction_events([]) == []


def test_single_step_trajectory_yields_no_events():
    # No transitions to inspect — can't compact across zero steps.
    assert detect_compaction_events([_step(["sys", "u1"], ["a1"])]) == []


def test_monotonic_growth_yields_no_events():
    # Each step's prompt extends the prior step's accumulated conversation,
    # which is the non-compaction case. No event should fire.
    traj = [
        _step(["sys", "u1"], ["a1"]),
        _step(["sys", "u1", "a1", "u2"], ["a2"]),
        _step(["sys", "u1", "a1", "u2", "a2", "u3"], ["a3"]),
    ]
    assert detect_compaction_events(traj) == []


def test_single_compaction_event_detected():
    # Step 0 accumulates a conversation of length 3 (prompt) + 1 (completion) = 4.
    # Step 1's prompt of length 2 is shorter than the accumulated 4 → compaction.
    traj = [
        _step(["sys", "u1", "a_prev"], ["assistant_summary"]),
        _step(["sys", "u_framing_with_summary"], ["a2"]),
    ]
    events = detect_compaction_events(traj)
    assert events == [CompactionEvent(step_index=1, pre_compaction_message_count=4)]


def test_multiple_compaction_events_in_one_trajectory():
    # Long rollout that compacts twice. Each compaction shows up as a
    # prompt-shrink relative to the prior step's accumulated length.
    traj = [
        # First batch grows organically.
        _step(["sys", "u1"], ["a1"]),  # acc=3 after
        _step(["sys", "u1", "a1", "u2"], ["a2"]),  # acc=5 after
        _step(["sys", "u1", "a1", "u2", "a2", "u3"], ["a3"]),  # acc=7 after
        # Compaction #1: prompt collapses to system + summary framing.
        _step(["sys", "u_summary_1"], ["a4"]),  # acc=3 after
        _step(["sys", "u_summary_1", "a4", "u5"], ["a5"]),  # acc=5 after
        # Compaction #2: another collapse.
        _step(["sys", "u_summary_2"], ["a6"]),  # acc=3 after
    ]
    events = detect_compaction_events(traj)
    assert events == [
        CompactionEvent(step_index=3, pre_compaction_message_count=7),
        CompactionEvent(step_index=5, pre_compaction_message_count=5),
    ]


def test_equal_length_is_not_a_compaction():
    # If a step's prompt is exactly the accumulated length (no growth, no
    # shrink), that's not a compaction — the strict ``<`` inequality
    # matters. This isn't a realistic case from the harness, but it pins
    # the boundary so a refactor that flips to ``<=`` would fail this
    # test loudly.
    traj = [
        _step(["sys", "u1"], ["a1"]),  # acc = 3
        _step(["sys", "u1", "a1"], []),  # prompt len = 3 = prev_len
    ]
    assert detect_compaction_events(traj) == []


def test_first_step_completion_counted_into_accumulator():
    # Regression check on the seed: the very first step's
    # ``completion`` must count toward the accumulator, otherwise an
    # immediately-compacting second step would be missed.
    traj = [
        _step(["sys"], ["a_long_summary"]),  # acc=2 after (1 prompt + 1 completion)
        _step(["sys"], ["a2"]),  # prompt len = 1 < acc 2 → compaction
    ]
    events = detect_compaction_events(traj)
    assert events == [CompactionEvent(step_index=1, pre_compaction_message_count=2)]


def test_accepts_arbitrary_iterables_not_just_lists():
    # The parser materializes the input via ``list(...)`` so a generator
    # works too — useful when trajectories are streamed from disk.
    def gen():
        yield _step(["sys", "u1"], ["a1"])
        yield _step(["sys", "u_summary"], ["a2"])

    events = detect_compaction_events(gen())
    assert events == [CompactionEvent(step_index=1, pre_compaction_message_count=3)]


def test_augmenter_injects_events_as_plain_dicts():
    # save_rollouts uses json.dump; the augmenter has to emit plain dicts
    # so the JSONL round-trips without needing a CompactionEvent encoder.
    rollouts = [
        {
            "trajectory": [
                _step(["sys", "u1"], ["a1"]),
                _step(["sys", "u_summary"], ["a2"]),
            ]
        }
    ]
    augment_rollouts_with_compaction_events(rollouts)
    assert rollouts[0]["compaction_events"] == [{"step_index": 1, "pre_compaction_message_count": 3}]


def test_augmenter_emits_empty_list_for_non_compacting_rollouts():
    # The augmenter always runs — non-RLM rollouts (no compaction) get an
    # explicit empty list, signaling "no compaction here" to post-hoc
    # tooling. The cost is ~22 JSONL bytes per rollout; the benefit is
    # that downstream consumers don't have to special-case the missing
    # key vs the empty-list case.
    rollouts = [
        {
            "trajectory": [
                _step(["sys", "u1"], ["a1"]),
                _step(["sys", "u1", "a1", "u2"], ["a2"]),
            ]
        }
    ]
    augment_rollouts_with_compaction_events(rollouts)
    assert rollouts[0]["compaction_events"] == []


def test_augmenter_handles_rollout_without_trajectory_key():
    # Defensive: if a rollout dict is malformed (no trajectory key), the
    # augmenter falls back to an empty trajectory rather than raising —
    # crashing the save path because one rollout is malformed would be
    # worse than persisting an empty events list.
    rollouts = [{}]
    augment_rollouts_with_compaction_events(rollouts)
    assert rollouts[0]["compaction_events"] == []


def test_augmenter_mutates_in_place():
    # Single source of truth — the orchestrator passes the same list
    # straight to save_rollouts, no rebinding. Pins the in-place
    # contract.
    rollouts = [{"trajectory": [_step(["sys"], ["a1"])]}]
    original_id = id(rollouts[0])
    augment_rollouts_with_compaction_events(rollouts)
    assert id(rollouts[0]) == original_id
    assert "compaction_events" in rollouts[0]
