from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import branch_rollout, interleave_rollout


@pytest.fixture
def single_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            )
        ],
    )
    return state


@pytest.fixture
def multi_step_trajectory_state():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    prompt_logprobs=[-0.01, -0.02, -0.1, -0.2, 0.0, 0.0],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )
    return state


@pytest.fixture
def multi_step_trajectory_with_tool_calls():
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1 + TC1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1 + TC1"},
                    {"role": "tool", "tool_call_id": "TR1", "content": "TR1"},
                ],
                completion=[{"role": "assistant", "content": "A2 + TC2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    prompt_logprobs=[-0.01, -0.02, -0.1, -0.2, 0.0, 0.0],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
        reward=1.0,
        advantage=None,
        stop_condition=None,
        metrics={"has_error": 0.0, "tool_calls": 1.0},
    )
    return state


def test_branching_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = branch_rollout(single_step_trajectory_state)

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]
    assert rollout["completion_ids"] == [3, 4]
    assert rollout["completion_mask"] == [1, 1]
    assert rollout["completion_logprobs"] == [-0.1, -0.2]


def test_branching_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = branch_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]
    assert rollout["completion_ids"] == [3, 4]
    assert rollout["completion_mask"] == [1, 1]
    assert rollout["completion_logprobs"] == [-0.1, -0.2]

    # second step
    rollout = rollouts[1]
    assert rollout["prompt_ids"] == [1, 2, 3, 4, 5, 6]
    assert rollout["prompt_mask"] == [0, 0, 0, 0, 0, 0]
    assert rollout["completion_ids"] == [7, 8]
    assert rollout["completion_mask"] == [1, 1]
    assert rollout["completion_logprobs"] == [-0.3, -0.4]


def test_branching_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = branch_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 2

    # first step
    rollout = rollouts[0]
    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]
    assert rollout["completion_ids"] == [3, 4]
    assert rollout["completion_mask"] == [1, 1]
    assert rollout["completion_logprobs"] == [-0.1, -0.2]

    # second step
    rollout = rollouts[1]
    assert rollout["prompt_ids"] == [1, 2, 3, 4, 5, 6]
    assert rollout["prompt_mask"] == [0, 0, 0, 0, 0, 0]
    assert rollout["completion_ids"] == [7, 8]
    assert rollout["completion_mask"] == [1, 1]
    assert rollout["completion_logprobs"] == [-0.3, -0.4]


def test_interleave_rollout_single_step_trajectory(single_step_trajectory_state):
    rollouts = interleave_rollout(single_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]
    assert rollout["completion_ids"] == [3, 4]
    assert rollout["completion_mask"] == [1, 1]
    assert rollout["completion_logprobs"] == [-0.1, -0.2]


def test_interleave_rollout_multi_step_trajectory(multi_step_trajectory_state):
    rollouts = interleave_rollout(multi_step_trajectory_state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]
    assert rollout["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert rollout["completion_mask"] == [1, 1, 0, 0, 1, 1]
    # Uses prompt_logprobs[2:] + completion_logprobs
    assert rollout["completion_logprobs"] == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]


def test_interleave_rollout_multi_step_trajectory_with_tool_calls(multi_step_trajectory_with_tool_calls):
    rollouts = interleave_rollout(multi_step_trajectory_with_tool_calls)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]
    assert rollout["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert rollout["completion_mask"] == [1, 1, 0, 0, 1, 1]
    # Uses prompt_logprobs[2:] + completion_logprobs
    assert rollout["completion_logprobs"] == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]


# ============================================================================
# BASELINE TESTS: Demonstrate the re-tokenization bug
# These tests show the current BROKEN behavior when tokens change across turns.
# ============================================================================


@pytest.fixture
def multi_step_with_token_mismatch():
    """
    Multi-turn trajectory where re-tokenization changed token IDs.

    Turn 1: completion tokens [3, 4] with logprobs [-0.1, -0.2]
    Turn 2: history re-tokenized to [1, 2, 99, 4, 5, 6] (token 3 → 99!)

    This simulates BPE re-tokenization changing token boundaries.
    """
    state = vf.State(
        example_id="test_mismatch_001",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],  # Original tokenization
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],  # Logprobs for tokens [3, 4]
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    # Re-tokenized: token 3 became 99 due to BPE context change!
                    prompt_ids=[1, 2, 99, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )
    return state


def test_baseline_current_behavior_with_token_mismatch(multi_step_with_token_mismatch):
    """
    BASELINE TEST: Shows that missing prompt_logprobs raises error.

    Without prompt_logprobs, multi-turn trajectories cannot be processed correctly
    due to re-tokenization issues. The code now raises ValueError to surface this.
    """
    # Without prompt_logprobs, should raise ValueError
    with pytest.raises(ValueError, match="prompt_logprobs not available"):
        interleave_rollout(multi_step_with_token_mismatch)


def test_baseline_mismatch_detection():
    """
    BASELINE TEST: Verify that multi-turn with prompt_logprobs works correctly.

    With prompt_logprobs available, interleave_rollout uses the aligned
    data from the final turn for correct importance ratio computation.
    """
    # Create a state with exact token match and prompt_logprobs
    state_matching = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    # Tokens MATCH: [1, 2, 3, 4] prefix is same as turn 1
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    # prompt_logprobs aligned with prompt_ids
                    prompt_logprobs=[-0.01, -0.02, -0.1, -0.2, 0.0, 0.0],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )

    # With prompt_logprobs available, interleave_rollout works correctly
    rollouts = interleave_rollout(state_matching)
    rollout = rollouts[0]

    # Uses prompt_logprobs[2:] + completion_logprobs
    assert rollout["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert rollout["completion_logprobs"] == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]


# ============================================================================
# FIX TESTS: Verify the fix using prompt_logprobs
# These tests verify CORRECT behavior when prompt_logprobs are available.
# ============================================================================


@pytest.fixture
def multi_step_with_prompt_logprobs():
    """
    Multi-turn trajectory WITH prompt_logprobs for testing the fix.

    Turn 1: completion tokens [3, 4] with logprobs [-0.1, -0.2]
    Turn 2: Has prompt_logprobs for entire prompt including re-tokenized history
            prompt_logprobs: [lp1, lp2, lp3, lp4, lp5, lp6] aligned with prompt_ids

    The key: prompt_logprobs are aligned with prompt_ids from turn 2!
    """
    state = vf.State(
        example_id="test_prompt_logprobs_001",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                    # First turn may not have prompt_logprobs
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    # Has prompt_logprobs from vLLM!
                    prompt_logprobs=[-0.05, -0.15, -0.12, -0.18, -0.25, -0.35],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )
    return state


@pytest.fixture
def multi_step_with_token_mismatch_and_prompt_logprobs():
    """
    Multi-turn trajectory with re-tokenization AND prompt_logprobs.

    Turn 1: completion tokens [3, 4] with logprobs [-0.1, -0.2]
    Turn 2: history re-tokenized to [1, 2, 99, 4, 5, 6] (token 3 → 99!)
            prompt_logprobs: [lp1, lp2, lp_99, lp4, lp5, lp6] aligned with [1, 2, 99, 4, 5, 6]

    This is the KEY test case - the fix should use:
    - Token IDs from final turn's prompt (99, not 3!)
    - Logprobs from final turn's prompt_logprobs (aligned with 99)
    """
    state = vf.State(
        example_id="test_mismatch_with_fix_001",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],
                    prompt_mask=[0, 0],
                    completion_ids=[3, 4],  # Original tokenization
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],  # Logprobs for tokens [3, 4]
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    # Re-tokenized: token 3 became 99 due to BPE context change!
                    prompt_ids=[1, 2, 99, 4, 5, 6],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[7, 8],
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                    # prompt_logprobs aligned with re-tokenized prompt_ids!
                    # Position 2 has logprob for token 99, not token 3
                    prompt_logprobs=[-0.05, -0.15, -0.55, -0.18, -0.25, -0.35],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )
    return state


def test_fix_uses_prompt_logprobs(multi_step_with_prompt_logprobs):
    """
    FIX TEST: Verify that multi-turn uses prompt_logprobs from final turn.

    When prompt_logprobs are available, the fix should:
    1. Use token IDs from final turn's prompt (for past turns)
    2. Use prompt_logprobs for past turn logprobs (aligned!)
    3. Use completion_logprobs for final turn's completion
    """
    rollouts = interleave_rollout(multi_step_with_prompt_logprobs)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    # Prompt should be first turn's prompt
    assert rollout["prompt_ids"] == [1, 2]
    assert rollout["prompt_mask"] == [0, 0]

    # Completion should be: [past_assistant + user2 + final_completion]
    # From final turn's prompt[2:] + final turn's completion
    assert rollout["completion_ids"] == [3, 4, 5, 6, 7, 8]

    # Logprobs should use prompt_logprobs for positions 0-3, completion_logprobs for 4-5
    # prompt_logprobs[2:6] = [-0.12, -0.18, -0.25, -0.35] for tokens [3, 4, 5, 6]
    # completion_logprobs = [-0.3, -0.4] for tokens [7, 8]
    expected_logprobs = [-0.12, -0.18, -0.25, -0.35, -0.3, -0.4]
    assert rollout["completion_logprobs"] == expected_logprobs

    # Mask: assistant tokens (positions 0,1) = 1, user tokens (2,3) = 0, assistant (4,5) = 1
    assert rollout["completion_mask"] == [1, 1, 0, 0, 1, 1]


def test_fix_handles_token_mismatch_correctly(multi_step_with_token_mismatch_and_prompt_logprobs):
    """
    FIX TEST: Verify correct handling when tokens change AND prompt_logprobs available.

    This is the KEY test that shows the fix works:
    - Turn 1 had token 3, turn 2 re-tokenized to 99
    - With prompt_logprobs, we use the ALIGNED data from turn 2

    After fix:
    - completion_ids[0] = 99 (from final turn's re-tokenized prompt)
    - completion_logprobs[0] = -0.55 (logprob for token 99 from prompt_logprobs)

    The importance ratio will now be computed correctly because both
    token ID and logprob are from the same tokenization run.
    """
    rollouts = interleave_rollout(multi_step_with_token_mismatch_and_prompt_logprobs)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    # Prompt should be first turn's prompt
    assert rollout["prompt_ids"] == [1, 2]

    # Completion IDs should use FINAL turn's tokenization (99, not 3!)
    # completion_ids = final_prompt_ids[2:] + final_completion_ids
    #                = [99, 4, 5, 6] + [7, 8]
    assert rollout["completion_ids"] == [99, 4, 5, 6, 7, 8]

    # Completion logprobs should use prompt_logprobs (aligned with 99!)
    # prompt_logprobs[2:6] = [-0.55, -0.18, -0.25, -0.35]
    # completion_logprobs = [-0.3, -0.4]
    expected_logprobs = [-0.55, -0.18, -0.25, -0.35, -0.3, -0.4]
    assert rollout["completion_logprobs"] == expected_logprobs

    # This is the KEY difference from baseline:
    # - Baseline: completion_ids[0] = 3, completion_logprobs[0] = -0.1 (WRONG!)
    # - Fix: completion_ids[0] = 99, completion_logprobs[0] = -0.55 (CORRECT!)
    #
    # Now importance_ratio = trainer_logprobs[0] - inference_logprobs[0]
    #                     = logP(99 | context) - (-0.55)
    # Both refer to the SAME token 99!


def test_fix_raises_error_without_prompt_logprobs(multi_step_with_token_mismatch):
    """
    FIX TEST: Verify error raised when prompt_logprobs not available for multi-turn.

    Multi-turn REQUIRES prompt_logprobs for correct importance ratio computation.
    If not available, raise ValueError to surface the misconfiguration early.
    """
    # multi_step_with_token_mismatch has no prompt_logprobs
    with pytest.raises(ValueError, match="prompt_logprobs not available"):
        interleave_rollout(multi_step_with_token_mismatch)


@pytest.fixture
def three_turn_trajectory_with_prompt_logprobs():
    """
    Three-turn trajectory to test more complex scenarios.

    Turn 1: User asks question, model responds
    Turn 2: User follow-up, model responds with tool call
    Turn 3: Tool result, model final response

    Tests that mask is computed correctly for all turns.
    """
    state = vf.State(
        example_id="test_three_turn_001",
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2],  # 2 tokens
                    prompt_mask=[0, 0],
                    completion_ids=[10, 11],  # 2 tokens
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.1, -0.2],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    # 2 + 2 + 2 = 6 prompt tokens
                    prompt_ids=[1, 2, 10, 11, 20, 21],
                    prompt_mask=[0, 0, 0, 0, 0, 0],
                    completion_ids=[30, 31],  # 2 tokens
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.3, -0.4],
                    overlong_prompt=False,
                    is_truncated=False,
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                    {"role": "tool", "tool_call_id": "tc1", "content": "TR"},
                ],
                completion=[{"role": "assistant", "content": "A3"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    # 6 + 2 + 2 = 10 prompt tokens
                    prompt_ids=[1, 2, 10, 11, 20, 21, 30, 31, 40, 41],
                    prompt_mask=[0] * 10,
                    completion_ids=[50, 51],  # 2 tokens
                    completion_mask=[1, 1],
                    completion_logprobs=[-0.5, -0.6],
                    overlong_prompt=False,
                    is_truncated=False,
                    # prompt_logprobs for all 10 prompt tokens
                    prompt_logprobs=[-0.01, -0.02, -0.11, -0.12, -0.21, -0.22, -0.31, -0.32, -0.41, -0.42],
                ),
                reward=None,
                advantage=None,
                extras={},
            ),
        ],
    )
    return state


def test_fix_three_turn_trajectory(three_turn_trajectory_with_prompt_logprobs):
    """
    FIX TEST: Verify correct handling of 3-turn trajectory.

    Structure:
    - prompt_ids: [1, 2] (first turn's prompt)
    - completion_ids: [10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
      - [10, 11]: Turn 1 assistant (mask=1)
      - [20, 21]: Turn 2 user (mask=0)
      - [30, 31]: Turn 2 assistant (mask=1)
      - [40, 41]: Turn 3 tool result (mask=0)
      - [50, 51]: Turn 3 assistant (mask=1)
    """
    rollouts = interleave_rollout(three_turn_trajectory_with_prompt_logprobs)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    # Prompt should be first turn's prompt
    assert rollout["prompt_ids"] == [1, 2]

    # Completion should be everything after first prompt
    # final_prompt_ids[2:] + final_completion_ids
    assert rollout["completion_ids"] == [10, 11, 20, 21, 30, 31, 40, 41, 50, 51]

    # Logprobs from prompt_logprobs[2:] + completion_logprobs
    expected_logprobs = [-0.11, -0.12, -0.21, -0.22, -0.31, -0.32, -0.41, -0.42, -0.5, -0.6]
    assert rollout["completion_logprobs"] == expected_logprobs

    # Mask: assistant tokens get 1, user/tool tokens get 0
    # [10, 11] = assistant turn 1 = [1, 1]
    # [20, 21] = user turn 2 = [0, 0]
    # [30, 31] = assistant turn 2 = [1, 1]
    # [40, 41] = tool result = [0, 0]
    # [50, 51] = assistant turn 3 = [1, 1]
    expected_mask = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    assert rollout["completion_mask"] == expected_mask


def test_fix_single_turn_unchanged():
    """
    FIX TEST: Single-turn trajectories should work the same as before.

    Single-turn has no re-tokenization issue, so behavior should be identical.
    """
    state = vf.State(
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=vf.TrajectoryStepTokens(
                    prompt_ids=[1, 2, 3],
                    prompt_mask=[0, 0, 0],
                    completion_ids=[4, 5, 6],
                    completion_mask=[1, 1, 1],
                    completion_logprobs=[-0.1, -0.2, -0.3],
                    overlong_prompt=False,
                    is_truncated=False,
                    # Even if prompt_logprobs is present, single-turn uses completion_logprobs
                    prompt_logprobs=[-0.01, -0.02, -0.03],
                ),
                reward=None,
                advantage=None,
                extras={},
            )
        ],
    )

    rollouts = interleave_rollout(state)
    assert len(rollouts) == 1
    rollout = rollouts[0]

    # Should use original behavior (completion_logprobs, not prompt_logprobs)
    assert rollout["prompt_ids"] == [1, 2, 3]
    assert rollout["completion_ids"] == [4, 5, 6]
    assert rollout["completion_logprobs"] == [-0.1, -0.2, -0.3]
    assert rollout["completion_mask"] == [1, 1, 1]
