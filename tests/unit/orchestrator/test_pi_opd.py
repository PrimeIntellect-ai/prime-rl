import json

from prime_rl.orchestrator.pi_opd import (
    build_pi_prefix,
    compute_pi_advantage_weights,
    extract_tool_calls,
    select_donor_rollout,
)

# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------


def _make_rollout(completion_messages):
    return {"completion": completion_messages}


def test_extract_tool_calls_single():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_web",
                        "arguments": json.dumps({"queries": ["test query"]}),
                    }
                }
            ],
        }
    ]
    result = extract_tool_calls(_make_rollout(messages))
    assert "search_web" in result
    assert "test query" in result


def test_extract_tool_calls_multiple_turns():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search_web", "arguments": '{"queries": ["q1"]}'}},
            ],
        },
        {"role": "tool", "content": "result1"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "scan_page", "arguments": '{"url": "http://example.com"}'}},
            ],
        },
        {"role": "tool", "content": "result2"},
        {"role": "assistant", "content": "Final answer"},
    ]
    result = extract_tool_calls(_make_rollout(messages))
    assert "search_web" in result
    assert "scan_page" in result
    # Tool responses should NOT appear
    assert "result1" not in result
    assert "result2" not in result


def test_extract_tool_calls_no_tools():
    messages = [
        {"role": "assistant", "content": "Just text, no tools"},
    ]
    result = extract_tool_calls(_make_rollout(messages))
    assert result == ""


def test_extract_tool_calls_empty_completion():
    assert extract_tool_calls({"completion": []}) == ""
    assert extract_tool_calls({"completion": None}) == ""
    assert extract_tool_calls({}) == ""


def test_extract_tool_calls_skips_non_assistant():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "tool response", "tool_calls": [{"function": {"name": "fake"}}]},
    ]
    result = extract_tool_calls(_make_rollout(messages))
    assert result == ""


# ---------------------------------------------------------------------------
# select_donor_rollout
# ---------------------------------------------------------------------------


def test_select_donor_picks_winner():
    group = [
        (0, {"reward": 0.0, "id": "loser"}),
        (1, {"reward": 1.0, "id": "winner1"}),
        (2, {"reward": 1.0, "id": "winner2"}),
    ]
    donor = select_donor_rollout(group, current_idx=0)
    assert donor is not None
    assert donor["reward"] == 1.0


def test_select_donor_excludes_self():
    group = [
        (0, {"reward": 1.0, "id": "only_winner"}),
        (1, {"reward": 0.0, "id": "loser"}),
    ]
    # Current idx is the only winner
    donor = select_donor_rollout(group, current_idx=0)
    assert donor is None


def test_select_donor_no_winners():
    group = [
        (0, {"reward": 0.0}),
        (1, {"reward": 0.0}),
    ]
    donor = select_donor_rollout(group, current_idx=0)
    assert donor is None


def test_select_donor_multiple_winners_excludes_self():
    group = [
        (0, {"reward": 1.0, "id": "w0"}),
        (1, {"reward": 1.0, "id": "w1"}),
        (2, {"reward": 1.0, "id": "w2"}),
    ]
    donor = select_donor_rollout(group, current_idx=0)
    assert donor is not None
    assert donor["id"] != "w0"


# ---------------------------------------------------------------------------
# build_pi_prefix
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def test_build_pi_prefix():
    template = "Tool calls:\n{tool_calls}\nEnd."
    tool_calls_text = "search_web(query)"
    ids = build_pi_prefix(tool_calls_text, template, FakeTokenizer())
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# compute_pi_advantage_weights
# ---------------------------------------------------------------------------


def test_weights_uniform_when_equal_logprobs():
    n = 10
    teacher_lps = [-1.0] * n
    student_lps = [-1.0] * n
    mask = [True] * n
    weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=1.0, advantage=1.0)
    assert len(weights) == n
    for w in weights:
        assert abs(w - 1.0) < 1e-6


def test_weights_upweight_teacher_preferred_tokens():
    n = 4
    # Teacher gives higher logprob to first 2 tokens
    teacher_lps = [-0.5, -0.5, -2.0, -2.0]
    student_lps = [-1.0, -1.0, -1.0, -1.0]
    mask = [True] * n
    weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=1.0, advantage=1.0)
    assert len(weights) == n
    assert weights[0] > weights[2]
    assert weights[1] > weights[3]
    # Mean should be ~1.0
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-6


def test_weights_dampen_reduces_spread():
    n = 4
    teacher_lps = [-0.5, -0.5, -2.0, -2.0]
    student_lps = [-1.0, -1.0, -1.0, -1.0]
    mask = [True] * n
    full_weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=1.0, advantage=1.0)
    half_weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=0.5, advantage=1.0)

    full_spread = max(full_weights) - min(full_weights)
    half_spread = max(half_weights) - min(half_weights)
    assert half_spread < full_spread


def test_weights_zero_dampen_is_uniform():
    n = 4
    teacher_lps = [-0.5, -0.5, -2.0, -2.0]
    student_lps = [-1.0, -1.0, -1.0, -1.0]
    mask = [True] * n
    weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=0.0, advantage=1.0)
    for w in weights:
        assert abs(w - 1.0) < 1e-6


def test_weights_respects_mask():
    teacher_lps = [-0.5, -0.5, -2.0, -2.0]
    student_lps = [-1.0, -1.0, -1.0, -1.0]
    mask = [False, True, True, False]
    weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=1.0, advantage=1.0)
    assert weights[0] == 0.0
    assert weights[3] == 0.0
    assert weights[1] > 0.0
    assert weights[2] > 0.0


def test_weights_length_mismatch_returns_empty():
    weights = compute_pi_advantage_weights([0.0, 0.0], [0.0], [True, True], dampen=1.0, advantage=1.0)
    assert weights == []


def test_weights_sign_flip_negative_advantage():
    n = 4
    teacher_lps = [-0.5, -0.5, -2.0, -2.0]
    student_lps = [-1.0, -1.0, -1.0, -1.0]
    mask = [True] * n

    pos_weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=0.5, advantage=1.0)
    neg_weights = compute_pi_advantage_weights(teacher_lps, student_lps, mask, dampen=0.5, advantage=-1.0)

    # With positive advantage, teacher-preferred tokens (0,1) have weight > 1
    # With negative advantage + sign flip, those same tokens should have weight < 1
    assert pos_weights[0] > 1.0
    assert neg_weights[0] < 1.0
