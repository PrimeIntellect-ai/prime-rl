"""Unit tests for orchestrator-side echo annotation logic (pure, no GPU)."""

import pytest

from prime_rl.configs.losses import (
    AssistantRoleEchoConfig,
    EchoLossConfig,
    SystemRoleEchoConfig,
    ToolRoleEchoConfig,
)
from prime_rl.orchestrator.advantage import RenderHints, echo_advantage
from prime_rl.orchestrator.echo import _build_step_echo_alpha, apply_echo_filter
from prime_rl.orchestrator.trajectories import step_token_roles


def test_role_alpha_maps_only_enabled_roles():
    attribution = {
        "message_roles": ["system", "user"],
        "message_indices": [0, 0, 1, 1],
        "is_content": [True, True, True, True],
    }
    out = _build_step_echo_alpha(
        prompt_attribution=attribution,
        prompt_len=4,
        completion_len=0,
        echo_config=EchoLossConfig(system=SystemRoleEchoConfig(alpha=0.5)),
    )
    # System tokens get alpha; user tokens stay None (user echo disabled).
    assert out == [0.5, 0.5, None, None]


def test_assistant_alpha_covers_completion():
    out = _build_step_echo_alpha(
        prompt_attribution=None,
        prompt_len=2,
        completion_len=2,
        echo_config=EchoLossConfig(assistant=AssistantRoleEchoConfig(alpha=0.3)),
    )
    assert out == [None, None, 0.3, 0.3]


def test_tool_names_filter():
    attribution = {
        "message_roles": ["tool"],
        "message_indices": [0, 0],
        "is_content": [True, True],
        "message_tool_names": ["lookup"],
    }
    kept = _build_step_echo_alpha(
        prompt_attribution=attribution,
        prompt_len=2,
        completion_len=0,
        echo_config=EchoLossConfig(tool=ToolRoleEchoConfig(alpha=0.5, tool_names={"lookup"})),
    )
    assert kept == [0.5, 0.5]

    dropped = _build_step_echo_alpha(
        prompt_attribution=attribution,
        prompt_len=2,
        completion_len=0,
        echo_config=EchoLossConfig(tool=ToolRoleEchoConfig(alpha=0.5, tool_names={"other"})),
    )
    assert dropped == [None, None]


def test_filter_mask_narrows_role_baseline():
    attribution = {
        "message_roles": ["system"],
        "message_indices": [0, 0, 0, 0],
        "is_content": [True, True, True, True],
    }
    out = _build_step_echo_alpha(
        prompt_attribution=attribution,
        prompt_len=4,
        completion_len=0,
        echo_config=EchoLossConfig(system=SystemRoleEchoConfig(alpha=0.5)),
        filter_mask=[True, False, True, True],
    )
    assert out == [0.5, None, 0.5, 0.5]


def _rollout(prompt_ids, completion_ids):
    return {"trajectory": [{"tokens": {"prompt_ids": prompt_ids, "completion_ids": completion_ids}}]}


def test_apply_echo_filter_happy_path():
    result = apply_echo_filter(_rollout([1, 2], [3]), lambda r: [[True, False, True]])
    assert result == [[True, False, True]]


def test_apply_echo_filter_rejects_non_list():
    with pytest.raises(TypeError, match="must return list"):
        apply_echo_filter(_rollout([1, 2], [3]), lambda r: "nope")


def test_apply_echo_filter_rejects_wrong_inner_length():
    with pytest.raises(ValueError, match="mask length"):
        apply_echo_filter(_rollout([1, 2], [3]), lambda r: [[True, False]])


def test_apply_echo_filter_rejects_non_bool():
    with pytest.raises(TypeError, match="plain bool"):
        apply_echo_filter(_rollout([1, 2], [3]), lambda r: [[True, 1, True]])


def test_echo_advantage_matches_build_step_echo_alpha():
    """echo_advantage on step_token_roles' output reproduces _build_step_echo_alpha's role->alpha
    masking — the bit-identity guard for moving echo onto the advantage_fn (transitional; removed
    when build_echo_annotations is retired)."""
    tokens = {
        "prompt_ids": [1, 2, 3, 4],
        "completion_ids": [5, 6],
        "prompt_attribution": {
            "message_roles": ["system", "user", "tool"],
            "message_indices": [0, 1, 2, 2],
            "is_content": [True, True, True, True],
            "message_tool_names": [None, None, "calc"],
        },
    }
    roles, tool_names = step_token_roles(tokens)
    hints = RenderHints(
        token_id=tokens["prompt_ids"] + tokens["completion_ids"],
        role=roles,
        tool_name=tool_names,
        is_sampled=[False, False, False, False, True, True],
        inference_logprob=[0.0] * 6,
    )
    adv = echo_advantage([hints], roles=["system", "tool"], tool_names={"calc"}, alpha=0.5)[0]
    expected = _build_step_echo_alpha(
        prompt_attribution=tokens["prompt_attribution"],
        prompt_len=4,
        completion_len=2,
        echo_config=EchoLossConfig(
            system=SystemRoleEchoConfig(alpha=0.5),
            tool=ToolRoleEchoConfig(alpha=0.5, tool_names={"calc"}),
        ),
    )
    # _build_step_echo_alpha uses None for "not echoed"; echo_advantage uses 0.0.
    assert adv == [0.5 if e is not None else 0.0 for e in expected]
