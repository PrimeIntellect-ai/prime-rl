"""Unit tests for orchestrator-side echo annotation logic (pure, no GPU)."""

import pytest

from prime_rl.configs.orchestrator import (
    AssistantRoleEchoConfig,
    EchoConfig,
    SystemRoleEchoConfig,
    ToolRoleEchoConfig,
)
from prime_rl.orchestrator.echo import _build_step_echo_alpha, apply_echo_filter


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
        echo_config=EchoConfig(system=SystemRoleEchoConfig(alpha=0.5)),
    )
    # System tokens get alpha; user tokens stay None (user echo disabled).
    assert out == [0.5, 0.5, None, None]


def test_assistant_alpha_covers_completion():
    out = _build_step_echo_alpha(
        prompt_attribution=None,
        prompt_len=2,
        completion_len=2,
        echo_config=EchoConfig(assistant=AssistantRoleEchoConfig(alpha=0.3)),
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
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5, tool_names={"lookup"})),
    )
    assert kept == [0.5, 0.5]

    dropped = _build_step_echo_alpha(
        prompt_attribution=attribution,
        prompt_len=2,
        completion_len=0,
        echo_config=EchoConfig(tool=ToolRoleEchoConfig(alpha=0.5, tool_names={"other"})),
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
        echo_config=EchoConfig(system=SystemRoleEchoConfig(alpha=0.5)),
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
