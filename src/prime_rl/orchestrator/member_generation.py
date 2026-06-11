from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import verifiers as vf

from prime_rl.configs.multi_agent import (
    FixedMemberTargetConfig,
    MultiAgentConfig,
    RequestMode,
    stable_train_member,
)
from prime_rl.utils.client import build_client

DISPATCH_ID_FIELD = "multi_agent_dispatch_id"


def _client_type(request_mode: RequestMode) -> vf.ClientType:
    match request_mode:
        case "chat":
            return "openai_chat_completions"
        case "token":
            return "openai_chat_completions_token"
        case "renderer":
            return "renderer"


def _dispatch_index(values: list[str], *, target_name: str, member_id: str, group_id: object) -> int:
    # Group-stable on purpose: consecutive turns of the same fixed member
    # within one group hit the same server (KV prefix-cache reuse), while
    # different groups still spread across the pool.
    payload = f"{target_name}:{member_id}:{group_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % len(values)


def _fixed_client(
    target_name: str,
    target: FixedMemberTargetConfig,
    *,
    member_id: str,
    group_id: object,
) -> vf.ClientConfig:
    idx = _dispatch_index(
        target.base_url,
        target_name=target_name,
        member_id=member_id,
        group_id=group_id,
    )
    return build_client(
        target,
        api_base_url=target.base_url[idx],
        client_type=_client_type(target.request_mode),
        extra_headers=dict(target.headers),
        renderer_config=target.renderer,
        renderer_model_name=target.renderer_model_name,
        renderer_pool_size=target.renderer_pool_size,
    )


def fixed_member_targets(config: MultiAgentConfig) -> dict[str, str]:
    members: dict[str, str] = {}
    for target_name, target in config.fixed.items():
        for member in target.members:
            members[member] = target_name
    return members


def dispatch_id_for_rollout(rollout: Mapping[str, Any]) -> object:
    dispatch_id = rollout.get(DISPATCH_ID_FIELD)
    if dispatch_id is not None:
        return dispatch_id
    info = rollout.get("info")
    if isinstance(info, Mapping):
        dispatch_id = info.get(DISPATCH_ID_FIELD)
        if dispatch_id is not None:
            return dispatch_id
        prime_rl = info.get("prime_rl")
        if isinstance(prime_rl, Mapping) and prime_rl.get(DISPATCH_ID_FIELD) is not None:
            return prime_rl[DISPATCH_ID_FIELD]
    return rollout.get("trajectory_id") or rollout.get("example_id") or ""


def trainable_member_ids(config: MultiAgentConfig, *, dispatch_id: object) -> set[str] | None:
    if config.train_one is None:
        return None
    selected = stable_train_member(
        config.train_one.members,
        seed=config.train_one.seed,
        dispatch_id=dispatch_id,
    )
    return {selected}


def is_trainable_member(config: MultiAgentConfig, rollout: Mapping[str, Any], member_id: str) -> bool:
    if member_id in fixed_member_targets(config):
        return False
    trainable = trainable_member_ids(config, dispatch_id=dispatch_id_for_rollout(rollout))
    if trainable is None:
        return True
    if config.train_one is not None and member_id in config.train_one.members:
        return member_id in trainable
    return True


def _target_name_for_member(
    config: MultiAgentConfig,
    *,
    member_id: str,
    dispatch_id: object,
) -> str | None:
    fixed_members = fixed_member_targets(config)
    if member_id in fixed_members:
        return fixed_members[member_id]
    if config.train_one is None or member_id not in config.train_one.members:
        return None
    selected = stable_train_member(
        config.train_one.members,
        seed=config.train_one.seed,
        dispatch_id=dispatch_id,
    )
    if member_id == selected:
        return None
    return config.train_one.unselected


def compile_member_generation_plan(
    config: MultiAgentConfig,
    *,
    member_ids: Iterable[str],
    default_client: vf.ClientConfig,
    default_model: str,
    learner_sampling_args: Mapping[str, Any],
    fixed_sampling_args: Mapping[str, Any],
    dispatch_id: object,
    group_id: object,
) -> vf.MemberGenerationPlan | None:
    if not config.enabled:
        return None

    default_target = vf.GenerationTarget(
        client=default_client,
        model=default_model,
        sampling_args=dict(learner_sampling_args),
    )
    targets: dict[str, vf.GenerationTarget] = {}
    for member_id in member_ids:
        target_name = _target_name_for_member(
            config,
            member_id=member_id,
            dispatch_id=dispatch_id,
        )
        if target_name is None:
            targets[member_id] = default_target
            continue

        fixed = config.fixed[target_name]
        sampling_args = dict(fixed_sampling_args)
        overrides = dict(fixed.sampling)
        if "extra_body" in overrides:
            # extra_body merges per-key (target keys win, inherited keys survive);
            # every other sampling field replaces wholesale.
            overrides["extra_body"] = {**sampling_args.get("extra_body", {}), **overrides["extra_body"]}
        sampling_args.update(overrides)
        targets[member_id] = vf.GenerationTarget(
            client=_fixed_client(
                target_name,
                fixed,
                member_id=member_id,
                group_id=group_id,
            ),
            model=fixed.model,
            sampling_args=sampling_args,
        )

    return vf.MemberGenerationPlan(members=targets)


def validate_member_references(config: MultiAgentConfig, members_by_env: Mapping[str, Sequence[str]]) -> None:
    """Fail fast when the multi-agent config references member ids that exist in
    no loaded multi-agent env protocol. A typo here would otherwise silently
    reroute the real member to the learner (served and trained as policy).

    ``members_by_env`` maps each loaded multi-agent env name to its protocol
    member ids; config-time validation cannot see these, so this runs at
    orchestrator startup, right after the envs are loaded.
    """
    if not config.enabled:
        return
    if not members_by_env:
        raise ValueError(
            "[orchestrator.multi_agent] is configured but no loaded environment exposes protocol members. "
            "Remove the multi_agent block or run a multi-agent environment."
        )
    known = {member for members in members_by_env.values() for member in members}
    referenced = set(fixed_member_targets(config))
    if config.train_one is not None:
        referenced.update(config.train_one.members)
    unknown = sorted(referenced - known)
    if unknown:
        valid = ", ".join(f"{env}={sorted(members)}" for env, members in sorted(members_by_env.items()))
        raise ValueError(
            f"multi_agent config references unknown member id(s) {unknown}. Valid member ids per env: {valid}"
        )


def uncovered_trainable_members(config: MultiAgentConfig, member_ids: Sequence[str]) -> list[str]:
    """Member ids that are trainable-by-default under ``train_one``: neither a
    ``train_one`` candidate nor bound to a fixed target, so *every* rollout
    trains them. Empty when ``train_one`` is unset (all-trainable is then the
    explicit mode, not an oversight)."""
    if config.train_one is None:
        return []
    fixed_members = fixed_member_targets(config)
    candidates = set(config.train_one.members)
    return [member for member in member_ids if member not in fixed_members and member not in candidates]
