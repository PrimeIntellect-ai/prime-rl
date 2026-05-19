from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import verifiers as vf

from prime_rl.configs.multi_agent import (
    FixedMemberTargetConfig,
    MultiAgentConfig,
    RequestMode,
    stable_train_member,
)

DISPATCH_ID_FIELD = "multi_agent_dispatch_id"


def _client_type(request_mode: RequestMode) -> vf.ClientType:
    match request_mode:
        case "chat":
            return "openai_chat_completions"
        case "token":
            return "openai_chat_completions_token"
        case "renderer":
            return "renderer"


def _dispatch_index(values: list[str], *, target_name: str, member_id: str, dispatch_id: object) -> int:
    payload = f"{target_name}:{member_id}:{dispatch_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % len(values)


def _fixed_client(
    target_name: str,
    target: FixedMemberTargetConfig,
    *,
    member_id: str,
    dispatch_id: object,
) -> vf.ClientConfig:
    idx = _dispatch_index(
        target.base_url,
        target_name=target_name,
        member_id=member_id,
        dispatch_id=dispatch_id,
    )
    return vf.ClientConfig(
        client_type=_client_type(target.request_mode),
        api_base_url=target.base_url[idx],
        api_key_var=target.api_key_var,
        timeout=target.timeout,
        connect_timeout=target.connect_timeout,
        max_connections=8192,
        max_keepalive_connections=8192,
        max_retries=10,
        extra_headers=dict(target.headers),
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
        sampling_args.update(dict(fixed.sampling))
        targets[member_id] = vf.GenerationTarget(
            client=_fixed_client(
                target_name,
                fixed,
                member_id=member_id,
                dispatch_id=dispatch_id,
            ),
            model=fixed.model,
            sampling_args=sampling_args,
        )

    return vf.MemberGenerationPlan(members=targets)
