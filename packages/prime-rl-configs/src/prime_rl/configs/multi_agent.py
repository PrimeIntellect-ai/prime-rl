from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from prime_rl.utils.config import BaseConfig

RequestMode = Literal["chat", "token", "renderer"]


class FixedMemberTargetConfig(BaseConfig):
    """A fixed generation target bound to one or more protocol members."""

    members: Annotated[
        list[str],
        Field(description="Protocol member ids routed to this fixed target."),
    ] = Field(default_factory=list)
    model: Annotated[str, Field(description="Model name served by this target.")]
    base_url: Annotated[
        list[str],
        Field(description="OpenAI-compatible base URL(s) for this target."),
    ]
    api_key_var: Annotated[
        str,
        Field(description="Environment variable containing the API key."),
    ] = "VLLM_API_KEY"
    request_mode: Annotated[
        RequestMode,
        Field(description="Wire protocol for this member target."),
    ] = "chat"
    sampling: Annotated[
        dict[str, Any],
        Field(description="Sampling overrides for this fixed target."),
    ] = Field(default_factory=dict)
    headers: Annotated[
        dict[str, str],
        Field(description="Static HTTP headers for this target."),
    ] = Field(default_factory=dict)
    timeout: Annotated[float, Field(description="Request timeout in seconds.")] = 1200.0
    connect_timeout: Annotated[
        float,
        Field(description="TCP connect timeout in seconds."),
    ] = 30.0

    @field_validator("members")
    @classmethod
    def validate_members(cls, members: list[str]) -> list[str]:
        if len(members) != len(set(members)):
            raise ValueError(f"fixed target members contains duplicates: {members}")
        if any(not member.strip() for member in members):
            raise ValueError("fixed target members must be non-empty strings")
        return members

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, base_url: list[str]) -> list[str]:
        if not base_url:
            raise ValueError("fixed target base_url must contain at least one URL")
        if any(not url.strip() for url in base_url):
            raise ValueError("fixed target base_url entries must be non-empty strings")
        return base_url


class TrainOneConfig(BaseConfig):
    """Train the policy in exactly one member role per rollout."""

    members: Annotated[
        list[str],
        Field(description="Candidate protocol member ids for the trainable policy."),
    ]
    seed: Annotated[
        int,
        Field(description="Deterministic seed for rollout-scoped member selection."),
    ] = 0
    unselected: Annotated[
        str,
        Field(
            description=("Fixed target used by train_one candidates not selected as the learner role for this rollout.")
        ),
    ]

    @field_validator("members")
    @classmethod
    def validate_members(cls, members: list[str]) -> list[str]:
        if not members:
            raise ValueError("multi_agent.train_one.members cannot be empty")
        if len(members) != len(set(members)):
            raise ValueError(f"multi_agent.train_one.members contains duplicates: {members}")
        if any(not member.strip() for member in members):
            raise ValueError("multi_agent.train_one.members must be non-empty strings")
        return members


class MultiAgentConfig(BaseConfig):
    """Runtime generation policy keyed by Verifiers protocol member ids."""

    train_one: Annotated[
        TrainOneConfig | None,
        Field(description="Optional one-trainable-member policy."),
    ] = None
    fixed: Annotated[
        dict[str, FixedMemberTargetConfig],
        Field(description="Named fixed generation targets."),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self):
        fixed_members: dict[str, str] = {}
        for target_name, target in self.fixed.items():
            if not target_name.strip():
                raise ValueError("multi_agent.fixed target names must be non-empty")
            for member in target.members:
                previous = fixed_members.get(member)
                if previous is not None:
                    raise ValueError(f"member {member!r} appears in fixed targets {previous!r} and {target_name!r}")
                fixed_members[member] = target_name

        if self.train_one is not None:
            missing = self.train_one.unselected not in self.fixed
            if missing:
                raise ValueError("multi_agent.train_one.unselected must name an entry in multi_agent.fixed")
            overlap = sorted(set(self.train_one.members) & set(fixed_members))
            if overlap:
                raise ValueError(f"train_one members cannot also be statically fixed: {overlap}")
        return self

    @property
    def enabled(self) -> bool:
        return self.train_one is not None or bool(self.fixed)


def stable_train_member(members: list[str], *, seed: int, dispatch_id: object) -> str:
    payload = f"{seed}:{dispatch_id}".encode()
    digest = hashlib.sha256(payload).digest()
    idx = int.from_bytes(digest[:8], "big") % len(members)
    return members[idx]
