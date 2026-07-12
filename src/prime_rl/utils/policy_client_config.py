"""Resolve the generated DGD policy-client deployment boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prime_rl.configs.shared import ClientConfig

DYNAMO_TOPOLOGY_ENV = "DYN_RL_TOPOLOGY"


class _DynamoClientTopologyEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    admin_api: Literal["dynamo"]
    base_url: tuple[str, ...] = Field(min_length=1)
    rl_base_url: tuple[str, ...] = Field(min_length=1)
    dynamo_worker_roles: tuple[Literal["agg", "prefill", "decode"], ...] = Field(min_length=1)
    dynamo_gpus_per_worker: int = Field(ge=1)


def policy_client_config_from_environment(
    client_config: ClientConfig,
    environment: Mapping[str, str] | None = None,
) -> ClientConfig:
    """Apply the generated policy topology without mutating user config."""
    source = environment if environment is not None else os.environ
    serialized = source.get(DYNAMO_TOPOLOGY_ENV)
    if serialized is None:
        return client_config
    topology = _DynamoClientTopologyEnvironment.model_validate_json(serialized)
    updates = {
        "admin_api": topology.admin_api,
        "base_url": list(topology.base_url),
        "rl_base_url": list(topology.rl_base_url),
        "dynamo_worker_roles": topology.dynamo_worker_roles,
        "dynamo_gpus_per_worker": topology.dynamo_gpus_per_worker,
    }
    for field, expected in updates.items():
        if field in client_config.model_fields_set and getattr(client_config, field) != expected:
            raise ValueError(
                f"orchestrator.model.client.{field} conflicts with the generated "
                f"{DYNAMO_TOPOLOGY_ENV} deployment boundary"
            )
    return client_config.model_copy(update=updates)
