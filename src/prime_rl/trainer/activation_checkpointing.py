from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.checkpoint import checkpoint

from prime_rl.configs.trainer import ActivationCheckpointConfig

SCOPED_ACTIVATION_CHECKPOINT_FIELDS = (
    "attention",
    "attn_norm",
    "feed_forward",
    "ffn_norm",
    "moe",
    "shared_expert",
)


@dataclass(frozen=True)
class ActivationCheckpointScopes:
    attention: bool = False
    attn_norm: bool = False
    feed_forward: bool = False
    ffn_norm: bool = False
    moe: bool = False
    shared_expert: bool = False

    @classmethod
    def from_config(cls, config: ActivationCheckpointConfig) -> "ActivationCheckpointScopes":
        return cls(**{field: getattr(config, field) for field in SCOPED_ACTIVATION_CHECKPOINT_FIELDS})

    def has_any(self) -> bool:
        return bool(self.enabled_scopes())

    def enabled_scopes(self) -> tuple[str, ...]:
        return tuple(field for field in SCOPED_ACTIVATION_CHECKPOINT_FIELDS if getattr(self, field))


class SelectiveActivationCheckpointingMixin:
    _activation_checkpoint_scopes: ActivationCheckpointScopes

    def _init_selective_activation_checkpointing(self) -> None:
        self._activation_checkpoint_scopes = ActivationCheckpointScopes()

    def set_activation_checkpoint_scopes(
        self, config: ActivationCheckpointConfig | ActivationCheckpointScopes | None
    ) -> None:
        if isinstance(config, ActivationCheckpointConfig):
            config = ActivationCheckpointScopes.from_config(config)
        self._activation_checkpoint_scopes = config if config is not None else ActivationCheckpointScopes()

    def _scope_enabled(self, scope: str) -> bool:
        return self.training and torch.is_grad_enabled() and getattr(self._activation_checkpoint_scopes, scope)

    def _maybe_checkpoint(self, scope: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._scope_enabled(scope):
            return function(*args, **kwargs)
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)

    def _run_feed_forward(
        self,
        function: Callable[..., Any],
        *args: Any,
        use_moe_recompute: bool = False,
        **kwargs: Any,
    ) -> Any:
        if use_moe_recompute:
            recompute = self._scope_enabled("feed_forward") or self._scope_enabled("moe")
            return function(*args, recompute=recompute, **kwargs)
        return self._maybe_checkpoint("feed_forward", function, *args, **kwargs)

    def _run_shared_expert(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        scope = "feed_forward" if self._scope_enabled("feed_forward") else "shared_expert"
        return self._maybe_checkpoint(scope, function, *args, **kwargs)

    def _forward_mlp_module(
        self,
        hidden_states: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        recompute: bool = False,
    ) -> Any:
        if recompute:
            return self.mlp(hidden_states, routed_experts=routed_experts, recompute=True)
        return self.mlp(hidden_states, routed_experts=routed_experts)
