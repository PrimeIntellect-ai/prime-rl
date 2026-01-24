from prime_rl.orchestrator.config import AdaptiveWeightConfig


class AdaptiveWeightManager:
    """Manages adaptive decay of reward weights based on running statistics."""

    def __init__(
        self,
        config: AdaptiveWeightConfig,
        reward_keys: list[str],
        base_weights: list[float],
    ):
        self.config = config
        self.reward_keys = reward_keys
        self.base_weights = {k: w for k, w in zip(reward_keys, base_weights)}

        if config.min_weights is not None:
            if len(config.min_weights) != len(reward_keys):
                raise ValueError(
                    f"min_weights length ({len(config.min_weights)}) must match reward_keys length ({len(reward_keys)})"
                )
            self.min_weights = {k: mw for k, mw in zip(reward_keys, config.min_weights)}
        else:
            self.min_weights = {k: 0.1 for k in reward_keys}

        self.ema_values: dict[str, float] = {k: 0.0 for k in reward_keys}
        self.current_weights: dict[str, float] = {k: w for k, w in zip(reward_keys, base_weights)}
        self.ratchet_floor: dict[str, float] = {k: w for k, w in zip(reward_keys, base_weights)}

    def update(self, batch_rewards: dict[str, float]) -> list[float]:
        """Update EMAs and compute new weights for this batch."""
        for key in self.reward_keys:
            if key not in batch_rewards:
                continue

            batch_mean = batch_rewards[key]

            self.ema_values[key] = (
                self.config.ema_alpha * batch_mean + (1 - self.config.ema_alpha) * self.ema_values[key]
            )

            min_weight = self.min_weights[key]

            if min_weight >= 1.0:
                self.current_weights[key] = self.base_weights[key]
                continue

            normalized = min(1.0, self.ema_values[key] / self.config.saturation_threshold)
            decay_factor = max(0.0, 1.0 - normalized**self.config.decay_exponent)

            base_weight = self.base_weights[key]
            raw_weight = base_weight * decay_factor
            raw_weight = max(raw_weight, min_weight * base_weight)

            if raw_weight < self.ratchet_floor[key]:
                self.ratchet_floor[key] = raw_weight
            else:
                self.ratchet_floor[key] += self.config.recovery_rate * (raw_weight - self.ratchet_floor[key])

            self.current_weights[key] = self.ratchet_floor[key]

        return [self.current_weights[k] for k in self.reward_keys]

    def get_weights_dict(self) -> dict[str, float]:
        """Get current weights as a dictionary."""
        return self.current_weights.copy()

    def get_ema_dict(self) -> dict[str, float]:
        """Get current EMA values as a dictionary."""
        return self.ema_values.copy()

    def get_state(self) -> dict:
        """Get state for checkpointing."""
        return {
            "ema_values": self.ema_values.copy(),
            "current_weights": self.current_weights.copy(),
            "ratchet_floor": self.ratchet_floor.copy(),
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.ema_values = state["ema_values"].copy()
        self.current_weights = state["current_weights"].copy()
        self.ratchet_floor = state["ratchet_floor"].copy()
