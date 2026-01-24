"""Adaptive reward weight decay based on running statistics.

This module implements dynamic weight decay for multi-reward RL training.
As an auxiliary reward signal approaches saturation, its weight is decayed
to reduce its influence on the advantage calculation.

The primary reward (e.g., correctness) is never decayed to ensure gradient
signal is always present. Only auxiliary rewards (e.g., length) are subject
to adaptive decay.

Uses a ratchet mechanism with slow leak to prevent oscillation while
allowing gradual recovery if rewards drop.
"""

from prime_rl.orchestrator.config import AdaptiveWeightConfig


class AdaptiveWeightManager:
    """Manages adaptive decay of reward weights based on running statistics.

    Only auxiliary rewards are decayed. The primary reward always keeps its
    full weight to ensure gradient signal is never lost.

    The decay mechanism for auxiliary rewards follows:
        weight_t = base_weight * decay_factor
        decay_factor = max(0, 1 - (EMA_t / saturation_threshold) ^ decay_exponent)

    A ratchet with slow leak prevents oscillation:
    - Once weight decays, it doesn't immediately recover
    - Small recovery_rate allows gradual recovery if reward drops
    """

    def __init__(
        self,
        config: AdaptiveWeightConfig,
        reward_keys: list[str],
        base_weights: list[float],
    ):
        """Initialize the adaptive weight manager.

        Args:
            config: Configuration for adaptive weight decay.
            reward_keys: List of reward metric keys to track.
            base_weights: Initial/base weights for each reward key.
        """
        self.config = config
        self.reward_keys = reward_keys
        self.base_weights = {k: w for k, w in zip(reward_keys, base_weights)}

        # Determine primary reward (never decayed)
        # If not specified, default to first reward key
        self.primary_reward = config.primary_reward or (reward_keys[0] if reward_keys else None)

        # Per-reward tracking (only for auxiliary rewards, but we track EMA for all for logging)
        self.ema_values: dict[str, float] = {k: 0.0 for k in reward_keys}
        self.current_weights: dict[str, float] = {k: w for k, w in zip(reward_keys, base_weights)}
        self.ratchet_floor: dict[str, float] = {k: w for k, w in zip(reward_keys, base_weights)}

    def update(self, batch_rewards: dict[str, float]) -> list[float]:
        """Update EMAs and compute new weights for this batch.

        Args:
            batch_rewards: Dictionary mapping reward keys to their batch mean values.

        Returns:
            List of current weights in the same order as reward_keys.
        """
        for key in self.reward_keys:
            if key not in batch_rewards:
                continue

            batch_mean = batch_rewards[key]

            # Update EMA (for all rewards, including primary, for logging purposes)
            self.ema_values[key] = (
                self.config.ema_alpha * batch_mean + (1 - self.config.ema_alpha) * self.ema_values[key]
            )

            # Skip decay for primary reward - always keep full weight
            if key == self.primary_reward:
                # Primary reward keeps its base weight, no decay applied
                self.current_weights[key] = self.base_weights[key]
                continue

            # For auxiliary rewards: compute decay factor
            # normalized is how close we are to saturation (0 = no reward, 1 = saturated)
            normalized = min(1.0, self.ema_values[key] / self.config.saturation_threshold)
            # decay_factor goes from 1 (no decay) to 0 (full decay) as normalized approaches 1
            decay_factor = max(0.0, 1.0 - normalized**self.config.decay_exponent)

            # Apply decay with minimum floor
            base_weight = self.base_weights[key]
            raw_weight = base_weight * decay_factor
            raw_weight = max(raw_weight, self.config.min_weight * base_weight)

            # Ratchet with slow leak
            if raw_weight < self.ratchet_floor[key]:
                # Weight decreased - update ratchet floor
                self.ratchet_floor[key] = raw_weight
            else:
                # Weight would increase - allow slow recovery via leak
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
        """Get state for checkpointing.

        Returns:
            Dictionary containing all state needed to restore the manager.
        """
        return {
            "ema_values": self.ema_values.copy(),
            "current_weights": self.current_weights.copy(),
            "ratchet_floor": self.ratchet_floor.copy(),
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary from get_state().
        """
        self.ema_values = state["ema_values"].copy()
        self.current_weights = state["current_weights"].copy()
        self.ratchet_floor = state["ratchet_floor"].copy()
