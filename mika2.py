import json
from typing import Annotated, Any, Dict, List

from pydantic import Field, field_validator, model_validator

from prime_rl.utils.pydantic_config import BaseSettings, parse_argv


class EvalConfig(BaseSettings):
    """Enhanced evaluation configuration with flexible environment specification."""

    # Simple fields that work well with CLI
    env_names: Annotated[
        List[str], Field(default_factory=list, description="Environment names (comma-separated string or list)")
    ]

    env_ids: Annotated[
        Dict[str, str],
        Field(default_factory=dict, description="Mapping from env names to env IDs (JSON string in CLI)"),
    ]

    # Global defaults
    num_examples: Annotated[int, Field(default=-1, description="Default number of examples per environment")]
    rollouts_per_example: Annotated[int, Field(default=1, description="Default rollouts per example")]
    max_concurrent: Annotated[int, Field(default=-1, description="Default max concurrent rollouts")]

    # Store parsed environment configs (exclude from serialization)
    environment_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, exclude=True)

    # Backwards compatibility
    environment_ids: List[str] = Field(default_factory=list)
    environment_args: Dict[str, Dict] = Field(default_factory=dict)

    @field_validator("env_names", mode="before")
    @classmethod
    def parse_env_names(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [name.strip() for name in v.split(",") if name.strip()]
        return v

    @field_validator("env_ids", mode="before")
    @classmethod
    def parse_env_ids(cls, v):
        """Parse JSON string into dict."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v

    @model_validator(mode="before")
    @classmethod
    def parse_dynamic_env_configs(cls, values):
        """Parse env_{name}_{param} style CLI arguments."""
        if not isinstance(values, dict):
            return values

        env_configs = {}
        keys_to_remove = []

        for key, value in values.items():
            # Look for env_{name}_{param} pattern
            if key.startswith("env_") and key.count("_") >= 2:
                parts = key.split("_")
                if len(parts) >= 3:
                    env_name = "_".join(parts[1:-1])  # Handle names with underscores
                    param = parts[-1]

                    if env_name not in env_configs:
                        env_configs[env_name] = {}

                    # Try to parse value as JSON, fallback to string
                    try:
                        if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                            value = json.loads(value)
                        elif isinstance(value, str) and value.isdigit():
                            value = int(value)
                        elif isinstance(value, str) and value.replace(".", "").isdigit():
                            value = float(value)
                    except (json.JSONDecodeError, ValueError):
                        pass  # Keep as string

                    env_configs[env_name][param] = value
                    keys_to_remove.append(key)

        # Remove parsed keys
        for key in keys_to_remove:
            del values[key]

        # Store for later processing
        if env_configs:
            values["parsed_env_configs"] = env_configs

        return values

    @model_validator(mode="after")
    def build_environment_configs(self):
        """Build final environment configuration."""
        # Get parsed configs from before validator
        parsed_configs = getattr(self, "parsed_env_configs", {})

        # Clear environment_configs to rebuild
        self.environment_configs = {}

        # Build environment list
        for env_name in self.env_names:
            env_config = {
                "name": env_name,
                "env_id": self.env_ids.get(env_name, env_name),
                "num_examples": self.num_examples,
                "rollouts_per_example": self.rollouts_per_example,
                "max_concurrent": self.max_concurrent,
                "env_args": {},
            }

            # Apply environment-specific overrides
            if env_name in parsed_configs:
                env_specific = parsed_configs[env_name]
                for param, value in env_specific.items():
                    if param == "args":
                        env_config["env_args"] = value
                    else:
                        env_config[param] = value

            self.environment_configs[env_name] = env_config

        # Populate legacy fields for backwards compatibility
        if not self.environment_ids and self.environment_configs:
            self.environment_ids = [cfg["env_id"] for cfg in self.environment_configs.values()]
            self.environment_args = {cfg["env_id"]: cfg["env_args"] for cfg in self.environment_configs.values()}

        return self

    def get_environment_configs(self) -> List[Dict[str, Any]]:
        """Get list of environment configurations for evaluation."""
        return list(self.environment_configs.values())


if __name__ == "__main__":
    config = parse_argv(EvalConfig)
    print("=== Parsed Configuration ===")
    print(f"env_names: {config.env_names}")
    print(f"env_ids: {config.env_ids}")
    print(f"num_examples: {config.num_examples}")
    print(f"rollouts_per_example: {config.rollouts_per_example}")
    print(f"max_concurrent: {config.max_concurrent}")

    print("\n=== Environment Configurations ===")
    for name, env_config in config.environment_configs.items():
        print(f"{name}: {env_config}")

    print("\n=== Legacy Fields ===")
    print(f"environment_ids: {config.environment_ids}")
    print(f"environment_args: {config.environment_args}")
