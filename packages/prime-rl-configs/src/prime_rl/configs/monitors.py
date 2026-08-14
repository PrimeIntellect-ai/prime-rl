from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig


class WandbConfig(BaseConfig):
    project: str = "prime-rl"
    """W&B project to log to."""

    entity: str | None = None
    """W&B entity to log to."""

    name: str | None = None
    """W&B run name."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool = False
    """Run W&B in offline mode."""


class FileMonitorConfig(BaseConfig):
    """Local JSONL metric sink (``<output_dir>/metrics.jsonl``). Metrics are the same
    scalars sent to W&B; useful for self-hosted dashboards or when W&B is disabled."""

    filename: str = "metrics.jsonl"
    """Name of the JSONL file written under the component's ``output_dir``."""


class EpisodeLogConfig(BaseConfig):
    interval: int = Field(10, ge=1)
    """Step interval between episode uploads."""

    sample_ratio: float | None = Field(None, ge=0.0, le=1.0)
    """Fraction of episodes to upload per logged step. 1.0 = all, 0.5 = half, 0.0 = none; None keeps all."""


class PrimeMonitorConfig(BaseConfig):
    base_url: str = "https://api.primeintellect.ai/api/v1/rft"
    """Base URL for the Prime Intellect monitoring API."""

    api_key_var: str = "PRIME_API_KEY"
    """Environment variable name containing the Prime Intellect API key, resolved via ``os.getenv``."""

    log_episodes: EpisodeLogConfig | None = EpisodeLogConfig()
    """Episode upload configuration. If None, no episodes are uploaded."""

    run_name: str | None = None
    """Run name shown on the platform. Defaults to the W&B run name when set, otherwise the platform auto-generates one."""

    team_id: str | None = None
    """Team ID to associate the run with."""

    frontend_url: str | None = None
    """Frontend base URL used for the dashboard link printed after registration. Defaults to the Prime CLI frontend URL when unset."""


class MonitorsConfig(BaseConfig):
    wandb: WandbConfig | None = None
    """Log metrics to Weights & Biases. If None, W&B logging is disabled."""

    file: FileMonitorConfig | None = None
    """Append metrics to a local JSONL file under the run's output directory. If None, disabled."""


class OrchestratorMonitorsConfig(MonitorsConfig):
    prime: PrimeMonitorConfig | None = None
    """Log metrics and episodes to the Prime Intellect platform. If None, disabled."""


class SharedWandbConfig(BaseConfig):
    project: str | None = "prime-rl"
    """W&B project."""

    entity: str | None = None
    """W&B entity."""

    name: str | None = None
    """W&B run name."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool | None = False
    """Run W&B in offline mode. Incompatible with shared mode, which is always on for the ``rl`` entrypoint."""

    @model_validator(mode="after")
    def validate_not_offline(self):
        if self.offline:
            raise ValueError(
                "W&B shared mode is always on for the rl entrypoint and requires server "
                "connectivity; monitors.wandb.offline = true is not supported. Use offline mode "
                "via the sub-config wandb blocks (trainer.monitors.wandb.offline, "
                "orchestrator.monitors.wandb.offline) if you really need it per-process."
            )
        return self


class SharedMonitorsConfig(BaseConfig):
    """The ``rl`` entrypoint's shared monitor configs, propagated to trainer and orchestrator."""

    wandb: SharedWandbConfig | None = None
    """Shared W&B config. Propagated to trainer and orchestrator."""

    file: FileMonitorConfig | None = None
    """Shared local JSONL metric sink. If set, enables ``<output_dir>/metrics.jsonl`` on both trainer and orchestrator."""
