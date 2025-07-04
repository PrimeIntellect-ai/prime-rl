from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from zeroband.utils.pydantic_config import BaseConfig


class ModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[
        str,
        Field(
            default="Qwen/Qwen3-0.6B",
            description="Name or path of the HF model to use.",
        ),
    ]


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        Literal["debug", "info", "sucess"],
        Field(description="Logging level for the process. Will determine the logging verbosity and format."),
    ] = "info"

    path: Annotated[
        Path | None,
        Field(
            default=Path("logs"),
            description="The file path to log to. If None, will only log to stdout. This field is particularly useful to distinguish logs when multi-processing.",
        ),
    ]

    utc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time.",
        ),
    ]


class FileMonitorConfig(BaseConfig):
    """Configures logging to a file."""

    path: Annotated[Path | None, Field(default=None, description="The file path to log to")]

    @model_validator(mode="after")
    def validate_path(self):
        if self.path is None:
            raise ValueError("File path must be set when FileMonitor is enabled. Try setting --monitor.file.path")
        return self


class SocketMonitorConfig(BaseConfig):
    """Configures logging to a Unix socket."""

    path: Annotated[Path | None, Field(default=None, description="The socket path to log to")]

    @model_validator(mode="after")
    def validate_path(self):
        if self.path is None:
            raise ValueError("Socket path must be set when SocketMonitor is enabled. Try setting --monitor.socket.path")
        return self


class APIMonitorConfig(BaseConfig):
    """Configures logging to an API via HTTP."""

    url: Annotated[str | None, Field(default=None, description="The API URL to log to")]

    auth_token: Annotated[str | None, Field(default=None, description="The API auth token to use")]

    @model_validator(mode="after")
    def validate_url(self):
        if self.url is None:
            raise ValueError("URL must be set when APIMonitor is enabled. Try setting --monitor.api.url")
        return self

    @model_validator(mode="after")
    def validate_auth_token(self):
        if self.auth_token is None:
            raise ValueError("Auth token must be set when APIMonitor is enabled. Try setting --monitor.api.auth_token")
        return self


class SampleLoggingConfig(BaseConfig):
    """Configures sample logging for W&B tables."""

    interval: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            description="Step interval at which to log samples to W&B table.",
        ),
    ]

    num_samples: Annotated[
        int,
        Field(
            default=8,
            ge=1,
            description="Number of samples to randomly select and log from each batch.",
        ),
    ]


class WandbMonitorConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    project: Annotated[str, Field(default="prime-rl", description="The W&B project to log to.")]
    group: Annotated[
        str | None,
        Field(
            default=None,
            description="The W&B group to log to. If None, it will not set the group. Use grouping if you want multiple associated runs (e.g. RL training has a training and inference run) log to the same dashboard.",
        ),
    ]
    name: Annotated[
        str | None,
        Field(
            default=None,
            description="The W&B name to to use for logging. If group and name are set, they will be automatically combined into a single name.",
        ),
    ]
    dir: Annotated[
        Path | None,
        Field(
            default=Path("logs"),
            description="Path to the directory to keep local logs. It will automatically create a `wandb` subdirectory to store run logs.",
        ),
    ]

    offline: Annotated[bool, Field(default=False, description="Whether to run W&B in offline mode.")]

    log_samples: Annotated[
        SampleLoggingConfig | None,
        Field(
            default=None,
            description="Configuration for logging prompt/response samples to W&B tables. If None, no samples are logged.",
        ),
    ]

    @model_validator(mode="after")
    def validate_name(self):
        # If group and name are set, the run name will be prefixed with the group
        if self.group and self.name:
            self.name = f"{self.group}-{self.name}"
        return self


class MultiMonitorConfig(BaseConfig):
    """Configures the monitoring system."""

    # All possible monitors (currently only supports one instance per type)
    file: Annotated[FileMonitorConfig, Field(default=None)]
    socket: Annotated[SocketMonitorConfig, Field(default=None)]
    api: Annotated[APIMonitorConfig, Field(default=None)]
    wandb: Annotated[WandbMonitorConfig, Field(default=None)]

    system_log_frequency: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Interval in seconds to log system metrics. If 0, no system metrics are logged)",
        ),
    ]

    def __str__(self) -> str:
        is_enabled = lambda x: "enabled" if x is not None else "disabled"  # noqa
        return f"file={is_enabled(self.file)}, socket={is_enabled(self.socket)}, api={is_enabled(self.api)}, wandb={is_enabled(self.wandb)}, system_log_frequency={self.system_log_frequency}"
