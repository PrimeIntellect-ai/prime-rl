from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from zeroband.utils.pydantic_config import BaseConfig


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
    name: Annotated[str | None, Field(default=None, description="The W&B name to to use for logging.")]
    dir: Annotated[
        Path | None,
        Field(
            default=Path("logs"),
            description="Path to the directory to keep local logs. It will automatically create a `wandb` subdirectory to store run logs.",
        ),
    ]

    prefix: Annotated[
        str,
        Field(
            default="",
            description="The prefix to add to each key in the metrics dictionary before logging to W&B. Useful for distinguishing between different runs in the same group (e.g. training and inference)",
        ),
    ]


class MultiMonitorConfig(BaseConfig):
    """Configures the monitoring system."""

    # All possible monitors (currently only supports one instance per type)
    file: Annotated[FileMonitorConfig, Field(default=None)]
    socket: Annotated[SocketMonitorConfig, Field(default=None)]
    api: Annotated[APIMonitorConfig, Field(default=None)]
    wandb: Annotated[WandbMonitorConfig, Field(default=None)]

    system_log_frequency: Annotated[
        int, Field(default=0, ge=0, description="Interval in seconds to log system metrics. If 0, no system metrics are logged)")
    ]

    def __str__(self) -> str:
        is_enabled = lambda x: "enabled" if x is not None else "disabled"
        return f"file={is_enabled(self.file)}, socket={is_enabled(self.socket)}, api={is_enabled(self.api)}, wandb={is_enabled(self.wandb)}, system_log_frequency={self.system_log_frequency}"
