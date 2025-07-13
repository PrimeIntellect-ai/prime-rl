from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from prime_rl.utils.pydantic_config import BaseConfig


class ModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[str, Field(description="Name or path of the HF model to use.")] = "Qwen/Qwen3-0.6B"


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        Literal["debug", "info", "sucess"],
        Field(description="Logging level for the process. Will determine the logging verbosity and format."),
    ] = "info"

    utc: Annotated[
        bool,
        Field(
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time."
        ),
    ] = False


class FileMonitorConfig(BaseConfig):
    """Configures logging to a file."""

    path: Annotated[Path, Field(description="The file path to log to")]


class SocketMonitorConfig(BaseConfig):
    """Configures logging to a Unix socket."""

    path: Annotated[Path, Field(description="The socket path to log to")]


class APIMonitorConfig(BaseConfig):
    """Configures logging to an API via HTTP."""

    url: Annotated[str, Field(description="The API URL to log to")]

    auth_token: Annotated[str, Field(description="The API auth token to use")]


class SampleLoggingConfig(BaseConfig):
    """Configures sample logging for W&B tables."""

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Step interval at which to log samples to W&B table.",
        ),
    ] = 10

    num_samples: Annotated[
        int,
        Field(
            ge=1,
            description="Number of samples to randomly select and log from each batch.",
        ),
    ] = 8


class WandbMonitorConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    project: Annotated[str, Field(description="The W&B project to log to.")] = "prime-rl"

    name: Annotated[
        str | None,
        Field(
            description="The W&B name to to use for logging.",
        ),
    ] = None

    id: Annotated[
        str | None,
        Field(
            description="The W&B run ID to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.",
        ),
    ] = None

    dir: Annotated[
        Path | None,
        Field(
            description="Path to the directory to keep local logs. It will automatically create a `wandb` subdirectory to store run logs.",
        ),
    ] = Path("logs")

    offline: Annotated[bool, Field(description="Whether to run W&B in offline mode.")] = False

    log_samples: Annotated[
        SampleLoggingConfig | None,
        Field(
            description="Configuration for logging prompt/response samples to W&B tables. If None, no samples are logged.",
        ),
    ] = None


class MultiMonitorConfig(BaseConfig):
    """Configures the monitoring system."""

    # All possible monitors (currently only supports one instance per type)
    file: FileMonitorConfig | None = None
    socket: SocketMonitorConfig | None = None
    api: APIMonitorConfig | None = None
    wandb: WandbMonitorConfig | None = None

    system_log_frequency: Annotated[
        int,
        Field(
            ge=0,
            description="Interval in seconds to log system metrics. If 0, no system metrics are logged.",
        ),
    ] = 0

    def __str__(self) -> str:
        is_enabled = lambda x: "enabled" if x is not None else "disabled"  # noqa
        return f"file={is_enabled(self.file)}, socket={is_enabled(self.socket)}, api={is_enabled(self.api)}, wandb={is_enabled(self.wandb)}, system_log_frequency={self.system_log_frequency}"
