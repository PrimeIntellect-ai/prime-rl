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


class MultiMonitorConfig(BaseConfig):
    """Configures the monitoring system."""

    # All possible monitors (currently only supports one instance per type)
    file: Annotated[FileMonitorConfig, Field(default=None)]
    socket: Annotated[SocketMonitorConfig, Field(default=None)]
    api: Annotated[APIMonitorConfig, Field(default=None)]

    system_log_frequency: Annotated[
        int, Field(default=0, ge=0, description="Interval in seconds to log system metrics. If 0, no system metrics are logged)")
    ]

    def __str__(self) -> str:
        file_str = "disabled" if self.file is None else f"path={self.file.path}"
        socket_str = "disabled" if self.socket is None else f"path={self.socket.path}"
        api_str = "disabled" if self.api is None else f"url={self.api.url}"
        return f"file={file_str}, socket={socket_str}, api={api_str}, system_log_frequency={self.system_log_frequency}"
