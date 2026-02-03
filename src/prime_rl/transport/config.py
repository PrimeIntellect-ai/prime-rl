from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


class BaseTransportConfig(BaseModel):
    """Base configuration for transport."""

    pass


class FileSystemTransportConfig(BaseTransportConfig):
    """Configures filesystem-based transport for training examples."""

    type: Literal["filesystem"] = "filesystem"


class ZMQTransportConfig(BaseTransportConfig):
    """Configures ZMQ-based transport for training examples."""

    type: Literal["zmq"] = "zmq"
    host: Annotated[str, Field(description="The host address for ZMQ transport.")] = "localhost"
    port: Annotated[int, Field(description="The base port for ZMQ transport.")] = 5555
    hwm: Annotated[int, Field(description="High water mark (max messages in queue) for ZMQ sockets.")] = 10


class TCPStoreTransportConfig(BaseTransportConfig):
    """Configures TCPStore-based transport for micro batches using torch.distributed.TCPStore."""

    type: Literal["tcpstore"] = "tcpstore"
    host: Annotated[str, Field(description="The host address for the TCPStore master.")] = "localhost"
    port: Annotated[int, Field(description="The port for the TCPStore.")] = 29600
    timeout_seconds: Annotated[int, Field(description="Timeout in seconds for TCPStore operations.")] = 300
    wait_timeout_seconds: Annotated[
        int, Field(description="Timeout in seconds for waiting on keys in the TCPStore.")
    ] = 60


TransportConfigType: TypeAlias = FileSystemTransportConfig | ZMQTransportConfig | TCPStoreTransportConfig
