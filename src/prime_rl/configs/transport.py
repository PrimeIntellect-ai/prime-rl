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


TransportConfigType: TypeAlias = FileSystemTransportConfig | ZMQTransportConfig
