from typing import Annotated, Literal

import zmq
from pydantic import Field

from zeroband.utils.pydantic_config import BaseConfig


class ZMQServerConfig(BaseConfig):
    """Configures a ZMQ server socket."""

    method: Annotated[Literal["tcp"], Field(default="tcp")]
    host: Annotated[str, Field(default="*")]
    port: Annotated[int, Field(default=5555)]


class ZMQClientConfig(BaseConfig):
    """Configures a ZMQ client socket."""

    method: Annotated[Literal["tcp"], Field(default="tcp")]
    host: Annotated[str, Field(default="localhost")]
    port: Annotated[int, Field(default=5555)]


def setup_server(config: ZMQServerConfig):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"{config.method}://{config.host}:{config.port}")

    return socket


def setup_client(config: ZMQClientConfig) -> zmq.SyncSocket:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"{config.method}://{config.host}:{config.port}")

    return socket


if __name__ == "__main__":

    def server():
        socket = setup_server(ZMQServerConfig())
        message = socket.recv()
        print(f"Server received: {message}")
        socket.send(b"World")
        socket.close()

    def client():
        socket = setup_client(ZMQClientConfig())
        socket.send(b"Hello")
        message = socket.recv()
        print(f"Client received: {message}")
        socket.close()

    from multiprocessing import Process

    Process(target=server).start()
    Process(target=client).start()
