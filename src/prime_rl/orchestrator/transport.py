"""
Abstract bidirectional transport using multiprocessing queues.
"""

from abc import ABC, abstractmethod
from multiprocessing import Queue
from multiprocessing.queues import Empty
from typing import Generic, TypeVar

TSend = TypeVar("TSend")
TRecv = TypeVar("TRecv")


class NoDataAvailableError(Exception):
    """Exception raised when no data is available."""

    pass


class Sender(ABC, Generic[TSend]):
    @abstractmethod
    def send(self, data: TSend) -> None:
        """Send data to the other end."""
        pass

    def close(self) -> None:
        """Close the transport."""
        pass


class Receiver(ABC, Generic[TRecv]):
    @abstractmethod
    def recv(self, block: bool = True) -> TRecv:
        """Receive data from the other end. If non-blocking and no data is available, raise NoDataAvailableError."""
        pass

    def close(self) -> None:
        """Close the transport."""
        pass


class Transport(ABC, Generic[TSend, TRecv]):
    @abstractmethod
    def send(self, data: TSend) -> None:
        """Send data to the other end."""

    @abstractmethod
    def recv(self, block: bool = True) -> TRecv:
        """Receive data from the other end."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the transport."""
        pass


class QueueTransport(Transport[TSend, TRecv]):
    def __init__(self, sender_queue: Queue[TSend], receiver_queue: Queue[TRecv]):
        self.sender_queue = sender_queue
        self.receiver_queue = receiver_queue

    def send(self, data: TSend) -> None:
        self.sender_queue.put(data)

    def recv(self, block: bool = True) -> TRecv:
        try:
            return self.receiver_queue.get_nowait()
        except Empty as e:
            raise NoDataAvailableError from e

    def close(self) -> None:
        self.sender_queue.close()
        self.receiver_queue.close()
