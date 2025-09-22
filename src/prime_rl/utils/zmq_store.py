import asyncio
import pickle
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import zmq
import zmq.asyncio
from torch import Tensor

from prime_rl.utils.logger import get_logger


class MessageType(Enum):
    STORE = "store"
    RETRIEVE = "retrieve"
    DELETE = "delete"
    LIST = "list"
    EXISTS = "exists"


class RolloutStoreServer:
    """
    ZeroMQ server that acts as a distributed store for rollout data.
    Supports store, retrieve, delete, list, and exists operations.
    """

    def __init__(self, port: int = 5555):
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.rollout_store: Dict[str, Any] = {}
        self.running = False
        self._logger = get_logger()

    async def start(self):
        """Start the rollout store server."""
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        self._logger.info(f"Rollout store server started on port {self.port}")

        while self.running:
            try:
                # Receive request
                message = await self.socket.recv()
                request = pickle.loads(message)

                # Process request
                response = self._handle_request(request)

                # Send response
                response_data = pickle.dumps(response)
                await self.socket.send(response_data)

            except zmq.Again:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                continue
            except Exception as e:
                self._logger.error(f"Error handling request: {e}")
                error_response = {"status": "error", "message": str(e)}
                await self.socket.send(pickle.dumps(error_response))

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming requests based on message type."""
        msg_type = MessageType(request.get("type"))
        rollout_key = request.get("rollout_key")

        if msg_type == MessageType.STORE:
            rollout_data = request.get("rollout_data")
            self.rollout_store[rollout_key] = rollout_data
            self._logger.debug(f"Stored rollout '{rollout_key}'")
            return {"status": "success", "message": f"Rollout '{rollout_key}' stored"}

        elif msg_type == MessageType.RETRIEVE:
            if rollout_key in self.rollout_store:
                rollout_data = self.rollout_store[rollout_key]
                self._logger.debug(f"Retrieved rollout '{rollout_key}'")
                return {
                    "status": "success",
                    "rollout_data": rollout_data,
                }
            else:
                return {"status": "error", "message": f"Rollout '{rollout_key}' not found"}

        elif msg_type == MessageType.DELETE:
            if rollout_key in self.rollout_store:
                del self.rollout_store[rollout_key]
                self._logger.debug(f"Deleted rollout '{rollout_key}'")
                return {"status": "success", "message": f"Rollout '{rollout_key}' deleted"}
            else:
                return {"status": "error", "message": f"Rollout '{rollout_key}' not found"}

        elif msg_type == MessageType.LIST:
            rollout_list = list(self.rollout_store.keys())
            return {"status": "success", "rollouts": rollout_list}

        elif msg_type == MessageType.EXISTS:
            exists = rollout_key in self.rollout_store
            return {"status": "success", "exists": exists}

        else:
            return {"status": "error", "message": f"Unknown message type: {msg_type}"}

    async def stop(self):
        """Stop the server."""
        self.running = False
        self.socket.close()
        self.context.term()
        self._logger.info("Rollout store server stopped")


class RolloutStoreClient:
    """
    ZeroMQ client for interacting with the rollout store server.
    Provides methods to store, retrieve, delete, list, and check existence of rollouts.
    """

    def __init__(self, server_address: str = "localhost", server_port: int = 5555, timeout: int = 30000):
        self.server_address = server_address
        self.server_port = server_port
        self.timeout = timeout
        self.context = zmq.asyncio.Context()
        self.socket = None
        self._logger = get_logger()

    async def _connect(self):
        """Connect to the rollout store server."""
        if self.socket:
            self.socket.close()

        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.socket.connect(f"tcp://{self.server_address}:{self.server_port}")
        self._logger.debug(f"Connected to rollout store at {self.server_address}:{self.server_port}")

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to server and return response."""
        if not self.socket:
            await self._connect()

        try:
            # Send request
            message = pickle.dumps(request)
            await self.socket.send(message)

            # Receive response
            response_data = await self.socket.recv()
            response = pickle.loads(response_data)
            return response

        except zmq.Again:
            self._logger.error("Request timeout - reconnecting...")
            await self._connect()
            raise TimeoutError("Request timed out")
        except Exception as e:
            self._logger.error(f"Communication error: {e}")
            await self._connect()
            raise

    async def store_rollout(self, rollout_key: str, rollout_data: Any) -> bool:
        """
        Store rollout data on the server.

        Args:
            rollout_key: Unique identifier for the rollout (e.g., "step_123_rank_0")
            rollout_data: Rollout data to store

        Returns:
            bool: True if successful, False otherwise
        """
        request = {
            "type": MessageType.STORE.value,
            "rollout_key": rollout_key,
            "rollout_data": rollout_data
        }

        try:
            response = await self._send_request(request)
            success = response.get("status") == "success"

            if success:
                self._logger.debug(f"Successfully stored rollout '{rollout_key}'")
            else:
                self._logger.error(f"Failed to store rollout '{rollout_key}': {response.get('message')}")

            return success
        except Exception as e:
            self._logger.error(f"Failed to store rollout '{rollout_key}': {e}")
            return False

    async def retrieve_rollout(self, rollout_key: str) -> Optional[Any]:
        """
        Retrieve rollout data from the server.

        Args:
            rollout_key: Unique identifier for the rollout

        Returns:
            Any or None: Retrieved rollout data or None if not found
        """
        request = {
            "type": MessageType.RETRIEVE.value,
            "rollout_key": rollout_key
        }

        try:
            response = await self._send_request(request)

            if response.get("status") == "success":
                rollout_data = response.get("rollout_data")
                self._logger.debug(f"Successfully retrieved rollout '{rollout_key}'")
                return rollout_data
            else:
                self._logger.error(f"Failed to retrieve rollout '{rollout_key}': {response.get('message')}")
                return None
        except Exception as e:
            self._logger.error(f"Failed to retrieve rollout '{rollout_key}': {e}")
            return None

    async def delete_rollout(self, rollout_key: str) -> bool:
        """
        Delete a rollout from the server.

        Args:
            rollout_key: Unique identifier for the rollout

        Returns:
            bool: True if successful, False otherwise
        """
        request = {
            "type": MessageType.DELETE.value,
            "rollout_key": rollout_key
        }

        try:
            response = await self._send_request(request)
            success = response.get("status") == "success"

            if success:
                self._logger.debug(f"Successfully deleted rollout '{rollout_key}'")
            else:
                self._logger.error(f"Failed to delete rollout '{rollout_key}': {response.get('message')}")

            return success
        except Exception as e:
            self._logger.error(f"Failed to delete rollout '{rollout_key}': {e}")
            return False

    async def list_rollouts(self) -> Optional[list]:
        """
        List all rollouts stored on the server.

        Returns:
            list: List of rollout keys or None if error
        """
        request = {"type": MessageType.LIST.value}

        try:
            response = await self._send_request(request)

            if response.get("status") == "success":
                rollouts = response.get("rollouts", [])
                self._logger.debug(f"Found {len(rollouts)} rollouts on server")
                return rollouts
            else:
                self._logger.error(f"Failed to list rollouts: {response.get('message')}")
                return None
        except Exception as e:
            self._logger.error(f"Failed to list rollouts: {e}")
            return None

    async def rollout_exists(self, rollout_key: str) -> bool:
        """
        Check if a rollout exists on the server.

        Args:
            rollout_key: Unique identifier for the rollout

        Returns:
            bool: True if rollout exists, False otherwise
        """
        request = {
            "type": MessageType.EXISTS.value,
            "rollout_key": rollout_key
        }

        try:
            response = await self._send_request(request)

            if response.get("status") == "success":
                exists = response.get("exists", False)
                self._logger.debug(f"Rollout '{rollout_key}' {'exists' if exists else 'does not exist'}")
                return exists
            else:
                self._logger.error(f"Failed to check rollout existence: {response.get('message')}")
                return False
        except Exception as e:
            self._logger.error(f"Failed to check rollout existence: {e}")
            return False

    async def close(self):
        """Close the client connection."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self._logger.debug("Rollout store client closed")


class SyncRolloutStoreClient:
    """
    Synchronous version of RolloutStoreClient for use in non-async contexts.
    """

    def __init__(self, server_address: str = "localhost", server_port: int = 5555, timeout: int = 30000):
        self.server_address = server_address
        self.server_port = server_port
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = None
        self._logger = get_logger()
        self._connect()

    def _connect(self):
        """Connect to the rollout store server."""
        if self.socket:
            self.socket.close()

        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.socket.connect(f"tcp://{self.server_address}:{self.server_port}")
        self._logger.debug(f"Connected to rollout store at {self.server_address}:{self.server_port}")

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to server and return response."""
        try:
            # Send request
            message = pickle.dumps(request)
            self.socket.send(message)

            # Receive response
            response_data = self.socket.recv()
            response = pickle.loads(response_data)
            return response

        except zmq.Again:
            self._logger.error("Request timeout - reconnecting...")
            self._connect()
            raise TimeoutError("Request timed out")
        except Exception as e:
            self._logger.error(f"Communication error: {e}")
            self._connect()
            raise

    def retrieve_rollout(self, rollout_key: str) -> Optional[Any]:
        """
        Retrieve rollout data from the server.

        Args:
            rollout_key: Unique identifier for the rollout

        Returns:
            Any or None: Retrieved rollout data or None if not found
        """
        request = {
            "type": MessageType.RETRIEVE.value,
            "rollout_key": rollout_key
        }

        try:
            response = self._send_request(request)

            if response.get("status") == "success":
                rollout_data = response.get("rollout_data")
                self._logger.debug(f"Successfully retrieved rollout '{rollout_key}'")
                return rollout_data
            else:
                self._logger.debug(f"Failed to retrieve rollout '{rollout_key}': {response.get('message')}")
                return None
        except Exception as e:
            self._logger.error(f"Failed to retrieve rollout '{rollout_key}': {e}")
            return None

    def rollout_exists(self, rollout_key: str) -> bool:
        """
        Check if a rollout exists on the server.

        Args:
            rollout_key: Unique identifier for the rollout

        Returns:
            bool: True if rollout exists, False otherwise
        """
        request = {
            "type": MessageType.EXISTS.value,
            "rollout_key": rollout_key
        }

        try:
            response = self._send_request(request)

            if response.get("status") == "success":
                exists = response.get("exists", False)
                self._logger.debug(f"Rollout '{rollout_key}' {'exists' if exists else 'does not exist'}")
                return exists
            else:
                self._logger.error(f"Failed to check rollout existence: {response.get('message')}")
                return False
        except Exception as e:
            self._logger.error(f"Failed to check rollout existence: {e}")
            return False

    def close(self):
        """Close the client connection."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self._logger.debug("Rollout store client closed")


async def wait_for_rollout(client: RolloutStoreClient, rollout_key: str, interval: float = 1.0, log_interval: int = 10) -> None:
    """
    Wait for a rollout to become available on the server.

    Args:
        client: RolloutStoreClient instance
        rollout_key: Key of the rollout to wait for
        interval: Time to wait between checks (seconds)
        log_interval: How often to log waiting status (in check cycles)
    """
    logger = get_logger()
    wait_cycles = 0
    logger.debug(f"Waiting for rollout '{rollout_key}'")

    while True:
        if await client.rollout_exists(rollout_key):
            logger.debug(f"Found rollout '{rollout_key}'")
            break

        if wait_cycles % log_interval == 0 and wait_cycles > 0:
            logger.debug(f"Waiting for rollout '{rollout_key}' for {wait_cycles * interval:.1f} seconds")

        await asyncio.sleep(interval)
        wait_cycles += 1


def wait_for_rollout_sync(client: SyncRolloutStoreClient, rollout_key: str, interval: float = 1.0, log_interval: int = 10) -> None:
    """
    Synchronous version of wait_for_rollout.

    Args:
        client: SyncRolloutStoreClient instance
        rollout_key: Key of the rollout to wait for
        interval: Time to wait between checks (seconds)
        log_interval: How often to log waiting status (in check cycles)
    """
    logger = get_logger()
    wait_cycles = 0
    logger.debug(f"Waiting for rollout '{rollout_key}'")

    while True:
        if client.rollout_exists(rollout_key):
            logger.debug(f"Found rollout '{rollout_key}'")
            break

        if wait_cycles % log_interval == 0 and wait_cycles > 0:
            logger.debug(f"Waiting for rollout '{rollout_key}' for {wait_cycles * interval:.1f} seconds")

        time.sleep(interval)
        wait_cycles += 1
