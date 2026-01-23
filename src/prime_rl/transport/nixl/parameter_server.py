#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter Server using NIXL for high-performance tensor distribution.

The server hosts a key-value map (string -> tensor) with memory registered for RDMA.
Clients can request tensors by key and receive data via high-performance RDMA transfers.
"""

import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger


def _configure_ucx_transports(tcp_only: bool = False):
    """Configure UCX to use best available transport with TCP fallback.

    UCX transport priority (best to worst):
    - rc_mlx5: Mellanox InfiniBand reliable connection (best for same-rack)
    - rc_verbs: Generic InfiniBand RC
    - dc_mlx5: Mellanox dynamically connected
    - ud_mlx5: Mellanox unreliable datagram
    - tcp: TCP sockets (works over internet, required for cross-datacenter)

    Args:
        tcp_only: If True, force TCP transport only (for internet/cross-datacenter).
                  If False, UCX picks the best available with TCP as fallback.
    """
    if "UCX_TLS" not in os.environ:
        if tcp_only:
            os.environ["UCX_TLS"] = "tcp"
        else:
            # Enable all transports, UCX will pick the best available
            os.environ["UCX_TLS"] = "all"

    # Reduce logging noise
    if "UCX_LOG_LEVEL" not in os.environ:
        os.environ["UCX_LOG_LEVEL"] = "warn"


logger = get_logger(__name__)


@dataclass
class TensorInfo:
    """Metadata about a tensor stored in the parameter server."""

    size: int  # Size in bytes
    shape: tuple  # Original tensor shape
    dtype: str  # e.g., "float32"
    device_id: int  # 0 for CPU, GPU index otherwise


@dataclass
class TensorEntry:
    """Internal entry for a registered tensor."""

    tensor: torch.Tensor  # The actual tensor
    info: TensorInfo  # Metadata
    reg_descs: object  # NIXL registration descriptors


class ParameterServer:
    """
    A tensor-based KV store using NIXL for RDMA-enabled data transfers.

    Tensors are registered individually as they are added via put().
    Clients pull data via RDMA READs (server is passive for better scalability).

    Example:
        server = ParameterServer("param_server", port=5555)

        # Add tensors - they get registered for RDMA automatically
        weight = torch.randn(512, 256)
        server.put("layer.0.weight", weight)

        # Or create and initialize in place
        bias = torch.zeros(512)
        server.put("layer.0.bias", bias)

        server.run()  # Blocking
    """

    def __init__(
        self,
        name: str,
        port: int = 5555,
        tcp_only: bool = False,
    ):
        """
        Initialize the parameter server.

        Args:
            name: Unique name for this server agent.
            port: Port for metadata exchange and notifications.
            tcp_only: Force TCP transport only (for internet/cross-datacenter).
                      Default False uses RDMA when available with TCP fallback.
        """
        self.name = name
        self.running = False

        # Configure UCX for best transport with TCP fallback
        _configure_ucx_transports(tcp_only=tcp_only)

        # Create NIXL agent with listener enabled
        config = nixl_agent_config(True, True, port)
        self.agent = nixl_agent(name, config)

        # KV store: key -> TensorEntry
        self._store: dict[str, TensorEntry] = {}
        self._store_lock = threading.Lock()

        # Track connected clients for cleanup
        self._connected_clients: set[str] = set()

        logger.info("ParameterServer '%s' initialized on port %d", name, port)

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """
        Add a tensor to the server and register it for RDMA.

        The tensor is stored by reference - modifications to the original
        tensor will be visible to clients.

        Args:
            key: Unique key to identify this tensor.
            tensor: The tensor to store. Must be contiguous.

        Raises:
            ValueError: If key already exists or tensor is not contiguous.
        """
        if not tensor.is_contiguous():
            raise ValueError("Tensor must be contiguous for RDMA registration")

        with self._store_lock:
            if key in self._store:
                raise ValueError(f"Key '{key}' already exists. Use update() to modify.")

            # Register the tensor with NIXL
            reg_descs = self.agent.register_memory(tensor)
            if not reg_descs:
                raise RuntimeError(f"Failed to register tensor '{key}' with NIXL")

            # Determine device ID
            device_id = tensor.get_device()
            if device_id == -1:  # CPU
                device_id = 0

            # Create metadata
            info = TensorInfo(
                size=tensor.numel() * tensor.element_size(),
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).replace("torch.", ""),
                device_id=device_id,
            )

            # Store entry
            self._store[key] = TensorEntry(
                tensor=tensor,
                info=info,
                reg_descs=reg_descs,
            )

            # logger.info(
            #     "Registered tensor '%s' with shape %s, dtype %s",
            #     key,
            #     info.shape,
            #     info.dtype,
            # )

    def update(self, key: str, tensor: torch.Tensor) -> None:
        """
        Update an existing tensor (deregisters old, registers new).

        Args:
            key: Key of the tensor to update.
            tensor: The new tensor.

        Raises:
            KeyError: If key doesn't exist.
        """
        with self._store_lock:
            if key not in self._store:
                raise KeyError(f"Key '{key}' not found. Use put() to add new tensors.")

            # Deregister old tensor
            old_entry = self._store[key]
            self.agent.deregister_memory(old_entry.reg_descs)

        # Remove and re-add (releases lock between operations)
        with self._store_lock:
            del self._store[key]

        self.put(key, tensor)

    def remove(self, key: str) -> None:
        """
        Remove a tensor from the server and deregister it.

        Args:
            key: Key of the tensor to remove.

        Raises:
            KeyError: If key doesn't exist.
        """
        with self._store_lock:
            if key not in self._store:
                raise KeyError(f"Key '{key}' not found")

            entry = self._store.pop(key)
            self.agent.deregister_memory(entry.reg_descs)
            logger.info("Removed tensor '%s'", key)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a tensor by key (returns the actual stored tensor).

        Args:
            key: The key to look up.

        Returns:
            The tensor if found, None otherwise.
        """
        with self._store_lock:
            entry = self._store.get(key)
            return entry.tensor if entry else None

    def keys(self) -> list[str]:
        """Return all keys in the store."""
        with self._store_lock:
            return list(self._store.keys())

    def _get_catalog(self) -> dict[str, TensorInfo]:
        """Get catalog of all tensor metadata."""
        with self._store_lock:
            return {key: entry.info for key, entry in self._store.items()}

    def run(self):
        """
        Main loop: handle client requests via notifications.

        This is a blocking call that runs until stop() is called.
        """
        self.running = True
        logger.info("ParameterServer '%s' started, waiting for clients...", self.name)

        while self.running:
            try:
                notifs = self.agent.get_new_notifs()
                for client_name, messages in notifs.items():
                    for msg in messages:
                        self._handle_request(client_name, msg)
            except Exception as e:
                if self.running:
                    logger.error("Error processing notifications: %s", e)

    def stop(self):
        """Stop the server main loop."""
        self.running = False
        logger.info("ParameterServer '%s' stopping...", self.name)

    def _wait_for_client_metadata(self, client: str, timeout: float = 5.0) -> bool:
        """Wait until we can reach the client (metadata is loaded)."""
        start = time.time()
        while time.time() - start < timeout:
            if self.agent.check_remote_metadata(client):
                return True
            time.sleep(0.01)
        return False

    def _handle_request(self, client: str, msg: bytes):
        """Handle a request from a client."""
        try:
            # Ensure we can respond to this client before processing
            if not self.agent.check_remote_metadata(client):
                logger.debug("Waiting for metadata from client '%s'", client)
                if not self._wait_for_client_metadata(client):
                    logger.error("Timeout waiting for metadata from client '%s'", client)
                    return

            if msg == b"CATALOG_REQ":
                self._handle_catalog_request(client)
            elif msg.startswith(b"DESCS_REQ:"):
                key = msg[10:].decode()
                self._handle_descs_request(client, key)
            elif msg.startswith(b"BATCH_DESCS_REQ:"):
                keys_data = msg[16:]
                keys = pickle.loads(keys_data)
                self._handle_batch_descs_request(client, keys)
            elif msg == b"HELLO":
                self._connected_clients.add(client)
                logger.info("Client '%s' connected", client)
            elif msg == b"BYE":
                self._connected_clients.discard(client)
                logger.info("Client '%s' disconnected", client)
            else:
                logger.warning("Unknown message from '%s': %s", client, msg[:50])
        except Exception as e:
            logger.error("Error handling request from '%s': %s", client, e)

    def _handle_catalog_request(self, client: str):
        """Send the catalog to a client."""
        catalog = self._get_catalog()
        catalog_data = pickle.dumps(catalog)
        self.agent.send_notif(client, b"CATALOG:" + catalog_data)
        logger.debug("Sent catalog to '%s' (%d entries)", client, len(catalog))

    def _handle_descs_request(self, client: str, key: str):
        """Send descriptors for a specific key to a client."""
        with self._store_lock:
            if key not in self._store:
                self.agent.send_notif(client, b"ERROR:Key not found: " + key.encode())
                return

            entry = self._store[key]
            tensor = entry.tensor

        # Build descriptor for this tensor
        descs = self.agent.get_xfer_descs([tensor])
        serialized = self.agent.get_serialized_descs(descs)
        self.agent.send_notif(client, b"DESCS:" + serialized)
        logger.debug("Sent descriptors for key '%s' to '%s'", key, client)

    def _handle_batch_descs_request(self, client: str, keys: list[str]):
        """Send descriptors for multiple keys to a client."""
        result = {}
        with self._store_lock:
            for key in keys:
                if key in self._store:
                    entry = self._store[key]
                    descs = self.agent.get_xfer_descs([entry.tensor])
                    result[key] = self.agent.get_serialized_descs(descs)

        response = pickle.dumps(result)
        self.agent.send_notif(client, b"BATCH_DESCS:" + response)
        logger.debug("Sent batch descriptors for %d keys to '%s'", len(result), client)

    def shutdown(self):
        """Cleanup resources."""
        self.stop()
        # Deregister all tensors
        with self._store_lock:
            for key, entry in self._store.items():
                try:
                    self.agent.deregister_memory(entry.reg_descs)
                except Exception as e:
                    logger.warning("Failed to deregister tensor '%s': %s", key, e)
            self._store.clear()
        logger.info("ParameterServer '%s' shut down", self.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIXL Parameter Server")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:N)")
    args = parser.parse_args()

    device = torch.device(args.device)

    server = ParameterServer("param_server", port=args.port)

    # Create and register tensors
    weight = torch.ones(512, 256, device=device)
    bias = torch.full((512,), 0.5, device=device)
    torch.manual_seed(42)
    embed = torch.randn(1000, 128, device=device)

    server.put("layer.0.weight", weight)
    server.put("layer.0.bias", bias)
    server.put("embedding", embed)

    print(f"Server ready with tensors: {server.keys()}")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
