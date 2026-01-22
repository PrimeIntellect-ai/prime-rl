"""
Parameter Client using NIXL for high-performance tensor fetching.

The client connects to a ParameterServer and requests tensors by key,
receiving data via high-performance RDMA transfers.
"""

import pickle
import time
from dataclasses import dataclass
from typing import Optional

import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TensorInfo:
    """Metadata about a tensor (must match server's TensorInfo)."""

    size: int  # Size in bytes
    shape: tuple  # Original tensor shape
    dtype: str  # e.g., "float32"
    device_id: int  # 0 for CPU, GPU index otherwise


class ParameterClient:
    """
    Client for fetching tensors from a ParameterServer via RDMA.

    The client initiates RDMA READs to pull data from the server.

    Example:
        client = ParameterClient("worker_0", "param_server", "192.168.1.100", 5555)

        # Fetch parameters
        weight = client.get("layer.0.weight")
        bias = client.get("layer.0.bias")

        # Batch fetch multiple parameters
        params = client.batch_get(["layer.0.weight", "layer.0.bias", "embedding"])
    """

    def __init__(
        self,
        name: str,
        server_name: str,
        server_ip: str,
        server_port: int,
        device: str = "cpu",
        timeout: float = 30.0,
    ):
        """
        Initialize the parameter client and connect to the server.

        Args:
            name: Unique name for this client agent.
            server_name: Name of the server agent to connect to.
            server_ip: IP address of the server.
            server_port: Port of the server.
            device: Device to allocate received tensors on ("cpu" or "cuda:N").
            timeout: Timeout in seconds for operations.
        """
        self.name = name
        self.server_name = server_name
        self.server_ip = server_ip
        self.server_port = server_port
        self.device = torch.device(device)
        self.timeout = timeout

        # Create NIXL agent (no listener needed for client)
        config = nixl_agent_config(True, True, 0)
        self.agent = nixl_agent(name, config)

        # Exchange metadata with server
        logger.info("Connecting to server '%s' at %s:%d", server_name, server_ip, server_port)
        self.agent.fetch_remote_metadata(server_name, server_ip, server_port)
        self.agent.send_local_metadata(server_ip, server_port)

        # Wait for metadata to be ready
        start_time = time.time()
        while not self.agent.check_remote_metadata(server_name):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for metadata from server '{server_name}'")
            time.sleep(0.01)

        # Send hello notification
        self.agent.send_notif(server_name, b"HELLO")

        # Request and cache catalog
        self.catalog: dict[str, TensorInfo] = {}
        self._fetch_catalog()

        logger.info(
            "Connected to server '%s', catalog has %d entries",
            server_name,
            len(self.catalog),
        )

    def _fetch_catalog(self):
        """Fetch the catalog from the server."""
        self.agent.send_notif(self.server_name, b"CATALOG_REQ")
        catalog_data = self._wait_for_response(b"CATALOG:")
        self.catalog = pickle.loads(catalog_data)

    def _wait_for_response(self, prefix: bytes, timeout: Optional[float] = None) -> bytes:
        """Wait for a response from the server with the given prefix."""
        if timeout is None:
            timeout = self.timeout

        start_time = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs:
                for msg in notifs[self.server_name]:
                    if msg.startswith(prefix):
                        return msg[len(prefix) :]
                    if msg.startswith(b"ERROR:"):
                        raise RuntimeError(f"Server error: {msg[6:].decode()}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for response with prefix {prefix}")
            time.sleep(0.001)

    def refresh_catalog(self):
        """Refresh the catalog from the server."""
        self._fetch_catalog()

    def keys(self) -> list[str]:
        """Return all available keys from the catalog."""
        return list(self.catalog.keys())

    def get_info(self, key: str) -> Optional[TensorInfo]:
        """Get tensor info for a key without fetching the data."""
        return self.catalog.get(key)

    def get(
        self,
        key: str,
        out: Optional[torch.Tensor] = None,
        timeout: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """
        Fetch a tensor by key from the server.

        Args:
            key: The key of the tensor to fetch.
            out: Optional pre-allocated output tensor (must match shape and dtype).
            timeout: Optional timeout override for this operation.

        Returns:
            The fetched tensor, or None if the key doesn't exist.
        """
        if key not in self.catalog:
            logger.warning("Key '%s' not found in catalog", key)
            return None

        info = self.catalog[key]
        dtype = getattr(torch, info.dtype)

        # Request remote descriptors for this key
        self.agent.send_notif(self.server_name, f"DESCS_REQ:{key}".encode())
        remote_descs_data = self._wait_for_response(b"DESCS:", timeout)
        remote_descs = self.agent.deserialize_descs(remote_descs_data)

        # Allocate local buffer if not provided
        if out is None:
            local_tensor = torch.empty(info.shape, dtype=dtype, device=self.device)
        else:
            if out.shape != info.shape:
                raise ValueError(f"Output tensor shape {out.shape} doesn't match expected {info.shape}")
            if out.dtype != dtype:
                raise ValueError(f"Output tensor dtype {out.dtype} doesn't match expected {dtype}")
            local_tensor = out

        # Register local memory and create descriptors
        local_reg = self.agent.register_memory(local_tensor)
        if not local_reg:
            raise RuntimeError("Failed to register local memory")

        try:
            local_descs = self.agent.get_xfer_descs([local_tensor])
            if not local_descs:
                raise RuntimeError("Failed to create local transfer descriptors")

            # Perform RDMA READ
            handle = self.agent.initialize_xfer("READ", local_descs, remote_descs, self.server_name)

            status = self.agent.transfer(handle)
            if status == "ERR":
                self.agent.release_xfer_handle(handle)
                raise RuntimeError("Transfer failed to start")

            # Wait for transfer completion
            start_time = time.time()
            actual_timeout = timeout if timeout is not None else self.timeout
            while status == "PROC":
                status = self.agent.check_xfer_state(handle)
                if time.time() - start_time > actual_timeout:
                    self.agent.release_xfer_handle(handle)
                    raise TimeoutError(f"Transfer timeout for key '{key}'")
                time.sleep(0.0001)

            self.agent.release_xfer_handle(handle)

            if status != "DONE":
                raise RuntimeError(f"Transfer failed with status: {status}")

        finally:
            self.agent.deregister_memory(local_reg)

        logger.debug("Fetched tensor '%s' with shape %s", key, info.shape)
        return local_tensor

    def batch_get(
        self,
        keys: list[str],
        timeout: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Fetch multiple tensors by key from the server.

        Args:
            keys: List of keys to fetch.
            timeout: Optional timeout override for this operation.

        Returns:
            Dictionary mapping keys to fetched tensors.
            Keys that don't exist are omitted.
        """
        # Filter valid keys
        valid_keys = [k for k in keys if k in self.catalog]
        if not valid_keys:
            return {}

        # Request batch descriptors
        self.agent.send_notif(self.server_name, b"BATCH_DESCS_REQ:" + pickle.dumps(valid_keys))
        response_data = self._wait_for_response(b"BATCH_DESCS:", timeout)
        descs_map = pickle.loads(response_data)

        results = {}
        for key in valid_keys:
            if key not in descs_map:
                continue

            info = self.catalog[key]
            dtype = getattr(torch, info.dtype)
            remote_descs = self.agent.deserialize_descs(descs_map[key])

            # Allocate local buffer
            local_tensor = torch.empty(info.shape, dtype=dtype, device=self.device)

            # Register local memory
            local_reg = self.agent.register_memory(local_tensor)
            if not local_reg:
                logger.error("Failed to register local memory for key '%s'", key)
                continue

            try:
                local_descs = self.agent.get_xfer_descs([local_tensor])
                if not local_descs:
                    logger.error("Failed to create local transfer descriptors for key '%s'", key)
                    continue

                # Perform RDMA READ
                handle = self.agent.initialize_xfer("READ", local_descs, remote_descs, self.server_name)

                status = self.agent.transfer(handle)
                if status == "ERR":
                    self.agent.release_xfer_handle(handle)
                    logger.error("Transfer failed to start for key '%s'", key)
                    continue

                # Wait for transfer completion
                start_time = time.time()
                actual_timeout = timeout if timeout is not None else self.timeout
                while status == "PROC":
                    status = self.agent.check_xfer_state(handle)
                    if time.time() - start_time > actual_timeout:
                        self.agent.release_xfer_handle(handle)
                        logger.error("Transfer timeout for key '%s'", key)
                        break
                    time.sleep(0.0001)

                self.agent.release_xfer_handle(handle)

                if status == "DONE":
                    results[key] = local_tensor
                else:
                    logger.error("Transfer failed for key '%s' with status: %s", key, status)

            finally:
                self.agent.deregister_memory(local_reg)

        logger.debug("Batch fetched %d/%d tensors", len(results), len(keys))
        return results

    def disconnect(self):
        """Disconnect from the server and cleanup resources."""
        try:
            self.agent.send_notif(self.server_name, b"BYE")
            self.agent.remove_remote_agent(self.server_name)
            self.agent.invalidate_local_metadata(self.server_ip, self.server_port)
        except Exception as e:
            logger.warning("Error during disconnect: %s", e)
        logger.info("Disconnected from server '%s'", self.server_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIXL Parameter Client")
    parser.add_argument("--server-ip", type=str, required=True, help="Server IP address")
    parser.add_argument("--server-port", type=int, default=5555, help="Server port")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:N)")
    args = parser.parse_args()

    client = ParameterClient(
        "worker_0",
        "param_server",
        args.server_ip,
        args.server_port,
        device=args.device,
    )

    print(f"Available tensors: {client.keys()}")

    # Fetch individual tensors
    for key in client.keys():
        info = client.get_info(key)
        print(f"\nFetching '{key}' (shape={info.shape}, dtype={info.dtype})...")
        tensor = client.get(key)
        if tensor is not None:
            print(f"  Received tensor with shape {tensor.shape}")
            print(f"  First few values: {tensor.flatten()[:5].tolist()}")

    client.disconnect()
