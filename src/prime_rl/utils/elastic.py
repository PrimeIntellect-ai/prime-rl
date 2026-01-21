"""
Elastic inference pool with DNS-based service discovery.

Discovers inference servers via DNS (any hostname that resolves to multiple IPs),
tracks which servers have the correct LoRA adapter loaded, and
only exposes ready servers to workers.
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from httpx import AsyncClient

from prime_rl.utils.client import load_lora_adapter, setup_admin_clients, setup_clients
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger


class WorkerServerDiscovery:
    """Lightweight server discovery for env workers.

    Unlike ElasticInferencePool, this doesn't manage adapters or admin clients.
    It just discovers and provides inference clients for rollouts, with
    round-robin client selection.
    """

    def __init__(self, client_config: ClientConfig, model_name: str):
        self._client_config = client_config
        self._model_name = model_name
        self._hostname = client_config.elastic.hostname
        self._port = client_config.elastic.port
        self._sync_interval = client_config.elastic.sync_interval
        self._clients: list = []
        self._client_index = 0
        self._last_refresh = 0.0
        self._last_urls: set[str] = set()
        self._logger = get_logger()

    @property
    def has_clients(self) -> bool:
        """Check if any clients are available."""
        return len(self._clients) > 0

    def get_next_client(self):
        """Get next client in round-robin fashion. Returns None if no clients."""
        if not self._clients:
            return None
        client = self._clients[self._client_index % len(self._clients)]
        self._client_index += 1
        return client

    async def refresh(self) -> bool:
        """Refresh clients via DNS discovery. Returns True if clients changed."""
        if not self._hostname:
            return False
        if time.time() - self._last_refresh < self._sync_interval:
            return False
        self._last_refresh = time.time()

        urls = await discover_ready_servers(self._hostname, self._port, self._model_name)
        if set(urls) == self._last_urls:
            return False
        self._last_urls = set(urls)

        if not urls:
            self._logger.debug("No ready inference servers found")
            self._clients = []
            return True

        self._logger.debug(f"Discovered {len(urls)} ready server(s)")
        self._clients = setup_clients(
            ClientConfig(
                timeout=self._client_config.timeout,
                base_url=urls,
                api_key_var=self._client_config.api_key_var,
                headers=self._client_config.headers,
            )
        )
        self._client_index = 0  # Reset round-robin on refresh
        return True

    async def close(self) -> None:
        """Close all clients."""
        for c in self._clients:
            await c.close()
        self._clients = []


def discover_server_ips(hostname: str) -> list[str]:
    """Discover server IPs via DNS lookup.

    Args:
        hostname: DNS hostname that resolves to one or more IP addresses

    Returns:
        List of server IP addresses (empty if lookup fails or no records)
    """
    try:
        _, _, ips = socket.gethostbyname_ex(hostname)
        return sorted(ips)  # Sort for deterministic ordering
    except socket.gaierror:
        return []


async def check_server_model(url: str, model_name: str, timeout: float = 5.0) -> tuple[bool, bool]:
    """Check if a server has a specific model loaded.

    Args:
        url: Base URL of the server (without /v1)
        model_name: Model name to check for
        timeout: Request timeout in seconds

    Returns:
        Tuple of (has_model, is_healthy) booleans
    """
    logger = get_logger()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            return model_name in models, len(models) > 0
    except Exception as e:
        logger.debug(f"Failed to check server {url}: {e}")
        return False, False


async def discover_ready_servers(hostname: str, port: int, model_name: str) -> list[str]:
    """Discover servers via DNS with majority vote logic.

    - If NO servers have the model: return all healthy servers (base model mode)
    - If ANY server has the model: return only those with it (adapter mode)

    Args:
        hostname: DNS hostname to resolve
        port: Port that servers listen on
        model_name: Model name to check for

    Returns:
        List of ready server URLs (with /v1 suffix)
    """
    loop = asyncio.get_event_loop()
    ips = await loop.run_in_executor(None, discover_server_ips, hostname)
    if not ips:
        return []

    checks = [check_server_model(f"http://{ip}:{port}", model_name) for ip in ips]
    results = await asyncio.gather(*checks, return_exceptions=True)

    with_model, healthy = set(), set()
    for ip, result in zip(ips, results):
        if isinstance(result, Exception):
            continue
        has_model, is_healthy = result
        url = f"http://{ip}:{port}/v1"
        if has_model:
            with_model.add(url)
        if is_healthy:
            healthy.add(url)

    return sorted(with_model) if with_model else sorted(healthy)


@dataclass
class LoadedAdapter:
    """Information about a loaded LoRA adapter from /v1/models."""

    name: str
    path: Path
    step: int


ServerStatus = Literal["discovering", "syncing", "ready", "unhealthy"]


@dataclass
class ServerState:
    """State of an individual inference server."""

    ip: str
    url: str
    status: ServerStatus = "discovering"
    loaded_adapter: LoadedAdapter | None = None
    sync_failures: int = 0


@dataclass
class DesiredAdapterState:
    """Desired adapter state for all pods."""

    lora_name: str | None = None
    lora_path: Path | None = None
    step: int = 0


class ElasticInferencePool:
    """Manages inference servers with DNS-based discovery and adapter sync.

    Discovers servers via DNS lookup, tracks which servers have the correct
    LoRA adapter loaded, and only exposes ready servers to workers.

    Key features:
    - DNS-based server discovery (works with any multi-IP DNS record)
    - Automatic adapter sync for new servers
    - Only exposes servers with correct adapter as "ready"
    - Callback notification when ready URLs change
    """

    def __init__(
        self,
        hostname: str,
        client_config: ClientConfig,
        base_model: str | None = None,
        port: int = 8000,
        sync_interval: float = 5.0,
    ):
        """Initialize the elastic inference pool.

        Args:
            hostname: DNS hostname that resolves to inference server IPs
            client_config: Client configuration for creating admin clients
            base_model: Base model name for adapter detection
            port: Port that inference servers listen on
            sync_interval: How often to check for new/removed servers in seconds
        """
        self.logger = get_logger()
        self.hostname = hostname
        self.client_config = client_config
        self.base_model = base_model
        self.port = port
        self.sync_interval = sync_interval

        # Track servers and their state
        self._servers: dict[str, ServerState] = {}  # ip -> ServerState
        self._admin_clients: dict[str, AsyncClient] = {}  # ip -> admin client
        self._lock = asyncio.Lock()

        # Desired adapter state
        self._desired: DesiredAdapterState = DesiredAdapterState()

        self._sync_task: asyncio.Task | None = None
        self._started = False

    @classmethod
    async def from_config(cls, config: ClientConfig, base_model: str | None = None) -> ElasticInferencePool:
        """Create and start an elastic pool from ClientConfig."""
        pool = cls(
            hostname=config.elastic.hostname,
            client_config=config,
            base_model=base_model,
            port=config.elastic.port,
            sync_interval=config.elastic.sync_interval,
        )
        await pool.start()
        return pool

    def _build_url(self, ip: str) -> str:
        """Build base URL from IP address."""
        return f"http://{ip}:{self.port}"

    def _build_inference_url(self, ip: str) -> str:
        """Build inference URL (with /v1) from IP address."""
        return f"http://{ip}:{self.port}/v1"

    @property
    def ready_urls(self) -> list[str]:
        """Get list of inference URLs for ready servers (with /v1 suffix)."""
        return [self._build_inference_url(ip) for ip, server in self._servers.items() if server.status == "ready"]

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get list of all admin clients."""
        return list(self._admin_clients.values())

    @property
    def num_servers(self) -> int:
        """Get number of discovered servers."""
        return len(self._servers)

    @property
    def num_ready_servers(self) -> int:
        """Get number of ready servers."""
        return sum(1 for server in self._servers.values() if server.status == "ready")

    async def _create_admin_client(self, ip: str) -> AsyncClient:
        """Create an admin client for the given IP."""
        url = self._build_url(ip)
        config = ClientConfig(
            timeout=self.client_config.timeout,
            base_url=[f"{url}/v1"],
            api_key_var=self.client_config.api_key_var,
            headers=self.client_config.headers,
        )
        clients = setup_admin_clients(config)
        return clients[0]

    async def _get_loaded_adapter(self, ip: str) -> LoadedAdapter | None:
        """Query /v1/models to get currently loaded adapter state."""
        if ip not in self._admin_clients:
            return None

        try:
            admin = self._admin_clients[ip]
            response = await admin.get("/v1/models")
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                parent = model.get("parent")
                if parent and (self.base_model is None or parent == self.base_model):
                    root = model.get("root", "")
                    path = Path(root)
                    try:
                        step_part = path.name
                        if step_part.startswith("step_"):
                            step = int(step_part.split("_")[1])
                        elif step_part.startswith("step-"):
                            step = int(step_part.split("-")[1])
                        else:
                            step = 0
                    except (ValueError, IndexError):
                        step = 0
                    return LoadedAdapter(name=model.get("id", ""), path=path, step=step)

            return None

        except Exception as e:
            self.logger.warning(f"Failed to query /v1/models on {ip}: {e}")
            return None

    def _adapter_matches_desired(self, loaded: LoadedAdapter | None) -> bool:
        """Check if loaded adapter matches desired state."""
        # If no adapter desired (base model inference), server is always ready
        if self._desired.lora_path is None:
            return True
        if loaded is None:
            return False
        # Match by path first
        if loaded.path == self._desired.lora_path:
            return True
        # Only match by step if step > 0 (avoid false match on default step=0)
        if self._desired.step > 0 and loaded.step == self._desired.step:
            return True
        return False

    async def _sync_server_adapter(self, ip: str) -> bool:
        """Sync a server to the desired adapter state."""
        server = self._servers.get(ip)
        if not server:
            return False

        # Check current adapter state
        loaded = await self._get_loaded_adapter(ip)
        server.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            server.status = "ready"
            return True

        # Need to sync - mark as syncing
        server.status = "syncing"

        if self._desired.lora_name and self._desired.lora_path:
            try:
                self.logger.debug(f"Loading adapter {self._desired.lora_name} on {ip}")
                await load_lora_adapter([self._admin_clients[ip]], self._desired.lora_name, self._desired.lora_path)
            except Exception as e:
                server.status = "unhealthy"
                server.sync_failures += 1
                self.logger.error(f"Failed to sync server {ip}: {e}")
                return False

        # Verify sync succeeded
        loaded = await self._get_loaded_adapter(ip)
        server.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            server.status = "ready"
            server.sync_failures = 0
            self.logger.debug(f"Successfully synced server {ip}")
            return True

        server.status = "unhealthy"
        server.sync_failures += 1
        return False

    async def _check_server_health(self, admin_client: AsyncClient, ip: str) -> bool:
        """Check if server is healthy and has the base model loaded."""
        try:
            # Check /health endpoint
            response = await admin_client.get("/health")
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Server {ip} health check failed: {e}")
            return False

        try:
            # Check if base model is available
            response = await admin_client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]

            if self.base_model is not None and self.base_model not in models:
                self.logger.debug(f"Server {ip} does not have base model {self.base_model}, found: {models}")
                return False
        except Exception as e:
            self.logger.debug(f"Server {ip} model check failed: {e}")
            return False

        return True

    async def _add_server(self, ip: str) -> bool:
        """Add a new server to the pool."""
        try:
            admin_client = await self._create_admin_client(ip)
        except Exception as e:
            self.logger.debug(f"Failed to create admin client for {ip}: {e}")
            return False

        # Check health and base model before announcing discovery
        if not await self._check_server_health(admin_client, ip):
            await admin_client.aclose()
            return False

        self.logger.debug(f"Discovered new inference server: {ip}")
        self._admin_clients[ip] = admin_client
        self._servers[ip] = ServerState(ip=ip, url=self._build_url(ip), status="discovering")
        await self._sync_server_adapter(ip)
        return True

    async def _remove_server(self, ip: str) -> None:
        """Remove a server from the pool."""
        self.logger.debug(f"Inference server removed: {ip}")
        self._servers.pop(ip, None)
        if ip in self._admin_clients:
            await self._admin_clients.pop(ip).aclose()

    async def sync(self) -> tuple[int, int]:
        """Sync the pool with discovered servers."""
        async with self._lock:
            # Run blocking DNS lookup in executor to avoid event loop stalls
            loop = asyncio.get_event_loop()
            discovered_ips = set(await loop.run_in_executor(None, discover_server_ips, self.hostname))
            known_ips = set(self._servers.keys())

            added = 0
            removed = 0

            # Add new servers (only if they pass health check)
            for ip in discovered_ips - known_ips:
                if await self._add_server(ip):
                    added += 1

            # Remove servers no longer in DNS
            for ip in known_ips - discovered_ips:
                await self._remove_server(ip)
                removed += 1

            # Health check known servers and remove unhealthy ones
            for ip in list(self._servers.keys()):
                if ip not in self._admin_clients:
                    continue
                if not await self._check_server_health(self._admin_clients[ip], ip):
                    self.logger.debug(f"Server {ip} failed health check, removing")
                    await self._remove_server(ip)
                    removed += 1
                elif self._servers[ip].status != "ready":
                    # Re-sync servers that aren't ready but are healthy
                    await self._sync_server_adapter(ip)

            return added, removed

    async def _sync_loop(self) -> None:
        """Background task that periodically syncs the server pool."""
        while True:
            try:
                added, removed = await self.sync()
                if added > 0 or removed > 0:
                    self.logger.debug(
                        f"Elastic pool sync: +{added} -{removed} servers "
                        f"(total: {self.num_servers}, ready: {self.num_ready_servers})"
                    )
            except Exception as e:
                self.logger.error(f"Error in elastic sync loop: {e}")
            await asyncio.sleep(self.sync_interval)

    async def start(self) -> None:
        """Start the elastic inference pool."""
        if self._started:
            return

        self.logger.debug(
            f"Starting elastic inference pool (hostname={self.hostname}, sync_interval={self.sync_interval})"
        )

        await self.sync()
        self.logger.debug(f"Initial discovery found {self.num_servers} server(s) ({self.num_ready_servers} ready)")

        self._sync_task = asyncio.create_task(self._sync_loop())
        self._started = True

    async def stop(self) -> None:
        """Stop the elastic inference pool."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling the sync task

        for ip in list(self._servers.keys()):
            await self._remove_server(ip)

        self._started = False

    async def sync_weights(self, weights_path: Path, lora_name: str | None = None, step: int = 0) -> None:
        """Sync weights/adapter across all servers with verification.

        Sets the desired adapter state, loads the adapter on each server, and verifies
        it was loaded correctly. Only servers that successfully load the adapter are
        marked as ready and will receive inference requests.
        """
        async with self._lock:
            self._desired = DesiredAdapterState(
                lora_name=lora_name,
                lora_path=weights_path if lora_name else None,
                step=step,
            )

            # Sync all servers to new desired state
            for ip in list(self._servers.keys()):
                await self._sync_server_adapter(ip)

    async def wait_for_ready(self, min_servers: int = 1, timeout: float = 300.0) -> None:
        """Wait for at least min_servers to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            await self.sync()
            if self.num_ready_servers >= min_servers:
                return
            self.logger.debug(f"Waiting for servers: {self.num_ready_servers}/{min_servers} ready")
            await asyncio.sleep(self.sync_interval)

        raise TimeoutError(f"Timed out waiting for {min_servers} ready servers (got {self.num_ready_servers})")

    def get_inference_clients(self) -> list:
        """Create inference clients for ready servers.

        Returns AsyncOpenAI clients for all ready servers.
        """
        ready_urls = self.ready_urls
        if not ready_urls:
            return []

        config = ClientConfig(
            timeout=self.client_config.timeout,
            base_url=ready_urls,
            api_key_var=self.client_config.api_key_var,
            headers=self.client_config.headers,
        )
        return setup_clients(config)

    def get_metrics(self) -> dict[str, float]:
        """Get metrics about the elastic pool."""
        return {
            "elastic/num_servers": self.num_servers,
            "elastic/num_ready_servers": self.num_ready_servers,
            "elastic/desired_step": self._desired.step,
        }
