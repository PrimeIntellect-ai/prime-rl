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

# --- Shared discovery functions ---


def discover_server_ips(hostname: str) -> list[str]:
    """Discover server IPs via DNS lookup."""
    try:
        _, _, ips = socket.gethostbyname_ex(hostname)
        return sorted(ips)
    except socket.gaierror:
        return []


async def check_server_model(url: str, model_name: str, timeout: float = 5.0) -> tuple[bool, bool]:
    """Check if a server has a specific model loaded."""
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
    """Discover servers via DNS that have the requested model loaded."""
    loop = asyncio.get_event_loop()
    ips = await loop.run_in_executor(None, discover_server_ips, hostname)
    if not ips:
        return []

    checks = [check_server_model(f"http://{ip}:{port}", model_name) for ip in ips]
    results = await asyncio.gather(*checks, return_exceptions=True)

    with_model = set()
    for ip, result in zip(ips, results):
        if isinstance(result, Exception):
            continue
        has_model, _ = result
        if has_model:
            with_model.add(f"http://{ip}:{port}/v1")

    return sorted(with_model)


# --- ServerDiscovery: for env workers ---


class ServerDiscovery:
    """DNS-based server discovery for env workers.

    Discovers servers via DNS, checks if they have the model, provides clients.
    """

    def __init__(
        self,
        hostname: str,
        port: int,
        model_name: str,
        client_config: ClientConfig,
        sync_interval: float = 5.0,
    ):
        self.logger = get_logger()
        self.hostname = hostname
        self.port = port
        self.model_name = model_name
        self.client_config = client_config
        self.sync_interval = sync_interval

        self._clients: list = []
        self._urls: list[str] = []
        self._client_index = 0
        self._last_refresh = 0.0

    @classmethod
    def from_config(cls, config: ClientConfig, model_name: str) -> ServerDiscovery:
        return cls(
            hostname=config.elastic.hostname,
            port=config.elastic.port,
            model_name=model_name,
            client_config=config,
            sync_interval=config.elastic.sync_interval,
        )

    @property
    def has_clients(self) -> bool:
        return len(self._clients) > 0

    def get_next_client(self):
        """Get next client in round-robin fashion."""
        if not self._clients:
            return None
        client = self._clients[self._client_index % len(self._clients)]
        self._client_index += 1
        return client

    async def refresh(self) -> bool:
        """Refresh clients via DNS discovery. Rate-limited by sync_interval."""
        if time.time() - self._last_refresh < self.sync_interval:
            return False
        self._last_refresh = time.time()

        urls = await discover_ready_servers(self.hostname, self.port, self.model_name)
        if set(urls) == set(self._urls):
            return False

        self._urls = urls
        if urls:
            self.logger.debug(f"Discovered {len(urls)} ready server(s)")
            self._clients = setup_clients(ClientConfig(
                timeout=self.client_config.timeout,
                base_url=urls,
                api_key_var=self.client_config.api_key_var,
                headers=self.client_config.headers,
            ))
        else:
            self.logger.debug("No ready inference servers found")
            self._clients = []
        self._client_index = 0
        return True

    async def stop(self) -> None:
        for client in self._clients:
            await client.close()
        self._clients = []


# --- ElasticInferencePool: for orchestrator ---


@dataclass
class AdapterState:
    """State of a LoRA adapter (loaded or desired)."""
    name: str | None = None
    path: Path | None = None
    step: int = 0


ServerStatus = Literal["discovering", "syncing", "ready", "unhealthy"]


@dataclass
class ServerState:
    """State of an individual inference server."""
    ip: str
    url: str
    status: ServerStatus = "discovering"
    loaded_adapter: AdapterState | None = None
    sync_failures: int = 0


class ElasticInferencePool:
    """Manages inference servers with DNS-based discovery and adapter sync.

    Discovers servers via DNS, tracks adapter state, syncs LoRA adapters.
    """

    def __init__(
        self,
        hostname: str,
        client_config: ClientConfig,
        base_model: str | None = None,
        port: int = 8000,
        sync_interval: float = 5.0,
    ):
        self.logger = get_logger()
        self.hostname = hostname
        self.client_config = client_config
        self.base_model = base_model
        self.port = port
        self.sync_interval = sync_interval

        self._servers: dict[str, ServerState] = {}
        self._admin_clients: dict[str, AsyncClient] = {}
        self._lock = asyncio.Lock()
        self._desired: AdapterState = AdapterState()

        self._clients: list = []
        self._client_urls: list[str] = []

        self._sync_task: asyncio.Task | None = None
        self._started = False

    @classmethod
    async def from_config(cls, config: ClientConfig, base_model: str | None = None) -> ElasticInferencePool:
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
        return f"http://{ip}:{self.port}"

    def _build_inference_url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}/v1"

    @property
    def ready_urls(self) -> list[str]:
        return [self._build_inference_url(ip) for ip, s in self._servers.items() if s.status == "ready"]

    @property
    def clients(self) -> list:
        urls = self.ready_urls
        if set(urls) != set(self._client_urls):
            self._client_urls = urls
            self._clients = setup_clients(ClientConfig(
                timeout=self.client_config.timeout,
                base_url=urls,
                api_key_var=self.client_config.api_key_var,
                headers=self.client_config.headers,
            )) if urls else []
        return self._clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return list(self._admin_clients.values())

    @property
    def num_servers(self) -> int:
        return len(self._servers)

    @property
    def num_ready_servers(self) -> int:
        return sum(1 for s in self._servers.values() if s.status == "ready")

    async def _create_admin_client(self, ip: str) -> AsyncClient:
        url = self._build_url(ip)
        config = ClientConfig(
            timeout=self.client_config.timeout,
            base_url=[f"{url}/v1"],
            api_key_var=self.client_config.api_key_var,
            headers=self.client_config.headers,
        )
        return setup_admin_clients(config)[0]

    async def _get_loaded_adapter(self, ip: str) -> AdapterState | None:
        if ip not in self._admin_clients:
            return None

        try:
            admin = self._admin_clients[ip]
            response = await admin.get("/v1/models")
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                parent = model.get("parent")
                model_id = model.get("id", "")

                if self._desired.name:
                    if model_id != self._desired.name:
                        continue
                elif not parent:
                    continue
                elif self.base_model is not None and parent != self.base_model:
                    continue

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
                return AdapterState(name=model_id, path=path, step=step)

            self.logger.debug(f"No matching adapter found on {ip} for desired={self._desired.name}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to query /v1/models on {ip}: {e}")
            return None

    def _adapter_matches_desired(self, loaded: AdapterState | None) -> bool:
        if self._desired.path is None:
            return True
        if loaded is None:
            return False
        if loaded.path == self._desired.path:
            return True
        if self._desired.step > 0 and loaded.step == self._desired.step:
            return True
        return False

    async def _sync_server_adapter(self, ip: str) -> bool:
        server = self._servers.get(ip)
        if not server:
            return False

        loaded = await self._get_loaded_adapter(ip)
        server.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            server.status = "ready"
            return True

        # Debug: log why pre-check failed (before attempting load)
        self.logger.debug(
            f"Pre-check failed on {ip}: loaded={loaded.path if loaded else None} "
            f"(step={loaded.step if loaded else None}), desired={self._desired.path} (step={self._desired.step})"
        )
        server.status = "syncing"

        if self._desired.name and self._desired.path:
            try:
                self.logger.debug(f"Loading adapter {self._desired.name} on {ip}")
                await load_lora_adapter([self._admin_clients[ip]], self._desired.name, self._desired.path)
            except Exception as e:
                server.status = "unhealthy"
                server.sync_failures += 1
                self.logger.error(f"Failed to sync server {ip}: {e}")
                return False

        loaded = await self._get_loaded_adapter(ip)
        server.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            server.status = "ready"
            server.sync_failures = 0
            self.logger.debug(f"Successfully synced server {ip}")
            return True

        # Debug: log why adapter didn't match after loading
        self.logger.warning(
            f"Adapter mismatch on {ip} after loading: "
            f"loaded={loaded.path if loaded else None} (step={loaded.step if loaded else None}), "
            f"desired={self._desired.path} (step={self._desired.step})"
        )
        server.status = "unhealthy"
        server.sync_failures += 1
        return False

    async def _check_server_health(self, admin_client: AsyncClient, ip: str) -> bool:
        try:
            response = await admin_client.get("/health")
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Server {ip} health check failed: {e}")
            return False

        try:
            response = await admin_client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]

            if self.base_model is not None and self.base_model not in models:
                self.logger.debug(f"Server {ip} does not have base model {self.base_model}")
                return False
        except Exception as e:
            self.logger.debug(f"Server {ip} model check failed: {e}")
            return False

        return True

    async def _add_server(self, ip: str) -> bool:
        try:
            admin_client = await self._create_admin_client(ip)
        except Exception as e:
            self.logger.debug(f"Failed to create admin client for {ip}: {e}")
            return False

        if not await self._check_server_health(admin_client, ip):
            await admin_client.aclose()
            return False

        self.logger.debug(f"Discovered new inference server: {ip}")
        self._admin_clients[ip] = admin_client
        self._servers[ip] = ServerState(ip=ip, url=self._build_url(ip), status="discovering")
        await self._sync_server_adapter(ip)
        return True

    async def _remove_server(self, ip: str) -> None:
        self.logger.debug(f"Inference server removed: {ip}")
        self._servers.pop(ip, None)
        if ip in self._admin_clients:
            await self._admin_clients.pop(ip).aclose()

    async def sync(self) -> tuple[int, int]:
        async with self._lock:
            discovered_ips = set(await asyncio.get_event_loop().run_in_executor(
                None, discover_server_ips, self.hostname
            ))
            known_ips = set(self._servers.keys())

            added = 0
            removed = 0

            for ip in discovered_ips - known_ips:
                if await self._add_server(ip):
                    added += 1

            for ip in known_ips - discovered_ips:
                await self._remove_server(ip)
                removed += 1

            for ip in list(self._servers.keys()):
                if ip not in self._admin_clients:
                    continue
                if not await self._check_server_health(self._admin_clients[ip], ip):
                    self.logger.debug(f"Server {ip} failed health check, removing")
                    await self._remove_server(ip)
                    removed += 1
                elif self._servers[ip].status != "ready":
                    await self._sync_server_adapter(ip)

            return added, removed

    async def _sync_loop(self) -> None:
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
        if self._started:
            return

        self.logger.debug(f"Starting elastic inference pool (hostname={self.hostname})")
        await self.sync()
        self.logger.debug(f"Initial discovery: {self.num_servers} server(s), {self.num_ready_servers} ready")
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._started = True

    async def stop(self) -> None:
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        for ip in list(self._servers.keys()):
            await self._remove_server(ip)

        # Close inference clients (admin clients are closed in _remove_server)
        for client in self._clients:
            await client.close()
        self._clients = []
        self._client_urls = []

        self._started = False

    async def sync_weights(self, weights_path: Path, lora_name: str | None = None, step: int = 0) -> None:
        async with self._lock:
            self._desired = AdapterState(
                name=lora_name,
                path=weights_path if lora_name else None,
                step=step,
            )
            # Sync all servers in parallel for faster weight updates
            await asyncio.gather(*[self._sync_server_adapter(ip) for ip in self._servers.keys()])

    async def wait_for_ready(self, model_name: str = "", timeout: int = 1800, min_servers: int = 1) -> None:
        start = time.time()
        while time.time() - start < timeout:
            await self.sync()
            if self.num_ready_servers >= min_servers:
                return
            self.logger.debug(f"Waiting for servers: {self.num_ready_servers}/{min_servers} ready")
            await asyncio.sleep(self.sync_interval)

        raise TimeoutError(f"Timed out waiting for {min_servers} ready servers (got {self.num_ready_servers})")

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        if lora_name is None:
            raise ValueError("Elastic inference pool requires LoRA training (lora_name must be set)")
        await self.sync_weights(weight_dir, lora_name, step)

    def get_metrics(self) -> dict[str, float]:
        return {
            "elastic/num_servers": self.num_servers,
            "elastic/num_ready_servers": self.num_ready_servers,
            "elastic/desired_step": self._desired.step,
        }
