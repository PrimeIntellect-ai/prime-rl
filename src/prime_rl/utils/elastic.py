"""
Elastic inference pool with DNS-based service discovery.

Automatically discovers inference pods via Kubernetes headless service DNS
and syncs new pods with current model weights.
"""

import asyncio
import socket
from pathlib import Path

from httpx import AsyncClient

from prime_rl.utils.client import load_lora_adapter, setup_admin_clients, update_weights
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger


class ElasticInferencePool:
    """Manages a dynamic pool of inference servers discovered via DNS.

    Uses Kubernetes headless service DNS to discover inference pods.
    Automatically syncs new pods with current model weights.
    """

    def __init__(
        self,
        headless_service: str,
        client_config: ClientConfig,
        port: int = 8000,
        sync_interval: float = 5.0,
    ):
        """Initialize the elastic inference pool.

        Args:
            headless_service: Kubernetes headless service hostname (e.g. "my-inference-headless")
            client_config: Client configuration for creating admin clients
            port: Port that inference servers listen on (default: 8000)
            sync_interval: How often to check for new/removed pods in seconds (default: 5.0)
        """
        self.logger = get_logger()
        self.headless_service = headless_service
        self.client_config = client_config
        self.port = port
        self.sync_interval = sync_interval

        # Track known clients by URL
        self._clients: dict[str, AsyncClient] = {}
        self._lock = asyncio.Lock()

        # Track current weights state for syncing new pods
        self._current_weights_path: Path | None = None
        self._current_lora_name: str | None = None

        self._sync_task: asyncio.Task | None = None

    def discover_pods(self) -> set[str]:
        """Discover inference pod URLs via DNS lookup.

        Returns:
            Set of URLs for all ready inference pods.
        """
        try:
            _, _, ips = socket.gethostbyname_ex(self.headless_service)
            return {f"http://{ip}:{self.port}" for ip in ips}
        except socket.gaierror as e:
            self.logger.warning(f"DNS lookup failed for {self.headless_service}: {e}")
            return set()

    @property
    def clients(self) -> list[AsyncClient]:
        """Get list of all active admin clients."""
        return list(self._clients.values())

    @property
    def urls(self) -> list[str]:
        """Get list of all active inference URLs."""
        return list(self._clients.keys())

    async def _create_client(self, url: str) -> AsyncClient:
        """Create an admin client for the given URL."""
        # Create a minimal ClientConfig with just this URL
        single_config = ClientConfig(
            timeout=self.client_config.timeout,
            base_url=[f"{url}/v1"],
            api_key_var=self.client_config.api_key_var,
            headers=self.client_config.headers,
        )
        clients = setup_admin_clients(single_config)
        return clients[0]

    async def _sync_new_pod(self, client: AsyncClient, url: str) -> bool:
        """Sync a new pod with current weights.

        Returns:
            True if sync succeeded, False otherwise.
        """
        if self._current_weights_path is None:
            self.logger.debug(f"No weights to sync for new pod {url}")
            return True

        try:
            self.logger.info(f"Syncing new pod {url} with weights from {self._current_weights_path}")
            if self._current_lora_name is not None:
                await load_lora_adapter([client], self._current_lora_name, self._current_weights_path)
            else:
                await update_weights([client], self._current_weights_path, lora_name=None)
            self.logger.info(f"Successfully synced new pod {url}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to sync new pod {url}: {e}")
            return False

    async def sync(self) -> tuple[int, int]:
        """Sync the client pool with discovered pods.

        Returns:
            Tuple of (pods_added, pods_removed).
        """
        async with self._lock:
            discovered = self.discover_pods()
            known = set(self._clients.keys())

            added = 0
            removed = 0

            # Add new pods
            for url in discovered - known:
                self.logger.info(f"Discovered new inference pod: {url}")
                try:
                    client = await self._create_client(url)
                    if await self._sync_new_pod(client, url):
                        self._clients[url] = client
                        added += 1
                    else:
                        await client.aclose()
                except Exception as e:
                    self.logger.error(f"Failed to add new pod {url}: {e}")

            # Remove dead pods
            for url in known - discovered:
                self.logger.info(f"Inference pod removed: {url}")
                client = self._clients.pop(url, None)
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                removed += 1

            return added, removed

    async def update_weights(self, weights_path: Path, lora_name: str | None = None) -> None:
        """Update weights on all inference pods and track for new pod sync.

        Args:
            weights_path: Path to the weights directory.
            lora_name: Optional LoRA adapter name.
        """
        async with self._lock:
            # Track current state for syncing new pods
            self._current_weights_path = weights_path
            self._current_lora_name = lora_name

            # Update all current clients
            if self._clients:
                if lora_name is not None:
                    await load_lora_adapter(list(self._clients.values()), lora_name, weights_path)
                else:
                    await update_weights(list(self._clients.values()), weights_path, lora_name=None)

    async def _sync_loop(self) -> None:
        """Background task that periodically syncs the pod pool."""
        while True:
            try:
                added, removed = await self.sync()
                if added > 0 or removed > 0:
                    self.logger.info(f"Elastic pool sync: +{added} -{removed} pods (total: {len(self._clients)})")
            except Exception as e:
                self.logger.error(f"Error in elastic sync loop: {e}")
            await asyncio.sleep(self.sync_interval)

    async def start(self) -> None:
        """Start the elastic inference pool and background sync task."""
        self.logger.info(
            f"Starting elastic inference pool (service={self.headless_service}, interval={self.sync_interval}s)"
        )

        # Initial sync
        await self.sync()
        self.logger.info(f"Initial discovery found {len(self._clients)} inference pod(s)")

        # Start background sync
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop the elastic inference pool and cleanup."""
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        # Close all clients
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception:
                pass
        self._clients.clear()

    def get_metrics(self) -> dict[str, float]:
        """Get metrics about the elastic pool."""
        return {
            "elastic/num_pods": len(self._clients),
        }
