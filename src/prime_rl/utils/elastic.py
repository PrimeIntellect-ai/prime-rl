"""
Elastic inference pool with DNS-based service discovery.

Discovers inference pods via Kubernetes headless service DNS,
tracks which pods have the correct LoRA adapter loaded, and
only exposes ready pods to workers.
"""

import asyncio
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from httpx import AsyncClient

from prime_rl.utils.client import load_lora_adapter, setup_admin_clients, setup_clients
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger


def discover_pod_ips(headless_service: str) -> list[str]:
    """Discover pod IPs via DNS lookup on headless service.

    Args:
        headless_service: Kubernetes headless service hostname

    Returns:
        List of pod IP addresses (empty if lookup fails or no pods)
    """
    try:
        _, _, ips = socket.gethostbyname_ex(headless_service)
        return sorted(ips)  # Sort for deterministic ordering
    except socket.gaierror:
        return []


@dataclass
class LoadedAdapter:
    """Information about a loaded LoRA adapter from /v1/models."""

    name: str
    path: Path
    step: int


@dataclass
class PodState:
    """State of an individual inference pod."""

    ip: str
    url: str
    status: str = "discovering"  # discovering, syncing, ready, unhealthy
    loaded_adapter: LoadedAdapter | None = None
    sync_failures: int = 0


@dataclass
class DesiredAdapterState:
    """Desired adapter state for all pods."""

    lora_name: str | None = None
    lora_path: Path | None = None
    step: int = 0


class ElasticInferencePool:
    """Manages inference pods with DNS-based discovery and adapter sync.

    Discovers pods via Kubernetes headless service DNS, tracks which pods
    have the correct LoRA adapter loaded, and only exposes ready pods.

    Key features:
    - DNS-based pod discovery via headless service
    - Automatic adapter sync for new pods
    - Only exposes pods with correct adapter as "ready"
    - Callback notification when ready URLs change
    """

    def __init__(
        self,
        headless_service: str,
        client_config: ClientConfig,
        base_model: str | None = None,
        port: int = 8000,
        sync_interval: float = 5.0,
    ):
        """Initialize the elastic inference pool.

        Args:
            headless_service: Kubernetes headless service hostname
            client_config: Client configuration for creating admin clients
            base_model: Base model name for adapter detection
            port: Port that inference servers listen on
            sync_interval: How often to check for new/removed pods in seconds
        """
        self.logger = get_logger()
        self.headless_service = headless_service
        self.client_config = client_config
        self.base_model = base_model
        self.port = port
        self.sync_interval = sync_interval

        # Track pods and their state
        self._pods: dict[str, PodState] = {}  # ip -> PodState
        self._admin_clients: dict[str, AsyncClient] = {}  # ip -> admin client
        self._lock = asyncio.Lock()

        # Desired adapter state
        self._desired: DesiredAdapterState = DesiredAdapterState()

        # Callback when ready URLs change
        self._on_ready_urls_changed: Callable[[list[str]], None] | None = None
        self._last_ready_urls: list[str] = []

        self._sync_task: asyncio.Task | None = None
        self._started = False

    @property
    def on_ready_urls_changed(self) -> Callable[[list[str]], None] | None:
        """Get the callback for when ready URLs change."""
        return self._on_ready_urls_changed

    @on_ready_urls_changed.setter
    def on_ready_urls_changed(self, callback: Callable[[list[str]], None] | None) -> None:
        """Set the callback for when ready URLs change."""
        self._on_ready_urls_changed = callback

    def _build_url(self, ip: str) -> str:
        """Build base URL from IP address."""
        return f"http://{ip}:{self.port}"

    def _build_inference_url(self, ip: str) -> str:
        """Build inference URL (with /v1) from IP address."""
        return f"http://{ip}:{self.port}/v1"

    @property
    def ready_urls(self) -> list[str]:
        """Get list of inference URLs for ready pods (with /v1 suffix)."""
        return [self._build_inference_url(ip) for ip, pod in self._pods.items() if pod.status == "ready"]

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get list of all admin clients."""
        return list(self._admin_clients.values())

    @property
    def num_pods(self) -> int:
        """Get number of discovered pods."""
        return len(self._pods)

    @property
    def num_ready_pods(self) -> int:
        """Get number of ready pods."""
        return sum(1 for pod in self._pods.values() if pod.status == "ready")

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
        # If no adapter desired (base model inference), pod is always ready
        if self._desired.lora_path is None:
            return True
        if loaded is None:
            return False
        # Match by path or step
        return loaded.path == self._desired.lora_path or loaded.step == self._desired.step

    async def _sync_pod_adapter(self, ip: str) -> bool:
        """Sync a pod to the desired adapter state."""
        pod = self._pods.get(ip)
        if not pod:
            return False

        # Check current adapter state
        loaded = await self._get_loaded_adapter(ip)
        pod.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            pod.status = "ready"
            return True

        # Need to sync - mark as syncing
        pod.status = "syncing"

        try:
            if self._desired.lora_name and self._desired.lora_path:
                self.logger.info(f"Loading adapter {self._desired.lora_name} on {ip}")
                await load_lora_adapter([self._admin_clients[ip]], self._desired.lora_name, self._desired.lora_path)

            # Verify sync succeeded
            loaded = await self._get_loaded_adapter(ip)
            pod.loaded_adapter = loaded

            if self._adapter_matches_desired(loaded):
                pod.status = "ready"
                pod.sync_failures = 0
                self.logger.info(f"Successfully synced pod {ip}")
                return True

            pod.status = "unhealthy"
            pod.sync_failures += 1
            return False

        except Exception as e:
            pod.status = "unhealthy"
            pod.sync_failures += 1
            self.logger.error(f"Failed to sync pod {ip}: {e}")
            return False

    async def _add_pod(self, ip: str) -> bool:
        """Add a new pod to the pool."""
        self.logger.info(f"Discovered new inference pod: {ip}")

        try:
            admin_client = await self._create_admin_client(ip)
            self._admin_clients[ip] = admin_client
            self._pods[ip] = PodState(ip=ip, url=self._build_url(ip), status="discovering")
            await self._sync_pod_adapter(ip)
            return True

        except Exception as e:
            self.logger.error(f"Failed to add pod {ip}: {e}")
            self._pods.pop(ip, None)
            if ip in self._admin_clients:
                try:
                    await self._admin_clients.pop(ip).aclose()
                except Exception:
                    pass
            return False

    async def _remove_pod(self, ip: str) -> None:
        """Remove a pod from the pool."""
        self.logger.info(f"Inference pod removed: {ip}")
        self._pods.pop(ip, None)
        if ip in self._admin_clients:
            try:
                await self._admin_clients.pop(ip).aclose()
            except Exception:
                pass

    def _notify_if_ready_urls_changed(self) -> None:
        """Notify callback if ready URLs have changed."""
        current_ready = self.ready_urls
        if current_ready != self._last_ready_urls:
            self._last_ready_urls = current_ready
            if self._on_ready_urls_changed is not None:
                self._on_ready_urls_changed(current_ready)

    async def sync(self) -> tuple[int, int]:
        """Sync the pool with discovered pods."""
        async with self._lock:
            # Run blocking DNS lookup in executor to avoid event loop stalls
            loop = asyncio.get_event_loop()
            discovered_ips = set(await loop.run_in_executor(None, discover_pod_ips, self.headless_service))
            known_ips = set(self._pods.keys())

            added = 0
            removed = 0

            # Add new pods
            for ip in discovered_ips - known_ips:
                if await self._add_pod(ip):
                    added += 1

            # Remove gone pods
            for ip in known_ips - discovered_ips:
                await self._remove_pod(ip)
                removed += 1

            # Re-sync pods that aren't ready
            for ip, pod in self._pods.items():
                if pod.status != "ready":
                    await self._sync_pod_adapter(ip)

            # Notify if ready URLs changed
            self._notify_if_ready_urls_changed()

            return added, removed

    async def _sync_loop(self) -> None:
        """Background task that periodically syncs the pod pool."""
        while True:
            try:
                added, removed = await self.sync()
                if added > 0 or removed > 0:
                    self.logger.info(
                        f"Elastic pool sync: +{added} -{removed} pods "
                        f"(total: {self.num_pods}, ready: {self.num_ready_pods})"
                    )
            except Exception as e:
                self.logger.error(f"Error in elastic sync loop: {e}")
            await asyncio.sleep(self.sync_interval)

    async def start(self) -> None:
        """Start the elastic inference pool."""
        if self._started:
            return

        self.logger.info(
            f"Starting elastic inference pool (service={self.headless_service}, interval={self.sync_interval}s)"
        )

        await self.sync()
        self.logger.info(f"Initial discovery found {self.num_pods} pod(s) ({self.num_ready_pods} ready)")

        self._sync_task = asyncio.create_task(self._sync_loop())
        self._started = True

    async def stop(self) -> None:
        """Stop the elastic inference pool."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        for ip in list(self._pods.keys()):
            await self._remove_pod(ip)

        self._started = False

    async def update_weights(self, weights_path: Path, lora_name: str | None = None, step: int = 0) -> None:
        """Update weights/adapter on all inference pods.

        This sets the desired adapter state and syncs all pods. Only pods
        that successfully load the adapter are marked as ready.
        """
        async with self._lock:
            self._desired = DesiredAdapterState(
                lora_name=lora_name,
                lora_path=weights_path if lora_name else None,
                step=step,
            )

            # Sync all pods to new desired state
            for ip in list(self._pods.keys()):
                await self._sync_pod_adapter(ip)

            # Notify if ready URLs changed
            self._notify_if_ready_urls_changed()

    async def wait_for_ready(self, min_pods: int = 1, timeout: float = 300.0) -> None:
        """Wait for at least min_pods to be ready."""
        import time

        start = time.time()
        while time.time() - start < timeout:
            await self.sync()
            if self.num_ready_pods >= min_pods:
                return
            self.logger.debug(f"Waiting for pods: {self.num_ready_pods}/{min_pods} ready")
            await asyncio.sleep(self.sync_interval)

        raise TimeoutError(f"Timed out waiting for {min_pods} ready pods (got {self.num_ready_pods})")

    def get_inference_clients(self) -> list:
        """Create inference clients for ready pods.

        Returns AsyncOpenAI clients for all ready pods.
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
            "elastic/num_pods": self.num_pods,
            "elastic/num_ready_pods": self.num_ready_pods,
            "elastic/desired_step": self._desired.step,
        }
