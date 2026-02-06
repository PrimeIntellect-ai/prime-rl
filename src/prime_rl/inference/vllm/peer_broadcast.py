"""
Peer-to-peer weight broadcasting for fast autoscaling.

When a new inference pod starts with --load-format dummy, it can discover
healthy peers via K8s DNS and request weights via NCCL instead of loading
from disk. This dramatically speeds up pod scaling.

Flow:
1. New pod starts with dummy weights (fast, no disk I/O)
2. Discovers healthy peers via K8s headless service DNS
3. Requests weights from a healthy peer via HTTP
4. Receives weights via NCCL broadcast
5. Loads weights into model and becomes ready
"""

import os
import socket
import threading
import time
from typing import TYPE_CHECKING

import requests
import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.nccl import receive_integer, receive_state_dict
from prime_rl.trainer.rl.broadcast.nccl import broadcast_integer, broadcast_state_dict

if TYPE_CHECKING:
    from torch.nn import Module

logger = init_logger("prime_rl.inference.peer_broadcast")

DEFAULT_NCCL_PORT = 29600
DEFAULT_HTTP_PORT = 8000
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 5

# Lock to prevent concurrent broadcasts from the same source
_broadcast_lock = threading.Lock()


def get_pod_ip() -> str:
    """Get the current pod's IP address."""
    return os.environ.get("POD_IP", socket.gethostbyname(socket.gethostname()))


def get_namespace() -> str | None:
    """Get the current K8s namespace from service account or env var."""
    # Try env var first (can be set via downward API)
    if ns := os.environ.get("POD_NAMESPACE"):
        return ns

    # Try service account mount
    sa_namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(sa_namespace_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def discover_peers(headless_service: str, namespace: str) -> list[str]:
    """
    Discover peer pod IPs via K8s headless service DNS.

    Args:
        headless_service: Name of the headless service (e.g., "my-app-inference-headless")
        namespace: K8s namespace

    Returns:
        List of peer IP addresses (excluding self)
    """
    my_ip = get_pod_ip()
    dns_name = f"{headless_service}.{namespace}.svc.cluster.local"

    logger.info(f"Discovering peers via DNS: {dns_name}")

    try:
        _, _, ips = socket.gethostbyname_ex(dns_name)
        peers = [ip for ip in ips if ip != my_ip]
        logger.info(f"Found {len(peers)} peers: {peers}")
        return peers
    except socket.gaierror as e:
        logger.warning(f"DNS lookup failed for {dns_name}: {e}")
        return []


def find_healthy_peer(
    peers: list[str],
    http_port: int = DEFAULT_HTTP_PORT,
    timeout: float = 5.0,
) -> str | None:
    """
    Find a peer that is healthy and has model loaded.

    Checks /v1/models endpoint to verify model is loaded and ready.

    Args:
        peers: List of peer IP addresses
        http_port: HTTP port to check health on
        timeout: Request timeout in seconds

    Returns:
        IP of healthy peer, or None if no healthy peer found
    """
    for ip in peers:
        try:
            # Check /v1/models to verify model is loaded and ready
            resp = requests.get(f"http://{ip}:{http_port}/v1/models", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # vLLM returns {"data": [{"id": "model-name", ...}]} when model is loaded
                models = data.get("data", [])
                if models:
                    logger.info(f"Found healthy peer with model: {ip} (models: {[m.get('id') for m in models]})")
                    return ip
                else:
                    logger.debug(f"Peer {ip} has no models loaded")
        except requests.RequestException as e:
            logger.debug(f"Peer {ip} not ready: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error checking peer {ip}: {e}")

    logger.warning("No healthy peer with model found")
    return None


def receive_weights_from_peer(
    peer_ip: str,
    http_port: int = DEFAULT_HTTP_PORT,
    nccl_port: int = DEFAULT_NCCL_PORT,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, torch.Tensor]:
    """
    Request and receive weights from a peer via NCCL.

    The peer acts as rank 0 (TCP store master), we are rank 1.
    We trigger the broadcast first (so peer starts TCP store), then connect.

    Args:
        peer_ip: IP address of the peer to receive from
        http_port: HTTP port for triggering broadcast on peer
        nccl_port: Port for NCCL communication
        timeout: Timeout for NCCL operations

    Returns:
        State dict with received weights

    Raises:
        RuntimeError: If weight transfer fails
    """
    import concurrent.futures

    logger.info(f"Requesting weights from peer {peer_ip}:{nccl_port}")

    received_weights: dict[str, torch.Tensor] = {}
    broadcast_error: Exception | None = None
    broadcast_started = threading.Event()

    def trigger_broadcast():
        """Trigger broadcast on peer in background thread."""
        nonlocal broadcast_error
        try:
            logger.info("Triggering broadcast on peer...")
            resp = requests.post(
                f"http://{peer_ip}:{http_port}/broadcast_weights_to_peer",
                json={"nccl_port": nccl_port},
                timeout=600,  # Weight transfer can take a while for large models
            )
            if resp.status_code != 200:
                broadcast_error = RuntimeError(f"Peer broadcast failed: {resp.text}")
            else:
                logger.info("Peer broadcast completed")
        except Exception as e:
            broadcast_error = e

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Start broadcast trigger in background - this makes peer start TCP store
        broadcast_future = executor.submit(trigger_broadcast)

        # Give peer time to start TCP store before we try to connect
        time.sleep(2)

        # Now connect to peer's TCP store (peer is rank 0 master, we're rank 1)
        logger.info(f"Connecting to NCCL at {peer_ip}:{nccl_port}...")
        try:
            pg = StatelessProcessGroup.create(
                host=peer_ip,
                port=nccl_port,
                rank=1,
                world_size=2,
                store_timeout=timeout,
            )
            comm = PyNcclCommunicator(pg, device=torch.cuda.current_device())
            logger.info("NCCL connected!")

            start = time.time()
            num_chunks = receive_integer(comm)
            logger.info(f"Expecting {num_chunks} weight chunks")

            for i in range(num_chunks):
                for key, tensor in receive_state_dict(comm):
                    received_weights[key] = tensor
                logger.debug(f"Received chunk {i + 1}/{num_chunks}")

            torch.cuda.synchronize()
            elapsed = time.time() - start
            size_gb = sum(p.numel() * p.element_size() for p in received_weights.values()) / 1e9
            logger.info(f"Received {len(received_weights)} tensors ({size_gb:.2f}GB) in {elapsed:.2f}s")

        except Exception as e:
            raise RuntimeError(f"NCCL receive failed: {e}")

        # Wait for broadcast thread to finish
        broadcast_future.result(timeout=30)
        if broadcast_error:
            raise RuntimeError(f"Broadcast failed: {broadcast_error}")

    return received_weights


def broadcast_weights_to_peer(
    model: "Module",
    nccl_port: int = DEFAULT_NCCL_PORT,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """
    Broadcast model weights to a requesting peer via NCCL.

    We act as rank 0 (TCP store master), peer is rank 1.
    Uses a lock to prevent concurrent broadcasts.

    Args:
        model: The model to broadcast weights from
        nccl_port: Port for NCCL communication
        timeout: Timeout for NCCL operations

    Raises:
        RuntimeError: If another broadcast is in progress
    """
    if not _broadcast_lock.acquire(blocking=False):
        raise RuntimeError("Another broadcast is already in progress")

    try:
        my_ip = get_pod_ip()
        logger.info(f"Broadcasting weights on {my_ip}:{nccl_port}")

        # Create TCP store as master (rank 0)
        pg = StatelessProcessGroup.create(
            host=my_ip,
            port=nccl_port,
            rank=0,
            world_size=2,
            store_timeout=timeout,
        )
        comm = PyNcclCommunicator(pg, device=torch.cuda.current_device())
        logger.info("NCCL connected, starting broadcast...")

        state_dict = model.state_dict()

        # Count layers for chunked transfer
        layer_keys = [k for k in state_dict if "model.layers." in k]
        if layer_keys:
            num_layers = max(int(k.split(".")[2]) for k in layer_keys) + 1
        else:
            num_layers = 0

        num_chunks = num_layers + 1  # layers + non-layer weights

        start = time.time()

        # Broadcast number of chunks
        broadcast_integer(num_chunks, comm)

        # Broadcast non-layer weights first
        non_layer = {k: v for k, v in state_dict.items() if "model.layers." not in k}
        broadcast_state_dict(non_layer, comm)

        # Broadcast each layer
        for i in range(num_layers):
            layer_dict = {k: v for k, v in state_dict.items() if f"model.layers.{i}." in k}
            broadcast_state_dict(layer_dict, comm)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        size_gb = sum(p.numel() * p.element_size() for p in state_dict.values()) / 1e9
        logger.info(f"Broadcast complete: {size_gb:.2f}GB in {elapsed:.2f}s ({size_gb / elapsed:.1f} GB/s)")

    finally:
        _broadcast_lock.release()


def fetch_weights_from_peer(
    model: "Module",
    headless_service: str | None = None,
    namespace: str | None = None,
    peer_ip: str | None = None,
    http_port: int = DEFAULT_HTTP_PORT,
    nccl_port: int = DEFAULT_NCCL_PORT,
    retries: int = MAX_RETRIES,
) -> bool:
    """
    Discover a healthy peer and fetch weights from it via NCCL.

    This is the main entry point for new pods starting with --load-format dummy.
    Call this after the model is initialized with dummy weights.

    Can use either K8s DNS discovery OR direct peer IP:
    - K8s mode: provide headless_service and namespace
    - Direct mode: provide peer_ip

    Args:
        model: The model to load weights into
        headless_service: K8s headless service name for peer discovery (optional)
        namespace: K8s namespace (optional)
        peer_ip: Direct peer IP address to fetch from (optional, skips discovery)
        http_port: HTTP port for health checks and broadcast trigger
        nccl_port: Port for NCCL communication
        retries: Number of retry attempts on failure

    Returns:
        True if weights were fetched from peer, False if no peer available
    """
    target_ip = peer_ip

    # If no direct peer IP, use K8s DNS discovery
    if not target_ip:
        if not headless_service:
            logger.error("Must provide either peer_ip or headless_service")
            return False

        # Auto-detect namespace if not provided
        resolved_namespace = namespace or get_namespace()
        if not resolved_namespace:
            logger.error("Namespace not provided and could not be auto-detected")
            return False

        # Discover peers via DNS
        peers = discover_peers(headless_service, resolved_namespace)
        if not peers:
            logger.info("No peers found, will need to load from disk")
            return False

        # Find a healthy peer with model loaded
        target_ip = find_healthy_peer(peers, http_port=http_port)
        if not target_ip:
            logger.info("No healthy peer with model, will need to load from disk")
            return False
    else:
        # Direct mode - verify peer is healthy
        logger.info(f"Using direct peer IP: {target_ip}")
        if not find_healthy_peer([target_ip], http_port=http_port):
            logger.warning(f"Peer {target_ip} not healthy or model not loaded")
            return False

    # Receive weights from peer with retries
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            logger.info(f"Fetching weights from peer {target_ip} (attempt {attempt + 1}/{retries})")
            weights = receive_weights_from_peer(
                target_ip,
                http_port=http_port,
                nccl_port=nccl_port,
            )

            # Validate we got weights
            if not weights:
                raise RuntimeError("Received empty weight dict")

            # Load weights into model
            logger.info(f"Loading {len(weights)} tensors into model...")
            model.load_weights(weights.items())

            logger.info("Weights loaded from peer successfully!")
            return True

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    logger.error(f"Failed to fetch weights after {retries} attempts: {last_error}")
    return False
