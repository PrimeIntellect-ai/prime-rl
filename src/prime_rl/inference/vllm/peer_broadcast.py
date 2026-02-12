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
import subprocess
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

# Lock to serialize peer broadcasts from a single source pod.
# Without this, concurrent bootstrap requests from multiple replicas would
# collide on the same NCCL port, causing deadlocks or connection failures.
_broadcast_lock = threading.Lock()


def get_pod_ip() -> str:
    """Get the current pod's IP address.

    Priority:
    1. POD_IP environment variable (set by K8s downward API or manually)
    2. Fallback to gethostbyname

    NOTE: For replica bootstrap to work, POD_IP must be set to a routable IP.
    In K8s, use the downward API. On bare metal, set it manually:
        export POD_IP=$(hostname -I | awk '{print $1}')
    """
    if pod_ip := os.environ.get("POD_IP"):
        return pod_ip

    ip = socket.gethostbyname(socket.gethostname())
    if ip.startswith("127."):
        logger.warning(
            f"POD_IP not set and gethostbyname returned localhost ({ip}). "
            "NCCL may fail to connect. Set POD_IP to a routable IP."
        )
    return ip


def configure_nccl_socket_interface() -> None:
    """Configure NCCL to use the correct network interface for TCP bootstrap.

    On multi-homed nodes (e.g., with InfiniBand), NCCL may select an interface
    whose IP is not routable for TCP (e.g., IB uses RDMA, not TCP). This causes
    StatelessProcessGroup.create() to hang waiting for TCP connections.

    This function detects which interface has the POD_IP and sets NCCL_SOCKET_IFNAME
    to force NCCL to use that interface for TCP bootstrap communication.
    """
    if os.environ.get("NCCL_SOCKET_IFNAME"):
        logger.info(f"NCCL_SOCKET_IFNAME already set: {os.environ['NCCL_SOCKET_IFNAME']}")
        return

    pod_ip = get_pod_ip()
    result = subprocess.run(
        ["ip", "-o", "-4", "addr", "show"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Failed to run 'ip addr show': {result.stderr}")
        return

    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[3].startswith(f"{pod_ip}/"):
            interface = parts[1]
            logger.info(f"Setting NCCL_SOCKET_IFNAME={interface} (detected from POD_IP={pod_ip})")
            os.environ["NCCL_SOCKET_IFNAME"] = interface
            os.environ["GLOO_SOCKET_IFNAME"] = interface
            return

    logger.warning(f"Could not find interface for POD_IP={pod_ip}, NCCL may select wrong interface")


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


def stream_weights_from_peer(
    model: "Module",
    peer_ip: str,
    http_port: int = DEFAULT_HTTP_PORT,
    nccl_port: int = DEFAULT_NCCL_PORT,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """
    Request and stream weights from a peer directly into the model via NCCL.

    This is memory-efficient: weights are received to CPU one chunk at a time,
    then loaded directly into the model. This avoids holding 2x model size in memory.

    The peer acts as rank 0 (TCP store master), we are rank 1.
    We trigger the broadcast first (so peer starts TCP store), then connect.

    Args:
        model: The model to load weights into
        peer_ip: IP address of the peer to receive from
        http_port: HTTP port for triggering broadcast on peer
        nccl_port: Port for NCCL communication
        timeout: Timeout for NCCL operations

    Raises:
        RuntimeError: If weight transfer fails
    """
    import concurrent.futures

    logger.info(f"Streaming weights from peer {peer_ip}:{nccl_port}")

    broadcast_error: Exception | None = None

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

        # Configure NCCL to use the correct interface for TCP bootstrap
        configure_nccl_socket_interface()

        # Now connect to peer's TCP store (peer is rank 0 master, we're rank 1)
        logger.info(f"Connecting to NCCL at {peer_ip}:{nccl_port}...")
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

        # Build a map from param names to their data buffers for in-place copy
        # This avoids 2x memory usage by receiving directly into existing tensors
        model_state = model.state_dict()
        param_map = {k: v for k, v in model_state.items()}

        # Receive weights via NCCL and copy directly into model parameters
        # NCCL broadcasts are synchronous collectives - both sides must stay in sync
        num_chunks = receive_integer(comm)
        logger.info(f"Expecting {num_chunks} weight chunks, copying in-place to avoid 2x memory")

        total_params = 0
        for i in range(num_chunks):
            for key, tensor in receive_state_dict(comm, receive_on_cpu=False):
                if key in param_map:
                    # Copy received tensor directly into existing parameter buffer
                    param_map[key].copy_(tensor)
                    total_params += 1
                    del tensor  # Free the temporary receive buffer immediately
                else:
                    logger.warning(f"Received unknown key: {key}")
            if (i + 1) % 10 == 0:
                logger.info(f"Received chunk {i + 1}/{num_chunks}")

        torch.cuda.synchronize()
        nccl_elapsed = time.time() - start
        size_gb = sum(p.numel() * p.element_size() for p in param_map.values()) / 1e9
        logger.info(f"NCCL transfer complete: {total_params} tensors ({size_gb:.2f}GB) in {nccl_elapsed:.2f}s")

        # Wait for broadcast thread to finish
        broadcast_future.result(timeout=30)
        if broadcast_error:
            raise RuntimeError(f"Broadcast failed: {broadcast_error}")

    # No need for load_state_dict - we already copied in-place above
    logger.info("Weights copied in-place, no additional load_state_dict needed")


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

        # Configure NCCL to use the correct interface for TCP bootstrap
        configure_nccl_socket_interface()

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

        logger.info("Getting state_dict from model...")
        state_dict = model.state_dict()
        logger.info(f"Got state_dict with {len(state_dict)} tensors")

        # Count layers for chunked transfer
        layer_keys = [k for k in state_dict if "model.layers." in k]
        if layer_keys:
            num_layers = max(int(k.split(".")[2]) for k in layer_keys) + 1
        else:
            num_layers = 0

        num_chunks = num_layers + 1  # layers + non-layer weights
        logger.info(f"Broadcasting {num_chunks} chunks ({num_layers} layers + 1 non-layer)")

        start = time.time()

        # Broadcast number of chunks
        logger.info("Broadcasting chunk count...")
        broadcast_integer(num_chunks, comm)
        logger.info("Chunk count sent, broadcasting non-layer weights...")

        # Broadcast non-layer weights first
        non_layer = {k: v for k, v in state_dict.items() if "model.layers." not in k}
        broadcast_state_dict(non_layer, comm)
        logger.info("Non-layer weights sent, broadcasting layers...")

        # Broadcast each layer
        for i in range(num_layers):
            layer_dict = {k: v for k, v in state_dict.items() if f"model.layers.{i}." in k}
            broadcast_state_dict(layer_dict, comm)
            if (i + 1) % 10 == 0:
                logger.info(f"Sent layer {i + 1}/{num_layers}")

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

    # Stream weights from peer directly into model with retries
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            logger.info(f"Fetching weights from peer {target_ip} (attempt {attempt + 1}/{retries})")
            stream_weights_from_peer(
                model,
                target_ip,
                http_port=http_port,
                nccl_port=nccl_port,
            )

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
