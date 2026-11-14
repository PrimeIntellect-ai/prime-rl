import torch
import torch.distributed as dist

MAX_COLLECTIVE_BYTES = 256 * 1024**2


def _collective_device(group: dist.ProcessGroup | None) -> torch.device:
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _validate_size(size: int, max_bytes: int) -> None:
    if not 0 <= size <= max_bytes:
        raise ValueError(f"Collective payload is {size} bytes, expected at most {max_bytes}")


def all_gather_bytes(
    data: bytes,
    group: dist.ProcessGroup | None = None,
    *,
    max_bytes: int = MAX_COLLECTIVE_BYTES,
) -> list[bytes]:
    """All-gather bounded byte strings using tensor collectives."""
    _validate_size(len(data), max_bytes)
    device = _collective_device(group)
    size = torch.tensor([len(data)], dtype=torch.int64, device=device)
    sizes = [torch.empty_like(size) for _ in range(dist.get_world_size(group))]
    dist.all_gather(sizes, size, group=group)
    lengths = [int(value.item()) for value in sizes]
    for length in lengths:
        _validate_size(length, max_bytes)

    width = max(lengths, default=0)
    if width == 0:
        return [b""] * len(lengths)

    send = torch.zeros(width, dtype=torch.uint8, device=device)
    if data:
        send[: len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
    received = [torch.empty_like(send) for _ in lengths]
    dist.all_gather(received, send, group=group)
    return [tensor[:length].cpu().numpy().tobytes() for tensor, length in zip(received, lengths)]


def broadcast_bytes(
    data: bytes | None,
    *,
    src: int = 0,
    group: dist.ProcessGroup | None = None,
    max_bytes: int = MAX_COLLECTIVE_BYTES,
) -> bytes:
    """Broadcast a bounded byte string using tensor collectives."""
    device = _collective_device(group)
    is_source = dist.get_rank() == src
    if is_source:
        if data is None:
            raise ValueError("The broadcast source must provide a payload")
        _validate_size(len(data), max_bytes)
    size = torch.tensor([len(data) if data is not None and is_source else 0], dtype=torch.int64, device=device)
    dist.broadcast(size, src=src, group=group)
    length = int(size.item())
    _validate_size(length, max_bytes)
    if length == 0:
        return b""

    payload = torch.empty(length, dtype=torch.uint8, device=device)
    if is_source:
        payload.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device))
    dist.broadcast(payload, src=src, group=group)
    return payload.cpu().numpy().tobytes()
