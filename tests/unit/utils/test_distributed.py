import os
import socket

import torch.distributed as dist
import torch.multiprocessing as mp

from prime_rl.utils.distributed import all_gather_bytes, broadcast_bytes


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_collectives(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        payload = bytes([rank]) * rank
        assert all_gather_bytes(payload) == [bytes([member]) * member for member in range(world_size)]
        assert broadcast_bytes(b"\x00metadata\xff" if rank == 0 else None) == b"\x00metadata\xff"
        assert broadcast_bytes(b"" if rank == 0 else None) == b""
    finally:
        dist.destroy_process_group()


def test_byte_collectives_preserve_variable_binary_payloads() -> None:
    world_size = 3
    mp.spawn(_run_collectives, args=(world_size, _free_port()), nprocs=world_size, join=True)
