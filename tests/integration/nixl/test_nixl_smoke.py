"""Smoke test for the low-level NIXL plumbing.

Runs two processes on two GPUs; one does a WRITE into the other's registered
memory and we verify the bytes landed. No prime-rl classes — just exercises
``NixlAgentWrapper`` + ``StatelessProcessGroup``. Catches UCX/NIXL/topology
issues before layering GLM MoE DSA on top.

Run directly with:

    uv run python tests/integration/nixl/test_nixl_smoke.py
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from contextlib import closing

import torch
import torch.multiprocessing as mp


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


SHARD_BYTES = 4 * 1024 * 1024  # 4 MiB


def _trainer(local_rank: int, global_rank: int, port: int, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.utils.nixl_transfer import NixlAgentWrapper

        agent = NixlAgentWrapper(name=f"trainer-r{global_rank}", local_rank=local_rank)
        buf = torch.full((SHARD_BYTES,), 0xAB, dtype=torch.uint8, device=f"cuda:{local_rank}")
        descs = agent.register_tensor(buf)
        local_prep = agent.prep_local(descs)

        spg = StatelessProcessGroup.create(host="localhost", port=port, rank=0, world_size=2)
        my_info = {
            "name": agent.name,
            "meta": agent.get_metadata(),
            "descs": agent.serialize_descs(descs),
        }
        peers = spg.all_gather_obj(my_info)
        peer = peers[1]

        agent.add_remote(peer["meta"])
        remote_descs = agent.deserialize_descs(peer["descs"])
        remote_prep = agent.prep_remote(peer["name"], remote_descs)
        agent.make_connection(peer["name"])

        handle = agent.post_write(local_prep, 0, remote_prep, 0)
        agent.wait(handle)

        spg.barrier()
        ready_q.put(("trainer", "ok"))
    except Exception as e:
        ready_q.put(("trainer", f"fail: {e}\n{traceback.format_exc()}"))


def _inference(local_rank: int, global_rank: int, port: int, ready_q: mp.Queue) -> None:
    try:
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.utils.nixl_transfer import NixlAgentWrapper

        agent = NixlAgentWrapper(name=f"inference-r{global_rank}", local_rank=local_rank)
        buf = torch.zeros(SHARD_BYTES, dtype=torch.uint8, device=f"cuda:{local_rank}")
        descs = agent.register_tensor(buf)

        spg = StatelessProcessGroup.create(host="localhost", port=port, rank=1, world_size=2)
        my_info = {
            "name": agent.name,
            "meta": agent.get_metadata(),
            "descs": agent.serialize_descs(descs),
        }
        peers = spg.all_gather_obj(my_info)
        peer = peers[0]
        agent.add_remote(peer["meta"])

        spg.barrier()
        ok = bool(torch.all(buf == 0xAB).item())
        ready_q.put(("inference", f"{'ok' if ok else f'fail: first={buf[0].item()} last={buf[-1].item()}'}"))
    except Exception as e:
        ready_q.put(("inference", f"fail: {e}\n{traceback.format_exc()}"))


def main() -> int:
    ctx = mp.get_context("spawn")
    port = _free_port()
    q: mp.Queue = ctx.Queue()

    p_trainer = ctx.Process(target=_trainer, args=(0, 0, port, q))
    p_infer = ctx.Process(target=_inference, args=(1, 1, port, q))
    p_trainer.start()
    p_infer.start()
    p_trainer.join(timeout=60)
    p_infer.join(timeout=60)

    results = {}
    while not q.empty():
        role, result = q.get_nowait()
        results[role] = result
    print(f"results: {results}")
    if p_trainer.is_alive() or p_infer.is_alive():
        p_trainer.kill()
        p_infer.kill()
        print("FAIL: processes hung")
        return 1
    if results.get("trainer") != "ok" or results.get("inference") != "ok":
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
