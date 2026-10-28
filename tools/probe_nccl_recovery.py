"""Fault injection against a disposable, single-GPU Qwen3-0.6B inference server.

Start prime-rl inference on physical GPU 1 with NCCL weight broadcast, then run
this script with both GPUs visible. The sender uses GPU 0 (8% memory cap).
The test overwrites part of the model, kills its sender, and reloads the original
checkpoint into the same engine. Do not use an inference server serving a run.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3-0.6B"


def trainer(pipe, port, crash, model_dir):
    import torch
    from safetensors.torch import load_file
    from vllm.distributed.utils import StatelessProcessGroup

    from prime_rl.transports.weights.nccl import broadcast_integer, broadcast_state_dict, filter_state_dict_by_layers
    from prime_rl.transports.weights.nccl_communicator import WeightBroadcastCommunicator

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.08)
    state = {}
    for file in model_dir.glob("*.safetensors"):
        state.update(load_file(file))
    assert state, "No safetensors found in model directory"
    config = json.loads((model_dir / "config.json").read_text())
    group = StatelessProcessGroup.create(host="127.0.0.1", port=port, rank=0, world_size=2, store_timeout=30)
    comm = WeightBroadcastCommunicator(group, 0, 15)
    pipe.send("initialized")
    broadcast_integer(config["num_hidden_layers"] + 1, comm)
    for layer, weights in filter_state_dict_by_layers(state, config["num_hidden_layers"], "model.layers."):
        packet = {key: value.cuda() for key, value in weights.items()}
        if crash:
            for value in packet.values():
                value.zero_()
        broadcast_state_dict(packet, comm)
        del packet
        if crash:
            pipe.send("partial_weights_sent")
            pipe.recv()
    pipe.send("complete")
    pipe.recv()
    comm.close()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def recv(pipe):
    if not await asyncio.to_thread(pipe.poll, 60):
        raise TimeoutError("sender unresponsive")
    return pipe.recv()


async def main(args):
    from prime_rl.configs.shared import ClientConfig
    from prime_rl.orchestrator.clients import AdminPlane

    ctx = mp.get_context("spawn")
    url = f"http://127.0.0.1:{args.port}"
    admin = AdminPlane(ClientConfig(base_url=url + "/v1", admin_base_url=[url]))
    processes = []
    async with httpx.AsyncClient(base_url=url, timeout=90) as client:

        async def completion():
            response = await client.post(
                "/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": "The capital of France is",
                    "temperature": 0,
                    "max_tokens": 12,
                    "seed": 0,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]

        for _ in range(90):
            try:
                response = await client.get("/health", timeout=2)
                if response.is_success:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(2)
        before = await completion()
        assert before
        print(json.dumps({"event": "initial_completion", "text": before}), flush=True)
        queued = None
        update = None
        try:
            for attempt, crash in enumerate((True, False)):
                port = free_port()
                parent, child = ctx.Pipe()
                process = ctx.Process(target=trainer, args=(child, port, crash, args.model_dir))
                processes.append(process)
                process.start()
                await admin.initialize_nccl(
                    host="127.0.0.1", port=port, inference_world_size=1, timeout=15, recoverable=True
                )
                assert await recv(parent) == "initialized"
                started = time.monotonic()
                update = asyncio.create_task(admin.update_weights(None, transport="nccl", step=attempt))
                if crash:
                    assert await recv(parent) == "partial_weights_sent"
                    process.kill()
                    await asyncio.to_thread(process.join, 10)
                    try:
                        await update
                    except httpx.HTTPStatusError as error:
                        assert error.response.status_code == 500
                        print(
                            json.dumps({"event": "transfer_failed", "seconds": time.monotonic() - started}), flush=True
                        )
                    else:
                        raise AssertionError("incomplete transfer succeeded")
                    (await client.get("/liveness")).raise_for_status()
                    assert (await client.post("/resume")).status_code == 409
                    queued = asyncio.create_task(completion())
                    await asyncio.sleep(1)
                    assert not queued.done(), "engine served partially overwritten weights"
                    print(json.dumps({"event": "engine_alive_and_paused"}), flush=True)
                else:
                    assert await recv(parent) == "complete"
                    await update
                    parent.send("close")
                    await asyncio.to_thread(process.join, 10)
                    assert process.exitcode == 0
                    after = await queued
                    assert before == after, (before, after)
                    print(
                        json.dumps(
                            {"event": "restored_and_resumed", "text": after, "seconds": time.monotonic() - started}
                        ),
                        flush=True,
                    )
            (await client.get("/liveness")).raise_for_status()
            print(
                "PASS: failed partial model reload, same live engine, full replacement, queued generation resumed",
                flush=True,
            )
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join(10)
            tasks = [task for task in (queued, update) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await admin.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Local Qwen/Qwen3-0.6B safetensors checkpoint directory")
    parser.add_argument("--port", type=int, default=18401, help="Disposable local inference server port")
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    if (
        config.get("model_type") != "qwen3"
        or config.get("hidden_size") != 1024
        or config.get("num_hidden_layers") != 28
    ):
        parser.error("This bounded probe requires Qwen3-0.6B")
    os.environ.update(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1", OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1")
    asyncio.run(main(args))
