"""An independent vLLM writer that never joins policy weight broadcasts."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from threading import Event, Thread

from prime_rl.inference.clef import partition_devices, relay_client


def start_writer(config, actor_gpus, engine_port=8000, admin_relay_port=8100):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not devices:
        devices = ",".join(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True,
            ).split()
        )
    actor_devices, writer_device = partition_devices(devices, actor_gpus)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": writer_device, "TOKENIZERS_PARALLELISM": "false"}
    # This process runs vanilla vLLM: it has no NCCL update worker or admin routes.
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            config.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(config.port),
            "--max-model-len",
            str(config.max_length),
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "1",
            "--reasoning-parser",
            "qwen3",
            "--gpu-memory-utilization",
            "0.9",
            "--generation-config",
            "vllm",
            "--hf-overrides",
            '{"head_dtype":"float32"}',
        ],
        env=env,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = actor_devices
    os.environ["PRIME_RL_FROZEN_WRITER_URL"] = f"http://127.0.0.1:{config.port}/v1/chat/completions"
    os.environ["PRIME_RL_FROZEN_WRITER_MODEL"] = config.model
    stopping = Event()

    async def relay():
        server = await asyncio.start_server(
            lambda r, w: relay_client(r, w, engine_port),
            "0.0.0.0",
            admin_relay_port,
        )
        async with server:
            await server.serve_forever()

    def watch():
        process.wait()
        if not stopping.is_set():
            logging.error("Frozen writer exited: code=%s", process.returncode)
            os.kill(os.getpid(), signal.SIGTERM)

    deadline = time.monotonic() + config.startup_timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Frozen writer startup failed: exit {process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config.port}/v1/models", timeout=2) as response:
                    models = json.load(response)
                if config.model not in {model["id"] for model in models["data"]}:
                    raise RuntimeError("Frozen writer serves an unexpected model")
                Thread(target=lambda: asyncio.run(relay()), daemon=True).start()
                Thread(target=watch, daemon=True).start()
                logging.info("Frozen base writer ready: model=%s", config.model)
                return process, stopping
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        raise TimeoutError("Frozen writer did not become ready")
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise
