"""Frozen Clef service on a GPU reserved from the inference allocation."""

import argparse
import asyncio
import importlib.util
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


def partition_devices(devices: str, actor_gpus: int) -> tuple[str, str]:
    visible = [device.strip() for device in devices.split(",") if device.strip()]
    if len(visible) != actor_gpus + 1 or len(set(visible)) != len(visible):
        raise ValueError(f"Expected {actor_gpus + 1} distinct visible GPUs, got {len(visible)}")
    return ",".join(visible[:-1]), visible[-1]


def start_judge(config, actor_gpus):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not devices:
        devices = ",".join(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True,
        ).split())
    actor_devices, judge_device = partition_devices(devices, actor_gpus)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": judge_device, "TOKENIZERS_PARALLELISM": "false"}
    process = subprocess.Popen([
        sys.executable, "-m", __name__, "--revision", config.revision,
        "--port", str(config.port), "--max-length", str(config.max_length),
    ], env=env)
    os.environ["CUDA_VISIBLE_DEVICES"] = actor_devices
    os.environ["PRIME_RL_CLEF_URL"] = f"http://127.0.0.1:{config.port}/v1/systemone"
    stopping = Event()

    def watch():
        process.wait()
        if not stopping.is_set():
            logging.error("Clef judge exited: code=%s", process.returncode)
            os.kill(os.getpid(), signal.SIGTERM)

    # Start supervision after readiness so startup failures propagate as errors.
    deadline = time.monotonic() + config.startup_timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Clef startup failed: exit {process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config.port}/health", timeout=2) as response:
                    health = json.load(response)
                if health["revision"] != config.revision:
                    raise RuntimeError("Clef health reports a different weight revision")
                Thread(target=watch, daemon=True).start()
                return process, stopping
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        raise TimeoutError("Clef judge did not become ready")
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise


def serve(revision, port, max_length):
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from huggingface_hub import snapshot_download

    logging.basicConfig(level=logging.INFO)
    path = snapshot_download("Cloudflare/clef-flash", revision=revision)
    spec = importlib.util.spec_from_file_location("rime_clef_release", path + "/joint_schema_model.py")
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    model, processor = release.load_release_model(path, device="cuda:0")
    app = FastAPI()
    gate = asyncio.Semaphore(1)

    def evaluate(body):
        # Reject oversized inputs: the reference encoder otherwise silently truncates the state.
        encoded = release.encode_record(processor.tokenizer, body, max_length=2**30, processor=processor)
        if len(encoded.input_ids) > max_length:
            raise ValueError("Clef input exceeds the configured token limit")
        with torch.inference_mode():
            return release.systemone(model, processor, body, max_length=max_length)

    @app.get("/health")
    async def health():
        return {"model": "Cloudflare/clef-flash", "revision": revision, "max_length": max_length}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        body = await request.json()
        if body.get("model", "").lower() != "cloudflare/clef-flash":
            raise HTTPException(400, "Unexpected judge model")
        if set(body) - {"model", "state", "questions"}:
            raise HTTPException(400, "Only text memory grading is supported")
        async with gate:
            try:
                started = time.monotonic()
                result = await asyncio.to_thread(evaluate, body)
                logging.info("Clef grade completed: seconds=%.3f", time.monotonic() - started)
                return result
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc

    logging.info("Clef ready: revision=%s gpu=%s", revision, torch.cuda.get_device_name(0))
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, default=9020)
    parser.add_argument("--max-length", type=int, default=24576)
    args = parser.parse_args()
    serve(args.revision, args.port, args.max_length)
