"""Check that a weight transport installs bit-exactly what vLLM's own loader produces.

The engine cold-loads MODEL_DIR, an HF checkpoint holding the trainer's initial weights W0, with vLLM's loader. The
per-tensor digests it then reports (F0) are the only oracle. The trainer loads the same checkpoint and, instead of
optimizer steps, publishes synthetic versions through the real transport (trainer sender, consumer-side receiver,
engine worker extension):

    step 0  W0        every engine tensor equals F0
    step 1  W0 + D1   every floating-point engine parameter differs from F0: the update was installed
    step 2  W0        every engine tensor equals F0: the install reproduces vLLM's loader, also after an update
    step 3  W0 + D2   every floating-point engine parameter differs from F0 and from step 1
    step 4  W0        every engine tensor equals F0

D1 and D2 are seeded perturbations of every floating-point parameter and persistent buffer of the trainer, so they
reach every floating-point tensor it publishes. Engine weights are its parameters and its bare tensor attributes
named like weights or biases (fused or post-processed copies, `@module.name`). Engine state is its buffers and other
bare tensor attributes except the KV cache: it must equal F0 at every W0 step but need not change with an update.
Bare weight attributes that an update leaves unchanged are reported (`unchanged_attributes`) but do not fail, since
some are constants (the unit gain of a norm without a learnable weight).

Before publishing, the trainer also checks that its W0, converted to the checkpoint format in the wire dtypes, equals
the checkpoint bit for bit, so that a difference at a W0 step is the transport's.

    uv run python tools/check_weight_update_bit_exact.py MODEL_DIR OUTPUT_DIR [--transport modelexpress]
        [--vllm JSON] [--model JSON] [--trainer-gpus 0] [--inference-gpus 1] [--ignore REGEX]

--vllm holds vLLM engine arguments, --model trainer model config fields. Writes OUTPUT_DIR/report.json and exits 1 if
a check fails. Processes inherit the environment, so PYTHONPATH can put another build of a transport client first.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import sysconfig
import time
from pathlib import Path

import httpx

# Synthetic versions: (step, perturbation seed or None for W0)
STEPS = [(0, None), (1, 1), (2, None), (3, 2), (4, None)]
READY_TIMEOUT_S = 1800


def is_weight(name: str, entry: dict) -> bool:
    if entry["kind"] == "parameter":
        return True
    attribute = name.rsplit(".", 1)[-1]
    return entry["kind"] == "attribute" and ("weight" in attribute or "bias" in attribute)


def is_floating(entry: dict) -> bool:
    return entry["dtype"].startswith(("torch.float", "torch.bfloat"))


def free_port() -> int:
    from prime_rl.utils.utils import get_free_port

    return get_free_port()


def published_tensors(model) -> dict:
    """Local shards of the trainer's floating-point parameters and persistent buffers, the state behind every tensor
    it publishes (state dict entries of packed parameters are copies of them)."""
    from torch.distributed.tensor import DTensor

    tensors = dict(model.named_parameters())
    for module_name, module in model.named_modules():
        for name, buffer in module.named_buffers(recurse=False):
            if name not in module._non_persistent_buffers_set:
                tensors[f"{module_name}.{name}" if module_name else name] = buffer
    return {
        name: tensor.to_local() if isinstance(tensor, DTensor) else tensor
        for name, tensor in tensors.items()
        if tensor.is_floating_point()
    }


def as_bytes(tensor):
    import torch

    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def checkpoint_mismatches(model, model_dir: Path) -> dict:
    """Compare the trainer's weights, gathered in their wire dtypes and converted to the checkpoint format as the
    filesystem broadcast does, with the checkpoint the engine cold-loaded."""
    import torch
    import torch.distributed as dist
    from safetensors import safe_open

    from prime_rl.utils.weights import convert_state_dict_to_hf, gather_weights_parallel

    location = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            location.update(dict.fromkeys(f.keys(), path))
    ours = convert_state_dict_to_hf(model, gather_weights_parallel(model))
    differ, cast = [], []
    for name, tensor in ours.items():
        if name not in location:
            continue
        with safe_open(location[name], framework="pt") as f:
            stored = f.get_tensor(name)
        if stored.dtype != tensor.dtype:
            # vLLM's loader casts the checkpoint into the parameter's dtype as torch does
            cast.append(f"{name}: checkpoint {stored.dtype}, trainer {tensor.dtype}")
            stored = stored.to(tensor.dtype)
        if stored.shape != tensor.shape or not torch.equal(as_bytes(stored), as_bytes(tensor)):
            differ.append(name)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (sorted(ours), differ, cast))
    produced = {name for names, _, _ in gathered for name in names}
    return {
        "compared": len(produced & set(location)),
        "differ": sorted(name for _, names, _ in gathered for name in names),
        "cast": sorted(name for _, _, names in gathered for name in names),
        "not_in_checkpoint": sorted(produced - set(location)),
        "not_published": sorted(set(location) - produced),
    }


def train(plan: dict) -> None:
    """One trainer rank, under torchrun: load the checkpoint, then publish each planned version."""
    import torch
    import torch.distributed as dist

    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.model import setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.trainer.world import get_world
    from prime_rl.transports.weights import setup_weight_sender
    from prime_rl.utils.logger import setup_logger

    setup_logger("info")
    setup_torch_distributed()
    world = get_world()
    output_dir, model_dir = Path(plan["output_dir"]), Path(plan["model_dir"])
    config = ModelConfig(name=plan["model_dir"], **{"compile": None, "ac": None, **plan["model"]})
    resolve_ep(config)
    model = setup_model(config, get_parallel_dims(config))

    check = checkpoint_mismatches(model, model_dir)
    if world.is_master:
        (output_dir / "trainer_checkpoint_check.json").write_text(json.dumps(check, indent=2))

    sender = setup_weight_sender(output_dir, transport_config(plan), model_name=plan["model_dir"])
    tensors = published_tensors(model)
    initial = {name: tensor.clone() for name, tensor in tensors.items()}
    for step, seed in plan["steps"]:
        with torch.no_grad():
            for name, tensor in tensors.items():
                tensor.copy_(initial[name])
            if seed is not None:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(seed * 1_000_003 + world.rank)
                for tensor in tensors.values():
                    if tensor.numel() == 0:
                        continue
                    scale = 0.1 * tensor.float().pow(2).mean().sqrt() + 1e-2
                    noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=torch.float32)
                    tensor.add_((noise * scale).to(tensor.dtype))
        sender.broadcast(model, step)
    dist.barrier()
    dist.destroy_process_group()


def transport_config(plan: dict):
    from prime_rl.configs.shared import FileSystemWeightBroadcastConfig, ModelExpressWeightBroadcastConfig

    if plan["transport"] == "modelexpress":
        return ModelExpressWeightBroadcastConfig(
            port=plan["modelexpress_port"],
            metadata_port=plan["metadata_port"],
            inference_world_size=plan["inference_world_size"],
            timeout=plan["timeout"],
        )
    return FileSystemWeightBroadcastConfig(timeout=plan["timeout"])


async def watched(awaitable, processes, timeout: float):
    """Await, failing as soon as a launched process dies."""
    task = asyncio.ensure_future(awaitable)
    deadline = time.monotonic() + timeout
    while not task.done():
        if processes.error_queue:
            task.cancel()
            raise RuntimeError(f"A process failed: {processes.error_queue[0]}")
        if time.monotonic() > deadline:
            task.cancel()
            raise TimeoutError(f"Timed out after {timeout}s")
        await asyncio.wait([task], timeout=1)
    return task.result()


async def fingerprint(engine_url: str) -> dict:
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.get(f"{engine_url}/weight_fingerprint/model", params={"buffers": "true"})
        response.raise_for_status()
    workers = response.json()["results"]
    if len(workers) == 1:
        return workers[0]
    return {f"worker{index}/{name}": entry for index, worker in enumerate(workers) for name, entry in worker.items()}


async def wait_healthy(engine_url: str) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            try:
                if (await client.get(f"{engine_url}/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(2)


async def consume(plan: dict, processes, engine_url: str, fingerprints: dict[int, dict]) -> None:
    """Record F0, start the trainer, then install every version it publishes and record the digests after each."""
    from prime_rl.configs.shared import ClientConfig
    from prime_rl.orchestrator.clients import AdminPlane
    from prime_rl.transports.weights import setup_weight_receiver
    from prime_rl.utils.pathing import get_broadcast_dir

    timeout, output_dir = plan["timeout"], Path(plan["output_dir"])
    fingerprints[-1] = await watched(fingerprint(engine_url), processes, timeout)
    print(f"cold load: fingerprinted {len(fingerprints[-1])} engine tensors", flush=True)
    # Ports are picked late, so that the engine's own startup cannot take them. Each trainer rank also serves
    # ModelExpress at MX_WORKER_GRPC_PORT plus its device id, which must not collide with another trainer on the node.
    plan["metadata_port"] = free_port()
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    processes.start(
        "trainer",
        [*plan["trainer_cmd"], f"--master-port={free_port()}", __file__, "--trainer", f"{output_dir}/plan.json"],
        env={**plan["trainer_env"], "MX_WORKER_GRPC_PORT": str(free_port())},
        log_path=output_dir / "logs" / "trainer.log",
    )
    admin_plane = AdminPlane(ClientConfig(admin_base_url=[f"{engine_url}/v1"]))
    receiver = setup_weight_receiver(
        get_broadcast_dir(output_dir), transport_config(plan), admin_plane, plan["model_dir"]
    )
    await watched(receiver.initialize(), processes, timeout)
    for step, _ in plan["steps"]:
        await watched(receiver.sync_startup(step, timeout), processes, timeout)
        fingerprints[step] = await watched(fingerprint(engine_url), processes, timeout)
        print(f"step {step}: installed and fingerprinted", flush=True)
    await admin_plane.aclose()


def compare(plan: dict, fingerprints: dict[int, dict], ignore: re.Pattern | None) -> dict:
    initial = fingerprints[-1]
    names = [name for name in initial if ignore is None or not ignore.search(name)]
    weights = [name for name in names if is_weight(name, initial[name])]
    steps = {}
    for step, seed in plan["steps"]:
        if step not in fingerprints:
            break
        current = fingerprints[step]
        result = {
            "version": "W0" if seed is None else f"W0 + D{seed}",
            "missing_weights": [name for name in weights if name not in current],
            "added_weights": [name for name in current if name not in initial and is_weight(name, current[name])],
            "missing_state": [name for name in names if name not in weights and name not in current],
            "added_state": [name for name in current if name not in initial and not is_weight(name, current[name])],
            "moved": [
                name for name in weights if name in current and current[name]["data_ptr"] != initial[name]["data_ptr"]
            ],
        }
        if seed is None:
            differ = [name for name in names if name in current and current[name]["sha256"] != initial[name]["sha256"]]
            result["differ_weights"] = [name for name in differ if name in weights]
            result["differ_state"] = [name for name in differ if name not in weights]
        else:
            # The perturbations reach only floating-point tensors. A bare weight attribute can be a constant (the
            # unit gain of a norm without a learnable weight), so only parameters must change.
            unchanged = [
                name
                for name in weights
                if name in current and current[name]["sha256"] == initial[name]["sha256"] and is_floating(initial[name])
            ]
            result["unchanged_weights"] = [name for name in unchanged if initial[name]["kind"] == "parameter"]
            result["unchanged_attributes"] = [name for name in unchanged if initial[name]["kind"] != "parameter"]
            result["unchanged_state"] = [
                name
                for name in names
                if name not in weights and name in current and current[name]["sha256"] == initial[name]["sha256"]
            ]
            earlier = [s for s, d in plan["steps"] if d is not None and s < step]
            if earlier:
                previous = fingerprints[earlier[-1]]
                result["same_as_previous_update"] = [
                    name
                    for name in weights
                    if name in current
                    and current[name]["sha256"] == previous[name]["sha256"]
                    and is_floating(initial[name])
                    and initial[name]["kind"] == "parameter"
                ]
        steps[step] = result
    return {"tensors": len(names), "weights": len(weights), "state": len(names) - len(weights), "steps": steps}


def last_error(log: Path) -> str | None:
    """The first transport error a log reports (``error=...``), else its last exception, for the report."""
    if not log.exists():
        return None
    lines = [
        line.strip()
        for line in log.read_text(errors="replace").splitlines()
        if not re.search(r"nixl_ep|nixl-cu13|_pytree|DeprecationWarning|HTTP/1.1", line)
    ]
    transport = [line for line in lines if "error=" in line]
    exceptions = [line for line in lines if re.search(r"\b\w*(Error|Exception)\b: ", line)]
    found = transport[:1] or exceptions[-1:]
    return found[0][:2000] if found else None


def failures(report: dict) -> list[str]:
    found = []
    if report["error"] is not None:
        found.append(f"stopped after installing steps {list(report['steps'])}: {report['error']}")
    check = report["trainer_checkpoint_check"]
    if check is not None and check["differ"]:
        found.append(f"trainer W0 differs from the checkpoint: {check['differ']}")
    for step, result in report["steps"].items():
        for key in (
            "missing_weights",
            "added_weights",
            "differ_weights",
            "differ_state",
            "unchanged_weights",
            "same_as_previous_update",
        ):
            if result.get(key):
                found.append(f"step {step} ({result['version']}), {key}: {result[key]}")
    return found


def launch(args: argparse.Namespace) -> int:
    from prime_rl.entrypoints.rl import start_modelexpress
    from prime_rl.utils.logger import setup_logger
    from prime_rl.utils.process import (
        DEFAULT_COMMON_ENV_VARS,
        DEFAULT_TRAINER_ENV_VARS,
        ProcessGroup,
    )

    setup_logger("info")
    output_dir, model_dir = args.output_dir.resolve(), args.model_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # GPU ids index this process's visible devices
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = visible.split(",") if visible else None
    trainer_gpus, inference_gpus = (
        [visible[int(gpu)] if visible else gpu for gpu in gpus.split(",")]
        for gpus in (args.trainer_gpus, args.inference_gpus)
    )
    engine_port = free_port()
    engine_url = f"http://127.0.0.1:{engine_port}"
    bin_dir = Path(sysconfig.get_path("scripts"))
    plan = {
        "model_dir": model_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "transport": args.transport,
        "model": json.loads(args.model),
        "steps": STEPS,
        "timeout": args.timeout,
        "modelexpress_port": free_port(),
        "inference_world_size": len(inference_gpus),
        "trainer_cmd": [str(bin_dir / "torchrun"), f"--nproc-per-node={len(trainer_gpus)}"],
    }
    base_env = {**os.environ, **DEFAULT_COMMON_ENV_VARS}
    plan["trainer_env"] = {**base_env, **DEFAULT_TRAINER_ENV_VARS, "CUDA_VISIBLE_DEVICES": ",".join(trainer_gpus)}
    engine_config = {
        "server": {"port": engine_port},
        "router": None,
        "weight_broadcast": {"type": args.transport},
        "vllm": {
            "model": model_dir.as_posix(),
            "tensor_parallel_size": len(inference_gpus),
            "data_parallel_rpc_port": free_port(),
            **json.loads(args.vllm),
        },
    }
    (output_dir / "inference.json").write_text(json.dumps(engine_config, indent=2))

    fingerprints: dict[int, dict] = {}
    error = None
    with ProcessGroup() as processes:
        if args.transport == "modelexpress":
            start_modelexpress(processes, plan["modelexpress_port"], output_dir / "logs", base_env)
        processes.start(
            "inference",
            [str(bin_dir / "inference"), "@", (output_dir / "inference.json").as_posix()],
            env={**base_env, "CUDA_VISIBLE_DEVICES": ",".join(inference_gpus)},
            log_path=output_dir / "logs" / "inference.log",
        )

        async def run() -> None:
            await watched(wait_healthy(engine_url), processes, READY_TIMEOUT_S)
            await consume(plan, processes, engine_url, fingerprints)

        # A failed install still reports the versions installed before it
        try:
            asyncio.run(run())
            processes.wait("trainer")
        except (Exception, SystemExit) as exc:
            error = f"{type(exc).__name__}: {exc}"
    if -1 not in fingerprints:
        raise RuntimeError(f"The engine never served its cold-loaded weights: {error}")

    ignore = re.compile(args.ignore) if args.ignore else None
    check_path = output_dir / "trainer_checkpoint_check.json"
    report = {
        "model_dir": plan["model_dir"],
        "transport": args.transport,
        "error": error,
        "engine_error": last_error(output_dir / "logs" / "inference.log") if error else None,
        "trainer_error": last_error(output_dir / "logs" / "trainer.log") if error else None,
        "trainer_checkpoint_check": json.loads(check_path.read_text()) if check_path.exists() else None,
        **compare(plan, fingerprints, ignore),
    }
    report["failures"] = failures(report)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "fingerprints.json").write_text(json.dumps(fingerprints))
    check = report["trainer_checkpoint_check"]
    print(
        f"{report['tensors']} engine tensors ({report['weights']} weights, {report['state']} state), "
        f"trainer W0 vs checkpoint: {check['compared'] if check else 'not checked'} tensors"
    )
    for step, result in report["steps"].items():
        counts = {key: len(value) for key, value in result.items() if isinstance(value, list) and value}
        print(f"step {step} ({result['version']}): {counts or 'ok'}")
    for failure in report["failures"]:
        print(f"FAIL {failure}")
    for key in ("engine_error", "trainer_error"):
        if report[key]:
            print(f"{key}: {report[key]}")
    print("PASS" if not report["failures"] else "FAIL")
    return 1 if report["failures"] else 0


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--trainer":
        train(json.loads(Path(sys.argv[2]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="HF checkpoint vLLM cold-loads, holding the trainer's W0")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--transport", choices=["modelexpress", "filesystem"], default="modelexpress")
    parser.add_argument("--vllm", default="{}", help="vLLM engine arguments, as JSON")
    parser.add_argument("--model", default="{}", help="Trainer model config fields, as JSON")
    parser.add_argument(
        "--trainer-gpus", default="0", help="Comma-separated visible GPU indices, one trainer rank each"
    )
    parser.add_argument("--inference-gpus", default="1", help="Comma-separated visible GPU indices of the engine (TP)")
    parser.add_argument("--ignore", default=None, help="Regex of engine tensor names to leave out of the checks")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds for each handshake, install and fetch")
    sys.exit(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
