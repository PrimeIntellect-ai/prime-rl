"""Verify trainer weight updates against independent vLLM cold loads of W0, W1 and W2."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import sysconfig
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import httpx

VERSIONS = (0, 1, 0, 2, 0)
READY_TIMEOUT_S = 1800
TENSOR_FIELDS = ("dtype", "shape", "stride", "storage_offset", "hashed_shape", "device", "sha256")


def free_port() -> int:
    from prime_rl.utils.utils import get_free_port

    return get_free_port()


def write_version(model_dir: Path, destination: Path, version: int) -> None:
    """Perturb whole checkpoint tensors before loading or sharding them on either side."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    destination.mkdir()
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors weights in {model_dir}")
    for path in model_dir.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            (destination / path.name).symlink_to(path)
    for shard in shards:
        with safe_open(shard, framework="pt") as source:
            tensors = {}
            for name in source.keys():
                tensor = source.get_tensor(name)
                if tensor.is_floating_point() and tensor.numel():
                    seed = hashlib.sha256(f"{version}:{name}".encode()).digest()[:8]
                    generator = torch.Generator().manual_seed(int.from_bytes(seed, "little"))
                    work = tensor.to(torch.float64 if tensor.dtype == torch.float64 else torch.float32)
                    noise = torch.randn(work.shape, generator=generator, dtype=work.dtype)
                    scale = work.square().mean().sqrt() * 0.1 + 0.01
                    tensor = (work + scale * noise).to(tensor.dtype)
                tensors[name] = tensor
            save_file(tensors, destination / shard.name, metadata=source.metadata())


def as_bytes(tensor):
    import torch

    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def checkpoint_mismatches(model, model_dir: Path) -> dict:
    """Check that the trainer can round-trip the input checkpoint before an install."""
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


def expected_wire_shards(model, fp32_pattern: str | None) -> dict:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    keep_fp32 = re.compile(fp32_pattern).search if fp32_pattern is not None else model.keep_in_fp32_for_weight_transfer
    expected = {}
    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point() or not tensor.numel():
            continue
        shape = tuple(tensor.shape)
        if isinstance(tensor, DTensor):
            local, offset = compute_local_shape_and_global_offset(shape, tensor.device_mesh, tensor.placements)
            if not math.prod(local):
                continue
        else:
            local, offset = shape, (0,) * len(shape)
        dtype, itemsize = ("torch.float32", 4) if keep_fp32(name) else ("torch.bfloat16", 2)
        expected[name] = [dtype, itemsize, list(shape), [[list(offset), list(local)]]]
    return expected


def read_wire_shards(port: int, version_uid: str) -> list[dict]:
    """Read the advertised wire tensors from the MX control and trainer worker RPCs."""
    import grpc
    from modelexpress import auth
    from modelexpress_rl import refit_pb2, refit_pb2_grpc
    from modelexpress_rl.refit.reshard.rendezvous import unwrap_rendezvous_blob

    with auth.with_auth(grpc.insecure_channel(f"localhost:{port}")) as channel:
        service = refit_pb2_grpc.RefitServiceStub(channel)
        publications = service.ListWeightVersionShards(
            refit_pb2.ListWeightVersionShardsRequest(version_id=version_uid), timeout=30
        ).shards
    manifests = []
    for publication in publications:
        if publication.version_id != version_uid:
            raise RuntimeError(f"MX returned metadata for {publication.version_id} while checking {version_uid}")
        options = [("grpc.max_receive_message_length", 64 * 1024**2)]
        with grpc.insecure_channel(publication.metadata_endpoint, options=options) as channel:
            worker = refit_pb2_grpc.RefitWorkerServiceStub(channel)
            response = worker.GetTrainerShardMetadata(
                refit_pb2.GetTrainerShardMetadataRequest(metadata_digest=publication.stable_metadata_digest), timeout=30
            )
        if hashlib.sha256(response.metadata).hexdigest() != publication.stable_metadata_digest:
            raise RuntimeError(f"MX wire manifest digest mismatch on {publication.worker_id}")
        tensors = unwrap_rendezvous_blob(response.metadata, stable=True).tensors
        manifests.append(
            {
                tensor.name: [
                    tensor.dtype,
                    tensor.elsize,
                    list(tensor.full_shape),
                    sorted([list(shard.shard_offset), list(shard.shape)] for shard in tensor.shards),
                ]
                for tensor in tensors
            }
        )
    return manifests


def wire_mismatches(expected: list[dict], actual: list[dict]) -> dict:
    # Worker IDs are assigned by MX. Compare every announced copy, including replica multiplicity.
    expected_keys = Counter(json.dumps(shard, sort_keys=True) for shard in expected)
    actual_keys = Counter(json.dumps(shard, sort_keys=True) for shard in actual)
    missing = [json.loads(shard) for shard in (expected_keys - actual_keys).elements()]
    added = [json.loads(shard) for shard in (actual_keys - expected_keys).elements()]

    def by_name(shards):
        counts = defaultdict(Counter)
        for shard in shards:
            for name, entry in shard.items():
                counts[name][json.dumps(entry)] += 1
        return counts

    want, got = by_name(missing), by_name(added)
    return {
        "expected_workers": len(expected),
        "actual_workers": len(actual),
        "differ": sorted(name for name in want.keys() | got.keys() if want.get(name) != got.get(name)),
        "matches": any(expected) and not missing and not added,
    }


def train(plan: dict) -> None:
    """One trainer rank under torchrun. Load each version into the same sharded model."""
    import torch.distributed as dist

    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.model import load_dcp_from_hf, setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.trainer.world import get_world
    from prime_rl.transports.weights import setup_weight_sender
    from prime_rl.utils.logger import setup_logger

    setup_logger("info")
    setup_torch_distributed()
    world = get_world()
    output_dir = Path(plan["output_dir"])
    configs = [
        ModelConfig(
            name=path,
            **{"compile": None, "ac": None, **plan["model"], "conversion_dir": Path(path) / "trainer_conversion"},
        )
        for path in plan["checkpoints"]
    ]
    resolve_ep(configs[0])
    model = setup_model(configs[0], get_parallel_dims(configs[0]))
    check = checkpoint_mismatches(model, Path(configs[0].name))
    if world.is_master:
        (output_dir / "trainer_checkpoint_check.json").write_text(json.dumps(check, indent=2))

    if plan["transport"] == "modelexpress":
        shards = [None] * world.world_size
        dist.all_gather_object(shards, expected_wire_shards(model, plan["wire_fp32"]))
        if world.is_master:
            (output_dir / "expected_wire.json").write_text(json.dumps(shards))
    sender = setup_weight_sender(output_dir, transport_config(plan), model_name=plan["model_dir"])
    for step, version in enumerate(plan["steps"]):
        if step:
            load_dcp_from_hf(model, configs[version], materialize=False)
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
            **plan["staging"],
        )
    return FileSystemWeightBroadcastConfig(timeout=plan["timeout"])


async def watched(awaitable, processes, timeout: float):
    """Await, failing as soon as a launched process dies."""
    task = asyncio.ensure_future(awaitable)
    try:
        async with asyncio.timeout(timeout):
            while not task.done():
                if processes.error_queue:
                    raise RuntimeError(f"A process failed: {processes.error_queue[0]}")
                await asyncio.wait([task], timeout=1)
        return task.result()
    finally:
        if not task.done():
            task.cancel()


async def fingerprint(engine_url: str, expected_workers: int) -> dict:
    from prime_rl.configs.shared import ADMIN_TOKEN_HEADER, VLLMClientConfig

    token = os.environ.get(VLLMClientConfig().admin_token_var)
    headers = {ADMIN_TOKEN_HEADER: token} if token else {}
    async with httpx.AsyncClient(timeout=600, headers=headers) as client:
        response = await client.get(f"{engine_url}/weight_fingerprint/model", params={"buffers": "true"})
        response.raise_for_status()
    workers = response.json()["results"]
    if len(workers) != expected_workers or any(not worker for worker in workers):
        raise RuntimeError(f"Expected {expected_workers} nonempty worker fingerprints, got {[len(w) for w in workers]}")
    if len(workers) == 1:
        return workers[0]
    return {f"worker{index}/{name}": entry for index, worker in enumerate(workers) for name, entry in worker.items()}


async def probe(engine_url: str, model: str, dp_size: int) -> list[dict]:
    """Greedy decode through each DP engine, including the single-token graph path."""
    api_key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        outputs = []
        for rank in range(dp_size):
            response = await client.post(
                f"{engine_url}/inference/v1/generate",
                headers={"X-data-parallel-rank": str(rank)},
                json={
                    "model": model,
                    "token_ids": [1, 2, 3, 4],
                    "sampling_params": {"max_tokens": 4, "temperature": 0, "ignore_eos": True, "logprobs": 2},
                },
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if len(choice["token_ids"]) != 4 or choice["logprobs"] is None:
                raise RuntimeError(f"Incomplete forward probe on DP rank {rank}")
            outputs.append({"token_ids": choice["token_ids"], "logprobs": choice["logprobs"]})
        return outputs


async def wait_healthy(engine_url: str) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            try:
                if (await client.get(f"{engine_url}/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(2)


async def consume(plan: dict, processes, engine_url: str, fingerprints: dict, probes: dict, wire: dict, env: dict):
    from prime_rl.configs.shared import VLLMClientConfig
    from prime_rl.orchestrator.clients import AdminPlane
    from prime_rl.transports.weights import setup_weight_receiver
    from prime_rl.utils.pathing import get_broadcast_dir

    timeout, output_dir = plan["timeout"], Path(plan["output_dir"])

    class AuditedAdminPlane(AdminPlane):
        async def update_modelexpress_weights(self, *, version_uid, step=0, on_paused=None):
            expected = json.loads((output_dir / "expected_wire.json").read_text())
            actual = await asyncio.to_thread(read_wire_shards, plan["modelexpress_port"], version_uid)
            wire[step] = wire_mismatches(expected, actual)
            if not wire[step]["matches"]:
                (output_dir / f"actual_wire_step_{step}.json").write_text(json.dumps(actual))
            await super().update_modelexpress_weights(version_uid=version_uid, step=step, on_paused=on_paused)

    plan["metadata_port"] = free_port()
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    processes.start(
        "trainer",
        [*plan["trainer_cmd"], f"--master-port={free_port()}", __file__, "--trainer", str(plan_path)],
        env={**env, "MX_WORKER_GRPC_PORT": str(free_port())},
        log_path=output_dir / "logs" / "trainer.log",
    )
    admin_plane = AuditedAdminPlane(VLLMClientConfig(admin_base_url=[f"{engine_url}/v1"]))
    try:
        receiver = setup_weight_receiver(
            get_broadcast_dir(output_dir), transport_config(plan), admin_plane, plan["model_dir"]
        )
        await watched(receiver.initialize(), processes, timeout)
        fingerprints[-1] = await watched(fingerprint(engine_url, plan["inference_world_size"]), processes, timeout)
        for step in range(len(plan["steps"])):
            await watched(receiver.sync_startup(step, timeout), processes, timeout)
            fingerprints[step] = await watched(
                fingerprint(engine_url, plan["inference_world_size"]), processes, timeout
            )
            if plan["forward"]:
                probes[step] = await watched(
                    probe(engine_url, plan["served_model"], plan["data_parallel_size"]), processes, timeout
                )
            print(f"step {step}: installed and fingerprinted", flush=True)
    finally:
        await admin_plane.aclose()


def storage_groups(tensors: dict) -> dict:
    groups = defaultdict(list)
    for name, entry in tensors.items():
        if entry["storage_ptr"] is not None:
            worker = name.partition("/")[0] if "/" in name else ""
            groups[worker, entry["device"], entry["storage_ptr"]].append(name)
    return {name: sorted(group) for group in groups.values() for name in group if len(group) > 1}


def compare_tensors(reference: dict, actual: dict, ignore: re.Pattern | None, *, mx: bool) -> dict:
    wanted = {name: entry for name, entry in reference.items() if ignore is None or not ignore.search(name)}
    got = {name: entry for name, entry in actual.items() if ignore is None or not ignore.search(name)}
    common = sorted(wanted.keys() & got.keys())
    differences = {
        name: [field for field in TENSOR_FIELDS if wanted[name][field] != got[name][field]] for name in common
    }
    aliases_wanted, aliases_got = storage_groups(wanted), storage_groups(got)
    for name in common:
        before, after = wanted[name], got[name]
        # MX deliberately keeps non-floating checkpoint parameters as immutable buffers.
        if before["kind"] != after["kind"] and not (
            mx
            and before["kind"] == "parameter"
            and after["kind"] == "buffer"
            and not before["dtype"].startswith(("torch.float", "torch.bfloat"))
        ):
            differences[name].append("kind")
        if aliases_wanted.get(name) != aliases_got.get(name):
            differences[name].append("aliases")
    return {
        "compared": len(common),
        "missing": sorted(wanted.keys() - got.keys()),
        "added": sorted(got.keys() - wanted.keys()),
        "differ": {name: fields for name, fields in differences.items() if fields},
    }


def probe_matches(reference, actual) -> bool:
    if isinstance(reference, float) and isinstance(actual, (float, int)):
        return math.isclose(reference, actual, rel_tol=0, abs_tol=1e-3)
    if isinstance(reference, list) and isinstance(actual, list):
        return len(reference) == len(actual) and all(probe_matches(x, y) for x, y in zip(reference, actual))
    if isinstance(reference, dict) and isinstance(actual, dict):
        return reference.keys() == actual.keys() and all(
            probe_matches(value, actual[key]) for key, value in reference.items()
        )
    return reference == actual


def compare(plan: dict, references: dict, installed: dict, probes: dict, wire: dict, ignore: re.Pattern | None) -> dict:
    results = {}
    for step, version in enumerate(plan["steps"]):
        if step not in installed:
            continue
        current = installed[step]
        result = compare_tensors(
            references[version]["tensors"], current, ignore, mx=plan["transport"] == "modelexpress"
        )
        live = installed[-1]
        result["moved_parameters"] = sorted(
            name
            for name, entry in live.items()
            if entry["kind"] == "parameter"
            and name in current
            and (ignore is None or not ignore.search(name))
            and entry["data_ptr"] != current[name]["data_ptr"]
        )
        result.update(version=version, wire=wire.get(step))
        if plan["forward"]:
            result["forward_matches"] = step in probes and probe_matches(references[version]["forward"], probes[step])
        results[step] = result
    return results


def failures(plan: dict, report: dict) -> list[str]:
    found = []
    if report["error"] is not None:
        found.append(report["error"])
    check = report["trainer_checkpoint_check"]
    if not check or not check["compared"]:
        found.append("No trainer checkpoint tensors were compared")
    elif check["differ"]:
        found.append(f"Trainer W0 differs from the checkpoint: {check['differ']}")
    if not report["reference_changes"]:
        found.append("The cold references do not have three distinguishable versions")
    for step in range(len(plan["steps"])):
        result = report["steps"].get(step)
        if result is None:
            found.append(f"Step {step} was not compared")
            continue
        if not result["compared"]:
            found.append(f"Step {step} compared no tensors")
        for key in ("missing", "added", "differ", "moved_parameters"):
            if result[key]:
                found.append(f"Step {step} {key}: {result[key]}")
        if plan["transport"] == "modelexpress" and not (result["wire"] and result["wire"]["matches"]):
            found.append(f"Step {step} MX wire manifest differs: {result['wire']}")
        if plan["forward"] and not result["forward_matches"]:
            found.append(f"Step {step} forward probe differs")
    return found


def launch(args: argparse.Namespace, checkpoint_dir: Path) -> int:
    from prime_rl.entrypoints.modelexpress_server import start_modelexpress
    from prime_rl.utils.logger import setup_logger
    from prime_rl.utils.process import (
        DEFAULT_COMMON_ENV_VARS,
        DEFAULT_INFERENCE_ENV_VARS,
        DEFAULT_TRAINER_ENV_VARS,
        ProcessGroup,
    )

    setup_logger("info")
    output_dir, model_dir = args.output_dir.resolve(), args.model_dir.resolve()
    checkpoints = [model_dir, checkpoint_dir / "w1", checkpoint_dir / "w2"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = visible.split(",") if visible else None
    trainer_gpus, inference_gpus = (
        [visible[int(gpu)] if visible else gpu for gpu in gpus.split(",")]
        for gpus in (args.trainer_gpus, args.inference_gpus)
    )
    if len(set(trainer_gpus + inference_gpus)) != len(trainer_gpus) + len(inference_gpus):
        raise ValueError("Trainer and inference GPUs must be distinct")
    bin_dir = Path(sysconfig.get_path("scripts"))
    engine_args = json.loads(args.vllm)
    dp = engine_args.get("data_parallel_size", 1)
    tp = engine_args.setdefault("tensor_parallel_size", len(inference_gpus) // dp)
    pp = engine_args.get("pipeline_parallel_size", 1)
    if dp * tp * pp != len(inference_gpus):
        raise ValueError("Inference TP × DP × PP must match the inference GPU count")
    engine_args.update(served_model_name=["mx-check"], enable_prefix_caching=False)
    plan = {
        "model_dir": model_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "checkpoints": [str(path) for path in checkpoints],
        "transport": args.transport,
        "model": json.loads(args.model),
        "steps": VERSIONS,
        "timeout": args.timeout,
        "modelexpress_port": free_port(),
        "inference_world_size": len(inference_gpus),
        "data_parallel_size": dp,
        "served_model": "mx-check",
        "staging": {
            "staging_buffer_bytes": args.staging_buffer_bytes,
            "staging_buffers_count": args.staging_buffers_count,
        },
        "wire_fp32": args.wire_fp32,
        "trainer_cmd": [str(bin_dir / "torchrun"), f"--nproc-per-node={len(trainer_gpus)}"],
        "forward": args.forward,
    }
    base_env = {**os.environ, **DEFAULT_COMMON_ENV_VARS}
    trainer_env = {**base_env, **DEFAULT_TRAINER_ENV_VARS, "CUDA_VISIBLE_DEVICES": ",".join(trainer_gpus)}
    inference_env = {
        **base_env,
        **DEFAULT_INFERENCE_ENV_VARS,
        "CUDA_VISIBLE_DEVICES": ",".join(inference_gpus),
    }

    # Each cold load starts a new process on the same devices used by the live engine.
    def start_engine(processes, path: Path, name: str) -> str:
        engine_port = free_port()
        config_path = output_dir / f"{name}.json"
        config_path.write_text(
            json.dumps(
                {
                    "server": {"host": "127.0.0.1", "port": engine_port},
                    "router": None,
                    "weight_broadcast": {"type": args.transport},
                    "vllm": {**engine_args, "model": str(path), "data_parallel_rpc_port": free_port()},
                },
                indent=2,
            )
        )
        processes.start(
            name,
            [str(bin_dir / "inference"), "@", str(config_path)],
            env=inference_env,
            log_path=output_dir / "logs" / f"{name}.log",
        )
        return f"http://127.0.0.1:{engine_port}"

    async def cold_reference(engine_url: str, processes) -> dict:
        await watched(wait_healthy(engine_url), processes, READY_TIMEOUT_S)
        result = {
            "tensors": await watched(fingerprint(engine_url, len(inference_gpus)), processes, args.timeout),
        }
        if args.forward:
            result["forward"] = await watched(probe(engine_url, plan["served_model"], dp), processes, args.timeout)
        return result

    references, installed, probes, wire = {}, {}, {}, {}
    error = None
    try:
        for version in (1, 2):
            write_version(model_dir, checkpoints[version], version)
        for version, checkpoint in enumerate(checkpoints):
            with ProcessGroup() as processes:
                engine_url = start_engine(processes, checkpoint, f"reference_w{version}")
                references[version] = asyncio.run(cold_reference(engine_url, processes))
                print(
                    f"cold W{version}: fingerprinted {len(references[version]['tensors'])} engine tensors", flush=True
                )
        with ProcessGroup() as processes:
            if args.transport == "modelexpress":
                start_modelexpress(processes, plan["modelexpress_port"], output_dir / "logs", base_env)
            engine_url = start_engine(processes, model_dir, "inference")

            async def run() -> None:
                await watched(wait_healthy(engine_url), processes, READY_TIMEOUT_S)
                if args.forward:
                    probes[-1] = await watched(probe(engine_url, plan["served_model"], dp), processes, args.timeout)
                await consume(plan, processes, engine_url, installed, probes, wire, trainer_env)

            asyncio.run(run())
            processes.wait("trainer")
    except (Exception, SystemExit) as exc:
        error = f"{type(exc).__name__}: {exc}"

    ignore = re.compile(args.ignore) if args.ignore else None
    check_path = output_dir / "trainer_checkpoint_check.json"
    reference_digests = {
        tuple(
            sorted(
                (name, entry["sha256"])
                for name, entry in version["tensors"].items()
                if ignore is None or not ignore.search(name)
            )
        )
        for version in references.values()
    }
    report = {
        "model_dir": plan["model_dir"],
        "transport": args.transport,
        "error": error,
        "ignored": args.ignore,
        "trainer_checkpoint_check": json.loads(check_path.read_text()) if check_path.exists() else None,
        "reference_changes": len(reference_digests) == len(checkpoints),
        "steps": compare(plan, references, installed, probes, wire, ignore),
    }
    if args.forward and -1 in probes:
        report["startup_forward_matches"] = probe_matches(references[0]["forward"], probes[-1])
    report["failures"] = failures(plan, report)
    if report.get("startup_forward_matches") is False:
        report["failures"].append("Live engine's startup forward probe differs from the cold W0 reference")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "fingerprints.json").write_text(
        json.dumps(
            {
                "reference": references,
                "installed": installed,
                "forward": probes,
            }
        )
    )
    for failure in report["failures"]:
        print(f"FAIL {failure}")
    print("PASS" if not report["failures"] else "FAIL")
    return int(bool(report["failures"]))


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--trainer":
        train(json.loads(Path(sys.argv[2]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Local HF safetensors checkpoint holding W0")
    parser.add_argument("output_dir", type=Path, help="New directory for the run and its diagnostics")
    parser.add_argument("--transport", choices=["modelexpress", "filesystem"], default="modelexpress")
    parser.add_argument("--vllm", default="{}", help="vLLM engine arguments, as JSON")
    parser.add_argument("--model", default="{}", help="Trainer model config fields, as JSON")
    parser.add_argument(
        "--trainer-gpus", default="0", help="Comma-separated visible GPU indices, one trainer rank each"
    )
    parser.add_argument("--inference-gpus", default="1", help="Comma-separated visible GPU indices of the engine")
    parser.add_argument("--ignore", help="Regex of engine tensor names to leave out of the checks")
    parser.add_argument("--wire-fp32", help="Regex specifying FP32 wire tensors by native trainer name")
    parser.add_argument("--staging-buffer-bytes", type=int)
    parser.add_argument("--staging-buffers-count", type=int, default=1)
    parser.add_argument("--forward", action="store_true", help="Compare deterministic generation on every DP engine")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per handshake, install or fingerprint")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("OUTPUT_DIR must be empty")
    with tempfile.TemporaryDirectory(prefix="checkpoints-", dir=args.output_dir) as checkpoint_dir:
        status = launch(args, Path(checkpoint_dir).resolve())
    sys.exit(status)


if __name__ == "__main__":
    main()
