"""vLLM on one node of the run: one engine per data-parallel rank (``tensor_parallel_size`` GPUs each, one API server
each), launched exactly like prime-rl's multi-node inference template (``_launch_rank.sh.j2``): ``inference @
inference.json`` with the DP size / rank / address overrides, the router nulled; and a vllm-router on node 0 over
every engine serving the same checkpoint (``_launch_router.sh.j2``), sticky per ``X-Session-ID`` so a rollout keeps
its engine (and its prefix cache)."""

import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Set by torchrun for its workers: never let them leak into vLLM (it would read RANK / WORLD_SIZE / MASTER_*).
TORCHRUN_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)
TORCHRUN_PREFIXES = ("TORCHELASTIC_",)


def clean_env(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if k not in TORCHRUN_VARS and not k.startswith(TORCHRUN_PREFIXES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def chat_template_path(serving, model_dir: Path) -> Path:
    path = Path(serving.chat_template)
    return path if path.is_absolute() else Path(model_dir) / path


def check_model_dir(serving, model_dir: Path) -> list[str]:
    """Why ``model_dir`` cannot be served as the eval's checkpoint (empty: it can). Cheap: no tensor reads."""
    model_dir = Path(model_dir)
    problems = []
    for name in ("config.json", "model.safetensors.index.json"):
        if not (model_dir / name).is_file():
            problems.append(f"{model_dir / name} missing")
    if problems:
        return problems
    try:
        weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    except Exception as exc:
        return [f"unreadable index: {exc!r}"]
    for name in sorted(set(weight_map.values())):
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{path} missing or empty")
    template = chat_template_path(serving, model_dir)
    if not template.is_file():
        problems.append(f"chat template {template} missing")
    elif serving.chat_template_sha256 is not None and sha256_file(template) != serving.chat_template_sha256:
        problems.append(f"chat template {template} sha256 {sha256_file(template)} != {serving.chat_template_sha256}")
    return problems


def inference_config(serving, model_dir: Path, output_dir: Path) -> dict[str, Any]:
    """The engines' shared ``inference.json`` (an ``InferenceConfig``; the per-engine port / DP rank come on the
    command line, as in the SLURM template)."""
    vllm: dict[str, Any] = {
        "model": str(model_dir),
        "served_model_name": serving.served_model_name,
        "max_model_len": serving.max_model_len,
        "chat_template": str(chat_template_path(serving, model_dir)),
        "default_chat_template_kwargs": dict(serving.default_chat_template_kwargs),
        "override_generation_config": dict(serving.override_generation_config),
        "reasoning_parser": serving.reasoning_parser,
        "tool_call_parser": serving.tool_call_parser,
        "enable_auto_tool_choice": True,
        "tensor_parallel_size": serving.tensor_parallel_size,
        "data_parallel_size": serving.data_parallel_size,
        "data_parallel_size_local": 1,
        "api_server_count": 1,
        "data_parallel_rpc_port": serving.rpc_port,
        "enable_expert_parallel": serving.enable_expert_parallel,
        "enable_prefix_caching": serving.enable_prefix_caching,
        "mamba_cache_dtype": serving.mamba_cache_dtype,
        "dtype": serving.dtype,
        "seed": serving.seed,
        "gpu_memory_utilization": serving.gpu_memory_utilization,
        "max_num_seqs": serving.max_num_seqs,
        "max_num_batched_tokens": serving.max_num_batched_tokens,
        **serving.vllm_extra,
    }
    return {
        "server": {"host": "0.0.0.0", "port": serving.engine_port},
        "router": None,
        "vllm": vllm,
        "env_vars": dict(serving.env_vars),
        "output_dir": str(output_dir),
    }


@dataclass
class EngineSpec:
    dp_rank: int
    port: int
    gpus: list[int]
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    """Overrides on top of the node's inference environment."""
    overrides: dict[str, Any] = field(default_factory=dict)
    """Per-engine ``[vllm]`` overrides, written to ``overrides_path`` (``--vllm @ <file>``, as the template does)."""
    overrides_path: Path | None = None


def engine_specs(serving, config_path: Path, node_ip: str, python: str) -> list[EngineSpec]:
    """One engine per data-parallel rank of the node: GPUs ``[d*tp, (d+1)*tp)``, port ``engine_port + d``. With
    expert parallelism the node's engines form one DP group (the experts spread over all of them); without it, each
    engine is a DP-1 replica of its own."""
    tp, dp = serving.tensor_parallel_size, serving.data_parallel_size
    dp_group = serving.enable_expert_parallel and dp > 1
    specs = []
    for d in range(dp):
        port = serving.engine_port + d
        cmd = [python, "-m", "prime_rl.entrypoints.inference", "@", str(config_path)]
        cmd += ["--server.host", "0.0.0.0", "--server.port", str(port)]
        cmd += ["--vllm.data-parallel-size", str(dp if dp_group else 1), "--vllm.data-parallel-size-local", "1"]
        cmd += ["--vllm.api-server-count", "1"]
        overrides: dict[str, Any] = {}
        overrides_path = None
        if dp_group:
            cmd += ["--vllm.data-parallel-rpc-port", str(serving.rpc_port)]
            overrides = {"data_parallel_rank": d, "data_parallel_address": node_ip}
            overrides_path = config_path.parent / f"vllm-overrides-{port}.json"
            cmd += ["--vllm", "@", str(overrides_path)]
        gpus = list(range(d * tp, (d + 1) * tp))
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            # One RPC / mooncake socket dir per engine (all engines of the node have data_parallel_rank_local 0).
            "VLLM_RPC_BASE_PATH": f"/tmp/vllm-rpc-inrun-eval-{port}",
        }
        specs.append(
            EngineSpec(
                dp_rank=d, port=port, gpus=gpus, cmd=cmd, env=env, overrides=overrides, overrides_path=overrides_path
            )
        )
    return specs


def engine_env(serving, base: dict[str, str]) -> dict[str, str]:
    """The node's inference environment: the process env without torchrun's variables, then prime-rl's launcher
    defaults for inference (e.g. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False``), then ``serving.env_vars``."""
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS

    return {**clean_env(base), **DEFAULT_COMMON_ENV_VARS, **DEFAULT_INFERENCE_ENV_VARS, **serving.env_vars}


def router_cmd(serving, router_bin: str, port: int, worker_urls: list[str]) -> list[str]:
    if not worker_urls:
        raise ValueError("a router needs at least one engine")
    return [
        router_bin,
        "--policy",
        serving.router_policy,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--worker-urls",
        *worker_urls,
        "--request-id-headers",
        "x-session-id",
        "--request-timeout-secs",
        str(serving.router_request_timeout_secs),
        "--prometheus-port",
        str(port + 21000),
        "--worker-startup-timeout-secs",
        "4200",
    ]


# Processes and HTTP


def http_get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception:
        return 0, b""


def http_post_json(url: str, body: dict, timeout: float = 120.0, headers: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception as exc:
        return 0, {"error": repr(exc)}


def serves_model(base_url: str, model: str) -> bool:
    status, body = http_get(f"{base_url}/v1/models")
    if status != 200:
        return False
    try:
        return any(item.get("id") == model for item in json.loads(body).get("data", []))
    except Exception:
        return False


def probe_chat(base_url: str, model: str, timeout: float = 600.0) -> tuple[bool, str]:
    """One short chat completion (with a tool, as the harness sends) through ``base_url``: template, parsers and
    routing work end to end. Not part of any rollout."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "max_tokens": 64,
    }
    status, payload = http_post_json(f"{base_url}/v1/chat/completions", body, timeout=timeout)
    if status != 200 or not payload.get("choices"):
        return False, f"HTTP {status}: {str(payload)[:300]}"
    return True, str(payload["choices"][0].get("finish_reason"))


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen
    log: Path

    def alive(self) -> bool:
        return self.popen.poll() is None


def start(name: str, cmd: list[str], env: dict[str, str], log: Path, cwd: Path | None = None) -> Proc:
    """A child in its own session (so its whole tree can be signalled), output appended to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "ab")
    popen = subprocess.Popen(
        cmd, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
    )
    handle.close()
    return Proc(name=name, popen=popen, log=log)


def stop(procs: list[Proc], grace: float = 120.0) -> None:
    """SIGTERM every process group, then SIGKILL whatever is left after ``grace`` seconds."""
    for proc in procs:
        if proc.alive():
            _signal_group(proc.popen.pid, signal.SIGTERM)
    end = time.monotonic() + grace
    for proc in procs:
        while proc.alive() and time.monotonic() < end:
            time.sleep(1)
    for proc in procs:
        # The group may outlive its leader (vLLM's workers): kill it either way.
        _signal_group(proc.popen.pid, signal.SIGKILL)
        try:
            proc.popen.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def tail(path: Path, lines: int = 40) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64 * 1024))
            return b"\n".join(f.read().splitlines()[-lines:]).decode(errors="replace")
    except OSError:
        return ""


def gpu_memory_used_mib() -> list[int] | None:
    """Used memory of every GPU (nvidia-smi), None if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return [int(float(line)) for line in out.stdout.split() if line.strip()]
    except Exception:
        return None


def wait_gpus_free(threshold_mib: int, timeout: float, log=print) -> bool:
    end = time.monotonic() + timeout
    used = None
    while time.monotonic() < end:
        used = gpu_memory_used_mib()
        if used is None:
            log("nvidia-smi unavailable, not waiting for free GPUs")
            return True
        if all(u <= threshold_mib for u in used):
            return True
        time.sleep(5)
    log(f"GPUs still in use after {timeout:.0f}s: {used} MiB")
    return False
