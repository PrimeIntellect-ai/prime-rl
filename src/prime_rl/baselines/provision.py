from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import tomli_w

from prime_rl.baselines.config import BaselineConfig

DP_COORDINATOR_STARTUP_TIMEOUT = "DP Coordinator process failed to report ZMQ addresses during startup."
TRANSIENT_HTTP_STATUS_CODES = {502, 503, 504}
TRANSIENT_READINESS_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key_var: str


def _normal_base_url(url: str) -> str:
    stripped = url.rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def _health_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/v1/models"


def _router_health_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/health"


def _api_key(api_key_var: str) -> str:
    return os.environ.get(api_key_var, "EMPTY")


def wait_for_endpoint(base_url: str, api_key_var: str, timeout_s: float, *, health_check: str = "models") -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    headers = {"Authorization": f"Bearer {_api_key(api_key_var)}"}
    with httpx.Client(timeout=10.0, headers=headers) as client:
        while time.time() < deadline:
            try:
                if health_check == "router_health":
                    response = client.get(_router_health_url(base_url))
                    response.raise_for_status()
                    return
                response = client.get(_health_url(base_url))
                response.raise_for_status()
                if response.json().get("data"):
                    return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in TRANSIENT_HTTP_STATUS_CODES:
                    raise
                last_error = exc
            except TRANSIENT_READINESS_ERRORS as exc:
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Inference endpoint {base_url} did not become ready. Last error: {last_error}")


def _wait_for_local_endpoint(
    base_url: str,
    api_key_var: str,
    timeout_s: float,
    proc: subprocess.Popen[str],
    log_path: Path,
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    headers = {"Authorization": f"Bearer {_api_key(api_key_var)}"}
    with httpx.Client(timeout=10.0, headers=headers) as client:
        while time.time() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Inference process exited with code {exit_code} before endpoint became ready. "
                    f"Last error: {last_error}. See {log_path}"
                )
            try:
                response = client.get(_health_url(base_url))
                response.raise_for_status()
                if response.json().get("data"):
                    return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in TRANSIENT_HTTP_STATUS_CODES:
                    raise
                last_error = exc
            except TRANSIENT_READINESS_ERRORS as exc:
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Inference endpoint {base_url} did not become ready. Last error: {last_error}")


def _wait_for_local_router(
    base_url: str,
    timeout_s: float,
    proc: subprocess.Popen[str],
    log_path: Path,
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    with httpx.Client(timeout=10.0) as client:
        while time.time() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Inference process exited with code {exit_code} before router became ready. "
                    f"Last error: {last_error}. See {log_path}"
                )
            try:
                response = client.get(_router_health_url(base_url))
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in TRANSIENT_HTTP_STATUS_CODES:
                    raise
                last_error = exc
            except TRANSIENT_READINESS_ERRORS as exc:
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Inference router {base_url} did not become ready. Last error: {last_error}")


def _log_has_dp_coordinator_startup_timeout(log_path: Path) -> bool:
    return log_path.exists() and DP_COORDINATOR_STARTUP_TIMEOUT in log_path.read_text(errors="replace")


def _slurm_hostnames(job_id: str | None = None) -> list[str]:
    cmd = ["scontrol", "show", "hostnames"]
    if job_id:
        node_list = subprocess.check_output(
            ["squeue", "-j", job_id, "-h", "-o", "%N"],
            text=True,
        ).strip()
        if not node_list:
            raise RuntimeError(f"No running Slurm job found for job_id={job_id!r}")
        cmd.append(node_list)
    result = subprocess.check_output(cmd, text=True)
    return [line.strip() for line in result.splitlines() if line.strip()]


def _multinode_hostnames(job_id: str | None = None) -> list[str]:
    if override := os.environ.get("PRIME_RL_MULTINODE_HOSTS"):
        return override.split()
    return _slurm_hostnames(job_id)


def _find_vllm_router() -> str | None:
    if router_bin := shutil.which("vllm-router"):
        return router_bin

    local_router = Path.cwd() / ".venv" / "bin" / "vllm-router"
    if local_router.is_file() and os.access(local_router, os.X_OK):
        return str(local_router)
    return None


def write_inference_config(config: BaselineConfig) -> Path:
    launch = config.launch
    output_dir = config.output_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "output_dir": str(output_dir),
        "gpu_memory_utilization": launch.gpu_memory_utilization,
        "server": {"host": "0.0.0.0", "port": launch.port},
        "model": {"name": config.model},
        "parallel": {"tp": launch.tp, "dp": launch.dp},
    }
    if launch.max_model_len is not None:
        payload["model"]["max_model_len"] = launch.max_model_len
    if launch.enforce_eager:
        payload["model"]["enforce_eager"] = True
    if launch.chat_template is not None:
        payload["model"]["chat_template"] = launch.chat_template
    if launch.api_server_count is not None:
        payload["api_server_count"] = int(launch.api_server_count)
    if launch.data_parallel_size_local is not None:
        payload["data_parallel_size_local"] = int(launch.data_parallel_size_local)
    if launch.data_parallel_rpc_port:
        payload["data_parallel_rpc_port"] = int(launch.data_parallel_rpc_port)
    if launch.enable_expert_parallel:
        payload["enable_expert_parallel"] = True
    if launch.use_deep_gemm:
        payload["use_deep_gemm"] = True
    if launch.vllm_extra:
        payload["vllm_extra"] = launch.vllm_extra

    path = output_dir / "inference.toml"
    with open(path, "wb") as f:
        tomli_w.dump(payload, f)
    return path


def _uses_multinode_tensor_parallel(config: BaselineConfig) -> bool:
    launch = config.launch
    gpus_per_node = launch.gpus_per_node or launch.dp * launch.tp
    pp = int(launch.vllm_extra.get("pipeline_parallel_size", 1) or 1)
    return (
        launch.mode == "srun_multinode"
        and launch.nodes > 1
        and launch.dp == 1
        and not launch.enable_expert_parallel
        and (launch.tp > gpus_per_node or pp > 1)
    )


def write_srun_multinode_script(config: BaselineConfig, config_path: Path, *, use_router: bool) -> Path:
    launch = config.launch
    output_dir = config.output_dir / "inference"
    script_path = output_dir / "launch_multinode.sh"
    gpus_per_node = launch.gpus_per_node or launch.dp * launch.tp
    router_port = launch.router_port or launch.port
    backend_port = launch.backend_port or (launch.port + 100)
    dp_local = launch.data_parallel_size_local or gpus_per_node
    router_extra_args = " ".join(shlex.quote(str(arg)) for arg in launch.router_extra_args)
    script = f"""#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR={str(Path.cwd())!r}
CONFIG_PATH={str(config_path)!r}
OUTPUT_DIR={str(output_dir)!r}
ROUTER_PORT={router_port}
BACKEND_PORT={backend_port}
DP_LOCAL={dp_local}
RPC_PORT={launch.data_parallel_rpc_port}
ENABLE_DISTRIBUTED_EP={1 if launch.enable_expert_parallel else 0}
ENABLE_MULTINODE_TP={1 if _uses_multinode_tensor_parallel(config) else 0}
USE_ROUTER={1 if use_router else 0}

cd "$PROJECT_DIR"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"
[ -f .env ] && source .env
mkdir -p "$OUTPUT_DIR"
ROUTER_POLICY="${{PRIME_RL_VLLM_ROUTER_POLICY:-round_robin}}"
ROUTER_EXTRA_ARGS=( {router_extra_args} )

if [ -n "${{PRIME_RL_MULTINODE_HOSTS:-}}" ]; then
    read -ra HOSTNAMES <<< "$PRIME_RL_MULTINODE_HOSTS"
else
    HOSTNAMES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
fi
HOSTNAMES_STR="${{HOSTNAMES[*]}}"
INFER_NODE_RANK=${{1:-${{SLURM_PROCID:-0}}}}
NODE_LOG="$OUTPUT_DIR/node_${{INFER_NODE_RANK}}.log"
HEAD_IP_FILE="$OUTPUT_DIR/head_hsn_ip"

ulimit -n 65536 || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Match production rl.sbatch: inference uses expandable_segments=False (True
# observed to crash vLLM v1 mid-run with api_server_count>1). Trainer uses
# True, but the baselines path is inference-only.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
# Avoid HF Hub 429 rate-limit storms during cold start. Gemma tokenizer init
# calls model_info() to probe "is this a Mistral base model?"; with 8 nodes ×
# 4 DP cores starting in parallel we exhaust the 1000-req/5min quota. Weights
# are pre-cached at HF_HUB_CACHE (Lustre), so offline mode is safe.
export HF_HUB_OFFLINE="${{HF_HUB_OFFLINE:-1}}"
export TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE:-1}}"
# Put torch.compile / triton caches on each node's local tmpfs to avoid the
# Lustre "Stale file handle" race when 32 workers autotune the same kernels
# in parallel on a cold cache. /tmp is 334G tmpfs per node on Isambard.
# Include job + hostname so concurrent jobs on shared nodes don't collide.
export TORCHINDUCTOR_CACHE_DIR="${{TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_${{SLURM_JOB_ID:-${{USER}}}}_$(hostname -s)}}"
export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-/tmp/triton_cache_${{SLURM_JOB_ID:-${{USER}}}}_$(hostname -s)}}"
# vLLM keeps its own torch.compile cache at $VLLM_CACHE_ROOT/torch_compile_cache;
# default is ~/.cache/vllm (Lustre on Isambard). Redirect to /tmp too.
export VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT:-/tmp/vllm_cache_${{SLURM_JOB_ID:-${{USER}}}}_$(hostname -s)}}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

module load brics/nccl 2>/dev/null || true
module load brics/aws-ofi-nccl 2>/dev/null || true
unset NCCL_ALGO

# Match production rl.sbatch: discover InfiniBand HCA so NCCL picks the
# right device for cross-node collectives (DP coordinator, weight bcast).
if command -v ibv_devinfo >/dev/null 2>&1; then
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[[:blank:]]' | paste -sd,)
    if [ -n "$IB_HCA" ]; then
        export NCCL_IB_HCA="$IB_HCA"
    fi
fi

LOCAL_IP=$(ip -o -4 addr show hsn0 2>/dev/null | awk '{{split($4,a,"/"); print a[1]; exit}}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(hostname -I | awk '{{print $1}}')
fi
export VLLM_HOST_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-hsn0}}"
export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-hsn}}"

if [ "$INFER_NODE_RANK" -eq 0 ]; then
    printf '%s\n' "$LOCAL_IP" > "$HEAD_IP_FILE"
else
    for _ in $(seq 1 120); do
        [ -s "$HEAD_IP_FILE" ] && break
        sleep 1
    done
    if [ ! -s "$HEAD_IP_FILE" ]; then
        echo "Timed out waiting for $HEAD_IP_FILE" | tee -a "$NODE_LOG"
        exit 1
    fi
fi
MASTER_HOST=$(cat "$HEAD_IP_FILE")

ROUTER_ARGS=""
for host in "${{HOSTNAMES[@]}}"; do
    ROUTER_ARGS="$ROUTER_ARGS http://${{host}}:${{BACKEND_PORT}}"
done

if [ "$USE_ROUTER" -eq 1 ] && [ "$INFER_NODE_RANK" -eq 0 ]; then
    ROUTER_LOG="$OUTPUT_DIR/router.log"
    echo "Starting vllm-router on $LOCAL_IP:$ROUTER_PORT for $ROUTER_ARGS" | tee "$ROUTER_LOG"
    vllm-router \\
        --policy "$ROUTER_POLICY" \\
        --worker-urls $ROUTER_ARGS \\
        --host 0.0.0.0 \\
        --port "$ROUTER_PORT" \\
        --intra-node-data-parallel-size "$DP_LOCAL" \\
        --api-key "${{VLLM_API_KEY:-EMPTY}}" \\
        --worker-startup-timeout-secs 4200 \\
        --log-level info \\
        "${{ROUTER_EXTRA_ARGS[@]}}" \\
        >> "$ROUTER_LOG" 2>&1 &
elif [ "$INFER_NODE_RANK" -eq 0 ]; then
    echo "vllm-router unavailable; exposing head backend directly on $LOCAL_IP:$BACKEND_PORT" | tee "$OUTPUT_DIR/router.log"
fi

VLLM_EXTRA="{{}}"
if [ "$ENABLE_DISTRIBUTED_EP" -eq 1 ]; then
    REPLICA_HEAD_HOST="$MASTER_HOST"
    START_RANK=$((INFER_NODE_RANK * DP_LOCAL))
    if [ "$INFER_NODE_RANK" -eq 0 ]; then
        VLLM_EXTRA="{{\\"data_parallel_address\\": \\"$REPLICA_HEAD_HOST\\"}}"
    else
        VLLM_EXTRA="{{\\"data_parallel_address\\": \\"$REPLICA_HEAD_HOST\\", \\"data_parallel_start_rank\\": $START_RANK, \\"headless\\": true, \\"api_server_count\\": 0}}"
    fi
elif [ "$ENABLE_MULTINODE_TP" -eq 1 ]; then
    if [ "$INFER_NODE_RANK" -eq 0 ]; then
        VLLM_EXTRA="{{\\"distributed_executor_backend\\": \\"mp\\", \\"nnodes\\": ${{#HOSTNAMES[@]}}, \\"node_rank\\": $INFER_NODE_RANK, \\"master_addr\\": \\"$MASTER_HOST\\", \\"master_port\\": $RPC_PORT, \\"distributed_timeout_seconds\\": 4200}}"
    else
        VLLM_EXTRA="{{\\"distributed_executor_backend\\": \\"mp\\", \\"nnodes\\": ${{#HOSTNAMES[@]}}, \\"node_rank\\": $INFER_NODE_RANK, \\"master_addr\\": \\"$MASTER_HOST\\", \\"master_port\\": $RPC_PORT, \\"distributed_timeout_seconds\\": 4200, \\"headless\\": true, \\"api_server_count\\": 0}}"
    fi
fi

echo "node_rank=$INFER_NODE_RANK host=$(hostname) local_ip=$LOCAL_IP master_host=$MASTER_HOST backend_port=$BACKEND_PORT gloo_if=$GLOO_SOCKET_IFNAME nccl_if=$NCCL_SOCKET_IFNAME nccl_net=${{NCCL_NET:-}} vllm_extra=$VLLM_EXTRA" | tee "$NODE_LOG"
uv run --no-sync --env-file .env inference \\
    @ "$CONFIG_PATH" \\
    --server.host 0.0.0.0 \\
    --server.port "$BACKEND_PORT" \\
    --vllm-extra "$VLLM_EXTRA" \\
    2>&1 | tee -a "$NODE_LOG"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def write_srun_multinode_driver_script(
    config: BaselineConfig,
    node_script: Path,
    *,
    hostnames: list[str],
) -> Path:
    launch = config.launch
    output_dir = config.output_dir / "inference"
    script_path = output_dir / "launch_multinode_driver.sh"
    gpus_per_node = launch.gpus_per_node or launch.dp * launch.tp
    hosts_literal = " ".join(hostnames)
    job_arg = f"--jobid={launch.srun_job_id}" if launch.srun_job_id else ""
    network_arg = f"--network={launch.srun_network}" if launch.srun_network else ""
    cpus_per_task = str(launch.srun_cpus_per_task or "")
    script = f"""#!/usr/bin/env bash
set -euo pipefail

NODE_SCRIPT={str(node_script)!r}
export PRIME_RL_MULTINODE_HOSTS={hosts_literal!r}
GPUS_PER_NODE={gpus_per_node}
JOB_ARG={job_arg!r}
NETWORK_ARG={network_arg!r}
CPUS_PER_TASK={cpus_per_task!r}
HOSTS=( $PRIME_RL_MULTINODE_HOSTS )
PIDS=()

NETWORK_ARGS=()
if [ -n "$NETWORK_ARG" ]; then
    NETWORK_ARGS=("$NETWORK_ARG")
fi
CPU_ARGS=()
if [ -n "$CPUS_PER_TASK" ]; then
    CPU_ARGS=(--cpus-per-task="$CPUS_PER_TASK")
fi
OVERLAP_ARGS=()
EXCLUSIVE_ARGS=(--exclusive)
if [ "${{PRIME_RL_SRUN_STEP_OVERLAP:-1}}" = "1" ]; then
    OVERLAP_ARGS=(--overlap)
    EXCLUSIVE_ARGS=()
fi

cleanup() {{
    for pid in "${{PIDS[@]:-}}"; do
        kill "$pid" 2>/dev/null || true
    done
}}
trap cleanup TERM INT EXIT

echo "[prime-rl] cleaning stale node-local inference state on ${{#HOSTS[@]}} hosts"
for host in "${{HOSTS[@]}}"; do
    srun "${{OVERLAP_ARGS[@]}}" $JOB_ARG \\
        "${{NETWORK_ARGS[@]}}" \\
        --nodes=1 \\
        --ntasks=1 \\
        --ntasks-per-node=1 \\
        --nodelist="$host" \\
        bash -c '
            # Target ONLY stale inference processes. Three layers of self-protection:
            #   1. Bracket-trick patterns ([V]LLM::APIServer etc) — the regex char class [V]
            #      matches "V" but the literal cmdline of this srun contains "[V]LLM" with
            #      brackets, so the regex does not match its own outer srun parent process.
            #   2. pgid filter skips bash/awk/ps subprocesses spawned by this script.
            #   3. Patterns are specific to inference subprocesses; baselines.cli is NEVER matched.
            self_pgid=$(ps -o pgid= -p "$$" | tr -d " ")
            ps -eo pid=,pgid=,args= | awk -v self_pgid="$self_pgid" '"'"'
                $2 == self_pgid {{ next }}
                /[V]LLM::|[E]ngineCore_DP|[P]RIME-RL::Infer|[v]llm-router|[p]rime_rl\\.entrypoints\\.inference/ {{ print $1 }}
            '"'"' | xargs -r kill -9
            sleep 2
            rm -rf /dev/shm/vllm-* /dev/shm/vllm_* /tmp/vllm-* /tmp/vllm_* /tmp/torch-* /tmp/torchelastic_* /tmp/vllm_cache_* /tmp/triton_cache_* /tmp/torch_inductor_* 2>/dev/null || true
            procs=$(ps -eo comm,args | grep -E "[V]LLM::|[E]ngineCore_DP|[P]RIME-RL::Infer" | wc -l)
            gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{{s+=\\$1}} END {{print s}}")
            echo "[node-cleanup] $(hostname) procs=$procs gpu_mem=${{gpu}}MiB"
        '
done

for idx in "${{!HOSTS[@]}}"; do
    host="${{HOSTS[$idx]}}"
    srun "${{OVERLAP_ARGS[@]}}" $JOB_ARG \\
        "${{NETWORK_ARGS[@]}}" \\
        --nodes=1 \\
        --ntasks=1 \\
        --ntasks-per-node=1 \\
        --nodelist="$host" \\
        --gpus-per-task="$GPUS_PER_NODE" \\
        "${{CPU_ARGS[@]}}" \\
        "${{EXCLUSIVE_ARGS[@]}}" \\
        bash "$NODE_SCRIPT" "$idx" &
    PIDS+=("$!")
done

status=0
for pid in "${{PIDS[@]}}"; do
    wait "$pid" || status=$?
done
exit "$status"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


class InferenceProvisioner(AbstractContextManager[Endpoint]):
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.proc: subprocess.Popen[str] | None = None
        self.log_file = None

    def __enter__(self) -> Endpoint:
        launch = self.config.launch
        if launch.mode == "external":
            if not self.config.base_url:
                raise ValueError("base_url is required when launch.mode='external'")
            base_url = _normal_base_url(self.config.base_url)
            wait_for_endpoint(
                base_url,
                self.config.api_key_var,
                timeout_s=launch.wait_timeout_s,
                health_check=launch.external_health_check,
            )
            return Endpoint(base_url=base_url, api_key_var=self.config.api_key_var)

        if launch.mode not in {"local", "srun", "srun_multinode"}:
            raise ValueError(f"Unsupported launch.mode={launch.mode!r}")

        config_path = write_inference_config(self.config)
        cmd = ["uv", "run", "--no-sync", "--env-file", ".env", "inference", "@", str(config_path)]
        if launch.extra_args:
            cmd.extend(launch.extra_args)
        multinode_hostnames: list[str] = []
        use_router = False
        if launch.mode == "srun_multinode":
            if launch.nodes < 2:
                raise ValueError("launch.nodes must be >= 2 when launch.mode='srun_multinode'")
            disable_router = os.environ.get("PRIME_RL_DISABLE_VLLM_ROUTER") == "1"
            allow_direct_backend = os.environ.get("PRIME_RL_ALLOW_DIRECT_BACKEND") == "1"
            uses_multinode_tp = _uses_multinode_tensor_parallel(self.config)
            router_bin = _find_vllm_router()
            use_router = (
                not disable_router
                and router_bin is not None
                and not uses_multinode_tp
            )
            if not use_router and not uses_multinode_tp and not allow_direct_backend:
                reason = (
                    "PRIME_RL_DISABLE_VLLM_ROUTER=1"
                    if disable_router
                    else "vllm-router was not found on PATH or in .venv/bin"
                )
                raise RuntimeError(
                    "vllm-router is required for srun_multinode inference. "
                    "Direct-backend fallback silently under-balances multi-server vLLM; "
                    f"{reason}. Set PRIME_RL_ALLOW_DIRECT_BACKEND=1 only for a deliberate debug run."
                )
            multinode_hostnames = _multinode_hostnames(launch.srun_job_id)
            # Optional: exclude the local hostname from the inference HOSTS list. Required on
            # Isambard, where running inference srun on the same node as the launcher CLI causes
            # the inference's DP coordinator to be SIGKILL'd at the ~4-minute mark under sustained
            # load (reproduced across 6+ attempts; smokes barely survive because they finish at
            # ~4 min; full runs always crash). Root cause not fully isolated (likely cgroup or
            # GPU-context contention with the interactive shell step on the same node). The fix
            # is operational: keep launcher + vllm-router on the local node, inference on others.
            effective_nodes = launch.nodes
            if os.environ.get("PRIME_RL_EXCLUDE_LOCAL_FROM_HOSTS") == "1":
                import socket
                local_short = socket.gethostname().split(".")[0]
                filtered = [h for h in multinode_hostnames if h.split(".")[0] != local_short]
                if len(filtered) < len(multinode_hostnames):
                    print(
                        f"[prime-rl] PRIME_RL_EXCLUDE_LOCAL_FROM_HOSTS=1: dropping local host "
                        f"{local_short!r}; effective nodes = {len(filtered)} (was {launch.nodes})"
                    )
                    multinode_hostnames = filtered
                    effective_nodes = min(effective_nodes, len(multinode_hostnames))
            if len(multinode_hostnames) < effective_nodes:
                raise RuntimeError(
                    f"Requested {effective_nodes} nodes but Slurm exposes only {multinode_hostnames}"
                )
            launch_script = write_srun_multinode_script(self.config, config_path, use_router=use_router)
            driver_script = write_srun_multinode_driver_script(
                self.config,
                launch_script,
                hostnames=multinode_hostnames[: effective_nodes],
            )
            cmd = ["bash", str(driver_script)]
        elif launch.mode == "srun":
            prefix = launch.launch_prefix or [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                f"--gpus={len((launch.gpus or '').split(',')) if launch.gpus else launch.dp * launch.tp}",
                "--exclusive",
            ]
            if not launch.launch_prefix:
                if launch.srun_job_id:
                    prefix.append(f"--jobid={launch.srun_job_id}")
                if launch.host == "127.0.0.1" and os.environ.get("SLURMD_NODENAME"):
                    prefix.append(f"--nodelist={os.environ['SLURMD_NODENAME']}")
            cmd = [*prefix, *cmd]

        env = os.environ.copy()
        env.setdefault(self.config.api_key_var, "EMPTY")
        if launch.gpus:
            env["CUDA_VISIBLE_DEVICES"] = launch.gpus
        if env.get("VLLM_RPC_BASE_PATH"):
            Path(env["VLLM_RPC_BASE_PATH"]).mkdir(parents=True, exist_ok=True)

        log_path = self.config.output_dir / "inference" / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        base_host = launch.host
        if launch.mode == "srun_multinode" and base_host == "127.0.0.1":
            base_host = multinode_hostnames[0]
        base_port = launch.port
        if launch.mode == "srun_multinode" and not use_router:
            base_port = launch.backend_port or (launch.port + 100)
        base_url = _normal_base_url(f"http://{base_host}:{base_port}")
        attempts = max(1, launch.server_start_retries + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                with log_path.open("a") as f:
                    f.write(
                        "\n[prime-rl] Retrying inference startup after vLLM DP coordinator "
                        f"startup timeout (attempt {attempt}/{attempts}).\n"
                    )
            self.log_file = open(log_path, "a" if attempt > 1 else "w")
            self.proc = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                if launch.mode == "srun_multinode" and use_router:
                    _wait_for_local_router(
                        base_url,
                        timeout_s=launch.wait_timeout_s,
                        proc=self.proc,
                        log_path=log_path,
                    )
                else:
                    _wait_for_local_endpoint(
                        base_url,
                        self.config.api_key_var,
                        timeout_s=launch.wait_timeout_s,
                        proc=self.proc,
                        log_path=log_path,
                    )
                return Endpoint(base_url=base_url, api_key_var=self.config.api_key_var)
            except RuntimeError as exc:
                last_error = exc
                self._stop_process()
                if attempt < attempts and _log_has_dp_coordinator_startup_timeout(log_path):
                    continue
                raise
        raise RuntimeError(f"Inference endpoint {base_url} did not become ready. Last error: {last_error}")

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_process()

    def _stop_process(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            pid = getattr(self.proc, "pid", None)
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if pid is not None:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
