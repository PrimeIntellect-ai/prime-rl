from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Sequence

from prime_rl.baselines.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
DEFAULT_COMPARE_OUTPUT = REPO_ROOT / "outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md"
DEFAULT_PATCHED_VERIFIERS = Path("/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs")
DEFAULT_OMNI_ENV = REPO_ROOT / "environments/omni_math2_singleturn"
DEFAULT_OFFLINE_EVAL_MAX_CONCURRENCY = 256


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _uv_python_module(module: str) -> list[str]:
    return ["uv", "run", "--no-sync", "python", "-m", module]


def _run(argv: Sequence[str], *, print_only: bool) -> int:
    rendered = shlex.join(str(part) for part in argv)
    print(rendered)
    if print_only:
        return 0
    return subprocess.run(list(argv), check=False).returncode


def _run_shell(script: str, *, print_only: bool) -> int:
    print(script)
    if print_only:
        return 0
    return subprocess.run(["bash", "-lc", script], check=False).returncode


def _shell_join(argv: Sequence[str]) -> str:
    parts = []
    for part in argv:
        if part == "${SLURM_JOB_ID}":
            parts.append('"$SLURM_JOB_ID"')
        else:
            parts.append(shlex.quote(str(part)))
    return " ".join(parts)


def _parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {value!r}") from exc


def _build_rlvr_command(args: argparse.Namespace) -> list[str]:
    command = ["uv", "run", "--no-sync", "rl"]
    for config in args.config:
        command.extend(["@", str(config)])
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if args.dry_run:
        command.append("--dry-run")
    command.extend(args.extra)
    return command


def _run_rlvr(args: argparse.Namespace) -> int:
    return _run(_build_rlvr_command(args), print_only=args.print_only)


def _offline_eval_args(args: argparse.Namespace) -> list[str]:
    weights_root = args.weights_root or args.run_root / "broadcasts"
    output_dir = args.output_dir or args.run_root.parent / "offline_eval_600x8_all_ckpts"
    command = _uv_python_module("scripts.evals.offline_omni_math2_ckpt_eval")
    command.extend(
        [
            "--arm",
            args.arm,
            "--run-root",
            str(args.run_root),
            "--weights-root",
            str(weights_root),
            "--output-dir",
            str(output_dir),
            "--base-model",
            args.base_model,
            "--num-examples",
            str(args.num_examples),
            "--rollouts-per-example",
            str(args.rollouts_per_example),
            "--max-concurrency",
            str(args.max_concurrency),
            "--score-max-concurrency",
            str(args.score_max_concurrency),
            "--max-retries",
            str(args.max_retries),
            "--ks",
            ",".join(str(k) for k in args.ks),
        ]
    )
    if args.steps:
        command.extend(["--steps", ",".join(str(step) for step in args.steps)])
    else:
        command.extend(
            [
                "--step-interval",
                str(args.step_interval),
                "--min-step",
                str(args.min_step),
                "--max-step",
                str(args.max_step),
            ]
        )

    if args.base_url:
        command.extend(["--base-url", args.base_url])
        for admin_url in args.admin_url:
            command.extend(["--admin-url", admin_url])
    else:
        command.extend(
            [
                "--launch-mode",
                "srun_multinode",
                "--launch-nodes",
                str(args.nodes),
                "--launch-gpus-per-node",
                str(args.gpus_per_node),
                "--launch-dp",
                str(args.dp_per_node),
                "--launch-tp",
                str(args.tp),
                "--launch-api-server-count",
                str(args.api_server_count),
                "--launch-data-parallel-size-local",
                str(args.dp_local),
                "--launch-port",
                str(args.port),
                "--launch-backend-port",
                str(args.backend_port),
                "--launch-srun-job-id",
                "${SLURM_JOB_ID}",
                "--launch-gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--launch-max-model-len",
                str(args.max_model_len),
                "--launch-max-num-seqs",
                str(args.max_num_seqs),
                "--launch-max-num-batched-tokens",
                str(args.max_num_batched_tokens),
            ]
        )
        if args.router_port is not None:
            command.extend(["--launch-router-port", str(args.router_port)])
    return command


def _offline_eval_env_script(args: argparse.Namespace, *, command: Sequence[str]) -> str:
    weights_root = args.weights_root or args.run_root / "broadcasts"
    output_dir = args.output_dir or args.run_root.parent / "offline_eval_600x8_all_ckpts"
    log_dir = output_dir / "logs"
    requested_steps = tuple(sorted(set(args.steps or ())))
    wait_block = ""
    if args.wait_step is not None:
        wait_block = textwrap.dedent(
            f"""
            wait_path={_q(weights_root)}/step_{args.wait_step}/STABLE
            echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for $wait_path"
            while [[ ! -f "$wait_path" ]]; do
                sleep 30
            done
            """
        ).strip()
    weight_preflight_block = ""
    if requested_steps:
        step_dirs = " ".join(_q(weights_root / f"step_{step}") for step in requested_steps)
        weight_preflight_block = textwrap.dedent(
            f"""
            echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] offline eval weight preflight: {_q(weights_root)}"
            print_available_weights() {{
                echo "available stable checkpoint markers under {_q(weights_root)}:" >&2
                find {_q(weights_root)} -maxdepth 2 \\( -name STABLE -o -name 'model.safetensors.index.json' -o -name '*.safetensors' \\) -printf '%p\\n' >&2 || true
            }}
            for step_dir in {step_dirs}; do
                if [[ ! -d "$step_dir" ]]; then
                    echo "missing requested checkpoint directory: $step_dir" >&2
                    print_available_weights
                    exit 6
                fi
                if [[ ! -f "$step_dir/STABLE" ]]; then
                    echo "requested checkpoint is not marked STABLE: $step_dir" >&2
                    print_available_weights
                    exit 6
                fi
                if [[ ! -f "$step_dir/model.safetensors.index.json" ]] && ! compgen -G "$step_dir/*.safetensors" >/dev/null; then
                    echo "requested checkpoint has no safetensors manifest or shards: $step_dir" >&2
                    print_available_weights
                    exit 6
                fi
            done
            """
        ).strip()
    host_setup = ""
    if not args.base_url:
        host_setup = textwrap.dedent(
            f"""
            if [[ -z "${{SLURM_JOB_ID:-}}" ]]; then
                echo "SLURM_JOB_ID is required for srun_multinode offline eval" >&2
                exit 1
            fi
            all_hosts=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
            if [[ "${{#all_hosts[@]}}" -lt {args.driver_node_count + args.nodes} ]]; then
                echo "expected at least {args.driver_node_count + args.nodes} allocated nodes, got: ${{all_hosts[*]}}" >&2
                exit 4
            fi
            driver_node_count={args.driver_node_count}
            if [[ "$driver_node_count" -lt 0 || "$driver_node_count" -ge "${{#all_hosts[@]}}" ]]; then
                echo "invalid driver_node_count=$driver_node_count for hosts: ${{all_hosts[*]}}" >&2
                exit 5
            fi
            eval_hosts=( "${{all_hosts[@]:$driver_node_count:{args.nodes}}}" )
            export PRIME_RL_MULTINODE_HOSTS="${{eval_hosts[*]}}"
            echo "driver_host=$(hostname)"
            echo "vllm_hosts=$PRIME_RL_MULTINODE_HOSTS"
            """
        ).strip()

    return textwrap.dedent(
        f"""
        set -euo pipefail
        cd {_q(args.root)}
        set -a
        [ -f .env ] && source .env
        set +a
        export PATH="$PWD/.venv/bin:$PATH"

        {host_setup}

        export PYTHONPATH="{_q(args.patched_verifiers)}:{_q(args.omni_env_path)}${{PYTHONPATH:+:$PYTHONPATH}}"
        export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/projects/a6r/joanv.a6r/tmp/xdg-cache}}"
        export XDG_CONFIG_HOME="${{XDG_CONFIG_HOME:-/projects/a6r/joanv.a6r/tmp/xdg-config}}"
        export VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT:-/projects/a6r/joanv.a6r/tmp/vllm-cache}}"
        export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-/tmp/triton-cache-${{SLURM_JOB_ID:-offline-eval}}}}"
        export TORCHINDUCTOR_CACHE_DIR="${{TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor-cache-${{SLURM_JOB_ID:-offline-eval}}}}"
        export INDUCTOR_CACHE_DIR="${{INDUCTOR_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR}}"
        export VLLM_TORCH_COMPILE_CACHE_DIR="${{VLLM_TORCH_COMPILE_CACHE_DIR:-/tmp/vllm-compile-${{SLURM_JOB_ID:-offline-eval}}}}"
        export VLLM_NO_USAGE_STATS=1
        export PRIME_RL_DISABLE_VLLM_ROUTER="{int(args.disable_router)}"
        export PRIME_RL_VLLM_ROUTER_POLICY="{args.router_policy}"

        mkdir -p {_q(log_dir)}
        {wait_block}
        {weight_preflight_block}

        log_path={_q(log_dir)}/launcher_$(date -u '+%Y%m%dT%H%M%SZ').log
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] logging to $log_path"
        set +e
        {_shell_join(command)} 2>&1 | tee "$log_path"
        status=${{PIPESTATUS[0]}}
        set -e
        uv run --no-sync python scripts/evals/compare_omni_math2_offline_evals.py --output {_q(args.compare_output)}
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] offline eval exited with status $status"
        exit "$status"
        """
    ).strip()


def _canonical_offline_eval_reentry(args: argparse.Namespace) -> list[str]:
    command = _uv_python_module("prime_rl.entrypoints.launch")
    command.extend(
        [
            "offline-eval",
            "--in-allocation",
            "--arm",
            args.arm,
            "--run-root",
            str(args.run_root),
            "--output-dir",
            str(args.output_dir or args.run_root.parent / "offline_eval_600x8_all_ckpts"),
            "--base-model",
            args.base_model,
            "--num-examples",
            str(args.num_examples),
            "--rollouts-per-example",
            str(args.rollouts_per_example),
            "--max-concurrency",
            str(args.max_concurrency),
            "--score-max-concurrency",
            str(args.score_max_concurrency),
            "--max-retries",
            str(args.max_retries),
            "--ks",
            ",".join(str(k) for k in args.ks),
            "--nodes",
            str(args.nodes),
            "--gpus-per-node",
            str(args.gpus_per_node),
            "--dp-per-node",
            str(args.dp_per_node),
            "--tp",
            str(args.tp),
            "--api-server-count",
            str(args.api_server_count),
            "--dp-local",
            str(args.dp_local),
            "--port",
            str(args.port),
            "--backend-port",
            str(args.backend_port),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--driver-node-count",
            str(args.driver_node_count),
            "--router-policy",
            args.router_policy,
            "--compare-output",
            str(args.compare_output),
            "--root",
            str(args.root),
            "--patched-verifiers",
            str(args.patched_verifiers),
            "--omni-env-path",
            str(args.omni_env_path),
        ]
    )
    if args.router_port is not None:
        command.extend(["--router-port", str(args.router_port)])
    if args.weights_root is not None:
        command.extend(["--weights-root", str(args.weights_root)])
    if args.steps:
        command.extend(["--steps", ",".join(str(step) for step in args.steps)])
    else:
        command.extend(
            [
                "--step-interval",
                str(args.step_interval),
                "--min-step",
                str(args.min_step),
                "--max-step",
                str(args.max_step),
            ]
        )
    if args.disable_router:
        command.append("--disable-router")
    return command


def _submit_offline_eval(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or args.run_root.parent / "offline_eval_600x8_all_ckpts"
    submit_dir = output_dir / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)
    stamp = subprocess.check_output(["date", "-u", "+%Y%m%dT%H%M%SZ"], text=True).strip()
    sbatch_path = submit_dir / f"offline_eval_after_{args.after_job_id}_{stamp}.sbatch"
    reentry = _canonical_offline_eval_reentry(args)
    sbatch_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=offline-{args.arm}
        #SBATCH --nodes={args.sbatch_nodes}
        #SBATCH --ntasks-per-node=1
        #SBATCH --gres=gpu:{args.gpus_per_node}
        #SBATCH --partition={args.partition}
        #SBATCH --account={args.account}
        #SBATCH --time={args.time_limit}
        #SBATCH --exclusive
        #SBATCH --dependency={args.dependency_type}:{args.after_job_id}
        #SBATCH --output={output_dir}/job_%j.log
        #SBATCH --error={output_dir}/job_%j.log

        set -euo pipefail
        cd {_q(args.root)}
        sleep {args.postrun_settle_seconds}
        {shlex.join(str(part) for part in reentry)}
        """
    )
    sbatch_path.write_text(sbatch_text)
    subprocess.run(["bash", "-n", str(sbatch_path)], check=True)
    print(f"wrote {sbatch_path}")
    if args.dry_run or args.print_only:
        return 0
    completed = subprocess.run(["sbatch", str(sbatch_path)], check=False, text=True, capture_output=True)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        return completed.returncode
    job_id = ""
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[:3] == ["Submitted", "batch", "job"]:
            job_id = parts[3]
            break
    if not job_id:
        print("failed to parse submitted eval job id", file=sys.stderr)
        return 1
    submission = textwrap.dedent(
        f"""\
        # Postrun Offline Eval Submission

        - eval_job_id: `{job_id}`
        - dependency: `{args.dependency_type}:{args.after_job_id}`
        - arm: `{args.arm}`
        - run_root: `{args.run_root}`
        - offline_output: `{output_dir}`
        - sbatch: `{sbatch_path}`
        - comparison: `{args.compare_output}`
        """
    )
    (output_dir / "POSTRUN_EVAL_SUBMISSION.md").write_text(submission)
    print(job_id)
    return 0


def _run_offline_eval(args: argparse.Namespace) -> int:
    if args.after_job_id and not args.in_allocation:
        return _submit_offline_eval(args)
    command = _offline_eval_args(args)
    script = _offline_eval_env_script(args, command=command)
    return _run_shell(script, print_only=args.print_only or args.dry_run)


def _run_data(args: argparse.Namespace) -> int:
    if not args.baseline_config and not args.baseline_rollouts:
        raise SystemExit("data requires --baseline-config and/or --baseline-rollouts")
    if args.filter_output and not args.dataset:
        raise SystemExit("--dataset is required when --filter-output is set")

    if args.baseline_config and not args.skip_baseline:
        status = _run(["uv", "run", "--no-sync", "baseline-eval", "--config", str(args.baseline_config)], print_only=args.print_only)
        if status != 0:
            return status

    baseline_rollouts = args.baseline_rollouts
    if baseline_rollouts is None and args.baseline_config is not None:
        baseline_rollouts = load_config(args.baseline_config).output_dir / "eval_rollouts.jsonl"

    if args.filter_output is None:
        return 0
    command = _uv_python_module("scripts.evals.make_perfectible_subset")
    command.extend(
        [
            "--baseline-rollouts",
            str(baseline_rollouts),
            "--dataset",
            str(args.dataset),
            "--output",
            str(args.filter_output),
            "--low",
            str(args.low),
            "--high",
            str(args.high),
            "--min-rollouts",
            str(args.min_rollouts),
            "--rollout-id-key",
            args.rollout_id_key,
            "--rollout-reward-key",
            args.rollout_reward_key,
            "--dataset-id-key",
            args.dataset_id_key,
        ]
    )
    if args.hf_repo:
        command.extend(["--hf-repo", args.hf_repo])
    if args.hf_private:
        command.append("--hf-private")
    return _run(command, print_only=args.print_only)


def _add_common_offline_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url")
    parser.add_argument("--admin-url", action="append", default=[])
    parser.add_argument("--steps", type=_parse_csv_ints)
    parser.add_argument("--step-interval", type=int, default=25)
    parser.add_argument("--min-step", type=int, default=25)
    parser.add_argument("--max-step", type=int, default=100)
    parser.add_argument("--num-examples", type=int, default=int(os.environ.get("OFFLINE_EVAL_NUM_EXAMPLES", "600")))
    parser.add_argument(
        "--rollouts-per-example",
        type=int,
        default=int(os.environ.get("OFFLINE_EVAL_ROLLOUTS_PER_EXAMPLE", "8")),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("OFFLINE_EVAL_MAX_CONCURRENCY", str(DEFAULT_OFFLINE_EVAL_MAX_CONCURRENCY))),
    )
    parser.add_argument(
        "--score-max-concurrency",
        type=int,
        default=int(os.environ.get("OFFLINE_EVAL_SCORE_MAX_CONCURRENCY", "1024")),
    )
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("OFFLINE_EVAL_MAX_RETRIES", "3")))
    parser.add_argument("--ks", type=_parse_csv_ints, default=_parse_csv_ints(os.environ.get("OFFLINE_EVAL_KS", "1,2,3,4,5,6,8")))
    parser.add_argument("--nodes", type=int, default=int(os.environ.get("OFFLINE_EVAL_NODES", "8")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("OFFLINE_EVAL_GPUS_PER_NODE", "4")))
    parser.add_argument("--dp-per-node", type=int, default=int(os.environ.get("OFFLINE_EVAL_DP_PER_NODE", "4")))
    parser.add_argument("--tp", type=int, default=int(os.environ.get("OFFLINE_EVAL_TP", "1")))
    parser.add_argument("--api-server-count", type=int, default=int(os.environ.get("OFFLINE_EVAL_API_SERVER_COUNT", "4")))
    parser.add_argument("--dp-local", type=int, default=int(os.environ.get("OFFLINE_EVAL_DP_LOCAL", "4")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OFFLINE_EVAL_PORT", "9800")))
    parser.add_argument(
        "--router-port",
        type=int,
        default=int(os.environ["OFFLINE_EVAL_ROUTER_PORT"]) if "OFFLINE_EVAL_ROUTER_PORT" in os.environ else None,
    )
    parser.add_argument("--backend-port", type=int, default=int(os.environ.get("OFFLINE_EVAL_BACKEND_PORT", "9900")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("OFFLINE_EVAL_GPU_MEMORY_UTILIZATION", "0.95")),
    )
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("OFFLINE_EVAL_MAX_MODEL_LEN", "16384")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("OFFLINE_EVAL_MAX_NUM_SEQS", "192")))
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("OFFLINE_EVAL_MAX_NUM_BATCHED_TOKENS", "65536")),
    )
    parser.add_argument("--driver-node-count", type=int, default=int(os.environ.get("OFFLINE_EVAL_DRIVER_NODE_COUNT", "0")))
    parser.add_argument("--router-policy", default=os.environ.get("PRIME_RL_VLLM_ROUTER_POLICY", "round_robin"))
    parser.add_argument("--disable-router", action="store_true", default=os.environ.get("PRIME_RL_DISABLE_VLLM_ROUTER", "0") == "1")
    parser.add_argument("--wait-step", type=int)
    parser.add_argument("--compare-output", type=Path, default=DEFAULT_COMPARE_OUTPUT)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--patched-verifiers", type=Path, default=DEFAULT_PATCHED_VERIFIERS)
    parser.add_argument("--omni-env-path", type=Path, default=DEFAULT_OMNI_ENV)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Prime-RL launch commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rlvr = subparsers.add_parser("rlvr", help="Launch or dry-run an RLVR config.")
    rlvr.add_argument("--config", type=Path, action="append", required=True, help="TOML config; may be repeated.")
    rlvr.add_argument("--output-dir", type=Path)
    rlvr.add_argument("--dry-run", action="store_true")
    rlvr.add_argument("--print-only", action="store_true")
    rlvr.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to `rl`; prefix with --.")
    rlvr.set_defaults(func=_run_rlvr)

    offline = subparsers.add_parser("offline-eval", help="Run or submit post-run OmniMath2 checkpoint eval.")
    _add_common_offline_eval_args(offline)
    offline.add_argument("--after-job-id", help="Submit an sbatch dependency instead of running in this shell.")
    offline.add_argument("--in-allocation", action="store_true", help="Internal flag used by generated sbatch jobs.")
    offline.add_argument("--partition", default=os.environ.get("PRIME_RL_SLURM_PARTITION", "workq"))
    offline.add_argument("--account", default=os.environ.get("PRIME_RL_SLURM_ACCOUNT", "brics.a6r"))
    offline.add_argument("--time-limit", default=os.environ.get("OFFLINE_EVAL_TIME_LIMIT", "06:00:00"))
    offline.add_argument("--dependency-type", default=os.environ.get("OFFLINE_EVAL_DEPENDENCY_TYPE", "afterany"))
    offline.add_argument(
        "--postrun-settle-seconds",
        type=int,
        default=int(os.environ.get("OFFLINE_EVAL_POSTRUN_SETTLE_SECONDS", "120")),
    )
    offline.add_argument("--sbatch-nodes", type=int, default=8)
    offline.set_defaults(func=_run_offline_eval)

    data = subparsers.add_parser("data", help="Generate baseline rollouts and/or filter a dataset from those rollouts.")
    data.add_argument("--baseline-config", type=Path, help="Run `baseline-eval --config` first.")
    data.add_argument("--skip-baseline", action="store_true", help="Do not run baseline generation even if --baseline-config is set.")
    data.add_argument("--baseline-rollouts", type=Path, help="Existing eval_rollouts.jsonl; inferred from --baseline-config if omitted.")
    data.add_argument("--dataset", type=Path, help="Source dataset JSONL for filtering.")
    data.add_argument("--filter-output", type=Path, help="Filtered dataset JSONL output.")
    data.add_argument("--low", type=float, default=0.2)
    data.add_argument("--high", type=float, default=0.8)
    data.add_argument("--min-rollouts", type=int, default=8)
    data.add_argument("--rollout-id-key", default="example_id")
    data.add_argument("--rollout-reward-key", default="reward")
    data.add_argument("--dataset-id-key", default="id")
    data.add_argument("--hf-repo")
    data.add_argument("--hf-private", action="store_true")
    data.add_argument("--print-only", action="store_true")
    data.set_defaults(func=_run_data)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
