"""Run a quick base-vs-checkpoint smoke eval against a local PRIME-RL inference server.

The script starts a single vLLM-backed inference server on the base model,
runs a prompt suite, hot-reloads checkpoint weights via /update_weights, and
runs the same prompts again. Results are written as JSON and Markdown for easy
inspection.

Usage:
    uv run --no-sync python scripts/smoke_eval_sft.py \
      --ckpt /scratch/.../weights/step_500
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

DEFAULT_PROMPT_PATH = Path(__file__).with_name("smoke_eval_sft_prompts.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/smoke_eval_sft")


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="HF-compatible checkpoint directory to evaluate.")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name to boot the server with. If omitted, infer from checkpoint config when possible.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="JSON prompt suite. Defaults to scripts/smoke_eval_sft_prompts.json.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Local inference server port.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Inference max model length.")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Start vLLM with --model.enforce-eager. Enabled by default for conservative smoke evals.",
    )
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help="Disable the default eager-mode safeguard and let vLLM use its normal compile path.",
    )
    parser.add_argument("--max-tokens", type=int, default=2400, help="Default max completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results into. Defaults to outputs/smoke_eval_sft/<timestamp>.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave the inference server running after the script finishes.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def infer_base_model(ckpt_dir: Path) -> str | None:
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        config = {}
    else:
        config = load_json(config_path)

    for key in ("_name_or_path", "name_or_path"):
        value = config.get(key)
        if isinstance(value, str) and value and value != str(ckpt_dir):
            return value

    # PRIME-RL weight exports do not always preserve the original HF source.
    # Fall back to the sibling training config when evaluating outputs/<run>/weights/step_N.
    try:
        run_root = ckpt_dir.parents[1]
    except IndexError:
        return None
    sft_config_path = run_root / "configs" / "sft.toml"
    if sft_config_path.exists():
        with open(sft_config_path, "rb") as f:
            sft_config = tomllib.load(f)
        model_name = sft_config.get("model", {}).get("name")
        if isinstance(model_name, str) and model_name:
            return model_name

    return None


def ensure_output_dir(path: Path | None) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = DEFAULT_OUTPUT_ROOT / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def start_inference_server(
    base_model: str,
    port: int,
    max_model_len: int,
    log_path: Path,
    *,
    enforce_eager: bool,
) -> subprocess.Popen[str]:
    cmd = [
        "uv",
        "run",
        "inference",
        "--model.name",
        base_model,
        "--model.max-model-len",
        str(max_model_len),
        "--server.port",
        str(port),
    ]
    if enforce_eager:
        cmd.extend(["--model.enforce-eager"])
    env = os.environ.copy()
    env.setdefault("VLLM_API_KEY", "EMPTY")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    proc._log_file_handle = log_file  # type: ignore[attr-defined]
    return proc


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        log_file = getattr(proc, "_log_file_handle", None)
        if log_file is not None:
            log_file.close()
        return

    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    log_file = getattr(proc, "_log_file_handle", None)
    if log_file is not None:
        log_file.close()


def wait_for_server(port: int, api_key: str, timeout_s: float = 300.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = None
    with httpx.Client(timeout=10.0, headers=auth_headers(api_key)) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{base_url(port)}/v1/models")
                response.raise_for_status()
                payload = response.json()
                if payload.get("data"):
                    return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Inference server on port {port} did not become ready. Last error: {last_error}")


def update_weights(port: int, ckpt_dir: Path, api_key: str) -> None:
    with httpx.Client(timeout=httpx.Timeout(120.0, read=120.0), headers=auth_headers(api_key)) as client:
        response = client.post(f"{base_url(port)}/update_weights", json={"weight_dir": str(ckpt_dir)})
        response.raise_for_status()


def get_model_id(port: int, api_key: str) -> str:
    payload = wait_for_server(port, api_key, timeout_s=30.0)
    return payload["data"][0]["id"]


def completion_metrics(text: str, usage: Any) -> dict[str, Any]:
    paragraphs = [chunk for chunk in text.split("\n\n") if chunk.strip()]
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(lines),
        "paragraphs": len(paragraphs),
        "bullets": sum(line.lstrip().startswith(("-", "*")) for line in lines),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def render_completion_prompt(prompt: str) -> str:
    return (
        "You are being evaluated for instruction following, clarity, and usefulness.\n"
        "Write a complete answer to the request below.\n\n"
        f"Request:\n{prompt}\n\n"
        "Answer:\n"
    )


def run_prompt_suite(
    *,
    prompts: list[dict[str, Any]],
    model_name: str,
    client: OpenAI,
    label: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    results = []
    for idx, prompt in enumerate(prompts, start=1):
        prompt_max_tokens = int(prompt.get("max_tokens", max_tokens))
        started = time.perf_counter()
        response = client.completions.create(
            model=model_name,
            prompt=render_completion_prompt(prompt["prompt"]),
            temperature=temperature,
            top_p=top_p,
            max_tokens=prompt_max_tokens,
        )
        elapsed = time.perf_counter() - started
        text = response.choices[0].text or ""
        metrics = completion_metrics(text, response.usage)
        result = {
            "phase": label,
            "id": prompt["id"],
            "source": prompt["source"],
            "category": prompt["category"],
            "prompt": prompt["prompt"],
            "response": text,
            "latency_s": elapsed,
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"[{label}] {idx:02d}/{len(prompts)} {prompt['id']}: "
            f"{metrics['completion_tokens'] or '?'} completion tok, "
            f"{metrics['paragraphs']} paragraphs, {elapsed:.1f}s"
        )
    return results


def write_markdown_report(
    output_path: Path,
    *,
    base_model: str,
    ckpt_dir: Path,
    prompts_path: Path,
    model_name: str,
    base_results: list[dict[str, Any]],
    ckpt_results: list[dict[str, Any]],
) -> None:
    ckpt_by_id = {item["id"]: item for item in ckpt_results}
    lines = [
        "# SFT Smoke Eval",
        "",
        f"- Base model boot: `{base_model}`",
        f"- Checkpoint loaded: `{ckpt_dir}`",
        f"- Served model id: `{model_name}`",
        f"- Prompt suite: `{prompts_path}`",
        "",
        "## Summary",
        "",
        "| Prompt | Source | Base comp tok | Ckpt comp tok | Base latency (s) | Ckpt latency (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for base_item in base_results:
        ckpt_item = ckpt_by_id[base_item["id"]]
        lines.append(
            "| "
            f"{base_item['id']} | {base_item['source']} | "
            f"{base_item['metrics']['completion_tokens'] or '?'} | "
            f"{ckpt_item['metrics']['completion_tokens'] or '?'} | "
            f"{base_item['latency_s']:.2f} | {ckpt_item['latency_s']:.2f} |"
        )

    for base_item in base_results:
        ckpt_item = ckpt_by_id[base_item["id"]]
        lines.extend(
            [
                "",
                f"## {base_item['id']}",
                "",
                f"Source: `{base_item['source']}`",
                "",
                "### Prompt",
                "",
                base_item["prompt"],
                "",
                "### Base",
                "",
                f"Metrics: `{json.dumps(base_item['metrics'], sort_keys=True)}`",
                "",
                base_item["response"],
                "",
                "### Checkpoint",
                "",
                f"Metrics: `{json.dumps(ckpt_item['metrics'], sort_keys=True)}`",
                "",
                ckpt_item["response"],
                "",
            ]
        )

    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    ckpt_dir = args.ckpt.resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    prompts_path = args.prompts.resolve()
    prompts = load_json(prompts_path)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt suite must be a non-empty JSON list: {prompts_path}")

    base_model = args.base_model or infer_base_model(ckpt_dir)
    if base_model is None:
        raise ValueError("Could not infer base model from checkpoint config. Pass --base-model explicitly.")

    enforce_eager = args.enforce_eager and not args.allow_compile

    output_dir = ensure_output_dir(args.output_dir)
    server_log = output_dir / "inference.log"
    results_json = output_dir / "results.json"
    report_md = output_dir / "comparison.md"

    proc = start_inference_server(
        base_model,
        args.port,
        args.max_model_len,
        server_log,
        enforce_eager=enforce_eager,
    )
    if not args.keep_server:
        atexit.register(stop_process, proc)

    try:
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        wait_for_server(args.port, api_key)
        client = OpenAI(api_key=api_key, base_url=f"{base_url(args.port)}/v1")
        model_name = get_model_id(args.port, api_key)
        print(f"Server ready on {base_url(args.port)}/v1 using model id: {model_name}")

        print("\nRunning prompt suite on base model...\n")
        base_results = run_prompt_suite(
            prompts=prompts,
            model_name=model_name,
            client=client,
            label="base",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"\nHot-loading checkpoint weights from {ckpt_dir} ...\n")
        update_weights(args.port, ckpt_dir, api_key)

        print("Running prompt suite on checkpoint...\n")
        ckpt_results = run_prompt_suite(
            prompts=prompts,
            model_name=model_name,
            client=client,
            label="ckpt",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        payload = {
            "base_model": base_model,
            "checkpoint_dir": str(ckpt_dir),
            "served_model_id": model_name,
            "port": args.port,
            "max_model_len": args.max_model_len,
            "enforce_eager": enforce_eager,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompts_path": str(prompts_path),
            "base": base_results,
            "ckpt": ckpt_results,
        }
        with open(results_json, "w") as f:
            json.dump(payload, f, indent=2)

        write_markdown_report(
            report_md,
            base_model=base_model,
            ckpt_dir=ckpt_dir,
            prompts_path=prompts_path,
            model_name=model_name,
            base_results=base_results,
            ckpt_results=ckpt_results,
        )

        print(f"\nSaved JSON results to {results_json}")
        print(f"Saved Markdown report to {report_md}")
        print(f"Inference server log: {server_log}")
        if args.keep_server:
            print(f"Keeping inference server alive on port {args.port}")

    finally:
        if not args.keep_server:
            stop_process(proc)


if __name__ == "__main__":
    main()
