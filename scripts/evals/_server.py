"""Shared vLLM inference-server harness for bespoke evals.

Pattern: each bespoke eval (smoke, sycophancy, mtbench) runs as a **paired
base/SFT comparison**:

  1. Boot vLLM on the base model weights.
  2. Run the eval's `run_phase` against base (phase="base").
  3. Hot-reload ckpt weights via `/update_weights`.
  4. Run `run_phase` again against ckpt (phase="ckpt").
  5. Orchestrator diffs the two phases.

Protocol B (empirically verified 2026-04-19 on marin-8b): client-side
rendering with the SFT tokenizer, sent via `/v1/completions` as a raw string.
Both base and ckpt see **byte-identical** prompts — the only variable is the
weights. Chat API is not used because base tokenizers often lack chat_template;
the SFT tokenizer's template is what the ckpt was trained on, and vocabs are
compatible across base/SFT tokenizers so token IDs match exactly.

The olmes-driven evals don't use this — olmes spins its own inference backend.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

Phase = Literal["base", "ckpt"]

import asyncio

import httpx
from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def resolve_path_args(args: argparse.Namespace, *fields: str) -> argparse.Namespace:
    """Resolve selected `Path`-typed argparse fields in-place to absolute paths.

    Absolute paths survive subprocess boundaries (vLLM server boot, HTTP
    `/update_weights` payloads) and log cleanly. Mutates and returns `args`.
    """
    for f in fields:
        val = getattr(args, f, None)
        if isinstance(val, Path):
            setattr(args, f, val.resolve())
    return args


@dataclass(frozen=True)
class ServerHandle:
    proc: subprocess.Popen[str]
    port: int
    api_key: str
    base_url: str
    model_id: str
    client: OpenAI
    log_path: Path


@dataclass(frozen=True)
class PhaseHandle:
    """Handle a single phase of a paired base/ckpt run."""

    phase: Phase
    client: OpenAI
    async_client: AsyncOpenAI
    model_id: str
    sft_tokenizer: PreTrainedTokenizerBase
    base_url: str
    api_key: str
    stop_token_ids: tuple[int, ...]


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _read_sft_sibling_config(ckpt_dir: Path) -> dict[str, Any]:
    """Locate the sibling `configs/sft.toml` next to a ckpt weights dir."""
    try:
        run_root = ckpt_dir.parents[1]
    except IndexError:
        raise ValueError(f"{ckpt_dir} has no grandparent; can't find sibling config")
    sft_config_path = run_root / "configs" / "sft.toml"
    if not sft_config_path.exists():
        raise ValueError(f"No sibling configs/sft.toml next to {ckpt_dir}")
    with open(sft_config_path, "rb") as f:
        return tomllib.load(f)


def infer_base_model(ckpt_dir: Path) -> str:
    """Read the base-model name from a checkpoint's saved config.

    Prime-rl weight exports do not always preserve `_name_or_path`, so fall back
    to the sibling SFT config.
    """
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        for key in ("_name_or_path", "name_or_path"):
            value = cfg.get(key)
            if isinstance(value, str) and value and value != str(ckpt_dir):
                return value
    cfg = _read_sft_sibling_config(ckpt_dir)
    model_name = cfg.get("model", {}).get("name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Sibling SFT config has no model.name for {ckpt_dir}")
    return model_name


def load_sft_tokenizer(ckpt_dir: Path) -> PreTrainedTokenizerBase:
    """Load the tokenizer that was used at SFT training time.

    Priority:
      1. The ckpt dir itself (prime-rl saves tokenizer alongside weights).
      2. The sibling `configs/sft.toml`'s `tokenizer.name` field.

    The tokenizer's `chat_template` is the exact protocol the SFT ckpt learned
    to respond to. Using it to render prompts client-side and sending raw
    strings to the server is the canonical "Protocol B" pattern.
    """
    ckpt_tokenizer_file = ckpt_dir / "tokenizer_config.json"
    if ckpt_tokenizer_file.exists():
        return AutoTokenizer.from_pretrained(str(ckpt_dir))
    cfg = _read_sft_sibling_config(ckpt_dir)
    tokenizer_name = cfg.get("tokenizer", {}).get("name")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError(f"Cannot locate SFT tokenizer for {ckpt_dir}")
    return AutoTokenizer.from_pretrained(tokenizer_name)


def render_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Apply the SFT chat template to a list of messages, return rendered string."""
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no chat_template; cannot render messages.")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _wait_for_server(port: int, api_key: str, timeout_s: float = 300.0) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    with httpx.Client(timeout=10.0, headers=_auth_headers(api_key)) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{_base_url(port)}/v1/models")
                response.raise_for_status()
                payload = response.json()
                if payload.get("data"):
                    return payload
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Inference server on port {port} did not become ready. Last error: {last_error}")


def start_server(
    base_model: str,
    *,
    port: int = 8000,
    max_model_len: int = 4096,
    log_path: Path,
    enforce_eager: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    """Boot a prime-rl inference server in the background."""
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
        cmd.append("--model.enforce-eager")
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env.setdefault("VLLM_API_KEY", "EMPTY")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    proc._log_file_handle = log_file  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    handle = getattr(proc, "_log_file_handle", None)
    if handle is not None:
        handle.close()


def update_weights(port: int, ckpt_dir: Path, api_key: str) -> None:
    """Hot-reload checkpoint weights into the running server."""
    with httpx.Client(timeout=httpx.Timeout(120.0, read=120.0), headers=_auth_headers(api_key)) as client:
        response = client.post(f"{_base_url(port)}/update_weights", json={"weight_dir": str(ckpt_dir)})
        response.raise_for_status()


@contextmanager
def running_server(
    base_model: str,
    *,
    port: int = 8000,
    max_model_len: int = 4096,
    log_path: Path,
    enforce_eager: bool = True,
    ckpt_dir: Path | None = None,
    keep_alive: bool = False,
) -> Iterator[ServerHandle]:
    """Context manager that yields a ready server handle.

    If `ckpt_dir` is given, the server boots on `base_model` and then hot-loads
    checkpoint weights before yielding — so the handle always points at the
    evaluated checkpoint, never the base model.
    """
    proc = start_server(
        base_model,
        port=port,
        max_model_len=max_model_len,
        log_path=log_path,
        enforce_eager=enforce_eager,
    )
    if not keep_alive:
        atexit.register(stop_server, proc)
    try:
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        _wait_for_server(port, api_key)
        client = OpenAI(api_key=api_key, base_url=f"{_base_url(port)}/v1")
        models_payload = _wait_for_server(port, api_key, timeout_s=30.0)
        model_id = models_payload["data"][0]["id"]
        if ckpt_dir is not None:
            update_weights(port, ckpt_dir, api_key)
        yield ServerHandle(
            proc=proc,
            port=port,
            api_key=api_key,
            base_url=_base_url(port),
            model_id=model_id,
            client=client,
            log_path=log_path,
        )
    finally:
        if not keep_alive:
            stop_server(proc)


def _snapshot_download(repo_id: str) -> Path:
    """Fetch an HF repo into the local cache, return the local path."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"],
    )
    return Path(local)


@contextmanager
def paired_run(
    *,
    base_model: str,
    ckpt_dir: Path,
    sft_tokenizer: PreTrainedTokenizerBase | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
    log_path: Path,
    enforce_eager: bool = True,
) -> Iterator[tuple[Callable[[], PhaseHandle], Callable[[], PhaseHandle]]]:
    """Context manager for base/ckpt paired evaluation.

    Boot strategy: vLLM boots on the **base model** as `--model.name`. Base
    phase runs immediately (no weight swap). When `enter_ckpt()` is called,
    we `/update_weights` once to the ckpt dir, then run the ckpt phase.
    Single server boot, single weight swap.

    EOS correctness: each `complete()` / `complete_batch()` call passes
    `stop_token_ids` = the full Llama-3 EOS set (`<|eot_id|>`,
    `<|end_of_text|>`, `<|eom_id|>`) extracted from the SFT tokenizer. The
    server's own tokenizer doesn't need to match, so we can boot on base
    without triggering the EOS-mismatch bug that plagued the earlier design.
    Protocol B (client-side SFT-template rendering) ensures both phases see
    byte-identical token sequences; base and SFT tokenizers share vocab.

    Yields `(enter_base, enter_ckpt)` — two nullary functions.

    Usage:
        with paired_run(base_model=..., ckpt_dir=...) as (enter_base, enter_ckpt):
            base_handle = enter_base()
            base_results = run_phase(base_handle, ...)
            ckpt_handle = enter_ckpt()
            ckpt_results = run_phase(ckpt_handle, ...)
    """
    if sft_tokenizer is None:
        sft_tokenizer = load_sft_tokenizer(ckpt_dir)

    with running_server(
        base_model=base_model,  # boot directly on base; no swap needed for base phase
        port=port,
        max_model_len=max_model_len,
        log_path=log_path,
        enforce_eager=enforce_eager,
        ckpt_dir=None,
    ) as server:
        entered = {"base": False, "ckpt": False}

        stop_ids = tuple(_stop_token_ids_for(sft_tokenizer))
        async_client = AsyncOpenAI(api_key=server.api_key, base_url=f"{server.base_url}/v1")

        def _handle(phase: Phase) -> PhaseHandle:
            return PhaseHandle(
                phase=phase,
                client=server.client,
                async_client=async_client,
                model_id=server.model_id,
                sft_tokenizer=sft_tokenizer,
                base_url=server.base_url,
                api_key=server.api_key,
                stop_token_ids=stop_ids,
            )

        def enter_base() -> PhaseHandle:
            if entered["base"]:
                raise RuntimeError("enter_base called twice")
            entered["base"] = True
            return _handle("base")

        def enter_ckpt() -> PhaseHandle:
            if entered["ckpt"]:
                raise RuntimeError("enter_ckpt called twice")
            entered["ckpt"] = True
            print(f"[_server] swapping to CKPT weights: {ckpt_dir}", flush=True)
            update_weights(server.port, ckpt_dir, server.api_key)
            return _handle("ckpt")

        yield enter_base, enter_ckpt


def _stop_token_ids_for(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Llama-3-lineage chat models have **multiple** valid EOS tokens:
      - <|end_of_text|> (128001) — base-model EOS
      - <|eot_id|> (128009) — chat turn terminator (what the SFT model emits)
      - <|eom_id|> (128008) — end-of-message in Llama-3 tool-call format

    vLLM's default stop uses the tokenizer's `eos_token_id` which is a single
    value. Different ckpts disagree on which single value it should be — the
    SFT tokenizer says 128009, the base marin tokenizer says 128001. We always
    pass both (plus 128008 for completeness) so generations stop correctly
    regardless of which one the model is taught to emit.

    Resolves IDs by name from the tokenizer's vocabulary; any missing token is
    silently skipped (so non-Llama-3 bases just fall back to the tokenizer's
    own eos_token_id).
    """
    candidates = ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]
    ids: list[int] = []
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.append(tid)
    # Always include the tokenizer's declared eos_token_id as a safety net.
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def complete(
    handle: PhaseHandle,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    echo: bool = False,
) -> tuple[str, Any]:
    """Synchronous single-prompt Protocol-B completion.

    Only use for small-N debugging. For real evals use `complete_batch`.
    """
    rendered = render_messages(handle.sft_tokenizer, messages, add_generation_prompt=True)
    effective_stop_ids = list(stop_token_ids) if stop_token_ids is not None else list(handle.stop_token_ids)
    response = handle.client.completions.create(
        model=handle.model_id,
        prompt=rendered,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        echo=echo,
        extra_body={"stop_token_ids": effective_stop_ids},
    )
    return response.choices[0].text or "", response.usage


async def _complete_one_async(
    handle: PhaseHandle,
    rendered: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    stop_token_ids: list[int],
    seed: int | None,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Any]:
    async with semaphore:
        resp = await handle.async_client.completions.create(
            model=handle.model_id,
            prompt=rendered,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
            extra_body={"stop_token_ids": stop_token_ids},
        )
        return resp.choices[0].text or "", resp.usage


def complete_batch(
    handle: PhaseHandle,
    batch_messages: list[list[dict[str, str]]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    seeds: list[int] | None = None,
    max_concurrency: int = 64,
) -> list[tuple[str, Any]]:
    """Protocol-B completion for a batch of prompts, **executed concurrently**.

    Each entry in `batch_messages` is a separate conversation (list of messages).
    We render all with the SFT template, then fire all requests concurrently via
    `AsyncOpenAI` gated by a semaphore (default 64 in-flight). vLLM's continuous
    batcher schedules them for actual parallel execution on the GPU.

    For multi-turn evals, call this once per turn, threading the previous
    assistant turn into each messages list before the next call.

    Returns list of (text, usage) in the same order as `batch_messages`.
    """
    n = len(batch_messages)
    if n == 0:
        return []
    rendered = [render_messages(handle.sft_tokenizer, ms, add_generation_prompt=True) for ms in batch_messages]
    effective_stop_ids = list(stop_token_ids) if stop_token_ids is not None else list(handle.stop_token_ids)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_all():
        tasks = [
            _complete_one_async(
                handle,
                r,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stop_token_ids=effective_stop_ids,
                seed=(seeds[i] if seeds is not None else None),
                semaphore=semaphore,
            )
            for i, r in enumerate(rendered)
        ]
        # Fail-fast: any exception aborts the batch and propagates to caller.
        # Per-request retry (if desired) belongs in the async func, not here.
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "complete_batch called from inside a running event loop — call from sync context or refactor caller."
            )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_run_all())


def load_suite(suite_path: Path | None = None) -> dict:
    """Load the locked eval suite config."""
    if suite_path is None:
        suite_path = Path(__file__).resolve().parents[2] / "configs" / "evals" / "rung6_suite.toml"
    with open(suite_path, "rb") as f:
        return tomllib.load(f)


@dataclass
class AccResult:
    """Shared per-phase result for single-accuracy evals (gsm8k, mmlu).

    Bespoke evals (ifeval: 4-metric; smoke/sycophancy/mtbench: rich per-phase
    structures) keep their own Result types — forcing them under this shape
    would require `Optional[Any]` sprawl, so we don't.
    """

    n: int = 0
    correct: int = 0
    per_row: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "acc": self.correct / self.n if self.n else 0.0,
        }


def paired_eval(
    *,
    eval_name: str,
    rows: list[dict],
    ckpt: Path,
    output_dir: Path,
    run_phase_fn: Callable[..., dict[str, Any]],
    headline_metrics: list[str],
    base_model: str | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
    log_path: Path | None = None,
    phase_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared paired base/ckpt scaffold for bespoke-server evals.

    Handles: output dir, base-model inference, SFT tokenizer load, server boot,
    two phases (base + ckpt), elapsed timing, headline deltas, json write.

    `run_phase_fn(handle, rows, **phase_kwargs) -> dict` owns all per-eval
    generation + scoring. Each returned phase dict must contain every key in
    `headline_metrics`; deltas are computed as `ckpt - base`.

    Output json path: `<output_dir>/<eval_name>.json`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_model if base_model is not None else infer_base_model(ckpt)
    sft_tokenizer = load_sft_tokenizer(ckpt)
    log_path = log_path if log_path is not None else (output_dir / "inference.log")
    phase_kwargs = phase_kwargs if phase_kwargs is not None else {}

    summary: dict[str, Any] = {
        "base_model": base,
        "ckpt": str(ckpt),
        "sft_tokenizer": getattr(sft_tokenizer, "name_or_path", "unknown"),
        "n": len(rows),
        "phases": {},
    }

    with paired_run(
        base_model=base,
        ckpt_dir=ckpt,
        sft_tokenizer=sft_tokenizer,
        port=port,
        max_model_len=max_model_len,
        log_path=log_path,
    ) as (enter_base, enter_ckpt):
        t0 = time.time()
        base_handle = enter_base()
        print(f"\n[{eval_name}] BASE pass (model_id={base_handle.model_id})\n", flush=True)
        summary["phases"]["base"] = run_phase_fn(base_handle, rows=rows, **phase_kwargs)
        ckpt_handle = enter_ckpt()
        print(f"\n[{eval_name}] CKPT pass\n", flush=True)
        summary["phases"]["ckpt"] = run_phase_fn(ckpt_handle, rows=rows, **phase_kwargs)
        summary["elapsed_s"] = time.time() - t0

    summary["headline"] = {}
    for m in headline_metrics:
        b = summary["phases"]["base"][m]
        c = summary["phases"]["ckpt"][m]
        summary["headline"][m] = {"base": b, "ckpt": c, "delta": c - b}

    json_path = output_dir / f"{eval_name}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{eval_name}] wrote {json_path}", flush=True)
    print(json.dumps(summary["headline"], indent=2))
    return summary
