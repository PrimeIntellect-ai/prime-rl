"""Pre-flight smoke test for LoRA-self vs base on one vLLM.

RUN THIS BEFORE the first GPU training run that uses a hot-swapped LoRA
adapter concurrently with base-model requests on the same vLLM server.
Probes three known-problematic vLLM codepaths (see
``skills/preflight-lora-smoke/SKILL.md`` for full context and rationale).

Prereqs
-------
  * vLLM server already running with ``--enable-lora`` on
    ``VLLM_URL`` (defaults to ``http://localhost:8000/v1``).
  * Two adapter checkpoint directories with distinct weights
    (same rank/target_modules). If you only have one, copy it to a
    second path and perturb a single weight to synthesize "different
    weights, same alias" — the probe needs genuine divergence.
  * The base model identity under which vLLM was launched (we address
    it by name in the ``model`` field).

Usage
-----
    uv run scripts/preflight_lora_smoke.py \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter-a /path/to/adapter_a \
        --adapter-b /path/to/adapter_b \
        --vllm-url http://localhost:8000/v1

Exit codes
----------
  0  all three probes PASS
  1  any probe FAILED (see stderr for which, and interpretation)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass

import httpx

# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


async def _load_adapter(client: httpx.AsyncClient, url: str, alias: str, path: str) -> None:
    """POST /load_lora_adapter. Tolerates the adapter already being loaded
    under a different path (reload behavior under prime-rl's monkeypatch)."""
    r = await client.post(
        f"{url}/load_lora_adapter",
        json={"lora_name": alias, "lora_path": path},
        timeout=120.0,
    )
    r.raise_for_status()


async def _complete(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    *,
    seed: int = 0,
    max_tokens: int = 64,
) -> tuple[str, list[int]]:
    """Fire one chat completion. Returns (text, token_ids). Uses greedy
    sampling + fixed seed so repeated calls give stable outputs (modulo
    the #7977 atomic-nondeterminism hazard we're partly probing)."""
    r = await client.post(
        f"{url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": seed,
            "logprobs": True,
            "top_logprobs": 5,
        },
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    text = choice["message"]["content"]
    ids = [tok["token"] for tok in choice.get("logprobs", {}).get("content", [])]
    return text, ids


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    name: str
    passed: bool
    detail: str


_PROMPT = "Explain the Kolmogorov complexity of a constant function in one paragraph."


async def probe_mixed_batch(client: httpx.AsyncClient, url: str, base_model: str, adapter_alias: str) -> ProbeResult:
    """Fire concurrent requests to both base and adapter. Both must return
    200, and their outputs must differ (else adapter isn't influencing)."""
    base_task = asyncio.create_task(_complete(client, url, base_model, _PROMPT))
    ada_task = asyncio.create_task(_complete(client, url, adapter_alias, _PROMPT))
    base_text, base_ids = await base_task
    ada_text, ada_ids = await ada_task

    if base_text == ada_text:
        return ProbeResult(
            "mixed_batch_correctness",
            False,
            "Base and adapter returned byte-identical outputs. Either the "
            "adapter isn't influencing the forward pass, or the server is "
            "collapsing both to the same path. Check adapter actually loaded, "
            "and that the adapter is non-trivial (nonzero weights).",
        )
    return ProbeResult(
        "mixed_batch_correctness",
        True,
        f"Base and adapter outputs differ (base first token={base_ids[:1]}, adapter first token={ada_ids[:1]}).",
    )


async def probe_hot_swap_idempotence(
    client: httpx.AsyncClient,
    url: str,
    alias: str,
    adapter_a_path: str,
    adapter_b_path: str,
) -> ProbeResult:
    """The #18372 probe. Load A -> record output. Load B (same alias!) ->
    output must differ. Load A again -> output must match first read."""
    await _load_adapter(client, url, alias, adapter_a_path)
    t_a, _ = await _complete(client, url, alias, _PROMPT)

    await _load_adapter(client, url, alias, adapter_b_path)
    t_b, _ = await _complete(client, url, alias, _PROMPT)

    await _load_adapter(client, url, alias, adapter_a_path)
    t_a_prime, _ = await _complete(client, url, alias, _PROMPT)

    if t_a == t_b:
        return ProbeResult(
            "hot_swap_idempotence",
            False,
            "Loading adapter B had no visible effect — output still matches "
            "adapter A. This is vLLM issue #18372 (3rd+ swap silently ignored). "
            "DO NOT proceed with a training run on this pin. Fall back to the "
            "two-instance topology.",
        )
    if t_a != t_a_prime:
        return ProbeResult(
            "hot_swap_idempotence",
            False,
            "Loading adapter A again didn't restore its output — server is "
            "stuck on B or somewhere else. Adapter hot-swap is unreliable; "
            "two-instance fallback recommended.",
        )
    return ProbeResult(
        "hot_swap_idempotence",
        True,
        "A -> B -> A round trip produced distinct then restored outputs.",
    )


async def probe_perf_delta(
    client: httpx.AsyncClient,
    url: str,
    base_model: str,
    n_requests: int = 8,
) -> ProbeResult:
    """Microbench base-only throughput. We can't compare to a non-LoRA
    server from here (that needs a second launch), but we can at least
    measure the LoRA-enabled baseline and flag if it's absurdly slow."""
    start = time.perf_counter()
    await asyncio.gather(*[_complete(client, url, base_model, _PROMPT) for _ in range(n_requests)])
    elapsed = time.perf_counter() - start
    per_req = elapsed / n_requests
    # Heuristic: on a 4B model with 64-token completion, >2s per request
    # for base-only on a warm server is a red flag. Tune to your hw.
    if per_req > 2.0:
        return ProbeResult(
            "perf_delta",
            False,
            f"Base-only requests averaging {per_req:.2f}s on LoRA-enabled "
            f"server. This is the #10898 / #10062 tax. Compare against a "
            f"plain `--model` server (no --enable-lora) to quantify; if the "
            f"ratio > 1.6x consider the two-instance topology for "
            f"throughput-critical runs.",
        )
    return ProbeResult(
        "perf_delta",
        True,
        f"Base-only requests averaging {per_req:.2f}s — within heuristic "
        f"budget. Still worth a side-by-side measurement vs a plain server "
        f"to get exact overhead numbers.",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


async def _main(args: argparse.Namespace) -> int:
    url = args.vllm_url.rstrip("/")
    alias = "preflight_probe"

    async with httpx.AsyncClient() as client:
        # Sanity: ensure the server is alive and has --enable-lora
        r = await client.get(f"{url}/models", timeout=10.0)
        r.raise_for_status()
        served = [m["id"] for m in r.json().get("data", [])]
        if args.base_model not in served:
            print(
                f"FAIL: base model {args.base_model!r} not in /v1/models served "
                f"set {served!r}. Did you launch vLLM with --model "
                f"{args.base_model}?",
                file=sys.stderr,
            )
            return 1

        # Prime probe 1 needs the adapter loaded up front
        await _load_adapter(client, url, alias, args.adapter_a)

        probes = [
            await probe_mixed_batch(client, url, args.base_model, alias),
            await probe_hot_swap_idempotence(client, url, alias, args.adapter_a, args.adapter_b),
            await probe_perf_delta(client, url, args.base_model),
        ]

    all_pass = all(p.passed for p in probes)
    for p in probes:
        status = "PASS" if p.passed else "FAIL"
        print(f"[{status}] {p.name}: {p.detail}")

    if not all_pass:
        print(
            "\nOne or more probes failed. Do NOT proceed with a training run "
            "until resolved. See skills/preflight-lora-smoke/SKILL.md for "
            "interpretation and the two-instance escape hatch.",
            file=sys.stderr,
        )
        return 1

    print("\nAll probes passed. LoRA-self-vs-base deployment is safe on this pin.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-model",
        required=True,
        help="Base model id as advertised by vLLM's /v1/models (e.g. 'Qwen/Qwen3-4B-Instruct-2507').",
    )
    ap.add_argument(
        "--adapter-a",
        required=True,
        help="Filesystem path to adapter A (first hot-load).",
    )
    ap.add_argument(
        "--adapter-b",
        required=True,
        help="Filesystem path to adapter B (distinct weights from A, same rank).",
    )
    ap.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
        help="vLLM OpenAI-compatible endpoint. Default: $VLLM_URL or localhost:8000/v1.",
    )
    args = ap.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
