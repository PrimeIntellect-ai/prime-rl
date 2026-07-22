#!/usr/bin/env python3
"""Compare OpenAI-compatible inference backends under RL-style rollout traffic.

The benchmark is intentionally endpoint-level. It can compare the built-in vLLM
server, a vLLM router deployment, or an experimental Dynamo deployment as long
as each exposes OpenAI-compatible chat completions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

COUNTER_KEYS = {
    "vllm:prompt_tokens": "prompt_tokens_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens": "generation_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success": "request_success_total",
    "vllm:request_success_total": "request_success_total",
    "vllm:prefix_cache_queries": "prefix_cache_queries",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    "vllm:prefix_cache_hits": "prefix_cache_hits",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:nixl_num_failed_transfers_total": "nixl_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total": "nixl_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total": "nixl_kv_expired_requests_total",
}


@dataclass(frozen=True)
class BackendConfig:
    label: str
    base_url: str
    metrics_url: str | None
    api_key_var: str


@dataclass
class MetricsRollup:
    values: dict[str, float]

    def summed(self, key: str) -> float:
        return self.values.get(key, 0.0)


@dataclass
class RequestSample:
    backend: str
    request_index: int
    session_id: str
    ok: bool
    status_code: int | None
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    output_chars: int
    error: str | None = None


@dataclass
class BackendSummary:
    label: str
    requests: int
    succeeded: int
    failed: int
    error_rate: float
    wall_time_s: float
    requests_per_second: float
    output_tokens_per_second: float | None
    output_chars_per_second: float
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_p99_s: float | None
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    ttft_p99_s: float | None
    mean_output_tokens: float | None
    mean_output_chars: float
    metrics_delta: dict[str, float]


@dataclass(frozen=True)
class RegressionGate:
    min_request_throughput_ratio: float | None
    min_output_throughput_ratio: float | None
    max_latency_p95_ratio: float | None
    max_error_rate: float | None


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    model: str
    requests: int
    warmup_requests: int
    concurrency: int
    sessions: int
    prompt_words: int
    max_tokens: int
    temperature: float
    stream: bool
    gate: RegressionGate


@dataclass
class ScenarioResult:
    name: str
    model: str
    summaries: list[BackendSummary]
    gate_failures: list[str]
    samples: list[RequestSample]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def clean_base_url(url: str) -> str:
    return url.rstrip("/")


def metrics_url_from_base_url(base_url: str) -> str:
    return clean_base_url(base_url).removesuffix("/v1") + "/metrics"


def parse_backend(value: str, default_api_key_var: str) -> BackendConfig:
    try:
        label, raw_url = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("backend must be LABEL=URL") from exc
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("backend label must not be empty")
    url = clean_base_url(raw_url.strip())
    if not url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("backend URL must start with http:// or https://")
    return BackendConfig(
        label=label,
        base_url=url,
        metrics_url=metrics_url_from_base_url(url),
        api_key_var=default_api_key_var,
    )


def build_messages(request_index: int, session_id: str, prompt_words: int) -> list[dict[str, str]]:
    context = " ".join(
        [
            "rollout",
            "policy",
            "reward",
            "trajectory",
            "tool",
            "state",
            "verifier",
            "advantage",
        ]
        * max(1, prompt_words // 8)
    )
    return [
        {
            "role": "system",
            "content": "You are a concise RL policy used inside a benchmark. Return only the final answer.",
        },
        {
            "role": "user",
            "content": (
                f"Session {session_id}. Request {request_index}. "
                f"Use the shared rollout context, then answer with a short deterministic summary.\n\n{context}"
            ),
        },
    ]


def parse_metrics_text(text: str) -> MetricsRollup:
    values: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = COUNTER_KEYS.get(sample.name)
            if key is not None:
                values[key] = values.get(key, 0.0) + float(sample.value)
    return MetricsRollup(values=values)


async def fetch_metrics(client: httpx.AsyncClient, url: str | None) -> MetricsRollup | None:
    if url is None:
        return None
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
    except Exception:
        return None
    return parse_metrics_text(response.text)


def rollup_delta(before: MetricsRollup | None, after: MetricsRollup | None) -> dict[str, float]:
    if before is None or after is None:
        return {}
    keys = [
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "prefix_cache_queries",
        "prefix_cache_hits",
        "nixl_failed_transfers_total",
        "nixl_failed_notifications_total",
        "nixl_kv_expired_requests_total",
    ]
    delta = {
        key: after.summed(key) - before.summed(key)
        for key in keys
        if after.summed(key) >= before.summed(key)
    }
    queries = delta.get("prefix_cache_queries", 0.0)
    if queries > 0:
        delta["prefix_cache_hit_rate"] = delta.get("prefix_cache_hits", 0.0) / queries
    return delta


def auth_headers(api_key_var: str) -> dict[str, str]:
    api_key = os.getenv(api_key_var)
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


async def run_one_request(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    model: str,
    request_index: int,
    session_id: str,
    prompt_words: int,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> RequestSample:
    payload = {
        "model": model,
        "messages": build_messages(request_index, session_id, prompt_words),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    headers = {"X-Session-ID": session_id, **auth_headers(backend.api_key_var)}
    url = f"{backend.base_url}/chat/completions"
    start = time.perf_counter()
    output_chars = 0
    prompt_tokens = None
    output_tokens = None
    total_tokens = None
    ttft_s = None
    status_code = None
    try:
        if stream:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=None) as response:
                status_code = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - start
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    output_chars += len(delta.get("content") or "")
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens")
                        output_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
        else:
            response = await client.post(url, json=payload, headers=headers, timeout=None)
            status_code = response.status_code
            response.raise_for_status()
            body = response.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content") or ""
            output_chars = len(content)
            usage = body.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        latency_s = time.perf_counter() - start
        return RequestSample(
            backend=backend.label,
            request_index=request_index,
            session_id=session_id,
            ok=True,
            status_code=status_code,
            latency_s=latency_s,
            ttft_s=ttft_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_chars=output_chars,
        )
    except Exception as exc:
        latency_s = time.perf_counter() - start
        return RequestSample(
            backend=backend.label,
            request_index=request_index,
            session_id=session_id,
            ok=False,
            status_code=status_code,
            latency_s=latency_s,
            ttft_s=ttft_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_chars=output_chars,
            error=repr(exc),
        )


async def run_backend(
    backend: BackendConfig,
    model: str,
    requests: int,
    warmup_requests: int,
    concurrency: int,
    sessions: int,
    prompt_words: int,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> tuple[BackendSummary, list[RequestSample]]:
    limits = httpx.Limits(max_connections=max(concurrency * 2, 16), max_keepalive_connections=max(concurrency, 8))
    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)
        session_ids = [f"{backend.label}-{uuid.uuid4().hex[:8]}-{idx}" for idx in range(max(1, sessions))]

        async def guarded(index: int) -> RequestSample:
            async with semaphore:
                return await run_one_request(
                    client=client,
                    backend=backend,
                    model=model,
                    request_index=index,
                    session_id=session_ids[index % len(session_ids)],
                    prompt_words=prompt_words,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )

        if warmup_requests > 0:
            await asyncio.gather(*(guarded(-(index + 1)) for index in range(warmup_requests)))

        before = await fetch_metrics(client, backend.metrics_url)
        wall_start = time.perf_counter()
        samples = await asyncio.gather(*(guarded(index) for index in range(requests)))
        wall_time_s = time.perf_counter() - wall_start
        after = await fetch_metrics(client, backend.metrics_url)
    return summarize_backend(backend.label, samples, wall_time_s, rollup_delta(before, after)), samples


def summarize_backend(
    label: str,
    samples: list[RequestSample],
    wall_time_s: float,
    metrics_delta: dict[str, float] | None = None,
) -> BackendSummary:
    succeeded = [sample for sample in samples if sample.ok]
    latencies = [sample.latency_s for sample in succeeded]
    ttfts = [sample.ttft_s for sample in succeeded if sample.ttft_s is not None]
    output_tokens = [sample.output_tokens for sample in succeeded if sample.output_tokens is not None]
    output_chars = sum(sample.output_chars for sample in succeeded)
    token_sum = sum(output_tokens) if output_tokens else None
    return BackendSummary(
        label=label,
        requests=len(samples),
        succeeded=len(succeeded),
        failed=len(samples) - len(succeeded),
        error_rate=(len(samples) - len(succeeded)) / len(samples) if samples else 0.0,
        wall_time_s=wall_time_s,
        requests_per_second=len(succeeded) / wall_time_s if wall_time_s > 0 else 0.0,
        output_tokens_per_second=(token_sum / wall_time_s if token_sum is not None and wall_time_s > 0 else None),
        output_chars_per_second=output_chars / wall_time_s if wall_time_s > 0 else 0.0,
        latency_p50_s=percentile(latencies, 0.50),
        latency_p95_s=percentile(latencies, 0.95),
        latency_p99_s=percentile(latencies, 0.99),
        ttft_p50_s=percentile(ttfts, 0.50),
        ttft_p95_s=percentile(ttfts, 0.95),
        ttft_p99_s=percentile(ttfts, 0.99),
        mean_output_tokens=statistics.mean(output_tokens) if output_tokens else None,
        mean_output_chars=statistics.mean([sample.output_chars for sample in succeeded]) if succeeded else 0.0,
        metrics_delta=metrics_delta or {},
    )


def format_optional(value: float | None, suffix: str = "", precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def relative_change(value: float | None, baseline: float | None, higher_is_better: bool) -> str:
    if value is None or baseline is None or baseline == 0:
        return ""
    change = (value - baseline) / baseline * 100
    if not higher_is_better:
        change *= -1
    return f" ({change:+.1f}%)"


def throughput_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return value / baseline


def latency_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return value / baseline


def evaluate_regression_gate(summaries: list[BackendSummary], gate: RegressionGate) -> list[str]:
    if len(summaries) < 2:
        return []
    baseline = summaries[0]
    failures: list[str] = []
    for summary in summaries[1:]:
        request_ratio = throughput_ratio(summary.requests_per_second, baseline.requests_per_second)
        output_ratio = throughput_ratio(summary.output_tokens_per_second, baseline.output_tokens_per_second)
        p95_ratio = latency_ratio(summary.latency_p95_s, baseline.latency_p95_s)
        if gate.min_request_throughput_ratio is not None and (
            request_ratio is None or request_ratio < gate.min_request_throughput_ratio
        ):
            failures.append(
                f"{summary.label} request throughput ratio {format_optional(request_ratio, precision=3)} "
                f"is below {gate.min_request_throughput_ratio:.3f}"
            )
        if gate.min_output_throughput_ratio is not None and (
            output_ratio is None or output_ratio < gate.min_output_throughput_ratio
        ):
            failures.append(
                f"{summary.label} output token throughput ratio {format_optional(output_ratio, precision=3)} "
                f"is below {gate.min_output_throughput_ratio:.3f}"
            )
        if gate.max_latency_p95_ratio is not None and (p95_ratio is None or p95_ratio > gate.max_latency_p95_ratio):
            failures.append(
                f"{summary.label} p95 latency ratio {format_optional(p95_ratio, precision=3)} "
                f"is above {gate.max_latency_p95_ratio:.3f}"
            )
        if gate.max_error_rate is not None and summary.error_rate > gate.max_error_rate:
            failures.append(f"{summary.label} error rate {summary.error_rate:.1%} is above {gate.max_error_rate:.1%}")
    return failures


def build_markdown(summaries: Iterable[BackendSummary], model: str, gate_failures: list[str] | None = None) -> str:
    summaries = list(summaries)
    baseline = summaries[0] if summaries else None
    lines = [
        "# Inference Backend Benchmark",
        "",
        f"Model: `{model}`",
        "",
        "| Backend | OK | Error rate | Req/s | Out tok/s | Out chars/s | TTFT p95 | E2E p95 | E2E p99 | Prefix hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        prefix_hit = summary.metrics_delta.get("prefix_cache_hit_rate")
        lines.append(
            "| "
            f"{summary.label} | "
            f"{summary.succeeded}/{summary.requests} | "
            f"{summary.error_rate:.1%} | "
            f"{summary.requests_per_second:.2f}{relative_change(summary.requests_per_second, baseline.requests_per_second if baseline else None, True)} | "
            f"{format_optional(summary.output_tokens_per_second, precision=1)}"
            f"{relative_change(summary.output_tokens_per_second, baseline.output_tokens_per_second if baseline else None, True)} | "
            f"{summary.output_chars_per_second:.1f}"
            f"{relative_change(summary.output_chars_per_second, baseline.output_chars_per_second if baseline else None, True)} | "
            f"{format_optional(summary.ttft_p95_s, 's')} | "
            f"{format_optional(summary.latency_p95_s, 's')} | "
            f"{format_optional(summary.latency_p99_s, 's')} | "
            f"{format_optional(prefix_hit, precision=3)} |"
        )
    if gate_failures:
        lines.extend(["", "## Regression gate", ""])
        lines.extend(f"- {failure}" for failure in gate_failures)
    return "\n".join(lines) + "\n"


def build_suite_markdown(results: Iterable[ScenarioResult]) -> str:
    results = list(results)
    lines = ["# Inference Backend Benchmark Suite", ""]
    for result in results:
        lines.extend(
            [
                f"## {result.name}",
                "",
                build_markdown(result.summaries, model=result.model, gate_failures=result.gate_failures).strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    output_json: Path,
    output_markdown: Path,
    results: list[ScenarioResult],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenarios": [
            {
                "name": result.name,
                "model": result.model,
                "summaries": [asdict(summary) for summary in result.summaries],
                "gate_failures": result.gate_failures,
                "samples": [asdict(sample) for sample in result.samples],
            }
            for result in results
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_markdown.write_text(build_suite_markdown(results), encoding="utf-8")


def non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    return float(value)


def scenario_from_mapping(raw: dict[str, object], defaults: BenchmarkScenario) -> BenchmarkScenario:
    gate_raw = raw.get("gate", {})
    if not isinstance(gate_raw, dict):
        raise ValueError("scenario gate must be an object")
    return BenchmarkScenario(
        name=str(raw.get("name", defaults.name)),
        model=str(raw.get("model", defaults.model)),
        requests=positive_int(raw.get("requests", defaults.requests), "requests"),
        warmup_requests=non_negative_int(raw.get("warmup_requests", defaults.warmup_requests), "warmup_requests"),
        concurrency=positive_int(raw.get("concurrency", defaults.concurrency), "concurrency"),
        sessions=positive_int(raw.get("sessions", defaults.sessions), "sessions"),
        prompt_words=positive_int(raw.get("prompt_words", defaults.prompt_words), "prompt_words"),
        max_tokens=positive_int(raw.get("max_tokens", defaults.max_tokens), "max_tokens"),
        temperature=float(raw.get("temperature", defaults.temperature)),
        stream=bool(raw.get("stream", defaults.stream)),
        gate=RegressionGate(
            min_request_throughput_ratio=optional_float(
                gate_raw.get("min_request_throughput_ratio", defaults.gate.min_request_throughput_ratio),
                "gate.min_request_throughput_ratio",
            ),
            min_output_throughput_ratio=optional_float(
                gate_raw.get("min_output_throughput_ratio", defaults.gate.min_output_throughput_ratio),
                "gate.min_output_throughput_ratio",
            ),
            max_latency_p95_ratio=optional_float(
                gate_raw.get("max_latency_p95_ratio", defaults.gate.max_latency_p95_ratio),
                "gate.max_latency_p95_ratio",
            ),
            max_error_rate=optional_float(
                gate_raw.get("max_error_rate", defaults.gate.max_error_rate),
                "gate.max_error_rate",
            ),
        ),
    )


def scenario_from_args(args: argparse.Namespace) -> BenchmarkScenario:
    return BenchmarkScenario(
        name="default",
        model=args.model,
        requests=args.requests,
        warmup_requests=args.warmup_requests,
        concurrency=args.concurrency,
        sessions=args.sessions,
        prompt_words=args.prompt_words,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=not args.no_stream,
        gate=RegressionGate(
            min_request_throughput_ratio=args.min_request_throughput_ratio,
            min_output_throughput_ratio=args.min_output_throughput_ratio,
            max_latency_p95_ratio=args.max_latency_p95_ratio,
            max_error_rate=args.max_error_rate,
        ),
    )


def load_scenarios(path: Path | None, defaults: BenchmarkScenario) -> list[BenchmarkScenario]:
    if path is None:
        return [defaults]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario file must contain a JSON object")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenario file must contain a non-empty scenarios list")
    suite_defaults = defaults
    if "model" in payload:
        suite_defaults = BenchmarkScenario(
            name=defaults.name,
            model=str(payload["model"]),
            requests=defaults.requests,
            warmup_requests=defaults.warmup_requests,
            concurrency=defaults.concurrency,
            sessions=defaults.sessions,
            prompt_words=defaults.prompt_words,
            max_tokens=defaults.max_tokens,
            temperature=defaults.temperature,
            stream=defaults.stream,
            gate=defaults.gate,
        )
    scenarios = []
    names = set()
    for raw in raw_scenarios:
        if not isinstance(raw, dict):
            raise ValueError("each scenario must be a JSON object")
        scenario = scenario_from_mapping(raw, suite_defaults)
        if scenario.name in names:
            raise ValueError(f"duplicate scenario name: {scenario.name}")
        names.add(scenario.name)
        scenarios.append(scenario)
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", required=True, help="Backend in LABEL=URL form. URL should include /v1.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming and TTFT measurement.")
    parser.add_argument("--api-key-var", default="OPENAI_API_KEY")
    parser.add_argument("--scenario-json", type=Path, help="JSON file with a scenarios list for multi-profile runs.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/inference_backend_benchmark.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("outputs/inference_backend_benchmark.md"))
    parser.add_argument("--min-request-throughput-ratio", type=float)
    parser.add_argument("--min-output-throughput-ratio", type=float)
    parser.add_argument("--max-latency-p95-ratio", type=float)
    parser.add_argument("--max-error-rate", type=float)
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    backends = [parse_backend(raw, args.api_key_var) for raw in args.backend]
    scenarios = load_scenarios(args.scenario_json, scenario_from_args(args))
    results = []
    for scenario in scenarios:
        summaries: list[BackendSummary] = []
        all_samples: list[RequestSample] = []
        for backend in backends:
            summary, samples = await run_backend(
                backend=backend,
                model=scenario.model,
                requests=scenario.requests,
                warmup_requests=scenario.warmup_requests,
                concurrency=scenario.concurrency,
                sessions=scenario.sessions,
                prompt_words=scenario.prompt_words,
                max_tokens=scenario.max_tokens,
                temperature=scenario.temperature,
                stream=scenario.stream,
            )
            summaries.append(summary)
            all_samples.extend(samples)
        results.append(
            ScenarioResult(
                name=scenario.name,
                model=scenario.model,
                summaries=summaries,
                gate_failures=evaluate_regression_gate(summaries, scenario.gate),
                samples=all_samples,
            )
        )
    gate_failures = [failure for result in results for failure in result.gate_failures]
    write_outputs(args.output_json, args.output_markdown, results)
    print(build_suite_markdown(results))
    print(f"Wrote JSON to {args.output_json}")
    print(f"Wrote Markdown to {args.output_markdown}")
    if gate_failures:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
