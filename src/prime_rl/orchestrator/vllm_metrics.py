import asyncio
import re
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from httpx import AsyncClient, Timeout

from prime_rl.orchestrator.config import VllmMetricsConfig
from prime_rl.utils.logger import get_logger

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+(?P<ts>\d+))?\s*$"
)


def parse_prometheus_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus exposition format (text) into a dict of metric -> samples.

    Notes:
    - Ignores comments and HELP/TYPE lines.
    - Preserves labels as dict[str, str].
    - Drops timestamps (if present).
    """
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        value = float(m.group("value"))
        labels_raw = m.group("labels")
        labels: dict[str, str] = {}
        if labels_raw:
            # Very small parser: split on commas not inside quotes (prometheus uses quoted label values).
            parts: list[str] = []
            buf = []
            in_quotes = False
            esc = False
            for ch in labels_raw:
                if esc:
                    buf.append(ch)
                    esc = False
                    continue
                if ch == "\\":
                    buf.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    buf.append(ch)
                    in_quotes = not in_quotes
                    continue
                if ch == "," and not in_quotes:
                    parts.append("".join(buf).strip())
                    buf = []
                    continue
                buf.append(ch)
            if buf:
                parts.append("".join(buf).strip())

            for part in parts:
                if not part:
                    continue
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                    # Keep it simple: unescape common sequences
                    v = v[1:-1]
                    v = v.replace(r"\\", "\\").replace(r"\n", "\n").replace(r"\"", '"')
                labels[k] = v

        out.setdefault(name, []).append((labels, value))
    return out


def _sanitize_wandb_key(s: str) -> str:
    # W&B supports "/" as a namespace; keep it, but avoid punctuation that creates messy keys.
    return re.sub(r"[^a-zA-Z0-9/_\-\.]+", "_", s)


def _endpoint_id_from_base_url(base_url: str, index: int) -> str:
    # Prefer host:port-ish stable id; fall back to index.
    try:
        p = urlparse(base_url)
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        scheme = p.scheme or "http"
        if host:
            return _sanitize_wandb_key(f"{scheme}__{host}{port}")
    except Exception:
        pass
    return f"endpoint_{index}"


def _metric_sum(metrics: dict[str, list[tuple[dict[str, str], float]]], name: str) -> float | None:
    samples = metrics.get(name)
    if not samples:
        return None
    return float(sum(v for _, v in samples))


class VllmMetricsCollector:
    """Poll vLLM `/metrics` from each inference endpoint and return metrics for logging."""

    def __init__(self, admin_clients: list[AsyncClient], config: VllmMetricsConfig):
        self.admin_clients = admin_clients
        self.config = config
        self.logger = get_logger()

        self._last_poll_t = 0.0
        self._last_token_counters: dict[str, tuple[float, float, float]] = {}
        # endpoint_id -> (t, prompt_total, gen_total)
        self._last_prefix_cache_counters: dict[str, tuple[float, float, float]] = {}
        # endpoint_id -> (t, hits_total, misses_total)

        self._collector_id = uuid4().hex[:8]

    async def maybe_collect(self, *, step: int) -> dict[str, Any]:
        now = time.monotonic()
        if self._last_poll_t and (now - self._last_poll_t) < self.config.interval_seconds:
            return {}

        self._last_poll_t = now

        async def _fetch_one(i: int, client: AsyncClient) -> tuple[str, str | None]:
            endpoint_id = _endpoint_id_from_base_url(str(client.base_url), i)
            try:
                resp = await client.get("/metrics", timeout=Timeout(self.config.timeout_seconds))
                resp.raise_for_status()
                return endpoint_id, resp.text
            except Exception as e:
                self.logger.debug(f"Failed to fetch /metrics from {client.base_url}: {e}")
                return endpoint_id, None

        results = await asyncio_gather_safe([_fetch_one(i, c) for i, c in enumerate(self.admin_clients)])

        out: dict[str, Any] = {}
        for endpoint_id, text in results:
            prefix = f"vllm/{endpoint_id}"
            if text is None:
                out[f"{prefix}/up"] = 0
                continue
            out[f"{prefix}/up"] = 1

            metrics = parse_prometheus_text(text)

            # --- handful of special metrics (derived from counters) ---
            prompt_total = (
                _metric_sum(metrics, "vllm:prompt_tokens_total")
                or _metric_sum(metrics, "vllm_prompt_tokens_total")
                or _metric_sum(metrics, "prompt_tokens_total")
            )
            gen_total = (
                _metric_sum(metrics, "vllm:generated_tokens_total")
                or _metric_sum(metrics, "vllm_generated_tokens_total")
                or _metric_sum(metrics, "generated_tokens_total")
            )

            if prompt_total is not None and gen_total is not None:
                prev = self._last_token_counters.get(endpoint_id)
                if prev is not None:
                    prev_t, prev_prompt, prev_gen = prev
                    dt = max(1e-6, now - prev_t)
                    out[f"{prefix}/prompt_tokens_per_sec"] = (prompt_total - prev_prompt) / dt
                    out[f"{prefix}/generated_tokens_per_sec"] = (gen_total - prev_gen) / dt
                    out[f"{prefix}/tokens_per_sec"] = (prompt_total - prev_prompt + gen_total - prev_gen) / dt
                self._last_token_counters[endpoint_id] = (now, prompt_total, gen_total)

            prefix_hits_total = (
                _metric_sum(metrics, "vllm:prefix_cache_hits_total")
                or _metric_sum(metrics, "vllm_prefix_cache_hits_total")
                or _metric_sum(metrics, "prefix_cache_hits_total")
            )
            prefix_misses_total = (
                _metric_sum(metrics, "vllm:prefix_cache_misses_total")
                or _metric_sum(metrics, "vllm_prefix_cache_misses_total")
                or _metric_sum(metrics, "prefix_cache_misses_total")
            )
            if prefix_hits_total is not None:
                out[f"{prefix}/prefix_cache_hits_total"] = prefix_hits_total
            if prefix_misses_total is not None:
                out[f"{prefix}/prefix_cache_misses_total"] = prefix_misses_total
            if prefix_hits_total is not None and prefix_misses_total is not None:
                prev = self._last_prefix_cache_counters.get(endpoint_id)
                if prev is not None:
                    prev_t, prev_hits, prev_misses = prev
                    dt = max(1e-6, now - prev_t)
                    hits_d = prefix_hits_total - prev_hits
                    misses_d = prefix_misses_total - prev_misses
                    out[f"{prefix}/prefix_cache_hits_per_sec"] = hits_d / dt
                    out[f"{prefix}/prefix_cache_misses_per_sec"] = misses_d / dt
                    denom = hits_d + misses_d
                    if denom > 0:
                        out[f"{prefix}/prefix_cache_hit_ratio"] = hits_d / denom
                self._last_prefix_cache_counters[endpoint_id] = (now, prefix_hits_total, prefix_misses_total)

            # Stable key for kv-cache utilization (keep best-effort name mapping).
            kv_candidates = [
                "vllm:gpu_kv_cache_usage_percent",
                "vllm:kv_cache_usage_percent",
                "vllm_gpu_kv_cache_usage_percent",
                "vllm_kv_cache_usage_percent",
            ]
            for name in kv_candidates:
                kv_val = _metric_sum(metrics, name)
                if kv_val is not None:
                    out[f"{prefix}/kv_cache_utilization_percent"] = kv_val
                    break

            # --- bulk log Prometheus metrics (no special heuristics) ---
            if self.config.log_all_vllm_metrics:
                emitted = 0
                for name, samples in metrics.items():
                    # Keep it simple: log the metric "as a scalar" by summing across label series.
                    out[f"{prefix}/prom/{_sanitize_wandb_key(name)}"] = float(sum(v for _, v in samples))
                    out[f"{prefix}/prom/{_sanitize_wandb_key(name)}__series"] = len(samples)
                    emitted += 1
                    if emitted >= self.config.max_metrics_per_endpoint:
                        out[f"{prefix}/prom/_truncated"] = 1
                        out[f"{prefix}/prom/_max_metrics_per_endpoint"] = self.config.max_metrics_per_endpoint
                        out[f"{prefix}/prom/_collector_id"] = self._collector_id
                        break

            # Always include collector metadata for debugging (cheap).
            out[f"{prefix}/collector_id"] = self._collector_id

        # Tie to orchestrator step for W&B x-axis consistency.
        out["step"] = step
        return out


async def asyncio_gather_safe(awaitables: list[Any]) -> list[Any]:
    """Gather awaitables (no return_exceptions)."""
    return await asyncio.gather(*awaitables, return_exceptions=False)

