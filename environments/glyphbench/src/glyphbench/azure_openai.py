"""Shared Azure OpenAI endpoint and authentication helpers.

Azure exposes both the legacy resource API (``*.openai.azure.com`` with an
``api-key`` header) and the newer AI Foundry v1 API
(``*.services.ai.azure.com/openai/v1`` with OpenAI-compatible bearer auth).
The eval proxy and the standalone pro harness use these helpers so one set of
credentials behaves consistently in both paths.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def redact_endpoint_url(endpoint: str | None) -> str | None:
    """Remove URL credentials before logging or saving endpoint metadata."""
    if not endpoint:
        return endpoint
    parts = urlsplit(endpoint)
    query = urlencode([
        (key, value if key.lower() == "api-version" else "REDACTED")
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ])
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, query, ""))


def normalize_responses_endpoint(endpoint: str) -> str:
    """Accept either a Responses URL or an ``/openai/v1`` base URL."""

    value = endpoint.strip()
    if not value:
        return value
    parts = urlsplit(value)
    path = parts.path.rstrip("/")
    if not path.endswith("/responses"):
        path = f"{path}/responses"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def azure_auth_headers(
    endpoint: str,
    api_key: str,
    *,
    auth_mode: str = "auto",
) -> dict[str, str]:
    """Build headers for legacy Azure OpenAI or the Foundry v1 endpoint.

    ``auth_mode`` may be ``auto``, ``bearer``, or ``api-key``. Auto selects
    bearer authentication for the OpenAI-compatible Foundry v1 URL and the
    legacy header everywhere else.
    """

    mode = auth_mode.strip().lower().replace("_", "-")
    if mode not in {"auto", "bearer", "api-key"}:
        raise ValueError(
            "Azure auth mode must be one of auto/bearer/api-key; "
            f"got {auth_mode!r}"
        )
    if mode == "auto":
        parts = urlsplit(endpoint)
        foundry_v1 = (
            parts.hostname is not None
            and parts.hostname.lower().endswith(".services.ai.azure.com")
            and "/openai/v1" in parts.path.rstrip("/").lower()
        )
        mode = "bearer" if foundry_v1 else "api-key"
    if mode == "bearer":
        return {"Authorization": f"Bearer {api_key}"}
    return {"api-key": api_key}


__all__ = ["azure_auth_headers", "normalize_responses_endpoint", "redact_endpoint_url"]
