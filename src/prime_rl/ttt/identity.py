"""Identity and filesystem invariants shared by the TTT service engines."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_ROLLOUT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def validate_rollout_id(rollout_id: str) -> str:
    """Return a safe single-path-component rollout ID or raise ``ValueError``."""
    if not _ROLLOUT_ID.fullmatch(rollout_id):
        raise ValueError("rollout_id must be 1-128 ASCII letters, digits, '_' or '-', starting with a letter or digit")
    return rollout_id


def validate_adapter_name(adapter_name: str) -> str:
    """Reject empty, oversized, or control-character adapter names."""
    if not adapter_name or len(adapter_name) > 256:
        raise ValueError("adapter_name must contain 1-256 characters")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in adapter_name):
        raise ValueError("adapter_name must not contain control characters")
    return adapter_name


def expected_adapter_name(rollout_id: str, adapter_prefix: str) -> str:
    """Derive the only adapter identity a configured service accepts for a rollout."""
    validate_rollout_id(rollout_id)
    # ``TTTServiceConfig`` validates the prefix and caps it at 127 characters. Keep this
    # runtime boundary defensive because tests and third-party callers can construct
    # config-like objects without going through Pydantic.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,126}", adapter_prefix):
        raise ValueError(
            "adapter_prefix must be 1-127 ASCII letters, digits, '_' or '-', starting with a letter or digit"
        )
    return f"{adapter_prefix}-{rollout_id}"


def validate_adapter_identity(rollout_id: str, adapter_name: str, adapter_prefix: str) -> str:
    """Reject caller-selected names outside the service's rollout-bound namespace."""
    validate_adapter_name(adapter_name)
    expected = expected_adapter_name(rollout_id, adapter_prefix)
    if adapter_name != expected:
        raise ValueError(f"rollout {rollout_id!r} must use adapter {expected!r}; got {adapter_name!r}")
    return adapter_name


def checkpoint_rollout_dir(root: Path, rollout_id: str) -> Path:
    """Resolve a rollout checkpoint directory and prove it remains below ``root``."""
    validate_rollout_id(rollout_id)
    resolved_root = root.resolve()
    candidate = (resolved_root / rollout_id).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"rollout_id {rollout_id!r} escapes the checkpoint root") from exc
    return candidate


def update_fingerprint(
    *,
    rollout_id: str,
    adapter_name: str,
    token_ids: list[int],
    loss_mask: list[bool],
    seq_no: int,
    qa_pairs: list[dict] | None,
    train_rollout: bool,
    system_prompt: str | None,
    tools: list[dict] | None,
) -> str:
    """Fingerprint every semantic field used by an update for exact retry checks."""
    payload = {
        "rollout_id": rollout_id,
        "adapter_name": adapter_name,
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "seq_no": seq_no,
        "qa_pairs": qa_pairs,
        "train_rollout": train_rollout,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"update payload is not JSON-serializable: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReleaseResult:
    adapter_name: str
    released: bool
