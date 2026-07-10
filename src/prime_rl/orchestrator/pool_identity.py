"""Serving-resource identity used by the mutable-policy barrier."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.utils.client import InferencePool


def _normalized_url(value: object) -> str:
    return str(value).rstrip("/").removesuffix("/v1")


def serving_identity(pool: InferencePool) -> tuple[str, frozenset[str], frozenset[str]]:
    """Return model, request endpoints, and admin endpoints for one pool."""
    request_endpoints = frozenset(_normalized_url(client.base_url) for client in getattr(pool, "train_clients", ()))
    admin_endpoints = frozenset(_normalized_url(client.base_url) for client in getattr(pool, "admin_clients", ()))
    return str(pool.model_name), request_endpoints, admin_endpoints


def pools_may_alias(left: InferencePool, right: InferencePool) -> bool:
    """Conservatively detect two pool objects backed by one mutable model.

    Inline frozen references construct a separate Python pool, so object
    identity is only the fast path. Equal model names plus an overlapping
    request or admin endpoint identify an alias. Missing endpoint information
    is ambiguous and therefore treated as mutable.
    """
    if left is right:
        return True
    left_model, left_requests, left_admin = serving_identity(left)
    right_model, right_requests, right_admin = serving_identity(right)
    if left_model != right_model:
        return False
    if not left_requests or not right_requests:
        return True
    if left_requests & right_requests:
        return True
    if not left_admin or not right_admin:
        return True
    return bool(left_admin & right_admin)
