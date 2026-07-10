from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.pool_identity import pools_may_alias


def _pool(model: str, request: str | None, admin: str | None):
    return SimpleNamespace(
        model_name=model,
        train_clients=[] if request is None else [SimpleNamespace(base_url=request)],
        admin_clients=[] if admin is None else [SimpleNamespace(base_url=admin)],
    )


@pytest.mark.parametrize(
    ("left", "right", "aliases"),
    [
        (
            _pool("policy", "http://frontend/v1", "http://worker:8081"),
            _pool("policy", "http://frontend", "http://worker:8081/"),
            True,
        ),
        (
            _pool("policy", "http://policy/v1", "http://policy-worker:8081"),
            _pool("policy", "http://frozen/v1", "http://frozen-worker:8081"),
            False,
        ),
        (
            _pool("policy", "http://router-a/v1", "http://shared-worker:8081"),
            _pool("policy", "http://router-b/v1", "http://shared-worker:8081"),
            True,
        ),
        (
            _pool("policy", "http://frontend/v1", "http://worker:8081"),
            _pool("other-model", "http://other-frontend/v1", "http://worker:8081"),
            True,
        ),
        (
            _pool("policy", None, None),
            _pool("policy", "http://frozen/v1", "http://frozen-worker:8081"),
            True,
        ),
    ],
)
def test_pool_aliasing_uses_model_request_and_admin_identity(left, right, aliases: bool):
    assert pools_may_alias(left, right) is aliases


def test_pool_aliasing_accepts_object_identity():
    pool = _pool("policy", "http://frontend/v1", "http://worker:8081")
    assert pools_may_alias(pool, pool)
