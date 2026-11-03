"""The env server ships its env config to workers as plain data: a JSON-mode
`model_dump` and `resolve_env_config` rebuilds the env's own config class, role-typed seats
included."""

import pytest
from verifiers.v1.utils.loaders import resolve_env_config


@pytest.mark.parametrize(
    "data",
    [
        {"taskset": {"id": "kuhn-poker"}, "player0": {"max_turns": 3}},
        {
            "id": "best-of-n",
            "n": 2,
            "taskset": {"id": "reverse-text"},
            "agent": {"harness": {"id": "null"}, "max_turns": 2},
        },
    ],
    ids=["role-typed-seats", "bundled-env"],
)
def test_env_config_survives_the_wire(data: dict) -> None:
    config = resolve_env_config(data)
    rebuilt = resolve_env_config(config.model_dump(mode="json"))
    assert type(rebuilt) is type(config)
    assert rebuilt == config


def test_broadcast_replies_merge() -> None:
    import msgpack

    from prime_rl.env.pool import merge_replies

    replies = [
        msgpack.packb({"success": True, "error": None, "running": 2, "saved": ["a"]}),
        msgpack.packb({"success": True, "error": None, "running": 1, "saved": ["b"]}),
    ]
    assert msgpack.unpackb(merge_replies(replies)) == {
        "success": True,
        "error": None,
        "running": 3,
        "saved": ["a", "b"],
    }
    failed = msgpack.packb({"success": False, "error": "boom"})
    merged = msgpack.unpackb(merge_replies([replies[0], failed]))
    assert merged["success"] is False and merged["error"] == "boom"
