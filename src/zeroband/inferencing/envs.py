from typing import TYPE_CHECKING, Any, List
import os

if TYPE_CHECKING:
    SHARDCAST_SERVERS: list[str] | None = None
    SHARDCAST_BACKLOG_VERSION: int = -1

_env = {
    "SHARDCAST_SERVERS": lambda: os.getenv("SHARDCAST_SERVERS", None).split(",")
    if os.getenv("SHARDCAST_SERVERS", None) is not None
    else None,
    "SHARDCAST_BACKLOG_VERSION": lambda: int(os.getenv("SHARDCAST_BACKLOG_VERSION", "-1")),
}


def __getattr__(name: str) -> Any:
    if name not in _env:
        raise AttributeError(f"Invalid environment variable: {name}")
    return _env[name]()


def __dir__() -> List[str]:
    return list(_env.keys())
