import json
import os
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_config import BaseConfig as BaseConfig  # noqa: F401
from pydantic_config import cli  # noqa: F401

if TYPE_CHECKING:
    from prime_rl.configs.rl import RLConfig


def default_output_dir() -> Path:
    """Default output directory: ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""
    return Path(os.environ.get("PRL_OUTPUT_DIR", "outputs"))


def dump_resolved_config(config: BaseModel, exclude: set[str] | dict[str, Any] | None = None) -> dict:
    """Dump a resolved config for a machine-written JSON artifact.

    Resolved configs are written as JSON, not TOML: JSON keeps nulls, so explicit None
    overrides (e.g. ``--trainer.optim.max-norm None``) round-trip exactly on re-parse.
    Hand-written configs stay TOML (sparse, commented); the format split is the marker.
    """
    return config.model_dump(exclude=exclude, mode="json")


def dump_rl_components(config: "RLConfig") -> dict[str, dict]:
    """Serialize a validated RL config without importing launchers or writing files.

    Keep arbitrary schema fields and explicit nulls. Hosted compilers can use
    this from the CPU-only configs distribution, alongside the local launcher.
    """
    components = {
        "trainer": dump_resolved_config(config.trainer),
        "orchestrator": dump_resolved_config(config.orchestrator),
    }
    if config.inference is not None:
        inference = dump_resolved_config(config.inference, exclude={"deployment", "slurm", "output_dir", "dry_run"})
        if config.deployment.type == "multi_node":
            inference["router"] = None
        components["inference"] = inference
    return components


def find_package_resource(subdir: str) -> Path | None:
    """Find a directory contributed to the `prime_rl` namespace package by any installed wheel.

    Returns None if `subdir` is not present in any wheel — e.g. on a slim
    `prime-rl-configs`-only install where `prime-rl`'s shipped resources
    (templates, etc.) are absent.
    """
    import prime_rl

    for p in prime_rl.__path__:
        candidate = Path(p) / subdir
        if candidate.is_dir():
            return candidate
    return None


def rgetattr(obj: Any, attr_path: str) -> Any:
    """Recursive getattr for dotted paths: rgetattr(cfg, "trainer.model.name")."""
    current = obj
    for attr in attr_path.split("."):
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)
    return current


def rsetattr(obj: Any, attr_path: str, value: Any) -> None:
    """Recursive setattr for dotted paths: rsetattr(cfg, "trainer.model.name", "foo")."""
    if "." not in attr_path:
        return setattr(obj, attr_path, value)
    parent_path, attr = attr_path.rsplit(".", 1)
    setattr(rgetattr(obj, parent_path), attr, value)


MODES = ("rl", "sft", "pretrain", "eval")
"""Run kinds a config can name with its top-level ``mode``. ``uv run launch`` runs the entrypoint of the same name."""


def config_files(args: list[str]) -> list[Path]:
    """Root ``@ file`` references in ``args`` (a ``--flag @ file`` is a nested reference)."""
    return [
        Path(args[i + 1])
        for i, arg in enumerate(args[:-1])
        if arg == "@" and (i == 0 or not args[i - 1].startswith("--"))
    ]


def config_mode(args: list[str]) -> str | None:
    """The ``mode`` that the root config files in ``args`` set, or None. Files that set different modes are an
    error."""
    modes = {}
    for path in config_files(args):
        # Hand-written configs are TOML; resolved configs that the launchers write and re-run are JSON.
        if path.suffix not in (".toml", ".json") or not path.is_file():
            continue
        with path.open("rb") as f:
            mode = (tomllib.load(f) if path.suffix == ".toml" else json.load(f)).get("mode")
        if mode is not None:
            modes[path] = mode
    if len(set(modes.values())) > 1:
        listed = ", ".join(dict.fromkeys(modes.values()))
        files = "".join(f"\n  {path}: {mode}" for path, mode in modes.items())
        raise SystemExit(f"config files set conflicting modes ({listed}):{files}")
    mode = next(iter(modes.values()), None)
    if mode is not None and mode not in MODES:
        raise SystemExit(f'unknown mode "{mode}": use one of {", ".join(MODES)}')
    return mode


def check_mode(entrypoint: str, args: list[str] | None = None) -> None:
    """Stop before parsing when the config files set conflicting modes or a mode other than ``entrypoint``."""
    mode = config_mode(sys.argv[1:] if args is None else args)
    if mode is not None and mode != entrypoint:
        raise SystemExit(f'this is a `mode = "{mode}"` config: run it with `uv run {mode}` or `uv run launch`')
