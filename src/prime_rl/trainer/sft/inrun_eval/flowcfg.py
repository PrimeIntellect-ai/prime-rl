"""The eval's flow configs: the collection runs' own ``configs/attempt_2/flow.toml``, with only the model, its
endpoint and the bookkeeping changed.

Everything that shapes a rollout (harness and its command timeout, runtime, timeouts, retries, sampling, taskset,
network notice, task selection, pieces) is kept byte for byte. ``derive`` refuses a source whose sha256 is not the
pinned one and returns the changed keys, which must all be in ``ALLOWED_CHANGES`` (the eval records them).
"""

import hashlib
import tomllib
from typing import Any

# Flattened keys (prefixes) the eval may change. Anything else differing from the source is a bug.
ALLOWED_CHANGES = (
    "run.name",
    "run.output_dir",
    "flow.store",
    "flow.model",
    "flow.solver.model",
    "flow.solver.client.base_url",
    "flow.solver.client.api_key_var",
    "flow.rollout_gate",
    # Only when configured: the tb-sci dataset with the 500th task, the smoke's k and task subset, and the landlock
    # bundle's path when its collection path cannot be created.
    "flow.taskset.dataset",
    "flow.pieces.rollouts.k",
    "flow.select.include.names",
    "flow.taskset.task.prime_kernels",
)


def flat(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flat(item, f"{prefix}.{key}" if prefix else str(key)))
        if not value and prefix:
            out[prefix] = {}
    else:
        out[prefix] = value
    return out


def changed_keys(before: dict, after: dict) -> list[str]:
    a, b = flat(before), flat(after)
    return sorted(key for key in a.keys() | b.keys() if a.get(key, _MISSING) != b.get(key, _MISSING))


_MISSING = object()


def run_name(checkpoint: str, label: str) -> str:
    return f"tb4-eval-{checkpoint}-{label}"


def derive(
    source: bytes,
    expected_sha256: str,
    *,
    label: str,
    checkpoint: str,
    model: str,
    base_url: str,
    api_key_var: str,
    root: str,
    gate_name: str,
    gate_dir: str,
    gate_max: int,
    dataset: str | None = None,
    k: int | None = None,
    include_names: list[str] | None = None,
    path_rewrites: dict[str, str] | None = None,
) -> tuple[dict, list[str]]:
    """The eval config for one source and checkpoint, and the keys it changed."""
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{label}: source flow.toml sha256 {digest} != pinned {expected_sha256}")
    before = tomllib.loads(source.decode())
    config = tomllib.loads(source.decode())
    flow = config["flow"]
    run = run_name(checkpoint, label)
    config.setdefault("run", {})
    config["run"]["name"] = run
    config["run"]["output_dir"] = root
    flow["store"] = f"{root}/{run}/store"
    flow["model"] = model
    solver = flow["solver"]
    solver["model"] = model
    solver["client"]["base_url"] = base_url
    solver["client"]["api_key_var"] = api_key_var
    # One gate shared by the checkpoint's runs (one per source) is its rollout concurrency. The collection's
    # priorities named its own runs and do not apply.
    flow["rollout_gate"] = {"name": gate_name, "max_concurrent": gate_max, "dir": gate_dir}
    if dataset is not None:
        flow["taskset"]["dataset"] = dataset
    if k is not None:
        flow["pieces"]["rollouts"]["k"] = k
    if include_names is not None:
        if not include_names:
            raise ValueError(f"{label}: an empty include list would select every task")
        flow["select"]["include"]["names"] = sorted(include_names)
    for old, new in (path_rewrites or {}).items():
        kernels = flow.get("taskset", {}).get("task", {}).get("prime_kernels", {})
        for spec in kernels.values():
            if isinstance(spec, dict) and spec.get("bundle") == old:
                spec["bundle"] = new
    changed = changed_keys(before, config)
    unexpected = [key for key in changed if not key.startswith(ALLOWED_CHANGES)]
    if unexpected:
        raise AssertionError(f"{label}: changed keys outside the allowed set: {unexpected}")
    return config, changed


def rollouts_k(source: bytes) -> int:
    """The k of the measure preset's rollouts piece."""
    return int(tomllib.loads(source.decode())["flow"]["pieces"]["rollouts"]["k"])


def taskset_fields(source: bytes) -> dict[str, Any]:
    taskset = tomllib.loads(source.decode())["flow"]["taskset"]
    return {key: taskset.get(key) for key in ("dataset", "repo", "registry_path")}


def kernel_bundles(source: bytes) -> list[str]:
    kernels = tomllib.loads(source.decode())["flow"].get("taskset", {}).get("task", {}).get("prime_kernels", {})
    return sorted({spec["bundle"] for spec in kernels.values() if isinstance(spec, dict) and spec.get("bundle")})
