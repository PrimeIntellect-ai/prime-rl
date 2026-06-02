"""Run setup — the *declared* protocol, distilled from a run's config tomls.

Feature #5 ("show what was set up instead of inferring"): a multi-agent run is
configured by three tomls (orchestrator / inference / trainer). The viewer reads
the orchestrator toml and projects the protocol-relevant fields into a compact,
frontend-ready ``setup`` dict — the env, the student model + LoRA, the sampling
knobs (including ``extra_body.thinking_token_budget``), the on/off-policy + size
knobs, and the multi-agent ``truth_member`` / ``subset``.

This is the *declared* side. The *observed* schedule (the exact turn-order we
recover from the rollout data) is attached by the serving layer (``app.py``), so
the frontend can render declared-config alongside recovered-protocol in one view.

Fail loud: a missing required field (``[student.model].name``, ``[[train.env]]``
``id``/``name``, the size knobs) raises with a path-qualified message rather than
emitting a silent ``None`` that the UI would render as a real (absent) value.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ORCHESTRATOR_FILENAME = "orchestrator.toml"


def load_run_setup(configs_dir: str | Path) -> dict[str, Any]:
    """Build the run-setup dict from a configs dir's ``orchestrator.toml``.

    Pulls (per the contract):
      - ``[student.model]`` name (+ ``[student.model.lora]`` name/rank/alpha)
      - ``[[train.env]]`` id / name, ``[train.env.args]`` truth_member / subset
      - ``[train.env.sampling]`` + ``extra_body.thinking_token_budget``
      - top-level group_size / max_off_policy_steps / max_steps / seq_len / batch_size
    """
    configs_dir = Path(configs_dir)
    orch_path = configs_dir / ORCHESTRATOR_FILENAME
    if not orch_path.exists():
        raise FileNotFoundError(f"no {ORCHESTRATOR_FILENAME} in {configs_dir}")
    with orch_path.open("rb") as f:
        orch = tomllib.load(f)

    model = _require(orch, "student", "model")
    lora = model.get("lora")
    env = _first_env(orch)
    env_args = env.get("args") or {}
    sampling = env.get("sampling") or {}
    extra_body = sampling.get("extra_body") or {}

    return {
        "env_id": _require(env, "id"),
        "env_name": _require(env, "name"),
        "model": {
            "name": _require(model, "name"),
            "lora": (
                None
                if lora is None
                else {
                    "name": lora.get("name"),
                    "rank": lora.get("rank"),
                    "alpha": lora.get("alpha"),
                }
            ),
        },
        "sampling": {
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "max_completion_tokens": sampling.get("max_completion_tokens"),
            "thinking_token_budget": extra_body.get("thinking_token_budget"),
            "repetition_penalty": sampling.get("repetition_penalty"),
        },
        "group_size": _require(orch, "group_size"),
        "max_off_policy_steps": _require(orch, "max_off_policy_steps"),
        "max_steps": _require(orch, "max_steps"),
        "seq_len": _require(orch, "seq_len"),
        "batch_size": _require(orch, "batch_size"),
        "truth_member": env_args.get("truth_member"),
        "subset": env_args.get("subset"),
    }


def _first_env(orch: dict[str, Any]) -> dict[str, Any]:
    """The first ``[[train.env]]`` table. Raises loud if the run declares none."""
    envs = _require(orch, "train", "env")
    if not isinstance(envs, list) or not envs:
        raise ValueError(
            "orchestrator.toml [[train.env]] is empty or not an array of tables — "
            "a run must declare at least one env to have a protocol to show"
        )
    return envs[0]


def _require(d: dict[str, Any], *path: str) -> Any:
    """Fetch a nested key, raising a path-qualified error if any segment is absent."""
    cur: Any = d
    for i, seg in enumerate(path):
        if not isinstance(cur, dict) or seg not in cur:
            qualified = ".".join(path[: i + 1])
            raise KeyError(
                f"orchestrator.toml missing required key {qualified!r} "
                f"(refusing to emit a silent None for a configured field)"
            )
        cur = cur[seg]
    return cur
