"""Static invariants for componentized orchestrator wiring.

These checks catch integration regressions that otherwise show up only after
starting a GPU smoke run: multi-agent config must reach the dispatcher, RAE
state must be shared with the train sink, and checkpoints must round-trip that
same state whenever it is active.
"""

from __future__ import annotations

import ast
from pathlib import Path

ORCHESTRATOR_SRC = Path(__file__).resolve().parents[3] / "src" / "prime_rl" / "orchestrator" / "orchestrator.py"


def _tree() -> ast.Module:
    return ast.parse(ORCHESTRATOR_SRC.read_text())


def _method(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ClassDef) and node.name == "Orchestrator":
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef | ast.AsyncFunctionDef) and body_item.name == name:
                    return body_item
    raise AssertionError(f"`Orchestrator.{name}` not found")


def _calls(fn: ast.AST, target: str) -> list[ast.Call]:
    return [node for node in ast.walk(fn) if isinstance(node, ast.Call) and ast.unparse(node.func) == target]


def _call_kw(call: ast.Call, name: str) -> ast.keyword | None:
    return next((kw for kw in call.keywords if kw.arg == name), None)


def test_setup_passes_multi_agent_config_to_dispatcher():
    calls = _calls(_method("setup"), "RolloutDispatcher")
    assert len(calls) == 1
    multi_agent_kw = _call_kw(calls[0], "multi_agent")
    assert multi_agent_kw is not None
    assert ast.unparse(multi_agent_kw.value) == "config.multi_agent"


def test_setup_passes_shared_rae_state_to_train_sink():
    calls = _calls(_method("setup"), "TrainSink")
    assert len(calls) == 1
    rae_state_kw = _call_kw(calls[0], "rae_state")
    assert rae_state_kw is not None
    assert ast.unparse(rae_state_kw.value) == "self.rae_state"


def test_all_ckpt_save_load_paths_pass_rae_state():
    missing: list[int] = []
    for method_name in ("setup", "start", "maybe_save_ckpt"):
        for node in ast.walk(_method(method_name)):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            direct_ckpt = target in {"self.ckpt_manager.save", "self.ckpt_manager.load"}
            threaded_ckpt = (
                target == "asyncio.to_thread"
                and node.args
                and ast.unparse(node.args[0]) in {"self.ckpt_manager.save", "self.ckpt_manager.load"}
            )
            if (direct_ckpt or threaded_ckpt) and _call_kw(node, "rae_state") is None:
                missing.append(node.lineno)
    assert not missing, f"ckpt save/load call(s) missing rae_state= at line(s) {missing}"


def test_train_rollout_persistence_honors_dump_trajectory():
    calls = [
        node
        for node in ast.walk(_method("finalize_train_batch"))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "asyncio.to_thread"
        and node.args
        and ast.unparse(node.args[0]) == "save_rollouts"
    ]
    assert len(calls) == 1
    exclude_kw = _call_kw(calls[0], "exclude_keys")
    assert exclude_kw is not None
    assert ast.unparse(exclude_kw.value) == "None if config.dump_trajectory else {'trajectory'}"


def test_eval_rollout_persistence_honors_dump_trajectory():
    calls = [
        node
        for node in ast.walk(_method("finalize_eval_batch"))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "asyncio.to_thread"
        and node.args
        and ast.unparse(node.args[0]) == "save_rollouts"
    ]
    assert len(calls) == 1
    exclude_kw = _call_kw(calls[0], "exclude_keys")
    assert exclude_kw is not None
    assert ast.unparse(exclude_kw.value) == "None if self.config.dump_trajectory else {'trajectory'}"


def test_failed_train_rollout_persistence_honors_config_and_dump_trajectory():
    method = _method("maybe_save_failed_train_rollout")
    assert "dump_failed_train_rollouts" in ast.unparse(method)
    assert "dump_failed_train_trajectory" in ast.unparse(method)
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "asyncio.to_thread"
        and node.args
        and ast.unparse(node.args[0]) == "append_rollouts"
    ]
    assert len(calls) == 1
    assert "train_failed_rollouts.jsonl" in ast.unparse(calls[0])
    exclude_kw = _call_kw(calls[0], "exclude_keys")
    assert exclude_kw is not None
    assert ast.unparse(exclude_kw.value) == "None if dump_trajectory else {'trajectory'}"


def test_main_loop_checks_pipeline_health():
    """The starvation watchdog only works if the main loop drives it: a dead
    or wedged dispatcher task cannot run its own health check."""
    calls = _calls(_method("main_loop"), "self.check_pipeline_health")
    assert len(calls) == 1
    calls = _calls(_method("check_pipeline_health"), "self.dispatcher.raise_if_starved")
    assert len(calls) == 1
