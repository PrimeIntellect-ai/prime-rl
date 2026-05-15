"""Static invariants on ``orchestrate`` that can't run end-to-end here.

The orchestrator function pulls in vLLM, scheduler, weight broadcast, and
torch — none of which load from this venv. These tests instead parse the
source AST and assert structural properties that a smoke run would catch
in production but only after burning hours of GPU time.

P1 reproducer (historical, fix landed 1e013eee0): ``is_ma`` (was
``use_rae``) must be assigned before any read. Python local-scope rule:
any function-level assignment makes the name local throughout the
function body, so an earlier read raises ``UnboundLocalError`` for EVERY
orchestrate invocation, single-agent or multi-agent.

P2 reproducer (historical, fix landed 1e013eee0): the final
``ckpt_manager.save`` (after the loop ends) must pass ``rae_state=`` so
multi-agent runs ending on a non-interval step write a complete final
checkpoint. Otherwise resume from the final ckpt hits the load-side
``FileNotFoundError`` that ckpt.py raises by design.
"""

from __future__ import annotations

import ast
from pathlib import Path

ORCHESTRATOR_SRC = Path(__file__).resolve().parents[3] / "src" / "prime_rl" / "orchestrator" / "orchestrator.py"


def _orchestrate_fn() -> ast.AsyncFunctionDef:
    tree = ast.parse(ORCHESTRATOR_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "orchestrate":
            return node
    raise AssertionError("`async def orchestrate` not found in orchestrator.py")


def _assert_assigned_before_first_use(fn: ast.AsyncFunctionDef, name: str) -> None:
    """Python local scoping: any function-level assignment promotes the
    name to local throughout. First Load (by source line) must come no
    earlier than first Store, else ``UnboundLocalError``. ``ast.walk`` is
    BFS — take ``min`` of linenumbers, not first-encountered."""
    store_lines = [
        n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
    ]
    load_lines = [
        n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
    ]
    if not store_lines and not load_lines:
        return  # name not present, nothing to check
    assert store_lines, f"{name} read but never assigned in orchestrate()"
    assert load_lines, f"{name} assigned but never read in orchestrate()"
    first_store = min(store_lines)
    first_load = min(load_lines)
    assert first_store <= first_load, (
        f"{name} first read at line {first_load} but first assigned at line "
        f"{first_store} → UnboundLocalError on every orchestrate() call"
    )


def test_is_ma_assigned_before_first_use():
    _assert_assigned_before_first_use(_orchestrate_fn(), "is_ma")


def test_advantage_state_assigned_before_first_use():
    _assert_assigned_before_first_use(_orchestrate_fn(), "advantage_state")


def test_advantage_type_assigned_before_first_use():
    _assert_assigned_before_first_use(_orchestrate_fn(), "advantage_type")


def _assert_all_calls_pass_kwarg(fn: ast.AsyncFunctionDef, receiver: str, method: str, kwarg: str) -> None:
    """Every ``receiver.method(...)`` call inside ``fn`` must include
    ``kwarg=``. Used to enforce that all ``ckpt_manager.save / load`` calls
    pass ``rae_state=`` so single-source-of-truth advantage state is
    consistently round-tripped."""
    call_lines: list[int] = []
    calls_with_kwarg: list[int] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == method
            and isinstance(func.value, ast.Name)
            and func.value.id == receiver
        ):
            continue
        call_lines.append(node.lineno)
        if any(kw.arg == kwarg for kw in node.keywords):
            calls_with_kwarg.append(node.lineno)
    missing = sorted(set(call_lines) - set(calls_with_kwarg))
    assert not missing, (
        f"{receiver}.{method}(...) call(s) missing {kwarg}= kwarg at line(s) "
        f"{missing}; advantage state will silently desync"
    )


def test_all_ckpt_saves_pass_rae_state_kwarg():
    _assert_all_calls_pass_kwarg(_orchestrate_fn(), "ckpt_manager", "save", "rae_state")


def test_all_ckpt_loads_pass_rae_state_kwarg():
    _assert_all_calls_pass_kwarg(_orchestrate_fn(), "ckpt_manager", "load", "rae_state")
