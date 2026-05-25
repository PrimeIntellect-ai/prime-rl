"""Articraft multi-turn RL environment for verifiers / prime-rl.

Phase 1: in-process compile-only reward, no external services.
Tool dispatch reuses articraft's ToolRegistry / Invocation classes directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import verifiers as vf

from agent.compiler import compile_urdf_report_maybe_timeout
from agent.feedback import compile_signal_bundle_from_exception, render_compile_signals
from agent.models import CompileSignalBundle
from agent.tools.base import ToolResult
from agent.tools.compile_model import CompileModelTool
from agent.tools.edit_code import ReplaceTool
from agent.tools.read_file import ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.write_code import WriteFileTool
from agent.workspace_docs import load_sdk_docs_reference

from .artifact_manager import ArticraftArtifactManager, ArtifactPolicy
from .dataset import build_dataset
from .prompts import build_turn0_messages, load_scaffold_text, load_system_prompt
from .rubric import ArticraftRubric
from .schema import Rollout, Task, TurnRecord, require_rollout

logger = logging.getLogger(__name__)


# ---- tool schema conversion ----

def _convert_schemas_to_vf_tools(
    oai_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert articraft's OpenAI-format tool schemas to verifiers Tool format.

    articraft: ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}``
    verifiers: ``{"name": ..., "description": ..., "parameters": ...}``
    """
    vf_tools: list[dict[str, Any]] = []
    for schema in oai_schemas:
        fn = schema.get("function", schema)
        vf_tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return vf_tools


# ---- tool_call accessor helpers (handle both dict and object formats) ----

def _extract_tool_calls(msg: Any) -> list[Any]:
    """Get tool_calls from the last assistant message (dict or object)."""
    if msg is None:
        return []
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return getattr(msg, "tool_calls", None) or []


def _tc_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        return fn.get("name", "") if isinstance(fn, dict) else ""
    fn = getattr(tc, "function", None)
    if fn is not None:
        return getattr(fn, "name", "")
    return getattr(tc, "name", "")


def _tc_args(tc: Any) -> dict[str, Any]:
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        raw = fn.get("arguments", {}) if isinstance(fn, dict) else {}
    else:
        fn = getattr(tc, "function", None)
        raw = getattr(fn, "arguments", {}) if fn is not None else getattr(tc, "arguments", {})
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def _tc_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return tc.get("id", "")
    return getattr(tc, "id", "")


class ArticraftEnv(vf.MultiTurnEnv):
    """Articraft placement environment over MultiTurnEnv.

    Full in-process pipeline: tool dispatch → compile → QC-based reward.
    No external render/score services (Phase 1).
    """

    def __init__(
        self,
        articraft_root: str | Path = "/data/work/articraft",
        max_turns: int = 50,
        work_root: str | Path = "/tmp/articraft_rl",
        keep_failed_only: bool = False,
        env_name: str = "articraft",
        split: str = "train",
        eval_split: str = "eval",
        eval_holdout: int = 50,
        sdk_package: str = "sdk",
        provider: str = "openrouter",
        reward_weights: dict[str, float] | None = None,
        max_rollouts_per_example: int = 0,
        **kwargs: Any,
    ) -> None:
        self.articraft_root = Path(articraft_root).expanduser().resolve()
        self.work_root = Path(work_root).expanduser().resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.env_name = env_name
        self.split = split
        self.eval_split = eval_split
        self.eval_holdout = eval_holdout
        self.sdk_package = sdk_package
        self.provider = provider

        # -- articraft tool registry (reuses articraft Tool/Invocation classes) --
        tools = [
            ReadFileTool(editable_model_only=True),
            ReplaceTool(),
            WriteFileTool(),
            CompileModelTool(),
        ]
        self.tool_registry = ToolRegistry(tools)
        self._vf_tool_defs = _convert_schemas_to_vf_tools(
            self.tool_registry.get_tool_schemas()
        )

        # -- shared state (immutable across rollouts) --
        self.sdk_docs_context: str = load_sdk_docs_reference(
            self.articraft_root, sdk_package=self.sdk_package,
        )
        self.system_prompt_text: str = load_system_prompt(
            self.articraft_root, provider=self.provider,
        )
        self.scaffold_text: str = load_scaffold_text(self.articraft_root)

        # -- artifact manager --
        policy = ArtifactPolicy(
            keep_failed_only=keep_failed_only,
            max_rollouts_per_example=max_rollouts_per_example,
        )
        self.artifact_manager = ArticraftArtifactManager(
            self.work_root,
            policy,
            articraft_root=self.articraft_root,
            sdk_package=self.sdk_package,
        )

        # -- rubric --
        rubric = ArticraftRubric(
            artifact_manager=self.artifact_manager,
            reward_weights=reward_weights,
        )

        # -- dataset builders --
        def _train_dataset_builder():
            return build_dataset(
                self.articraft_root,
                split=self.split,
                eval_holdout=self.eval_holdout,
                sdk_package=self.sdk_package,
            )

        def _eval_dataset_builder():
            return build_dataset(
                self.articraft_root,
                split=self.eval_split,
                eval_holdout=self.eval_holdout,
                sdk_package=self.sdk_package,
            )

        super().__init__(
            dataset=_train_dataset_builder,
            eval_dataset=_eval_dataset_builder,
            system_prompt=self.system_prompt_text,
            max_turns=max_turns,
            rubric=rubric,
            tool_defs=self._vf_tool_defs,
            **kwargs,
        )

    # ------------------------------------------------------------------- hooks

    async def setup_state(self, state: vf.State) -> vf.State:
        info = state.get("info") or {}
        task = Task.from_info(info)
        example_id = state.get("example_id", info.get("example_id"))

        mgr = self.artifact_manager
        work_dir = mgr.make_rollout_dir(
            traj_id=state["trajectory_id"],
            record_id=task.record_id,
            split=self.split,
            example_id=example_id,
        )
        script_path = mgr.script_path(work_dir)
        script_path.write_text(self.scaffold_text, encoding="utf-8")
        workspace = mgr.build_workspace(script_path)

        rollout = Rollout(
            task=task,
            trajectory_id=state["trajectory_id"],
            work_dir=work_dir,
            max_turns=self.max_turns,
            script_path=script_path,
            virtual_workspace=workspace,
            metadata={
                "env": self.env_name,
                "split": self.split,
                "example_id": example_id,
                "record_id": task.record_id,
                "category_slug": task.category_slug,
                "trajectory_id": state["trajectory_id"],
            },
        )
        state["rollout"] = rollout

        # verifiers already prepended system_prompt during dataset formatting.
        # Replace the (empty) user portion with task-specific turn-0 messages.
        existing = state.get("prompt") or []
        system_msgs = [m for m in existing if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "system"]
        state["prompt"] = [
            *system_msgs,
            *build_turn0_messages(
                task.prompt_text,
                sdk_docs_context=self.sdk_docs_context,
                provider=self.provider,
            ),
        ]
        return state

    async def add_model_response(
        self,
        state: vf.State,
        prompt_messages: vf.Messages,
        response: Any,
    ) -> None:
        await super().add_model_response(state, prompt_messages, response)
        if not state.get("trajectory"):
            return
        completion = state["trajectory"][-1]["completion"]

        # Fix thinking-only responses (model exhausted tokens during <think>)
        for msg in completion:
            if msg.get("role") == "assistant" and msg.get("content") is None and not msg.get("tool_calls"):
                msg["content"] = ""

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        rollout = require_rollout(state)
        last_msg = messages[-1] if messages else None

        tool_calls = _extract_tool_calls(last_msg)

        # -- no tool calls: freshness-based termination --
        if not tool_calls:
            if rollout.code_is_fresh():
                state["final_env_response"] = []
                return []
            rollout.compile_required_count += 1
            if rollout.compile_required_count > 3:
                state["final_env_response"] = []
                return []
            return [
                {
                    "role": "user",
                    "content": (
                        "<compile_required>\n"
                        "The latest code has changed since the last successful compile.\n"
                        "Run `compile_model` before concluding.\n"
                        "</compile_required>"
                    ),
                }
            ]

        # -- tool execution --
        result_messages: list[dict[str, Any]] = []
        tool_results: list[ToolResult] = []
        tc_names: list[str] = []

        for tc in tool_calls:
            tc_name = _tc_name(tc)
            tc_args = _tc_args(tc)
            tc_id = _tc_id(tc)
            tc_names.append(tc_name)

            result = await self._dispatch_tool(tc_name, tc_args, rollout)
            tool_results.append(result)

            content = json.dumps(
                {k: v for k, v in result.to_dict().items() if k != "tool_call_id"}
            )
            result_messages.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tc_id,
                }
            )

            if result.is_success():
                rollout.mark_code_mutated(tc_name)

        # -- turn record --
        has_compile = "compile_model" in tc_names
        turn = TurnRecord(
            turn=len(rollout.turns),
            tool_calls=[{"name": n} for n in tc_names],
            compile_attempted=has_compile,
            compile_success=rollout.code_is_fresh() if has_compile else None,
            compile_signals=rollout.last_compile_attempt_dict,
        )
        rollout.turns.append(turn)

        return result_messages

    # --------------------------------------------------------- tool dispatch

    async def _dispatch_tool(
        self, name: str, args: dict[str, Any], rollout: Rollout,
    ) -> ToolResult:
        """Dispatch a single tool call.  Mirrors harness.py _execute_tool."""

        # -- compile_model: parameter interception --
        if name == "compile_model":
            unexpected = sorted(args.keys())
            if unexpected:
                return ToolResult(
                    error=f"Invalid parameters for {name}. Unexpected parameters: {unexpected}"
                )
            return await self._dispatch_compile(rollout)

        # -- replace: empty old_string interception --
        if name == "replace":
            from agent.tools.code_region import extract_editable_code

            try:
                editable = extract_editable_code(
                    rollout.script_path.read_text("utf-8")
                )
            except Exception:
                editable = None

            if (
                args.get("old_string") == ""
                and editable is not None
                and editable.strip() != ""
            ):
                return ToolResult(
                    error=(
                        "old_string cannot be empty unless the editable code section is empty. "
                        'Call `read_file(path="model.py")` to copy exact current editable text and retry.'
                    )
                )
            if (
                editable is not None
                and editable.strip() == ""
                and args.get("old_string") != ""
            ):
                return ToolResult(
                    error=(
                        "Editable code section is empty. Initialize it with "
                        "`write_file(content=...)` or with replace using "
                        'old_string="" and new_string containing the initial '
                        "build_object_model() and run_tests() implementation."
                    )
                )

        # -- generic path: ToolRegistry → Invocation → execute --
        try:
            invocation = await self.tool_registry.build_invocation(name, args)

            if not invocation:
                return ToolResult(error=f"Tool {name} not found")

            if hasattr(invocation, "bind_file_path"):
                invocation.bind_file_path(str(rollout.script_path))
            if hasattr(invocation, "bind_virtual_workspace"):
                invocation.bind_virtual_workspace(rollout.virtual_workspace)

            result = await invocation.execute()
            return result

        except Exception as exc:
            from pydantic import ValidationError

            if isinstance(exc, ValidationError):
                errors = exc.errors()
                missing = [
                    err["loc"][0]
                    for err in errors
                    if err["type"] == "missing"
                ]
                invalid = [
                    f"{err['loc'][0]}: {err['msg']}"
                    for err in errors
                    if err["type"] != "missing"
                ]
                parts = [f"Invalid parameters for {name}."]
                if missing:
                    parts.append(f"Missing required: {missing}")
                if invalid:
                    parts.append(f"Invalid values: {invalid}")
                parts.append(f"Provided: {list(args.keys())}")
                return ToolResult(error=" ".join(parts))
            return ToolResult(error=f"Tool execution error: {exc!s}")

    # --------------------------------------------------------- compile

    async def _dispatch_compile(self, rollout: Rollout) -> ToolResult:
        """In-process compile with freshness cache.

        Mirrors harness_compile.py execute_compile_model() without
        CompileFeedbackLoop — uses Rollout fields directly.
        """
        # freshness cache
        if rollout.code_is_fresh() and rollout.last_compile_bundle_dict is not None:
            bundle = CompileSignalBundle.from_dict(rollout.last_compile_bundle_dict)
            cached_text = (
                "Fresh compile already exists for the current code revision; "
                "`compile_model` was not re-run.\n"
                "Treat that compile result as authoritative unless you are about "
                "to edit code for one specific unresolved defect.\n\n"
                + render_compile_signals(bundle)
            )
            return ToolResult(output=cached_text)

        t0 = time.monotonic()
        try:
            report = await asyncio.to_thread(
                compile_urdf_report_maybe_timeout,
                script_path=rollout.script_path,
                sdk_package=self.sdk_package,
            )
            rollout.last_compile_latency_ms = (time.monotonic() - t0) * 1000
            bundle = report.signal_bundle
            content = render_compile_signals(bundle)
            rollout.mark_compile_attempt(bundle)
            rollout.mark_compile_success(bundle)

            if report.urdf_xml:
                urdf_path = self.artifact_manager.checkpoint_urdf_path(rollout.work_dir)
                urdf_path.write_text(report.urdf_xml, encoding="utf-8")

            return ToolResult(output=content)

        except Exception as exc:
            rollout.last_compile_latency_ms = (time.monotonic() - t0) * 1000
            bundle = compile_signal_bundle_from_exception(exc)
            content = render_compile_signals(bundle)
            rollout.mark_compile_attempt(bundle)
            return ToolResult(output=content, error=str(exc))


def load_environment(**kwargs: Any) -> ArticraftEnv:
    """Factory used by ``verifiers.load_environment("articraft")``."""
    return ArticraftEnv(**kwargs)
