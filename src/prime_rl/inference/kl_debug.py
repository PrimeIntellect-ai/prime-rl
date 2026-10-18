"""Diagnostics for the native vLLM token endpoint; does not alter generation inputs."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path

from prime_rl.utils.kl_debug import append_json, code_provenance, image_summary, save_capture, tensor_tree_summary

active_request: contextvars.ContextVar[dict | None] = contextvars.ContextVar("kl_debug_request", default=None)


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class InferenceKLRecorder:
    def __init__(
        self,
        directory: Path,
        *,
        port: int,
        max_captures: int,
        max_capture_mb: int,
        max_events: int,
        request_every: int = 100,
        engine_id: str = "standalone",
        metadata: dict | None = None,
    ):
        self.directory = directory / f"{socket.gethostname()}-{port}" / engine_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.journal = self.directory / f"events-{self.session}.jsonl"
        # Shared by API processes serving this engine. Atomic replace prevents partial reads.
        self.policy_path = self.directory / "policy.json"
        self.max_captures, self.max_bytes, self.max_events = max_captures, max_capture_mb * 1024**2, max_events
        self.captures = 0
        self.events = 0
        self.requests = 0
        self.request_every = request_every
        (self.directory / f"manifest-{self.session}.json").write_text(
            json.dumps({"code": code_provenance(), "inference": metadata}, indent=2, default=str)
        )

    def policy(self) -> dict | None:
        return json.loads(self.policy_path.read_text()) if self.policy_path.exists() else None

    def event(self, kind: str, **fields) -> None:
        if self.events >= self.max_events:
            return
        self.events += 1
        append_json(self.journal, {"event": kind, "time": time.time(), **fields})
        if self.events == self.max_events:
            append_json(self.journal, {"event": "event_budget_exhausted", "time": time.time()})

    def set_policy(self, version: int | None, *, status: str) -> None:
        value = {"version": version, "status": status, "time": time.time()}
        temporary = self.policy_path.with_suffix(f".tmp-{self.session}")
        temporary.write_text(json.dumps(value))
        temporary.replace(self.policy_path)
        self.event("policy", **value)

    def begin(self, request) -> dict | None:
        capture = self.captures < self.max_captures
        self.captures += int(capture)
        self.requests += 1
        if not capture and self.requests % self.request_every:
            return None
        state = {
            "request_started": time.time(),
            "policy_at_start": self.policy(),
            "cache_salt": request.cache_salt,
            "logical_prompt_tokens": len(request.token_ids),
            "logical_prompt_sha256": token_hash(request.token_ids),
            "capture": capture,
            "capture_id": uuid.uuid4().hex if capture else None,
            "request_ordinal": self.requests,
            "sampling": {
                key: getattr(request.sampling_params, key, None)
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "max_tokens",
                    "seed",
                    "repetition_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                )
            },
        }
        if capture:
            # Raw inline images are saved only inside the bounded capture, never in the journal.
            state["content_parts"] = request.content_parts
            state["logical_token_ids"] = request.token_ids
        return state

    def engine_input(self, request_id: str, value: dict) -> None:
        state = active_request.get()
        if state is None:
            return
        state["request_id"] = request_id
        state["policy_at_engine_submission"] = self.policy()
        tokens = value.get("prompt_token_ids")
        if tokens is not None:
            state["expanded_prompt_tokens"] = len(tokens)
            state["expanded_prompt_sha256"] = token_hash(tokens)
        if state["capture"]:
            mm_kwargs = value.get("mm_kwargs")
            mm_items = (
                {
                    modality: [item.get_data() if item is not None else None for item in items]
                    for modality, items in mm_kwargs.items()
                }
                if mm_kwargs is not None
                else None
            )
            placeholders = {
                modality: [{"offset": item.offset, "length": item.length} for item in items]
                for modality, items in value.get("mm_placeholders", {}).items()
            }
            payload = {
                "schema_version": 1,
                "request_id": request_id,
                "token_ids": tokens,
                "logical_token_ids": state.get("logical_token_ids"),
                "mm_items": mm_items,
                "mm_placeholders": placeholders,
                "content_parts": state.get("content_parts"),
                "sampling": state["sampling"],
            }
            result = save_capture(self.directory / f"input-{state['capture_id']}.pt", payload, self.max_bytes)
            state["input_capture"] = result
            if "path" in result:
                state["mm_tensors"] = {
                    modality: [tensor_tree_summary(item) if item else None for item in items]
                    for modality, items in (mm_items or {}).items()
                }
                state["images"] = [
                    image_summary(part["url"])
                    for part in state.get("content_parts") or []
                    if part.get("type") == "image_url"
                ]
                state["mm_placeholders"] = placeholders
        self.event(
            "engine_submission",
            **{key: value for key, value in state.items() if key not in {"content_parts", "logical_token_ids"}},
        )

    def finish(self, state: dict, response) -> None:
        choices = getattr(response, "choices", None)
        details = {
            "request_id": state.get("request_id"),
            "request_started": state["request_started"],
            "policy_at_start": state["policy_at_start"],
            "policy_at_end": self.policy(),
            "cache_salt": state["cache_salt"],
            "response_type": type(response).__name__,
            "choices": [
                {
                    "index": choice.index,
                    "token_sha256": token_hash(choice.token_ids),
                    "tokens": len(choice.token_ids),
                    "finish_reason": choice.finish_reason,
                }
                for choice in choices or []
            ],
        }
        usage = getattr(response, "usage", None)
        if usage is not None:
            details["usage"] = usage.model_dump(mode="json")
        self.event("request_finished", **details)
