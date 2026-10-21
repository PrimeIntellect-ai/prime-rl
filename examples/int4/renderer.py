"""INTELLECT-4 renderer: the chat format for SFT and evals, with no Jinja template.

A conversation alternates environment blocks and agent turns. Every tag is one special
token of the INTELLECT-4 tokenizer (``build_tokenizer.py``)::

    <bos><｜environment｜>
      <｜system｜><｜metadata｜>{"depth": 64}</｜metadata｜>
        <｜tool｜><｜name｜>bash</｜name｜><｜content｜>{"description": ..., "parameters": ...}</｜content｜></｜tool｜>
        <｜content｜>You are ...</｜content｜></｜system｜>
      <｜user｜><｜content｜>Fix the bug.</｜content｜></｜user｜>
    </｜environment｜>
    reasoning<｜user｜><｜content｜>On it.</｜content｜></｜user｜>
      <｜tool｜><｜name｜>bash</｜name｜><｜content｜><｜arg｜><｜name｜>command</｜name｜><｜content｜>ls -la</｜content｜></｜arg｜></｜content｜></｜tool｜>
    <｜return｜>
    <｜environment｜><｜tool｜><｜name｜>bash</｜name｜><｜content｜>a.py</｜content｜></｜tool｜></｜environment｜>
    reasoning...

(Line breaks and indentation are for reading only; nothing separates the tags.)

- The first system item opens with its metadata as a JSON ``<｜metadata｜>`` object, which
  holds the reasoning depth budget: a positive number, or low/high/max = 16/32/128.
  ``None`` leaves the key out.
- The environment holds ``system``, ``user`` and ``tool`` (outputs) items. Consecutive
  non-assistant messages share one block. A message's ``name`` becomes its ``<｜name｜>``; a
  tool output's ``<｜name｜>`` is the name of the tool that produced it. A name says who or
  what an item comes from, not which instance: the same name can recur across turns. An
  item's optional ``metadata`` renders as ``<｜metadata｜>``, raw if a string, else as JSON.
- The agent turn is implicit. Text outside tags is reasoning. ``<｜user｜>`` carries the
  user-visible message, ``<｜tool｜>`` a tool call, and ``<｜return｜>`` yields control to the
  environment: it is the stop token. EOS is never trained but also stops, since a base
  model may still emit it. History keeps every turn's reasoning, so the next prompt extends
  the sampled tokens.
- An OpenAI assistant message renders as its reasoning, then its content as a ``<｜user｜>``
  message (none when empty), then its tool calls. Parsing joins all untagged text into
  ``reasoning_content`` and all ``<｜user｜>`` messages into ``content``.
- String arguments render raw; other values render as JSON. Parsing reads the tool schema
  to undo this, so a string argument never needs escaping.
- Text is encoded with ``split_special_tokens=True``: a literal ``<｜user｜>`` in any content
  is plain text, never the special token.

    [renderer]
    name = "custom"
    import_path = "../int4/renderer/renderer.py:Intellect4Renderer"
    depth = "max"  # a positive number, or low/high/max = 16/32/128
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field
from renderers.base import (
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    Tokenizer,
    ToolCallParseStatus,
    ToolSpec,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
)
from renderers.configs import BaseRendererConfig


def open_tag(name: str) -> str:
    # The fullwidth bar (U+FF5C) matches DeepSeek's own special tokens, such as BOS,
    # and sets the tags apart from the XML that plain text often contains.
    return f"<｜{name}｜>"


def close_tag(name: str) -> str:
    return f"</｜{name}｜>"


RETURN = open_tag("return")
ELEMENTS = ("environment", "system", "user", "tool", "name", "content", "metadata", "arg")
TAGS = tuple(tag for name in ELEMENTS for tag in (open_tag(name), close_tag(name))) + (RETURN,)
"""Tags that take reserved slots of the base vocabulary, in slot order."""

ENVIRONMENT_ROLES = {"system": "system", "developer": "system", "user": "user", "tool": "tool"}
DEPTH_ALIASES = {"low": 16, "high": 32, "max": 128}


def resolve_depth(depth: str | float | None) -> int | float | None:
    """Return the depth budget for an alias or a positive number; a whole float returns as an int."""
    if depth is None:
        return None
    if isinstance(depth, str):
        return DEPTH_ALIASES[depth]
    return int(depth) if float(depth).is_integer() else depth


class Intellect4RendererConfig(BaseRendererConfig):
    name: Literal["intellect-4"] = "intellect-4"
    _template_fields = frozenset({"depth"})

    depth: (
        Literal["low", "high", "max"]
        | Annotated[int, Field(strict=True, gt=0)]
        | Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
        | None
    ) = "max"
    """Reasoning depth budget, rendered as system metadata; ``None`` renders no depth."""


def _text(message: Message) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") != "text":
            raise TypeError(f"Intellect4Renderer renders text only, got a {part.get('type')!r} part.")
        parts.append(part["text"])
    return "".join(parts)


def _function(spec: Any) -> Mapping[str, Any]:
    if isinstance(spec, Mapping) and isinstance(spec.get("function"), Mapping):
        return spec["function"]
    if not isinstance(spec, Mapping):
        raise TypeError("Tool definitions and tool calls must be mappings.")
    return spec


def _arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    arguments = _function(call).get("arguments")
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise TypeError(f"Tool-call arguments must be an object, got {type(arguments).__name__}.")
    return arguments


def _argument_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _is_string_schema(schema: Any) -> bool:
    if not isinstance(schema, Mapping):
        return False
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = [k for k in kind if k != "null"]
        return kind == ["string"]
    if kind is None:
        return False
    return kind == "string"


class _Builder:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.message_indices: list[int] = []
        self.sampled_mask: list[bool] = []
        self.is_content: list[bool] = []

    def special(self, token_id: int, index: int, *, sampled: bool = False) -> None:
        self.token_ids.append(token_id)
        self.message_indices.append(index)
        self.sampled_mask.append(sampled)
        self.is_content.append(sampled)

    def text(self, text: str, index: int, *, sampled: bool = False, content: bool = False) -> None:
        if not text:
            return
        ids = self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
        self.token_ids.extend(ids)
        self.message_indices.extend([index] * len(ids))
        self.sampled_mask.extend([sampled] * len(ids))
        self.is_content.extend([content or sampled] * len(ids))

    def prior(self, token_ids: list[int]) -> None:
        self.token_ids.extend(token_ids)
        self.message_indices.extend([-1] * len(token_ids))
        self.sampled_mask.extend([False] * len(token_ids))
        self.is_content.extend([False] * len(token_ids))

    def build(self, messages: list[Message], tool_names: list[str | None]) -> RenderedTokens:
        return RenderedTokens(
            token_ids=self.token_ids,
            message_indices=self.message_indices,
            sampled_mask=self.sampled_mask,
            is_content=self.is_content,
            message_roles=[message.get("role") or "" for message in messages],
            message_tool_names=tool_names,
        )


class Intellect4Renderer:
    config_class = Intellect4RendererConfig
    supports_tools = True
    is_prefix_stable = True

    def __init__(self, tokenizer: Tokenizer, config: Intellect4RendererConfig | None = None):
        self._tokenizer = tokenizer
        self.config = config or Intellect4RendererConfig()
        self.effective_thinking_retention = resolve_thinking_retention(self.config, "all")
        # BOS and EOS are the base model's own, so one renderer serves every base vocabulary.
        self._bos = self._special(tokenizer.bos_token)
        self._eos = self._special(tokenizer.eos_token)
        self._return = self._special(RETURN)
        self._open = {name: self._special(open_tag(name)) for name in ELEMENTS}
        self._close = {name: self._special(close_tag(name)) for name in ELEMENTS}
        self._structural = {*self._open.values(), *self._close.values(), self._return}

    def _special(self, token: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        ids = self._tokenizer.encode(token, add_special_tokens=False)
        assert isinstance(token_id, int) and ids == [token_id], (
            f"{token!r} is not a single token; load the INTELLECT-4 tokenizer (int4/renderer/build_tokenizer.py)"
        )
        return token_id

    # Rendering

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")
        builder = _Builder(self._tokenizer)
        builder.special(self._bos, -1)
        tool_names = self._tool_names(messages)
        depth = resolve_depth(self.config.depth)
        in_environment = False
        if messages[0].get("role") not in ("system", "developer") and (tools or depth is not None):
            builder.special(self._open["environment"], -1)
            in_environment = True
            self._render_system({}, -1, builder, tools=tools, depth=depth)
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                if in_environment:
                    builder.special(self._close["environment"], -1)
                    in_environment = False
                self._render_assistant(message, index, builder)
                continue
            if not in_environment:
                builder.special(self._open["environment"], -1)
                in_environment = True
            if index == 0 and role in ("system", "developer"):
                self._render_system(message, index, builder, tools=tools, depth=depth)
            else:
                self._render_item(message, index, builder, tool_names[index])
        if in_environment:
            builder.special(self._close["environment"], -1)
        return builder.build(messages, tool_names)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def _field(
        self, name: str, text: str, index: int, builder: _Builder, *, sampled: bool = False, content: bool = False
    ) -> None:
        builder.special(self._open[name], index, sampled=sampled)
        builder.text(text, index, sampled=sampled, content=content)
        builder.special(self._close[name], index, sampled=sampled)

    def _render_system(
        self,
        message: Message,
        index: int,
        builder: _Builder,
        *,
        tools: list[ToolSpec] | None,
        depth: float | None,
    ) -> None:
        builder.special(self._open["system"], index)
        metadata = message.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("The first system message's metadata must be an object, to merge the depth into.")
        metadata = {**(metadata or {}), **({} if depth is None else {"depth": depth})}
        if metadata:
            self._field("metadata", json.dumps(metadata, ensure_ascii=False), index, builder)
        for spec in tools or []:
            function = _function(spec)
            definition = {key: value for key, value in function.items() if key not in ("name", "type")}
            builder.special(self._open["tool"], index)
            self._field("name", str(function["name"]), index, builder)
            self._field("content", json.dumps(definition, ensure_ascii=False), index, builder)
            builder.special(self._close["tool"], index)
        if message:
            self._field("content", _text(message), index, builder, content=True)
        builder.special(self._close["system"], index)

    def _render_item(self, message: Message, index: int, builder: _Builder, tool_name: str | None) -> None:
        role = message.get("role")
        element = ENVIRONMENT_ROLES.get(role)
        if element is None:
            raise ValueError(f"Intellect4Renderer cannot render role {role!r}.")
        name = tool_name if role == "tool" else message.get("name")
        if role == "tool" and not name:
            raise ValueError("A tool output needs the name of its tool: set `name` or a matching `tool_call_id`.")
        builder.special(self._open[element], index)
        if name:
            self._field("name", str(name), index, builder)
        metadata = message.get("metadata")
        if metadata is not None:
            self._field("metadata", _argument_text(metadata), index, builder)
        self._field("content", _text(message), index, builder, content=True)
        builder.special(self._close[element], index)

    def _render_assistant(self, message: Message, index: int, builder: _Builder) -> None:
        builder.text(message.get("reasoning_content") or message.get("reasoning") or "", index, sampled=True)
        content = _text(message)
        if content:
            builder.special(self._open["user"], index, sampled=True)
            self._field("content", content, index, builder, sampled=True)
            builder.special(self._close["user"], index, sampled=True)
        for call in message.get("tool_calls") or []:
            builder.special(self._open["tool"], index, sampled=True)
            self._field("name", str(_function(call)["name"]), index, builder, sampled=True)
            builder.special(self._open["content"], index, sampled=True)
            for key, value in _arguments(call).items():
                builder.special(self._open["arg"], index, sampled=True)
                self._field("name", str(key), index, builder, sampled=True)
                self._field("content", _argument_text(value), index, builder, sampled=True)
                builder.special(self._close["arg"], index, sampled=True)
            builder.special(self._close["content"], index, sampled=True)
            builder.special(self._close["tool"], index, sampled=True)
        builder.special(self._return, index, sampled=True)

    @staticmethod
    def _tool_names(messages: list[Message], previous_calls: list[str] | None = None) -> list[str | None]:
        """Each tool output's tool name, from its ``name`` or the closest earlier assistant turn.

        A ``tool_call_id`` is matched only against that turn's calls: clients such as verifiers
        number call ids per turn, so the same id repeats across turns. An output without a
        match takes the call at its position. ``previous_calls`` are the calls of the turn
        before ``messages``.
        """
        calls: list[tuple[str | None, str]] = [(None, name) for name in previous_calls or []]
        names: list[str | None] = []
        position = 0
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                calls = [(call.get("id"), str(_function(call)["name"])) for call in message.get("tool_calls") or []]
                position = 0
            name = None
            if role == "tool":
                name = message.get("name")
                if not name:
                    by_id = [n for call_id, n in calls if call_id and call_id == message.get("tool_call_id")]
                    name = by_id[0] if by_id else calls[position][1] if position < len(calls) else None
                position += 1
            names.append(name)
        return names

    # Parsing

    def get_stop_token_ids(self) -> list[int]:
        # <｜return｜> first: build_training_sample appends element 0 to an unterminated turn.
        # EOS is never trained; the base model may still emit it, so it also stops.
        return [self._return, self._eos]

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        stops = set(self.get_stop_token_ids())
        end = next((k for k, token in enumerate(token_ids) if token in stops), len(token_ids))
        ids = list(token_ids[:end])
        schemas = {str(_function(spec).get("name")): _function(spec).get("parameters") for spec in tools or []}
        reasoning: list[int] = []
        messages: list[str] = []
        calls: list[ParsedToolCall] = []
        position = 0
        in_reasoning = False
        while position < len(ids):
            token = ids[position]
            in_reasoning = token not in (self._open["user"], self._open["tool"])
            if in_reasoning:
                reasoning.append(token)
                position += 1
                continue
            closer = self._close["user"] if token == self._open["user"] else self._close["tool"]
            close = next((k for k in range(position + 1, len(ids)) if ids[k] == closer), None)
            if token == self._open["user"]:
                messages.append(self._message_text(ids[position + 1 : len(ids) if close is None else close]))
            elif close is None:
                calls.append(
                    ParsedToolCall(
                        raw=self._decode(ids[position:]),
                        token_span=(position, len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
            else:
                calls.append(self._parse_call(ids, position, close + 1, schemas))
            if close is None:
                break
            position = close + 1
        return ParsedResponse(
            content="".join(messages),
            reasoning_content=self._decode(reasoning) or None,
            tool_calls=calls,
            # Without a stop token, a completion that ends in untagged text ended mid-reasoning.
            reasoning_complete=not (in_reasoning and end == len(token_ids)),
        )

    def _message_text(self, inner: list[int]) -> str:
        """The text of a ``<｜user｜>`` message body, ``(<｜name｜>who</｜name｜>)?<｜content｜>text</｜content｜>``."""
        _, rest = self._take_field("name", inner)
        text, leftover = self._take_field("content", rest)
        if text is not None and not leftover:
            return text
        if rest[:1] == [self._open["content"]] and self._close["content"] not in rest:
            return self._decode(rest[1:])
        return self._decode(inner)

    def _decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False) if ids else ""

    def _parse_call(self, ids: list[int], start: int, end: int, schemas: dict[str, Any]) -> ParsedToolCall:
        """Parse ``<｜tool｜><｜name｜>tool</｜name｜><｜content｜>(<｜arg｜><｜name｜>key</｜name｜><｜content｜>value</｜content｜></｜arg｜>)*</｜content｜></｜tool｜>``."""
        call = ParsedToolCall(raw=self._decode(ids[start:end]), token_span=(start, end))
        body = ids[start + 1 : end - 1]
        name, rest = self._take_field("name", body)
        if not name:
            call.status = ToolCallParseStatus.MISSING_NAME
            return call
        call.name = name
        if len(rest) < 2 or rest[0] != self._open["content"] or rest[-1] != self._close["content"]:
            call.status = ToolCallParseStatus.MALFORMED_STRUCTURE
            return call
        rest = rest[1:-1]
        properties = (schemas.get(name) or {}).get("properties") or {}
        arguments: dict[str, Any] = {}
        while rest:
            if rest[0] != self._open["arg"] or self._close["arg"] not in rest:
                call.status = ToolCallParseStatus.MALFORMED_STRUCTURE
                return call
            close = rest.index(self._close["arg"])
            key, value_ids = self._take_field("name", rest[1:close])
            value, leftover = self._take_field("content", value_ids)
            if not key or value is None or leftover or key in arguments:
                call.status = ToolCallParseStatus.MALFORMED_STRUCTURE
                return call
            arguments[key] = self._argument_value(value, properties.get(key))
            rest = rest[close + 1 :]
        call.arguments = arguments
        return call

    def _take_field(self, element: str, ids: list[int]) -> tuple[str | None, list[int]]:
        """Read a leading ``<element>text</element>``; return its text and the ids after it."""
        if ids[:1] != [self._open[element]] or self._close[element] not in ids:
            return None, ids
        close = ids.index(self._close[element])
        inner = ids[1:close]
        if any(token in self._structural for token in inner):
            return None, ids
        return self._decode(inner), ids[close + 1 :]

    @staticmethod
    def _argument_value(text: str, schema: Any) -> Any:
        if schema is None or _is_string_schema(schema):
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    # Multi-turn

    def _ends_inside_block(self, ids: list[int]) -> bool:
        """Whether ``ids`` stop inside an open ``<｜user｜>`` or ``<｜tool｜>`` block."""
        closers = {self._open["user"]: self._close["user"], self._open["tool"]: self._close["tool"]}
        expected = None
        for token in ids:
            if expected is None and token in closers:
                expected = closers[token]
            elif token == expected:
                expected = None
        return expected is not None

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if not previous_prompt_ids or not new_messages or reject_assistant_in_extension(new_messages):
            return None
        if should_rerender_for_thinking_retention(self.effective_thinking_retention, new_messages):
            return None
        completion = list(previous_completion_ids)
        if self._eos in completion:
            return None
        if self._return in completion:
            completion = completion[: completion.index(self._return) + 1]
        elif self._ends_inside_block(completion):
            return None
        else:
            # Truncated turn: close it the way a canonical render would, as unsampled context.
            completion.append(self._return)
        previous_calls = [
            call.name
            for call in self.parse_response(completion, tools=tools, prompt_ids=previous_prompt_ids).tool_calls
            if call.name
        ]
        tool_names = self._tool_names(new_messages, previous_calls)
        builder = _Builder(self._tokenizer)
        builder.prior([*previous_prompt_ids, *completion])
        builder.special(self._open["environment"], -1)
        for index, message in enumerate(new_messages):
            self._render_item(message, index, builder, tool_names[index])
        builder.special(self._close["environment"], -1)
        return builder.build(new_messages, tool_names)


__all__ = ["TAGS", "Intellect4Renderer", "Intellect4RendererConfig"]
