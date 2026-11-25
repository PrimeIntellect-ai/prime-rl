# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-side reasoning and tool-call parsers for Inkling 2.

The model writes one framing token per block::

    <|message_model|><|content_thinking|>...<|end_message|>
    <|message_model|><|content_text|>...<|end_message|>
    <|message_model|><name><|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>

Generation ends with ``<|content_model_end_sampling|>``. The release's chat
template never emits an assistant header, so the model starts its answer with
``<|message_model|>`` itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)

MODEL_TOKEN = "<|message_model|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
_FRAMING = (
    MODEL_TOKEN,
    END_MESSAGE,
    END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_TOOL_ERROR,
    INVOKE_TOOL_JSON,
    INVOKE_TOOL_TEXT,
)

_FRAMING_PATTERN = re.compile("(" + "|".join(map(re.escape, _FRAMING)) + ")")
_TOOL_CALL_MARKERS = (INVOKE_TOOL_JSON, INVOKE_TOOL_TEXT)


def _open_block(text: str) -> tuple[str | None, str]:
    """The block left open at the end of ``text``: its content marker, or ``MODEL_TOKEN`` while only
    its header is written, with the header text; ``(None, "")`` between blocks."""
    start = text.rfind(MODEL_TOKEN)
    if start < 0:
        return None, ""
    block = text[start + len(MODEL_TOKEN) :]
    marker = _FRAMING_PATTERN.search(block)
    if marker is None:
        return MODEL_TOKEN, block
    if _FRAMING_PATTERN.search(block, marker.end()) is not None:
        return None, ""
    return marker.group(), block[: marker.start()]


def _clean(text: str) -> str:
    for token in _FRAMING:
        text = text.replace(token, "")
    return text.strip()


def _clean_content(text: str | None) -> str | None:
    """Strip framing from plain content, keep tool-call blocks for their parser.

    The serving layer extracts tool calls from the content the reasoning parser
    returns, so those blocks must keep the markers the tool parser splits on.
    """
    if not text:
        return None
    if INVOKE_TOOL_JSON in text or INVOKE_TOOL_TEXT in text:
        return text.strip() or None
    return _clean(text) or None


class Inkling2ReasoningParser(BaseThinkingReasoningParser):
    """Thinking block between ``<|content_thinking|>`` and ``<|end_message|>``."""

    # Without a tool parser, vLLM streams the text after the reasoning as is; this parser then
    # streams every block, so the content comes without its framing.
    opens_reasoning_in_content = True

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._other_role_ids = {
            token_id
            for token, token_id in self.vocab.items()
            if token.startswith("<|message_") and token != MODEL_TOKEN
        }

    @property
    def start_token(self) -> str:
        return CONTENT_THINKING

    @property
    def end_token(self) -> str:
        return END_MESSAGE

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # `<|end_message|>` closes every message, and a prompt ends with the one of its last message.
        # Reasoning has only ended when the model turn being continued holds a closed thinking block.
        turn_start = max((i for i, token_id in enumerate(input_ids) if token_id in self._other_role_ids), default=-1)
        turn = input_ids[turn_start + 1 :]
        return self.start_token_id in turn and super().is_reasoning_end(turn)

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        content_markers = (CONTENT_TEXT, CONTENT_TOOL_ERROR, INVOKE_TOOL_JSON, INVOKE_TOOL_TEXT)
        boundaries = {self.end_token_id, *(self.vocab.get(marker) for marker in content_markers)}
        return any(token_id in boundaries for token_id in delta_ids)

    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids
    ) -> DeltaMessage | None:
        kind, header = _open_block(previous_text)
        reasoning: list[str] = []
        content: list[str] = []
        # Each framing marker is a dedicated token. A vLLM delta may contain several blocks.
        for part in _FRAMING_PATTERN.split(delta_text):
            if not part:
                continue
            if part == MODEL_TOKEN:
                kind, header = MODEL_TOKEN, ""
            elif part in (END_MESSAGE, END_SAMPLING):
                if kind in _TOOL_CALL_MARKERS:
                    content.append(part)
                kind = None
            elif part in _TOOL_CALL_MARKERS:
                # The tool parser reads a call from its whole block: hand it over with its header,
                # which holds the tool's name.
                content.append(MODEL_TOKEN + header + part)
                kind = part
            elif part in (CONTENT_TEXT, CONTENT_TOOL_ERROR, CONTENT_THINKING):
                kind = part
            elif kind == CONTENT_THINKING:
                reasoning.append(part)
            elif kind == MODEL_TOKEN:
                header += part
            else:
                content.append(part)
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning="".join(reasoning) or None, content="".join(content) or None)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output:
            # No thinking block: every text block is content.
            return None, _clean_content(model_output)
        reasoning, content = super().extract_reasoning(model_output, request)
        reasoning = _clean(reasoning) if reasoning else None
        return reasoning or None, _clean_content(content)


class Inkling2ToolParser(ToolParser):
    """Parses ``<name><|content_invoke_tool_json|>{"name":..,"args":{..}}`` blocks."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._block_kind = CONTENT_TEXT
        self._block_parts: list[str] = []
        self._call_index = 0

    @staticmethod
    def extract_tool_calls(model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        calls: list[ToolCall] = []
        texts: list[str] = []
        blocks = model_output.split(MODEL_TOKEN)[1:]
        if not blocks:
            # Plain content (e.g. after reasoning extraction): nothing to extract.
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output or None)
        for block in blocks:
            header, sep, rest = block.partition("<|content_")
            name = header.strip()
            if not sep:
                texts.append(block)
                continue
            kind = "content_" + rest.split("|>", 1)[0]
            payload = rest.split("|>", 1)[1] if "|>" in rest else ""
            payload = payload.split(END_MESSAGE)[0].strip()
            if kind.endswith("invoke_tool_json") and payload:
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Inkling2: dropping malformed tool-call JSON: %.200s", payload)
                    continue
                if not isinstance(obj, dict):
                    logger.warning("Inkling2: tool-call payload must be a JSON object")
                    continue
                function_name = obj.get("name") or name
                if not isinstance(function_name, str) or not function_name:
                    logger.warning("Inkling2: tool call has no valid name")
                    continue
                args = obj.get("args", obj.get("arguments", {}))
                calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps(args, separators=(",", ":")),
                        )
                    )
                )
            elif kind.endswith("invoke_tool_text") and payload:
                calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps({"input": payload}, separators=(",", ":")),
                        )
                    )
                )
            else:
                texts.append(block)
        content = _clean(END_MESSAGE.join(texts)) if texts else None
        return ExtractedToolCallInformation(tools_called=bool(calls), tool_calls=calls, content=content or None)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        texts: list[str] = []
        calls: list[DeltaToolCall] = []
        # Each framing marker is a dedicated token. A vLLM delta may contain several blocks.
        for part in _FRAMING_PATTERN.split(delta_text):
            if not part:
                continue
            if part == MODEL_TOKEN:
                self._block_kind = MODEL_TOKEN
                self._block_parts = [part]
            elif part in (INVOKE_TOOL_JSON, INVOKE_TOOL_TEXT):
                if self._block_kind != MODEL_TOKEN:
                    self._block_parts = [MODEL_TOKEN]
                self._block_kind = part
                self._block_parts.append(part)
            elif part in (END_MESSAGE, END_SAMPLING):
                if self._block_kind in (INVOKE_TOOL_JSON, INVOKE_TOOL_TEXT):
                    parsed = self.extract_tool_calls("".join(self._block_parts), request)
                    for call in parsed.tool_calls:
                        calls.append(
                            DeltaToolCall(
                                index=self._call_index,
                                id=call.id,
                                type="function",
                                function=DeltaFunctionCall(name=call.function.name, arguments=call.function.arguments),
                            )
                        )
                        self._call_index += 1
                self._block_kind = CONTENT_TEXT
                self._block_parts.clear()
            elif part in (CONTENT_TEXT, CONTENT_TOOL_ERROR, CONTENT_THINKING):
                self._block_kind = part
                self._block_parts.clear()
            elif self._block_kind in (CONTENT_TEXT, CONTENT_TOOL_ERROR):
                texts.append(part)
            elif self._block_kind != CONTENT_THINKING:
                self._block_parts.append(part)
        if not texts and not calls:
            return None
        return DeltaMessage(content="".join(texts) or None, tool_calls=calls)
