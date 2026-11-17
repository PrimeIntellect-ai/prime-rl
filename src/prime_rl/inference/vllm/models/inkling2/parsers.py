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
from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
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

    @property
    def start_token(self) -> str:
        return CONTENT_THINKING

    @property
    def end_token(self) -> str:
        return END_MESSAGE

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
        self._buffer = ""

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
                args = obj.get("args", obj.get("arguments", {}))
                calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=obj.get("name") or name,
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
        # Tool calls are only emitted on the final chunk: streaming clients see
        # the text content as it arrives and get the calls when generation ends.
        return None
