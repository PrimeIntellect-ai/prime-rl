"""Streamed chat completions of the Inkling 2 parsers give what the non-streamed ones give.

The cases drive vLLM's chat stream processor, as its server does, with a reply cut into one chunk per token.
"""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.stream_processor import ChatStreamProcessor
from vllm.parser import ParserManager

from prime_rl.inference.vllm.models.inkling2 import parsers
from prime_rl.inference.vllm.models.inkling2.plugin import _register_parsers

ROLE_TOKENS = ("<|message_system|>", "<|message_user|>", "<|message_tool|>")
TEXT = "<|message_model|><|content_text|>Hello world, this is plain text.<|end_message|>"
THINKING = "<|message_model|><|content_thinking|>Let me think about it.<|end_message|>"
CALL = (
    '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}'
    "<|end_message|>"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
# A prompt ends with the `<|end_message|>` of its last message, which is also what closes a thinking block
PROMPT = "<|message_user|><|content_text|>hi<|end_message|>"


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    """Only the framing tokens need ids of their own: the parsers find them by id, and read the rest as text."""
    _register_parsers()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"text": 0}, unk_token="text")))
    tokenizer.add_special_tokens({"additional_special_tokens": [*parsers._FRAMING, *ROLE_TOKENS]})
    return tokenizer


def chunks(tokenizer: PreTrainedTokenizerFast, text: str) -> list[tuple[str, list[int]]]:
    """The text as (chunk text, token ids) pairs: every framing token and every word is one token."""
    vocab = tokenizer.get_vocab()
    pattern = parsers.re.compile("(" + "|".join(map(parsers.re.escape, (*parsers._FRAMING, *ROLE_TOKENS))) + r"|\s+)")
    return [(part, [vocab.get(part, 0)]) for part in pattern.split(text) if part]


def token_ids(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return [token_id for _, ids in chunks(tokenizer, text) for token_id in ids]


def complete(tokenizer: PreTrainedTokenizerFast, reply: str, tool_parser: bool, stream: bool, chunk_size: int = 1):
    """The reasoning, content and tool calls of a chat completion whose model wrote `reply`."""
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS if tool_parser else None,
        tool_choice="auto" if tool_parser else "none",
        skip_special_tokens=False,
        stream=stream,
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="inkling2" if tool_parser else None,
        reasoning_parser_name="inkling2",
        enable_auto_tools=tool_parser,
        tool_strict_level=None,
        model_name="model",
        is_harmony=False,
        tokenizer=tokenizer,
    )
    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs={}, model_config=None)
    if not stream:
        reasoning, content, calls = parser.parse(
            reply, request, enable_auto_tools=tool_parser, model_output_token_ids=token_ids(tokenizer, reply)
        )
        return reasoning, content, [(call.name, call.arguments) for call in calls or []]

    processor = ChatStreamProcessor(request, parser, enable_auto_tools=tool_parser, parser_cls=parser_cls)
    tokens = chunks(tokenizer, reply)
    steps = [
        (
            "".join(text for text, _ in tokens[i : i + chunk_size]),
            [t for _, ids in tokens[i : i + chunk_size] for t in ids],
        )
        for i in range(0, len(tokens), chunk_size)
    ]
    # Generation stops on the end-of-sampling token, whose text the engine does not send
    steps.append(("", token_ids(tokenizer, parsers.END_SAMPLING)))
    reasoning, content, calls = "", "", {}
    for index, (text, ids) in enumerate(steps):
        finish_reason = "stop" if index == len(steps) - 1 else None
        delta = processor.step(text, ids, finish_reason, token_ids(tokenizer, PROMPT)).delta
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            name, arguments = calls.get(call.index, ("", ""))
            calls[call.index] = (name + (call.function.name or ""), arguments + (call.function.arguments or ""))
    return reasoning or None, content or None, [calls[index] for index in sorted(calls)]


REASONING, CONTENT = "Let me think about it.", "Hello world, this is plain text."
WEATHER_CALL = [("get_weather", '{"city":"Paris"}')]


@pytest.mark.parametrize("chunk_size", [1, 3, 1000], ids=["one_token_chunks", "three_token_chunks", "one_chunk"])
@pytest.mark.parametrize(
    "tool_parser,reply,expected",
    [
        (False, TEXT, (None, CONTENT, [])),
        (False, THINKING + TEXT, (REASONING, CONTENT, [])),
        (True, TEXT, (None, CONTENT, [])),
        (True, THINKING + TEXT, (REASONING, CONTENT, [])),
        (True, CALL, (None, None, WEATHER_CALL)),
        (True, THINKING + CALL, (REASONING, None, WEATHER_CALL)),
        (True, TEXT + CALL, (None, CONTENT, WEATHER_CALL)),
        (True, THINKING + TEXT + CALL, (REASONING, CONTENT, WEATHER_CALL)),
    ],
    ids=[
        "text",
        "thinking_text",
        "tools-text",
        "tools-thinking_text",
        "tools-call",
        "tools-thinking_call",
        "tools-text_call",
        "tools-thinking_text_call",
    ],
)
def test_streaming_matches_non_streaming(tokenizer, tool_parser, reply, expected, chunk_size):
    assert complete(tokenizer, reply, tool_parser, stream=False) == expected
    assert complete(tokenizer, reply, tool_parser, stream=True, chunk_size=chunk_size) == expected


@pytest.mark.parametrize(
    "prompt,ended",
    [
        (PROMPT, False),
        # An earlier turn's thinking says nothing about the turn to generate
        (PROMPT + THINKING + TEXT + PROMPT, False),
        # The model turn is continued after its thinking block, or inside it
        (PROMPT + THINKING, True),
        (PROMPT + THINKING + "<|message_model|><|content_text|>Hello", True),
        (PROMPT + "<|message_model|><|content_thinking|>Let me", False),
    ],
    ids=["user_turn", "earlier_thinking", "after_thinking", "in_text_after_thinking", "in_thinking"],
)
def test_reasoning_end_in_prompt(tokenizer, prompt, ended):
    parser = parsers.Inkling2ReasoningParser(tokenizer)
    assert parser.is_reasoning_end(token_ids(tokenizer, prompt)) is ended
