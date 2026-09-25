import copy

from vllm.tokenizers.deepseek_v4_encoding import encode_messages

from prime_rl.inference.patches import monkey_patch_deepseek_v4_request_tools_placement


class _StubTokenizer:
    def get_added_vocab(self):
        return {}

    def encode(self, text, **_kwargs):
        return text


def _request_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _tokenizer():
    from vllm.tokenizers import deepseek_v4

    monkey_patch_deepseek_v4_request_tools_placement()
    return deepseek_v4.get_deepseek_v4_tokenizer(_StubTokenizer())


def _render(messages, tools):
    return _tokenizer().apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )


def _reference(messages):
    return encode_messages(messages, thinking_mode="thinking", reasoning_effort="low")


def test_request_tools_attach_to_first_existing_system_without_mutation():
    messages = [
        {"role": "user", "content": "This may precede the system message."},
        {"role": "system", "content": "You are helpful.", "tools": []},
        {"role": "system", "content": "A later system message."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    snapshot = copy.deepcopy(messages)
    tools = _request_tools()

    prompt = _render(messages, tools)

    expected_messages = messages.copy()
    expected_messages[1] = {**messages[1], "tools": tools}
    assert prompt == _reference(expected_messages)
    assert prompt.count("## Tools") == 1
    assert messages == snapshot


def test_request_tools_synthesize_system_only_when_none_exists():
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    tools = _request_tools()

    prompt = _render(messages, tools)

    assert prompt == _reference([{"role": "system", "tools": tools}, *messages])
