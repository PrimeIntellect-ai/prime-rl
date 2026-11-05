import json
import shutil

import pytest
from vllm.tokenizers import deepseek_v4, deepseek_v4_encoding, deepseek_v32, deepseek_v41

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.vllm.chat_encoding import patch_chat_encoding

# Messages as vLLM parses them: content as text blocks, tool call arguments as dicts.
MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
    {"role": "user", "content": [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Paris?"}]},
    {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "reasoning": "Call it.",
        "reasoning_content": "Call it.",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "Sunny"}]},
]


@pytest.fixture(autouse=True)
def restore_encodings(monkeypatch):
    for module in (deepseek_v32, deepseek_v4, deepseek_v41):
        monkeypatch.setattr(module, "encode_messages", module.encode_messages)


def test_a_copy_of_the_bundled_encoding_renders_identically(tmp_path):
    path = tmp_path / "encoding_dsv4.py"
    shutil.copy(deepseek_v4_encoding.__file__, path)
    patch_chat_encoding(str(path))
    for kwargs in ({"thinking_mode": "thinking", "reasoning_effort": "max"}, {"thinking_mode": "chat"}):
        expected = deepseek_v4_encoding.encode_messages(json.loads(json.dumps(MESSAGES)), **kwargs)
        assert deepseek_v4.encode_messages(MESSAGES, **kwargs) == expected


def test_the_encoding_gets_messages_in_the_openai_wire_form(tmp_path):
    path = tmp_path / "encoding.py"
    path.write_text("import json\n\ndef encode_messages(messages, **kwargs):\n    return json.dumps(messages)\n")
    patch_chat_encoding(str(path))
    for module in (deepseek_v32, deepseek_v4, deepseek_v41):
        system, user, assistant, tool = json.loads(module.encode_messages(MESSAGES, thinking_mode="chat"))
        assert (system["content"], user["content"], tool["content"]) == ("Be brief.", "Weather\n\nin Paris?", "Sunny")
        assert assistant["reasoning_content"] == "Call it."
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}


def test_the_encoding_rejects_images(tmp_path):
    path = tmp_path / "encoding.py"
    path.write_text("def encode_messages(messages, **kwargs):\n    return ''\n")
    patch_chat_encoding(str(path))
    image = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}
    with pytest.raises(ValueError, match="text-only"):
        deepseek_v4.encode_messages([image])


def test_the_config_passes_an_absolute_path(tmp_path, monkeypatch):
    (tmp_path / "encoding.py").write_text("")
    monkeypatch.chdir(tmp_path)
    assert InferenceConfig(chat_encoding="encoding.py").chat_encoding == tmp_path / "encoding.py"
    with pytest.raises(ValueError, match="not a file"):
        InferenceConfig(chat_encoding="missing.py")
