from prime_rl.utils.monitor.samples import render_prompt_completion_text, token_payload_ids, token_payload_length


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def test_token_payload_length_supports_full_and_compacted_tokens():
    assert token_payload_length({"prompt_ids": [1, 2, 3]}, "prompt_ids") == 3
    assert token_payload_length({"prompt_ids_len": 12}, "prompt_ids") == 12
    assert token_payload_length({"prompt_ids_len": "12"}, "prompt_ids") is None
    assert token_payload_length(None, "prompt_ids") is None


def test_token_payload_ids_returns_full_prompt_and_completion_ids_only():
    assert token_payload_ids({"prompt_ids": [1, 2], "completion_ids": [3, 4]}) == [1, 2, 3, 4]
    assert token_payload_ids({"prompt_ids_len": 2, "completion_ids_len": 2}) is None
    assert token_payload_ids(None) is None


def test_render_prompt_completion_text_uses_chat_template_after_tokens_are_compacted():
    text = render_prompt_completion_text(
        FakeTokenizer(),
        [{"role": "user", "content": "question"}],
        [{"role": "assistant", "content": "answer"}],
    )

    assert text == "user: question\nassistant: answer"
