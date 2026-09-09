import pytest
from renderers.base import RenderedTokens


class DummyRenderer:
    def render(self, messages, **kwargs):
        content_ids = [ord(char) + 2 for char in messages[-1]["content"]]
        token_ids = [0, *content_ids, 1]
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=[-1, *([len(messages) - 1] * (len(content_ids) + 1))],
            sampled_mask=[False, *([True] * (len(content_ids) + 1))],
        )

    def get_stop_token_ids(self):
        return [1]


@pytest.fixture
def dummy_renderer():
    return DummyRenderer()
