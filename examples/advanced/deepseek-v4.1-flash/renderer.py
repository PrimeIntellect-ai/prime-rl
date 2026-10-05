"""DeepSeek-V4.1 chat renderer, for text conversations without tools.

Selected from a config as::

    [renderer]
    name = "custom"
    import_path = "examples/advanced/deepseek-v4.1-flash/renderer.py:DeepSeekV41Renderer"

V4.1's reference encoder (`encoding/encoding.py` in the checkpoint) is V4's with three changes
that matter for plain chats, applied here on top of the V4 renderer:

- a `<｜System｜>` token leads the conversation whenever it opens with a reasoning-effort prefix
  or a system message;
- the reasoning effort is a numeric budget, `Reasoning Effort: {budget} (range 1-100, ...)`,
  rendered in thinking mode only, defaulting to "high" (75);
- thinking mode is the default.

Tool calls (V4.1 renamed the DSML tags) and mid-conversation system messages are not supported.
"""

from typing import Literal

from renderers.base import Message, RenderedTokens, ToolSpec
from renderers.configs import DeepSeekV4RendererConfig
from renderers.deepseek_v4 import DeepSeekV4Renderer

SYSTEM_TOKEN = "<｜System｜>"
EFFORT_BUDGETS = {"low": 50, "high": 75, "max": 100}


class DeepSeekV41RendererConfig(DeepSeekV4RendererConfig):
    name: Literal["deepseek-v4.1"] = "deepseek-v4.1"  # type: ignore[assignment]

    enable_thinking: bool = True
    """Thinking mode, the V4.1 encoder's default."""

    reasoning_effort: Literal["low", "high", "max"] | int = "high"  # type: ignore[assignment]
    """Thinking-only budget: an int in [1, 100] or "low" / "high" / "max" (50 / 75 / 100)."""


class DeepSeekV41Renderer(DeepSeekV4Renderer):
    config_class = DeepSeekV41RendererConfig

    def _reasoning_effort_prompt(self) -> str:
        effort = self.config.reasoning_effort
        budget = EFFORT_BUDGETS[effort] if isinstance(effort, str) else effort
        if not 1 <= budget <= 100:
            raise ValueError(f"reasoning_effort must be in [1, 100], got {budget}")
        return f"{SYSTEM_TOKEN}Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if tools:
            raise NotImplementedError("the DeepSeek-V4.1 renderer does not render tools")
        if any(message["role"] == "system" for message in messages[1:]):
            raise NotImplementedError("the DeepSeek-V4.1 renderer does not render mid-conversation system messages")
        if messages and messages[0]["role"] == "system" and not self.config.enable_thinking:
            # Without an effort prefix, the system message itself opens with the System token.
            first = {**messages[0], "content": SYSTEM_TOKEN + (messages[0].get("content") or "")}
            messages = [first, *messages[1:]]
        return super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
