"""A renderer that lives outside the ``renderers`` package.

``[renderer] name = "custom"`` loads it from ``import_path = "<this file>:InstructedQwen3Renderer"``
for SFT, and ``[orchestrator.renderer]`` does the same for RL. The renderer class points
``config_class`` at its own config, which validates the remaining ``[renderer]`` fields and
declares which of them are template controls.
"""

from dataclasses import replace
from typing import Literal

from renderers.base import Message, RenderedTokens, ToolSpec
from renderers.configs import PrimeQwen3RendererConfig
from renderers.prime_qwen3 import PrimeQwen3Renderer


class InstructedQwen3RendererConfig(PrimeQwen3RendererConfig):
    name: Literal["instructed-qwen3"] = "instructed-qwen3"
    _template_fields = frozenset({"instruction"})

    instruction: str = "Think step by step."
    """Appended to the system message. A conversation without one gets a system message."""


class InstructedQwen3Renderer(PrimeQwen3Renderer):
    """PrimeIntellect Qwen3 renderer that adds ``instruction`` to every system prompt."""

    config_class = InstructedQwen3RendererConfig

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        instruction = self.config.instruction
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content") or ""
            system: Message = {**messages[0], "content": f"{content}\n\n{instruction}"}
            return super().render([system, *messages[1:]], tools=tools, add_generation_prompt=add_generation_prompt)

        rendered = super().render(
            [{"role": "system", "content": instruction}, *messages],
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        # The inserted system message is scaffolding the caller never passed: its tokens
        # belong to no message (-1), and the caller's messages keep their own indices.
        return replace(
            rendered,
            message_indices=[index - 1 if index > 0 else -1 for index in rendered.message_indices],
            message_roles=rendered.message_roles[1:],
            message_tool_names=rendered.message_tool_names[1:],
        )
