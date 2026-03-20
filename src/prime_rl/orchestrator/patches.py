from typing import Any, Optional, TypedDict, Union

import openai.types.chat
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    Audio,
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call_union_param import ChatCompletionMessageToolCallUnionParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from typing_extensions import Literal, Required


def monkey_patch_oai_iterable_types():
    """
    This monkey patch is necessary to avoid Pydantic validating fields using
    typing.Iterable (e.g. in multimodal or tool call messages) lazily which
    leads to tokenization errors, for more info see
    https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
    """

    class ModdedChatCompletionDeveloperMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[Union[str, list[ChatCompletionContentPartTextParam]]]
        role: Required[Literal["developer"]]
        name: str

    class ModdedChatCompletionSystemMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[Union[str, list[ChatCompletionContentPartTextParam]]]
        role: Required[Literal["system"]]
        name: str

    class ModdedChatCompletionUserMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[Union[str, list[ChatCompletionContentPartParam]]]
        role: Required[Literal["user"]]
        name: str

    class ModdedChatCompletionAssistantMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        role: Required[Literal["assistant"]]
        audio: Optional[Audio]
        content: Union[str, list[ContentArrayOfContentPart], None]
        function_call: Optional[FunctionCall]
        name: str
        refusal: Optional[str]
        tool_calls: list[ChatCompletionMessageToolCallUnionParam]

    class ModdedChatCompletionToolMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[Union[str, list[ChatCompletionContentPartTextParam]]]
        role: Required[Literal["tool"]]
        tool_call_id: Required[str]

    # Patch OAI types
    openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam = (
        ModdedChatCompletionDeveloperMessageParam
    )
    openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam = (
        ModdedChatCompletionSystemMessageParam
    )
    openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam = (
        ModdedChatCompletionUserMessageParam
    )
    openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam = (
        ModdedChatCompletionAssistantMessageParam
    )
    openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam = (
        ModdedChatCompletionToolMessageParam
    )

    openai.types.chat.chat_completion_message_param.ChatCompletionMessageParam = Union[
        ChatCompletionDeveloperMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ModdedChatCompletionAssistantMessageParam,
        ModdedChatCompletionToolMessageParam,
        ChatCompletionFunctionMessageParam,
    ]


def monkey_patch_chat_completion_logprobs():
    """
    At large batch sizes and context, constructing OAI's Pydantic model
    ChatCompletion with logprobs is causing heavy CPU overhead (~200ms per
    object at 32K context, which translates to >10min overhead at 4K batch
    size). This function monkey-patches the OAI type and verifiers'
    post-processing utils to avoid validating the complex logprobs field.
    """

    class ChoiceAny(Choice):
        """Same as openai.types.chat.chat_completion.Choice, but without type validation for logprobs field."""

        logprobs: Optional[Any] = None

    class ModdedChatCompletion(ChatCompletion):
        """Same as openai.types.chat.chat_completion.ChatCompletion, but using ChoiceAny instead of Choice."""

        choices: list[ChoiceAny]  # type: ignore

    # Patch OAI types
    openai.types.chat.chat_completion.Choice = ChoiceAny
    openai.types.chat.chat_completion.ChatCompletion = ModdedChatCompletion


def monkey_patch_session_id_header():
    """Patches verifiers' OpenAI clients to send X-Session-ID as a per-request
    header derived from the rollout's example_id. This enables sticky routing
    at the inference router level for KV cache affinity."""
    from typing import cast

    from openai.types.chat import ChatCompletion

    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
        handle_openai_overlong_prompt,
    )
    from verifiers.clients.openai_chat_completions_token_client import (
        OpenAIChatCompletionsTokenClient,
        _has_multimodal_content,
    )
    from verifiers.types import SamplingArgs, State

    def _get_session_header(state) -> dict[str, str] | None:
        if state is None:
            return None
        example_id = state.get("example_id")
        if example_id is None:
            return None
        return {"X-Session-ID": str(example_id)}

    @handle_openai_overlong_prompt
    async def patched_chat_get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        def normalize_sampling_args(sa: SamplingArgs):
            sa = dict(sa)
            if "max_tokens" in sa:
                sa["max_completion_tokens"] = sa.pop("max_tokens")
            return {k: v for k, v in sa.items() if v is not None}

        extra_headers = _get_session_header(kwargs.get("state"))

        has_audio = False
        for message in prompt:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for part in content:
                    part_type = None
                    if isinstance(part, dict):
                        part_type = str(part.get("type", ""))
                    elif hasattr(part, "type"):
                        part_type = str(getattr(part, "type"))
                    if part_type and part_type.startswith("input_audio"):
                        has_audio = True
                        break
                if has_audio:
                    break

        if has_audio and "modalities" not in sampling_args:
            sampling_args = {**sampling_args, "modalities": ["text"]}

        if tools:
            return await self.client.chat.completions.create(
                model=model, messages=prompt, tools=tools,
                extra_headers=extra_headers, **normalize_sampling_args(sampling_args),
            )
        return await self.client.chat.completions.create(
            model=model, messages=prompt,
            extra_headers=extra_headers, **normalize_sampling_args(sampling_args),
        )

    OpenAIChatCompletionsClient.get_native_response = patched_chat_get_native_response

    @handle_openai_overlong_prompt
    async def patched_token_get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        def normalize_sampling_args(sa: SamplingArgs):
            sa = dict(sa)
            if "max_tokens" in sa:
                sa["max_completion_tokens"] = sa.pop("max_tokens")
            sa["logprobs"] = True
            extra_body = dict(return_token_ids=True)
            if "extra_body" in sa:
                sa["extra_body"] = {**sa["extra_body"], **extra_body}
            else:
                sa["extra_body"] = extra_body
            return {k: v for k, v in sa.items() if v is not None}

        sampling_args = normalize_sampling_args(sampling_args)
        state = cast(State, kwargs.pop("state"))

        extra_headers = _get_session_header(state)

        has_multimodal = _has_multimodal_content(prompt) or any(
            _has_multimodal_content(step["prompt"]) for step in state["trajectory"]
        )
        if len(state["trajectory"]) == 0 or has_multimodal:
            return await OpenAIChatCompletionsClient.get_native_response(
                self, prompt, model, sampling_args, tools, state=state
            )
        prompt_ids = await self.get_prompt_ids(state, prompt, tools)
        if prompt_ids is None:
            return await OpenAIChatCompletionsClient.get_native_response(
                self, prompt, model, sampling_args, tools, state=state
            )
        extra_body = sampling_args.pop("extra_body", {})
        body = dict(
            model=model, messages=prompt, tools=tools, tokens=prompt_ids,
            **sampling_args, **extra_body,
        )
        return await self.client.post(
            "/chat/completions/tokens", body=body, cast_to=ChatCompletion,
            extra_headers=extra_headers,
        )

    OpenAIChatCompletionsTokenClient.get_native_response = patched_token_get_native_response
