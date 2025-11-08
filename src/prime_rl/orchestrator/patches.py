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
        """Same as openai.types.chat.chat_completion.ChatCompletion, but but using ChoiceAny instead of Choice."""

        choices: list[ChoiceAny]  # type: ignore

    # Patch OAI types
    openai.types.chat.chat_completion.Choice = ChoiceAny
    openai.types.chat.chat_completion.ChatCompletion = ModdedChatCompletion

    # Patch verifiers parse_chat_completion_logprobs
    def patched_parse_chat_completion_logprobs(chat_completion: ModdedChatCompletion) -> list[float]:
        """Same as verifiers.utils.processing_utils.parse_chat_completion_logprobs, but using arbitrary logprobs type."""
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs["content"] is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        logprobs = [logprob["logprob"] for logprob in chat_completion.choices[0].logprobs["content"]]
        return logprobs

    # Patch verifiers parse_chat_completion_logprobs
    def patched_parse_chat_completion_tokens(chat_completion: ModdedChatCompletion) -> list[int]:
        """Same as verifiers.utils.processing_utils.parse_chat_completion_tokens, but using arbitrary logprobs type."""
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs["content"] is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        tokens = [
            # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm in GRPOTrainer
            int(token["token"].split(":")[-1])
            for token in chat_completion.choices[0].logprobs["content"]
        ]
        return tokens

    # Import verifiers here (after patching OpenAI types) so verifiers picks up the patched types
    import verifiers as vf

    vf.utils.processing_utils.parse_chat_completion_logprobs = patched_parse_chat_completion_logprobs
    vf.utils.processing_utils.parse_chat_completion_tokens = patched_parse_chat_completion_tokens
