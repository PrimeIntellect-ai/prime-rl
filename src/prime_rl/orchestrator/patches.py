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


def monkey_patch_parse_response_tokens_top_logprobs():
    """
    Monkey-patches verifiers' parse_response_tokens to also extract top_logprobs
    from the vLLM response. When top_logprobs are present, the returned dict
    will include 'completion_top_logprob_indices' and 'completion_top_logprob_values'
    alongside the existing 'completion_logprobs'.

    Requires return_token_ids=True in the vLLM request so that token strings
    are in "token_id:{id}" format, allowing us to recover integer token IDs.
    """
    import verifiers.utils.response_utils as response_utils

    _original_parse_response_tokens = response_utils.parse_response_tokens

    def _parse_token_id(token_str: str) -> int | None:
        """Parse token ID from vLLM's 'token_id:{id}' format."""
        if token_str.startswith("token_id:"):
            try:
                return int(token_str[9:])
            except ValueError:
                return None
        return None

    def _extract_top_logprobs_from_content(
        logprobs_content: list, is_dict: bool
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Extract top_logprob indices and values from logprobs content.

        Returns:
            (indices, values) where each is a list of lists (one per token).
        """
        all_indices = []
        all_values = []
        for token_entry in logprobs_content:
            if is_dict:
                top_lps = token_entry.get("top_logprobs", [])
            else:
                top_lps = getattr(token_entry, "top_logprobs", [])

            token_indices = []
            token_values = []
            if top_lps:
                for tlp in top_lps:
                    if is_dict:
                        tok_str = tlp.get("token", "")
                        logprob = tlp.get("logprob", float("-inf"))
                    else:
                        tok_str = getattr(tlp, "token", "")
                        logprob = getattr(tlp, "logprob", float("-inf"))
                    tid = _parse_token_id(tok_str)
                    if tid is not None:
                        token_indices.append(tid)
                        token_values.append(logprob)
            all_indices.append(token_indices)
            all_values.append(token_values)
        return all_indices, all_values

    _top_logprobs_debug_logged = False

    async def patched_parse_response_tokens(response, message_type, max_seq_len=None):
        nonlocal _top_logprobs_debug_logged
        result = await _original_parse_response_tokens(response, message_type, max_seq_len)
        if result is None:
            return result

        # Only attempt extraction for chat completions with dict-style logprobs
        # (which is the case after monkey_patch_chat_completion_logprobs)
        if message_type != "chat":
            return result

        try:
            logprobs = response.choices[0].logprobs
            if logprobs is None:
                if not _top_logprobs_debug_logged:
                    import logging

                    logging.getLogger(__name__).warning("top_logprobs extraction: logprobs is None")
                    _top_logprobs_debug_logged = True
                return result

            is_dict = isinstance(logprobs, dict)
            if is_dict:
                content = logprobs.get("content")
            else:
                content = getattr(logprobs, "content", None)

            if content is None:
                if not _top_logprobs_debug_logged:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"top_logprobs extraction: content is None (is_dict={is_dict}, "
                        f"logprobs type={type(logprobs).__name__}, keys={list(logprobs.keys()) if is_dict else 'N/A'})"
                    )
                    _top_logprobs_debug_logged = True
                return result

            # Log first token's structure for debugging
            if not _top_logprobs_debug_logged and content:
                import logging

                first = content[0]
                if is_dict:
                    top_lps = first.get("top_logprobs", [])
                    tok = first.get("token", "???")
                else:
                    top_lps = getattr(first, "top_logprobs", [])
                    tok = getattr(first, "token", "???")
                sample_tlp = None
                if top_lps:
                    sample_tlp = top_lps[0] if is_dict else repr(top_lps[0])
                logging.getLogger(__name__).warning(
                    f"top_logprobs extraction: is_dict={is_dict}, content_len={len(content)}, "
                    f"first_token={repr(tok)}, num_top_logprobs={len(top_lps)}, "
                    f"sample_top_logprob={sample_tlp}"
                )

            indices, values = _extract_top_logprobs_from_content(content, is_dict)

            if not _top_logprobs_debug_logged:
                import logging

                non_empty = sum(1 for idx in indices if len(idx) > 0)
                logging.getLogger(__name__).warning(
                    f"top_logprobs extraction: extracted {len(indices)} entries, "
                    f"{non_empty} non-empty, first_indices={indices[0] if indices else '[]'}"
                )
                _top_logprobs_debug_logged = True

            # Apply same truncation as the original function
            if max_seq_len is not None:
                prompt_len = len(getattr(response, "prompt_token_ids", []))
                if prompt_len > max_seq_len:
                    indices = []
                    values = []
                elif prompt_len + len(indices) > max_seq_len:
                    keep = max_seq_len - prompt_len
                    indices = indices[:keep]
                    values = values[:keep]

            # Only add if we got valid data (non-empty entries)
            if indices and any(len(idx) > 0 for idx in indices):
                result["completion_top_logprob_indices"] = indices
                result["completion_top_logprob_values"] = values
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to extract top_logprobs: {e}", exc_info=True)

        return result

    response_utils.parse_response_tokens = patched_parse_response_tokens

    # Also patch all known consumers that import parse_response_tokens via `from ... import`
    # (their local binding points to the original function, not the module attribute)
    try:
        import verifiers.envs.multiturn_env as multiturn_env

        multiturn_env.parse_response_tokens = patched_parse_response_tokens
    except ImportError:
        pass
    try:
        import verifiers.envs.experimental.rlm_env as rlm_env

        rlm_env.parse_response_tokens = patched_parse_response_tokens
    except ImportError:
        pass
