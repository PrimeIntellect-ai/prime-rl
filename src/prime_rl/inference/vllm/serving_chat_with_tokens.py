from typing import Any, Callable, Optional, Sequence, Union

from pydantic import Field
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
    apply_hf_chat_template,
    apply_mistral_chat_template,
    parse_chat_messages_futures,
    resolve_chat_template_content_format,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import RequestPrompt, TextTokensPrompt
from vllm.entrypoints.openai.tool_parsers import ToolParser
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import is_list_of

logger = init_logger(__name__)


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    tokens: list[int] = Field(description=("Prompt tokens to use for the request."))


class OpenAIServingChatCompletionWithTokens(OpenAIServingChat):
    """OpenAI-compatible generate API that allows token-in."""

    async def _preprocess_chat(
        self,
        request: ChatCompletionRequestWithTokens,
        tokenizer: AnyTokenizer,
        messages: list[ChatCompletionMessageParam],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None,
        add_special_tokens: bool = False,
    ) -> tuple[
        list[ConversationMessage],
        Sequence[RequestPrompt],
        list[EngineTokensPrompt],
    ]:
        model_config = self.model_config

        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tool_dicts,
            chat_template_content_format,
            tokenizer,
            model_config=model_config,
        )
        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tool_dicts,
            documents=documents,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        request_prompt: Union[str, list[int]]

        if tokenizer is None:
            request_prompt = "placeholder"
        elif isinstance(tokenizer, MistralTokenizer):
            request_prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                tokenizer=tokenizer,
                conversation=conversation,
                model_config=model_config,
                **_chat_template_kwargs,
            )

        mm_data = await mm_data_future

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        should_parse_tools = tool_parser is not None and (
            hasattr(request, "tool_choice") and request.tool_choice != "none"
        )

        if should_parse_tools:
            if not isinstance(request, ChatCompletionRequest):
                msg = "Tool usage is only supported for Chat Completions API"
                raise NotImplementedError(msg)

            request = tool_parser(tokenizer).adjust_request(  # type: ignore
                request=request
            )

        if tokenizer is None:
            assert isinstance(request_prompt, str), (
                "Prompt has to be a string",
                "when the tokenizer is not initialised",
            )
            prompt_inputs = TextTokensPrompt(prompt=request_prompt, prompt_token_ids=[1])
        elif isinstance(request_prompt, str):
            # NOTE: Main change from the parent class is that we skip the call
            # to _tokenize_prompt_input_async and directly use the tokens from
            # the request body

            # prompt_inputs = await self._tokenize_prompt_input_async(
            #     request,
            #     tokenizer,
            #     request_prompt,
            #     add_special_tokens=add_special_tokens,
            # )
            prompt_inputs = TextTokensPrompt(prompt=request_prompt, prompt_token_ids=request.tokens)
        else:
            # For MistralTokenizer
            assert is_list_of(request_prompt, int), "Prompt has to be either a string or a list of token ids"
            prompt_inputs = TextTokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt,
            )

        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data

        if mm_uuids is not None:
            engine_prompt["multi_modal_uuids"] = mm_uuids

        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        if hasattr(request, "cache_salt") and request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return conversation, [request_prompt], [engine_prompt]
