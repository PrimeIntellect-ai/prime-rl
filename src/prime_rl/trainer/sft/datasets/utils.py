from typing import Optional, Union, Callable
from transformers import PreTrainedTokenizer
from prime_rl.trainer.sft.config import LossMaskConfig
from typing import cast
import logging


logger = logging.getLogger(__name__)


def should_mask(message: dict, loss_mask_config: LossMaskConfig) -> bool:
    assert "role" in message, "Message must have a role"
    match message["role"]:
        case "user":
            return True if loss_mask_config.user else False
        case "assistant":
            return True if loss_mask_config.assistant else False
        case "system":
            return True if loss_mask_config.system else False
        case "tool":
            return True if loss_mask_config.tool else False
        case _:
            raise ValueError(f"Invalid message role: {message['role']}")


def build_loss_mask(messages, tokenizer, loss_mask_config: LossMaskConfig = LossMaskConfig(), tools: Optional[list[Union[dict, Callable]]] = None) -> list[bool]:
    loss_mask: list[bool] = []
    prev_ids, prev_len = [], 0
    for i, message in enumerate(messages):
        assert "role" in message, "Message must have a role"
        # Support parallel tool call outputs (treat them as one message for loss mask)
        if message["role"] == "tool" and i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
            continue
        cur_ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            tools=tools,
            # This is to mask out the generation prompt after user and tool messages
            # It leads to us not training on <|im_start|>assistant
            add_generation_prompt=True
            if (
                message["role"] in ["user", "tool"]
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "assistant"
            )
            else False,
        )
        if not prev_ids == cur_ids[:prev_len]:
            if prev_ids[-1] != 198 and cur_ids[-1] != 271: # the message boundary messes up with bpe, causing inconsistent encoding depending on how many newline characters are there, we can disregard this mismatch as we are only concerned about loss mask
                assert prev_ids == cur_ids[:prev_len], (
                    f"Got mismatch in incremental tokenization with chat template at message {i}. Previous ids: {prev_ids} != {cur_ids[:prev_len]=}.\nDecoded prev_ids:\n{tokenizer.decode(prev_ids)}\nDecoded cur_ids:\n{tokenizer.decode(cur_ids[:prev_len], skip_special_tokens=False)}\n Rendered so far:\n{tokenizer.decode(cur_ids, skip_special_tokens=False)}"
                )
        loss_mask.extend([should_mask(message, loss_mask_config)] * (len(cur_ids) - prev_len))
        prev_ids, prev_len = cur_ids, len(cur_ids)

    return loss_mask


def messages_to_sample(index: int, tokenizer: PreTrainedTokenizer, messages: list[dict], loss_mask_config: LossMaskConfig, tools: Optional[list[Union[dict, Callable]]] = None, with_text: bool = False) -> dict:
    input_ids = cast(list[int], tokenizer.apply_chat_template(messages, tools=tools))
    text = tokenizer.decode(input_ids) if with_text else None
    loss_mask = build_loss_mask(messages, tokenizer, loss_mask_config, tools)
    if not tokenizer.eos_token_id in input_ids:
        logger.warning(f"Did not find EOS token ID {tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token...")
        input_ids.append(cast(int, tokenizer.eos_token_id))
        loss_mask.append(True)

    target_ids = input_ids.copy()[1:]
    loss_mask = loss_mask[1:]
    input_ids = input_ids[:-1]

    prediction_text = tokenizer.decode([target_ids[i] for i, mask in enumerate(loss_mask) if mask], skip_special_tokens=False) if with_text else None

    return {
        "index": index,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "position_ids": list(range(len(input_ids))),
    } | ({"text": text, "prediction_text": prediction_text} if with_text else {})