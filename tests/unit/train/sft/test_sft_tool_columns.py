import json

import pytest
from datasets import Dataset, concatenate_datasets, load_dataset
from renderers.qwen3 import Qwen3Renderer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from prime_rl.trainer.sft.data import SFTDataset


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("column", ["tools", "tool_defs", "mixed", "both", "empty_tools"])
def test_mixed_tool_columns_keep_definitions(tmp_path, encoded, column):
    special = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ]
    vocab = ["[UNK]", "user", "assistant", "weather", "Question", "Answer", *special]
    backend = Tokenizer(WordLevel(dict(zip(vocab, range(len(vocab)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="<|im_end|>", additional_special_tokens=special
    )
    definition = {"name": "weather", "description": "Forecast", "parameters": {"type": "object"}}
    messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    native = [{"type": "function", "function": definition}]
    verifier = [definition]
    if encoded:
        native, verifier = json.dumps(native), json.dumps(verifier)
    native_data = Dataset.from_list([{"messages": messages, "tools": native}])
    verifier_data = Dataset.from_list([{"messages": messages, "tool_defs": verifier}])
    alternate = [{**definition, "name": "different_tool"}]
    if column in ("both", "empty_tools"):
        chosen = native if column == "both" else ("[]" if encoded else [])
        alternate = json.dumps(alternate) if encoded else alternate
        source = Dataset.from_list([{"messages": messages, "tools": chosen, "tool_defs": alternate}])
    elif column == "mixed":
        source = concatenate_datasets([native_data, verifier_data])
        assert source[1]["tools"] is None
    else:
        source = native_data if column == "tools" else verifier_data
    parquet_path = tmp_path / "data.parquet"
    source.to_parquet(parquet_path)
    source = load_dataset("parquet", data_files=str(parquet_path), split="train")
    dataset = SFTDataset(source, Qwen3Renderer(tokenizer), shuffle=False, max_epochs=1, seq_len=1024)
    samples = list(dataset)
    assert len(samples) == (2 if column == "mixed" else 1)
    weather_id = tokenizer.convert_tokens_to_ids("weather")
    for sample in samples:
        assert (weather_id in sample["input_ids"]) == (column != "empty_tools")
        assert sample == samples[0]
