from transformers import AutoTokenizer

from prime_rl.trainer.pretrain.bytes import token_byte_lengths


def test_token_byte_lengths_count_utf8_bytes():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    lengths = token_byte_lengths(tokenizer)
    text = "Grüße aus Zürich — 東京 🚀 naïve café"
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert int(lengths[token_ids].sum()) == len(text.encode())
    assert lengths[tokenizer.eos_token_id] == 0
