import torch
from transformers.tokenization_utils import PreTrainedTokenizer


def bytes_to_unicode() -> dict[int, str]:
    """The byte -> character map of byte-level BPE tokenizers (GPT-2, Llama 3, Qwen, DeepSeek)."""
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    n = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + n)
            n += 1
    return dict(zip(printable, map(chr, chars)))


def token_byte_lengths(tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """UTF-8 bytes each token id stands for. Special tokens count as 0 bytes. Exact for
    byte-level BPE tokenizers, approximate otherwise."""
    byte_chars = set(bytes_to_unicode().values())
    special_ids = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    lengths = torch.zeros(len(tokenizer), dtype=torch.long)
    for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if token is None or token_id in special_ids:
            continue
        if all(char in byte_chars for char in token):
            lengths[token_id] = len(token)
        else:
            lengths[token_id] = len(tokenizer.convert_tokens_to_string([token]).encode())
    return lengths
