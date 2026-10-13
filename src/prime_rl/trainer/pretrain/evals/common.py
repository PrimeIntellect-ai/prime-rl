from typing import TypeVar

import torch
import torch.distributed as dist
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.data import Sample
from prime_rl.trainer.world import get_world

T = TypeVar("T")


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


def local_rows(rows: list[T]) -> list[T | None]:
    """This rank's share of ``rows``, padded with ``None`` so every rank runs the same number of
    forward passes (each forward is an FSDP collective)."""
    world = get_world()
    per_rank = -(-len(rows) // world.world_size)
    mine: list[T | None] = list(rows[world.rank :: world.world_size])
    return mine + [None] * (per_rank - len(mine))


def to_device(row: Sample) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(row["input_ids"], device="cuda").unsqueeze(0),
        "position_ids": torch.tensor(row["position_ids"], device="cuda").unsqueeze(0),
        "target_ids": torch.tensor(row["target_ids"], device="cuda").unsqueeze(0),
        "loss_mask": torch.tensor(row["loss_mask"], device="cuda").unsqueeze(0),
        "seq_lens": torch.tensor(row["seq_lens"], device="cuda"),
    }


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
