import torch
import torch.distributed as dist
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.data import Sample
from prime_rl.trainer.world import get_world


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


BATCH_KEYS = ("input_ids", "position_ids", "target_ids", "loss_mask", "seq_lens")


def local_batches(rows: list[Sample], batch_size: int) -> list[dict[str, torch.Tensor]]:
    """This rank's share of ``rows``, ``batch_size`` rows per forward concatenated into one packed sample, like a
    training micro-batch. Every rank runs the same number of forwards (each is an FSDP collective): a rank that
    runs out of rows scores ``rows[0]`` with an empty loss mask."""
    world = get_world()
    mine = rows[world.rank :: world.world_size]
    batches = []
    for start in range(0, -(-len(rows) // world.world_size), batch_size):
        chunk = mine[start : start + batch_size]
        batch = to_device({key: [value for row in chunk or rows[:1] for value in row[key]] for key in BATCH_KEYS})
        if not chunk:
            batch["loss_mask"].zero_()
        batches.append(batch)
    return batches


def to_device(row: dict[str, list[int]]) -> dict[str, torch.Tensor]:
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
