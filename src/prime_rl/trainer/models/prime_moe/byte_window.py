import torch
from torch import nn

PAD_UNIT = 256
SPECIAL_UNIT = 257


class ByteWindowEmbedding(nn.Module):
    """Embeds, at every token, the last ``window`` UTF-8 bytes ending with that token, pulled from the previous tokens
    of the same document (mixture of tokenizers). One byte table (256 bytes, a pad and a special-token unit) is shared
    by the ``window`` slots, each filling ``dim / window`` dimensions. Tokens without bytes (special tokens) count as one
    special unit. ``table_path`` is a ``torch.save``d dict with the tokenizer's ``token_bytes`` (``[vocab, max_len]``,
    each token's UTF-8 bytes, left-aligned) and ``token_lengths`` (``[vocab]``, 0 for special tokens)."""

    def __init__(self, *, dim: int, window: int, table_path: str) -> None:
        super().__init__()
        self.window = window
        self.table_path = table_path
        table = torch.load(table_path)
        self.embed = nn.Embedding(SPECIAL_UNIT + 1, dim // window)
        self.register_buffer("token_bytes", torch.zeros(table["token_bytes"].shape, dtype=torch.long), persistent=False)
        self.register_buffer(
            "token_lengths", torch.zeros(table["token_lengths"].shape, dtype=torch.long), persistent=False
        )

    def reset_buffers(self) -> None:
        table = torch.load(self.table_path)
        self.token_bytes.copy_(table["token_bytes"])
        self.token_lengths.copy_(table["token_lengths"])

    def forward(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        tokens = input_ids.reshape(-1)
        lengths = self.token_lengths[tokens]
        units = lengths.clamp_min(1)
        ends = units.cumsum(0)
        starts = ends - units
        positions = torch.arange(tokens.numel(), device=tokens.device)
        document_starts = cu_seqlens[torch.searchsorted(cu_seqlens[1:], positions, right=True)]
        unit_positions = ends.unsqueeze(1) - self.window + torch.arange(self.window, device=tokens.device)
        source = torch.searchsorted(ends, unit_positions.clamp_min(0), right=True).clamp_max(tokens.numel() - 1)
        offsets = (unit_positions - starts[source]).clamp(0, self.token_bytes.shape[1] - 1)
        byte_units = torch.where(lengths[source] > 0, self.token_bytes[tokens[source], offsets], SPECIAL_UNIT)
        byte_units = torch.where(unit_positions >= starts[document_starts].unsqueeze(1), byte_units, PAD_UNIT)
        return self.embed(byte_units).flatten(-2).reshape(*input_ids.shape, -1)


__all__ = ["ByteWindowEmbedding"]
