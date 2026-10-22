import torch
from torch import nn

PAD_UNIT = 256
SPECIAL_UNIT = 257
# Byte-window sentinels next to the byte values 0..255: a special token, a position before the document start, a
# position after the token's end (first-K windows of long tokens).
SPECIAL_BYTE, PAD_BYTE, NO_BYTE = 256, 257, 258
UNITS_PER_CHAR = {"bytes": 1, "chars": 4, "script": 2}


class ByteWindowEmbedding(nn.Module):
    """Embeds, at every token, the last ``window`` units ending with that token, pulled from the previous tokens of the
    same document (mixture of tokenizers). One unit table is shared by the ``window`` slots, each filling
    ``dim / window`` dimensions. ``table_path`` is a ``torch.save``d dict with the tokenizer's ``token_bytes``
    (``[vocab, max_len]``, each token's UTF-8 bytes, left-aligned) and ``token_lengths`` (``[vocab]``, 0 for special
    tokens).

    ``units``:
    - ``bytes``: one unit per UTF-8 byte (256 bytes, a pad and a special-token unit); a special token is one unit.
    - ``chars``: four units per character (its UTF-8 bytes, padded to four), so the window is ``window / 4`` characters.
    - ``script``: two units per character, SCRIPT's block and index units (Land & Arnett 2025, arXiv 2505.24689;
      ``script_table_path`` maps code points to them), so the window is ``window / 2`` characters.
    In the character modes only complete characters count: a character split across tokens belongs to the token with
    its last byte, the earlier token sees a partial-character unit of its own bytes; bytes that form no valid character
    are raw-byte units, unassigned code points an unknown unit.

    ``first_k``: a token longer than the window (more than ``window`` bytes / ``window / units per char`` characters)
    sees its own first units instead of its last ones."""

    def __init__(
        self,
        *,
        dim: int,
        window: int,
        table_path: str,
        units: str = "bytes",
        first_k: bool = False,
        script_table_path: str | None = None,
    ) -> None:
        super().__init__()
        self.window = window
        self.units = units
        self.first_k = first_k
        self.table_path = table_path
        self.script_table_path = script_table_path
        self.chars = window // UNITS_PER_CHAR[units]
        table = torch.load(table_path)
        if units == "bytes":
            unit_count = SPECIAL_UNIT + 1
        elif units == "chars":
            # bytes 0..255, the pad of a character's unused byte slots, a special token, a position before the document
            self.char_pad_unit, self.special_char_unit, self.pad_char_unit = 256, 257, 258
            unit_count = 259
        else:
            script = torch.load(script_table_path)
            indices, blocks = script["index_units"], script["block_units"]
            # index units, block units, then the blocks of raw bytes, partial characters, special tokens, unassigned
            # code points and the document pad
            self.raw_unit, self.partial_unit, self.special_script_unit, self.unknown_unit, self.pad_script_unit = (
                indices + blocks + i for i in range(5)
            )
            unit_count = indices + blocks + 5
            self.register_buffer(
                "code_point_block", torch.zeros(script["block"].shape, dtype=torch.long), persistent=False
            )
            self.register_buffer(
                "code_point_index", torch.zeros(script["index"].shape, dtype=torch.long), persistent=False
            )
        self.embed = nn.Embedding(unit_count, dim // window)
        self.register_buffer("token_bytes", torch.zeros(table["token_bytes"].shape, dtype=torch.long), persistent=False)
        self.register_buffer(
            "token_lengths", torch.zeros(table["token_lengths"].shape, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "token_chars", torch.zeros(table["token_lengths"].shape, dtype=torch.long), persistent=False
        )

    def reset_buffers(self) -> None:
        table = torch.load(self.table_path)
        self.token_bytes.copy_(table["token_bytes"])
        self.token_lengths.copy_(table["token_lengths"])
        positions = torch.arange(self.token_bytes.shape[1], device=self.token_bytes.device)
        in_token = positions < self.token_lengths.unsqueeze(1)
        self.token_chars.copy_((in_token & (self.token_bytes & 0xC0 != 0x80)).sum(1))
        if self.units == "script":
            script = torch.load(self.script_table_path)
            # unassigned code points (-1) -> the unknown block
            self.code_point_block.copy_(torch.where(script["block"] >= 0, script["block"].long(), self.unknown_unit))
            self.code_point_index.copy_(torch.where(script["block"] >= 0, script["index"].long(), 0))

    def forward(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        return self.embed(self.unit_ids(input_ids, cu_seqlens)).flatten(-2).reshape(*input_ids.shape, -1)

    def unit_ids(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """``[tokens, window]`` unit ids of the flattened ``input_ids``."""
        tokens = input_ids.reshape(-1)
        if self.units == "bytes":
            long_tokens = self.token_lengths[tokens] > self.window if self.first_k else None
            window_bytes = self._window_bytes(tokens, cu_seqlens, self.window, long_tokens)
            pad = torch.full_like(window_bytes, PAD_UNIT)
            return torch.where(
                window_bytes == SPECIAL_BYTE, SPECIAL_UNIT, torch.where(window_bytes == PAD_BYTE, pad, window_bytes)
            )
        long_tokens = self.token_chars[tokens] > self.chars if self.first_k else None
        # the last `chars` characters fit in 4 bytes each, plus up to 3 bytes of a trailing partial character
        window_bytes = self._window_bytes(tokens, cu_seqlens, 4 * self.chars + 3, long_tokens)
        return self._char_units(window_bytes, long_tokens)

    def _window_bytes(
        self, tokens: torch.Tensor, cu_seqlens: torch.Tensor, width: int, long_tokens: torch.Tensor | None
    ) -> torch.Tensor:
        """The ``width`` bytes ending with each token (special tokens one ``SPECIAL_BYTE``, positions before the
        document ``PAD_BYTE``); for ``long_tokens`` the token's own first ``width`` bytes (``NO_BYTE`` after its end)."""
        lengths = self.token_lengths[tokens]
        units = lengths.clamp_min(1)
        ends = units.cumsum(0)
        starts = ends - units
        positions = torch.arange(tokens.numel(), device=tokens.device)
        document_starts = cu_seqlens[torch.searchsorted(cu_seqlens[1:], positions, right=True)]
        slots = torch.arange(width, device=tokens.device)
        unit_positions = ends.unsqueeze(1) - width + slots
        if long_tokens is not None:
            unit_positions = torch.where(long_tokens.unsqueeze(1), starts.unsqueeze(1) + slots, unit_positions)
        source = torch.searchsorted(ends, unit_positions.clamp_min(0), right=True).clamp_max(tokens.numel() - 1)
        offsets = (unit_positions - starts[source]).clamp(0, self.token_bytes.shape[1] - 1)
        window_bytes = torch.where(lengths[source] > 0, self.token_bytes[tokens[source], offsets], SPECIAL_BYTE)
        window_bytes = torch.where(unit_positions >= starts[document_starts].unsqueeze(1), window_bytes, PAD_BYTE)
        if long_tokens is not None:
            window_bytes = torch.where(
                long_tokens.unsqueeze(1) & (unit_positions >= ends.unsqueeze(1)), NO_BYTE, window_bytes
            )
        return window_bytes

    def _char_units(self, b: torch.Tensor, long_tokens: torch.Tensor | None) -> torch.Tensor:
        """Decodes byte windows ``[tokens, width]`` into the units of their last (first, for ``long_tokens``)
        ``chars`` elements (characters, raw bytes, a partial character, special tokens, pads)."""
        width = b.shape[1]
        real = b < 256
        continuation = real & (b & 0xC0 == 0x80)
        lead_length = torch.where(
            b < 0x80,
            1,
            torch.where(b < 0xC0, 0, torch.where(b < 0xE0, 2, torch.where(b < 0xF0, 3, torch.where(b < 0xF8, 4, 1)))),
        )
        lead_length = torch.where(real, lead_length, torch.where(b == NO_BYTE, 0, 1))

        def shifted(x: torch.Tensor, k: int, fill) -> torch.Tensor:
            """x[:, j + k] at column j (``fill`` past the edge)."""
            pad = torch.full_like(x[:, : abs(k)], fill)
            return torch.cat([x[:, k:], pad], 1) if k > 0 else torch.cat([pad, x[:, :k]], 1)

        # consecutive continuation bytes after each position (at most 3)
        follow = torch.zeros_like(b)
        run = torch.ones_like(continuation)
        for k in (1, 2, 3):
            run = run & shifted(continuation, k, False)
            follow = follow + run.long()
        last_real = torch.where(b == NO_BYTE, -1, torch.arange(width, device=b.device)).amax(1, keepdim=True)
        positions = torch.arange(width, device=b.device).unsqueeze(0)
        multi = real & (lead_length >= 2)
        complete = multi & (follow >= lead_length - 1)
        partial = multi & ~complete & (positions + follow == last_real)
        valid = multi & (complete | partial)
        covered = torch.zeros_like(continuation)
        for k in (1, 2, 3):
            lead_before = shifted(valid, -k, False)
            reach = shifted(torch.where(complete, lead_length, follow + 1), -k, 0)
            covered = covered | (lead_before & (reach > k))
        covered = covered & continuation
        left_cut = torch.cumprod(continuation.long(), 1).bool()
        starts = (b != NO_BYTE) & ~(continuation & (covered | left_cut))
        ascii_char = real & (b < 0x80)

        b1, b2, b3 = (shifted(b, k, 0) & 0x3F for k in (1, 2, 3))
        code_point = torch.where(
            lead_length == 2,
            (b & 0x1F) << 6 | b1,
            torch.where(lead_length == 3, (b & 0x0F) << 12 | b1 << 6 | b2, (b & 0x07) << 18 | b1 << 12 | b2 << 6 | b3),
        )
        code_point = torch.where(ascii_char, b, code_point).clamp(0, 0x10FFFF)
        is_char = ascii_char | complete
        special = b == SPECIAL_BYTE
        pad = b == PAD_BYTE

        if self.units == "script":
            block = self.code_point_block[code_point]
            index = self.code_point_index[code_point]
            block = torch.where(
                is_char,
                block,
                torch.where(
                    partial,
                    self.partial_unit,
                    torch.where(
                        special, self.special_script_unit, torch.where(pad, self.pad_script_unit, self.raw_unit)
                    ),
                ),
            )
            index = torch.where(
                is_char,
                index,
                torch.where(
                    partial,
                    follow + 1,
                    torch.where(pad, self.pad_script_unit, torch.where(special, 0, b.clamp_max(255))),
                ),
            )
            element_units = torch.stack([block, index], -1)
        else:
            byte_slots = [b, shifted(b, 1, 0), shifted(b, 2, 0), shifted(b, 3, 0)]
            length = torch.where(complete, lead_length, torch.where(partial, follow + 1, 1))
            element_units = torch.stack(
                [torch.where(length > k, slot, self.char_pad_unit) for k, slot in enumerate(byte_slots)], -1
            )
            first = torch.where(special, self.special_char_unit, torch.where(pad, self.pad_char_unit, b))
            element_units[..., 0] = first.clamp_max(self.pad_char_unit)
            element_units = torch.where(pad.unsqueeze(-1), self.pad_char_unit, element_units)

        # rank the elements from the end (from the start for long tokens) and scatter the selected ones into their slots;
        # unselected positions go to an extra slot that is dropped
        rank_from_end = starts.long().flip(1).cumsum(1).flip(1)
        slot = torch.where(starts & (rank_from_end <= self.chars), self.chars - rank_from_end, self.chars)
        if long_tokens is not None:
            rank_from_start = starts.long().cumsum(1)
            first_slot = torch.where(starts & (rank_from_start <= self.chars), rank_from_start - 1, self.chars)
            slot = torch.where(long_tokens.unsqueeze(1), first_slot, slot)
        per_char = element_units.shape[-1]
        empty = self.pad_script_unit if self.units == "script" else self.pad_char_unit
        out = torch.full((b.shape[0], self.chars + 1, per_char), empty, dtype=torch.long, device=b.device)
        out.scatter_(1, slot.unsqueeze(-1).expand(-1, -1, per_char), element_units)
        return out[:, : self.chars].reshape(b.shape[0], self.window)


__all__ = ["ByteWindowEmbedding"]
