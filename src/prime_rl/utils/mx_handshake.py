"""Fixed CPU message for one ModelExpress offer token."""

import struct

import torch

MAX_OFFER_TOKEN_BYTES = 4096
_LENGTH = struct.Struct("!I")


class OfferTokenMessage:
    """Reuse one CPU tensor while transporting the complete UTF-8 offer token."""

    def __init__(self) -> None:
        self._buffer = bytearray(_LENGTH.size + MAX_OFFER_TOKEN_BYTES)
        self.tensor = torch.frombuffer(self._buffer, dtype=torch.uint8)

    def encode(self, token: str) -> None:
        self.tensor.zero_()
        data = token.encode("utf-8")
        if not 0 < len(data) <= MAX_OFFER_TOKEN_BYTES:
            raise ValueError(f"offer token must contain 1..{MAX_OFFER_TOKEN_BYTES} UTF-8 bytes")
        _LENGTH.pack_into(self._buffer, 0, len(data))
        self._buffer[_LENGTH.size : _LENGTH.size + len(data)] = data

    def decode(self) -> str:
        size = _LENGTH.unpack_from(self._buffer)[0]
        if not 0 < size <= MAX_OFFER_TOKEN_BYTES:
            raise ValueError(f"invalid offer token message length: {size}")
        return self._buffer[_LENGTH.size : _LENGTH.size + size].decode("utf-8")
