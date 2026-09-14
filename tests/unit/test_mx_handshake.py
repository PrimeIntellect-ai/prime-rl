"""Offer message encoding and reuse without service or GPU dependencies."""

import struct
import uuid

import pytest
import torch

from prime_rl.utils.mx_handshake import MAX_OFFER_TOKEN_BYTES, OfferTokenMessage


def test_offer_message_preserves_fresh_tokens_and_storage():
    message = OfferTokenMessage()
    address = message.tensor.data_ptr()
    tokens = [f"run.\N{GREEK SMALL LETTER PI}.{uuid.uuid4().hex[:8]}" for _ in range(3)]
    tokens += ["x" * MAX_OFFER_TOKEN_BYTES, "short"]
    for token in tokens:
        message.encode(token)
        assert message.decode() == token
        assert message.tensor.data_ptr() == address
    assert len(set(tokens)) == len(tokens)
    assert message.tensor.device.type == "cpu"
    assert message.tensor.dtype == torch.uint8
    assert not message.tensor[4 + len("short") :].any()


@pytest.mark.parametrize("token", ["", "x" * (MAX_OFFER_TOKEN_BYTES + 1), "\N{GREEK SMALL LETTER PI}" * 2049])
def test_offer_message_rejects_invalid_encoded_length(token):
    message = OfferTokenMessage()
    message.encode("previous.12345678")
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        message.encode(token)
    with pytest.raises(ValueError, match="message length"):
        message.decode()


@pytest.mark.parametrize("size", [0, MAX_OFFER_TOKEN_BYTES + 1, 2**32 - 1])
def test_offer_message_rejects_invalid_received_length(size):
    message = OfferTokenMessage()
    message.tensor[:4] = torch.tensor(list(struct.pack("!I", size)), dtype=torch.uint8)
    with pytest.raises(ValueError, match="message length"):
        message.decode()


def test_offer_message_rejects_invalid_received_utf8():
    message = OfferTokenMessage()
    message.encode("x")
    message.tensor[4] = 255
    with pytest.raises(UnicodeDecodeError):
        message.decode()
