import random

import pytest
import torch

from prime_rl.trainer.models.prime_moe.byte_window import PAD_UNIT, SPECIAL_UNIT, ByteWindowEmbedding

TEXT = "Hello wörld, привет мир! 中文字符 テスト ٱلْعَرَبِيَّة 👍🏽 ok"
SPECIAL = b""


def tables(tmp_path, tokens: list[bytes]):
    lengths = torch.tensor([len(t) for t in tokens])
    token_bytes = torch.zeros(len(tokens), int(lengths.max()), dtype=torch.long)
    for i, data in enumerate(tokens):
        token_bytes[i, : len(data)] = torch.tensor(list(data))
    torch.save({"token_bytes": token_bytes, "token_lengths": lengths}, tmp_path / "bytes.pt")
    # toy SCRIPT table: block = code point // 64, index = code point % 64; code points >= 0x10000 unassigned
    code_points = torch.arange(0x110000)
    block = torch.where(code_points < 0x10000, code_points // 64, -1)
    torch.save(
        {"block": block, "index": code_points % 64, "index_units": 64, "block_units": 1024}, tmp_path / "script.pt"
    )
    return str(tmp_path / "bytes.pt"), str(tmp_path / "script.pt")


def embedding(tmp_path, tokens, units, window, first_k=False):
    byte_table, script_table = tables(tmp_path, tokens)
    module = ByteWindowEmbedding(
        dim=window * 2,
        window=window,
        table_path=byte_table,
        units=units,
        first_k=first_k,
        script_table_path=script_table,
    )
    module.reset_buffers()
    return module


def vocabulary(text: str, seed: int) -> list[bytes]:
    """Random byte chunks of the text (splitting characters), plus a special token (id 0)."""
    data, rng, pieces, i = text.encode(), random.Random(seed), [], 0
    while i < len(data):
        n = rng.randint(1, 5)
        pieces.append(data[i : i + n])
        i += n
    return [SPECIAL] + pieces


def script_units(module, text_bytes: bytes) -> list[int]:
    """Reference: SCRIPT units of the last characters of a byte string (a trailing partial character as one element)."""
    elements, i = [], 0
    while i < len(text_bytes):
        lead = text_bytes[i]
        length = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
        if i + length > len(text_bytes):
            elements.append((module.partial_unit, len(text_bytes) - i))
            break
        code_point = ord(text_bytes[i : i + length].decode())
        elements.append((code_point // 64, code_point % 64) if code_point < 0x10000 else (module.unknown_unit, 0))
        i += length
    elements = [(module.pad_script_unit, module.pad_script_unit)] * module.chars + elements
    return [unit for element in elements[-module.chars :] for unit in element]


def test_script_units_match_the_reference_encoding(tmp_path):
    tokens = vocabulary(TEXT, seed=0)
    module = embedding(tmp_path, tokens, "script", window=16)
    ids = torch.arange(1, len(tokens))
    units = module.unit_ids(ids, torch.tensor([0, len(ids)]))
    for t in range(len(ids)):
        assert units[t].tolist() == script_units(module, b"".join(tokens[1 : t + 2])), t


def test_split_character_completes_at_the_later_token(tmp_path):
    han = "中".encode()
    tokens = [SPECIAL, b"a", han[:2], han[2:] + b"b"]
    module = embedding(tmp_path, tokens, "script", window=6)
    units = module.unit_ids(torch.tensor([1, 2, 3]), torch.tensor([0, 3])).tolist()
    pad = module.pad_script_unit
    assert units[1] == [pad, pad, ord("a") // 64, ord("a") % 64, module.partial_unit, 2]
    assert units[2] == [ord("a") // 64, ord("a") % 64, ord("中") // 64, ord("中") % 64, ord("b") // 64, ord("b") % 64]


@pytest.mark.parametrize("units,window", [("bytes", 8), ("chars", 16), ("script", 8)])
@pytest.mark.parametrize("first_k", [False, True])
def test_no_unit_depends_on_later_tokens(tmp_path, units, window, first_k):
    tokens = vocabulary(TEXT * 2, seed=1) + [TEXT.encode()[:40]]
    module = embedding(tmp_path, tokens, units, window, first_k)
    rng = torch.Generator().manual_seed(0)
    ids = torch.randint(0, len(tokens), (64,), generator=rng)
    for t in (0, 10, 31, 62):
        changed = ids.clone()
        changed[t + 1 :] = torch.randint(0, len(tokens), (63 - t,), generator=rng)
        cu_seqlens = torch.tensor([0, 20, 64])
        before = module.unit_ids(ids, cu_seqlens)[: t + 1]
        assert torch.equal(before, module.unit_ids(changed, cu_seqlens)[: t + 1])


def test_documents_do_not_see_each_other(tmp_path):
    tokens = [SPECIAL, b"abc", b"def"]
    module = embedding(tmp_path, tokens, "bytes", window=4)
    units = module.unit_ids(torch.tensor([1, 2, 2]), torch.tensor([0, 2, 3])).tolist()
    assert units[2] == [PAD_UNIT] + list(b"def")
    assert module.unit_ids(torch.tensor([0, 1]), torch.tensor([0, 2])).tolist()[1] == [SPECIAL_UNIT] + list(b"abc")


def test_chars_pad_every_character_to_four_byte_slots(tmp_path):
    tokens = [SPECIAL, "aé中".encode()]
    module = embedding(tmp_path, tokens, "chars", window=12)
    pad = module.char_pad_unit
    assert module.unit_ids(torch.tensor([1]), torch.tensor([0, 1])).tolist()[0] == (
        [ord("a"), pad, pad, pad] + list("é".encode()) + [pad, pad] + list("中".encode()) + [pad]
    )


def test_first_k_takes_the_start_of_long_tokens_only(tmp_path):
    tokens = [SPECIAL, b"xy", b"abcdefghij", "αβγδεζ".encode()]
    first = embedding(tmp_path, tokens, "bytes", window=4, first_k=True)
    last = embedding(tmp_path, tokens, "bytes", window=4)
    ids, cu_seqlens = torch.tensor([1, 2, 1]), torch.tensor([0, 3])
    assert first.unit_ids(ids, cu_seqlens)[1].tolist() == list(b"abcd")
    assert last.unit_ids(ids, cu_seqlens)[1].tolist() == list(b"ghij")
    assert torch.equal(first.unit_ids(ids, cu_seqlens)[[0, 2]], last.unit_ids(ids, cu_seqlens)[[0, 2]])
    script = embedding(tmp_path, tokens, "script", window=8, first_k=True)
    units = script.unit_ids(torch.tensor([1, 3]), torch.tensor([0, 2])).tolist()[1]
    assert units == [unit for c in "αβγδ" for unit in (ord(c) // 64, ord(c) % 64)]
