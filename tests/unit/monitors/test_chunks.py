import pytest

from prime_rl.eval.resume import read_records
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl, chunk_path, open_chunk


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize(
    "prefix,tail,expected",
    [
        (b"", b"", []),
        (b'{"id":1}\n', b"", [{"id": 1}]),
        (b'{"id":1}\n', b'{"id":2}', [{"id": 1}, {"id": 2}]),
        (b'{"id":1}\n', b' { "id" : 2 } ', [{"id": 1}, {"id": 2}]),
        (b'{"id":1}\n', b'{"id":"\xe2\x82\xac"}', [{"id": 1}, {"id": "\u20ac"}]),
        (b'{"id":1}\n', b'{"id":2', [{"id": 1}]),
        (b'{"id":1}\n', b'{"id":"\xe2\x82', [{"id": 1}]),
    ],
)
def test_reopened_chunk_preserves_complete_records_and_recovers_torn_tail(tmp_path, compress, prefix, tail, expected):
    directory = tmp_path / "stream"
    directory.mkdir()
    chunk_path(directory, 0).write_bytes(prefix + tail)

    stream = ChunkedJsonl(directory, max_bytes=1 << 20, compress=compress)
    chunk, offset = stream.append(b'{"id":3}\n')
    stream.close()

    assert list(read_records(directory)) == [*expected, {"id": 3}]
    assert chunk == 0
    assert offset == len(prefix) + (len(tail) + 1 if len(expected) > 1 else 0)
    with open_chunk(directory, 0) as file:
        assert file.read() == prefix + (tail + b"\n" if len(expected) > 1 else b"") + b'{"id":3}\n'


@pytest.mark.parametrize("complete", [False, True])
def test_reopened_chunk_recovers_a_tail_larger_than_one_read_block(tmp_path, complete):
    directory = tmp_path / "stream"
    directory.mkdir()
    prefix = b'{"id":1}\n'
    text = "x" * ((1 << 20) + 17)
    tail = b'{"text":"' + text.encode() + (b'"}' if complete else b"")
    chunk_path(directory, 0).write_bytes(prefix + tail)

    stream = ChunkedJsonl(directory, max_bytes=1 << 22, compress=False)
    _, offset = stream.append(b'{"id":3}\n')
    stream.close()

    expected = [{"id": 1}, {"text": text}] if complete else [{"id": 1}]
    assert list(read_records(directory)) == [*expected, {"id": 3}]
    assert offset == len(prefix) + (len(tail) + 1 if complete else 0)


def test_reopened_stream_does_not_rewrite_sealed_chunks_or_unrelated_files(tmp_path):
    directory = tmp_path / "stream"
    stream = ChunkedJsonl(directory, max_bytes=1 << 20, compress=True)
    stream.append(b'{"id":1}\n')
    stream.close()
    sealed = directory / "00000.jsonl.zst"
    sealed_bytes = sealed.read_bytes()
    unrelated = directory / "notes.txt"
    unrelated.write_bytes(b"keep me")

    stream = ChunkedJsonl(directory, max_bytes=1 << 20, compress=True)
    assert stream.append(b'{"id":2}\n') == (1, 0)
    stream.close()

    assert sealed.read_bytes() == sealed_bytes
    assert unrelated.read_bytes() == b"keep me"
    assert list(read_records(directory)) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("compress", [False, True])
def test_valid_unterminated_record_is_unchanged_until_append(tmp_path, compress):
    directory = tmp_path / "stream"
    directory.mkdir()
    record = b'{"id":1}'
    chunk_path(directory, 0).write_bytes(record)

    stream = ChunkedJsonl(directory, max_bytes=1 << 20, compress=compress)
    stream.close()

    with open_chunk(directory, 0) as file:
        assert file.read() == record


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("fits", [False, True])
def test_recovered_separator_counts_toward_chunk_limit(tmp_path, compress, fits):
    directory = tmp_path / "stream"
    directory.mkdir()
    record = b'{"id":1}'
    line = b'{"id":2}\n'
    chunk_path(directory, 0).write_bytes(record)

    max_bytes = len(record) + 1 + len(line)
    if not fits:
        max_bytes -= 1
    stream = ChunkedJsonl(directory, max_bytes=max_bytes, compress=compress)
    expected_chunk = 0 if fits else 1
    expected_offset = len(record) + 1 if fits else 0
    assert stream.append(line) == (expected_chunk, expected_offset)
    stream.close()

    with open_chunk(directory, 0) as file:
        assert file.read() == record + b"\n" + (line if fits else b"")
    assert list(read_records(directory)) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("compress", [False, True])
def test_roll_after_empty_append_does_not_add_a_separator_to_new_chunk(tmp_path, compress):
    directory = tmp_path / "stream"
    directory.mkdir()
    record = b'{"id":1}'
    chunk_path(directory, 0).write_bytes(record)

    stream = ChunkedJsonl(directory, max_bytes=len(record) - 1, compress=compress)
    assert stream.append(b"") == (1, 0)
    assert stream.append(b'{"id":2}\n') == (1, 0)
    stream.close()

    with open_chunk(directory, 0) as file:
        assert file.read() == record
    assert list(read_records(directory)) == [{"id": 1}, {"id": 2}]
