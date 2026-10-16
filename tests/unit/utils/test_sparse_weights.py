import json

import torch

from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    MANIFEST,
    as_bits,
    bits_checksum,
    decode_gaps,
    encode_gaps,
    rank_part,
    rank_parts_marker,
    read_delta,
    write_rank_file,
    zigzag,
)


def test_sparse_delta_roundtrip_is_exact(tmp_path):
    torch.manual_seed(0)
    previous = {"a": torch.randn(64, 32, dtype=torch.bfloat16), "b": torch.randn(1000)}
    current = {name: tensor.clone() for name, tensor in previous.items()}
    current["a"].view(-1)[torch.randperm(2048)[:100]] += 1
    current["b"].view(-1)[torch.randperm(1000)[:10]] *= 2

    tensors = {name: [list(tensor.shape), str(tensor.dtype).removeprefix("torch.")] for name, tensor in current.items()}
    # Two ranks, each holding a row slice of "a"; rank 1 also holds "b".
    delta_dir = tmp_path / DELTA_DIR
    delta_dir.mkdir()
    for rank, names in enumerate([[("a", slice(0, 32))], [("a", slice(32, 64)), ("b", slice(None))]]):
        entries, gaps, values, checksums = [], [], [], {}
        for name, rows in names:
            offset = 0 if rows.start is None else rows.start * 32
            checksums[name] = bits_checksum(as_bits(current[name][rows]), offset)
            delta = as_bits(current[name][rows]).reshape(-1) - as_bits(previous[name][rows]).reshape(-1)
            changed = delta.nonzero().squeeze(1)
            starts = torch.zeros_like(changed, dtype=torch.bool)
            starts[:1] = True
            entries.append((name, len(changed)))
            gaps.append(encode_gaps(changed + offset, starts))
            values.append(zigzag(delta[changed]).view(torch.uint8))
        gaps, values = torch.cat(gaps).numpy(), torch.cat(values).numpy()
        # Small frames, so a frame boundary falls inside rank 1's entries
        write_rank_file(delta_dir / rank_part(rank, 0), entries, gaps, values, tensors, checksums, frame_size=8)
        (delta_dir / rank_parts_marker(rank)).write_text("1")
    (delta_dir / MANIFEST).write_text(json.dumps({"ranks": 2, "tensors": tensors}))

    def apply(names):
        patched = {name: as_bits(tensor.clone()).view(-1) for name, tensor in previous.items()}
        delta, checksums = read_delta(tmp_path, names)
        assert checksums == {name: bits_checksum(as_bits(tensor)) for name, tensor in current.items()}
        for entries, gaps, values in delta:
            gaps, values = torch.from_numpy(gaps), torch.from_numpy(values)
            indices = decode_gaps(gaps, torch.tensor([count for _, count in entries]))
            i = b = 0
            for name, count in entries:
                itemsize = patched[name].element_size()
                patched[name][indices[i : i + count]] += (
                    values[b : b + count * itemsize].clone().view(patched[name].dtype)
                )
                i += count
                b += count * itemsize
        return patched

    patched = apply(None)
    for name, tensor in current.items():
        assert torch.equal(patched[name], as_bits(tensor).view(-1))

    # A reader that holds only "a" (split over both rank files) skips the frame holding "b".
    patched = apply({"a"})
    assert torch.equal(patched["a"], as_bits(current["a"]).view(-1))
    assert torch.equal(patched["b"], as_bits(previous["b"]).view(-1))
