import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    MANIFEST,
    AnchorWriter,
    anchor_complete,
    as_bits,
    bits_checksum,
    decode_gaps,
    encode_gaps,
    rank_failed_marker,
    rank_part,
    rank_parts_marker,
    read_anchor_checksums,
    read_delta,
    rows_checksum,
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
        (delta_dir / rank_parts_marker(rank)).write_text(json.dumps({"parts": 1}))
    (delta_dir / MANIFEST).write_text(json.dumps({"ranks": 2, "tensors": tensors}))

    def apply(names):
        patched = {name: as_bits(tensor.clone()).view(-1) for name, tensor in previous.items()}
        with ThreadPoolExecutor(4) as pool:
            delta, checksums = read_delta(tmp_path, pool, names)
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


def test_anchor_checksums_and_unwritten_slices(tmp_path):
    values = torch.randn(12, dtype=torch.bfloat16)
    slices = [("a", [12], 0, 6, torch.bfloat16), ("a", [12], 6, 6, torch.bfloat16)]
    for step, written in (("step_5", slices), ("step_6", slices[:1])):
        step_dir = tmp_path / step
        step_dir.mkdir()
        writer = AnchorWriter(slices, step_dir, 0)
        for name, _, start, numel, _ in written:
            part = values[start : start + numel]
            writer.write_parts([(name, [12], start, part)], lambda tensor: [tensor.view(torch.uint8).numpy()])
            writer.add_sums([name], rows_checksum(as_bits(part), start).view(1))
        if written == slices:
            writer.close(1)
            assert anchor_complete(step_dir)
            assert read_anchor_checksums(step_dir) == {"a": bits_checksum(as_bits(values))}
        else:
            with pytest.raises(RuntimeError, match="a@6"):
                writer.close(1)
            assert not anchor_complete(step_dir)


def test_read_delta_fails_on_a_failed_trainer_rank(tmp_path):
    delta_dir = tmp_path / DELTA_DIR
    delta_dir.mkdir()
    (delta_dir / MANIFEST).write_text(json.dumps({"ranks": 2, "tensors": {}}))
    (delta_dir / rank_failed_marker(1)).write_text(json.dumps({"error": "RuntimeError('boom')"}))
    with ThreadPoolExecutor(1) as pool, pytest.raises(RuntimeError, match="rank 1 failed.*boom"):
        read_delta(tmp_path, pool)
