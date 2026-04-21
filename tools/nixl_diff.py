"""Byte-level trainer vs inference NIXL transport diff.

Run AFTER a job with ``NIXL_DUMP_DIR`` + ``NIXL_DUMP_PUSH`` set, which
produces ``trainer_r{RRR}.pt`` and ``inference_r{RRR}.pt`` files in the
dump directory. This tool reconstructs each inference-side tensor's
expected content from the trainer dumps and reports any byte-level
mismatch against the actual inference-side dump.

Example:
    NIXL_DUMP_DIR=/beegfs/outputs/nixl-broadcast/dump \\
    NIXL_DUMP_PUSH=1 sbatch …
    # After job:
    uv run python tools/nixl_diff.py /beegfs/outputs/nixl-broadcast/dump
"""

from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path

import torch


def load_dumps(dump_dir: Path):
    trainer_files = sorted(glob.glob(str(dump_dir / "trainer_r*.pt")))
    inference_files = sorted(glob.glob(str(dump_dir / "inference_r*.pt")))
    print(f"found {len(trainer_files)} trainer dumps, {len(inference_files)} inference dumps")

    trainer = {}  # {rank: {slot_key: entry}}
    for f in trainer_files:
        rank = int(Path(f).stem.split("_r")[-1])
        trainer[rank] = torch.load(f, map_location="cpu", weights_only=False)

    inference = {}  # {global_rank: {"params": {..}, "buffers": {..}, "expert_map": {..}}}
    for f in inference_files:
        gr = int(Path(f).stem.split("_r")[-1])
        inference[gr] = torch.load(f, map_location="cpu", weights_only=False)

    return trainer, inference


def _fetch(worker: dict, name: str):
    if name in worker["params"]:
        return worker["params"][name]
    if name in worker["buffers"]:
        return worker["buffers"][name]
    return None


def compare_tensors(expected: torch.Tensor, actual: torch.Tensor, tag: str) -> tuple[bool, str]:
    if expected.shape != actual.shape:
        return False, f"shape {tuple(expected.shape)} vs {tuple(actual.shape)}"
    if expected.dtype != actual.dtype:
        return False, f"dtype {expected.dtype} vs {actual.dtype}"
    eq = torch.equal(expected, actual) if expected.dtype != torch.float8_e4m3fn else torch.equal(
        expected.view(torch.uint8), actual.view(torch.uint8)
    )
    if eq:
        return True, "OK"
    # Summarize diff
    if expected.dtype == torch.float8_e4m3fn:
        a = expected.view(torch.uint8).to(torch.int32)
        b = actual.view(torch.uint8).to(torch.int32)
    else:
        a = expected.to(torch.float64)
        b = actual.to(torch.float64)
    diff = (a - b).abs()
    n_diff = (diff > 0).sum().item()
    max_diff = diff.max().item()
    # Locate first differing position
    flat_diff = diff.flatten()
    first = int(torch.where(flat_diff > 0)[0][0].item()) if n_diff > 0 else -1
    return False, f"differs at {n_diff}/{expected.numel()} positions, max|diff|={max_diff}, first_pos={first}"


def report_mismatches(trainer_dumps: dict, inference_dumps: dict):
    if not trainer_dumps or not inference_dumps:
        print("no dumps to compare")
        return

    # Group trainer slots by inference target name, so we can assemble
    # expected values.
    slots_by_inf_name: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    scales_by_inf_name: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    expert_slots: list[tuple[int, dict]] = []
    for rank, slots in trainer_dumps.items():
        for key, entry in slots.items():
            if entry["type"] == "ExpertSlot":
                expert_slots.append((rank, entry))
                continue
            inf = entry.get("inference_name")
            if inf:
                slots_by_inf_name[inf].append((rank, entry))
            inf_scale = entry.get("inference_scale_name")
            if inf_scale:
                scales_by_inf_name[inf_scale].append((rank, entry))

    total = 0
    mismatches = 0

    # --- Non-expert slots: assemble expected from trainer shards ---------- #
    for inf_name, entries in sorted(slots_by_inf_name.items()):
        # Group entries by slot type.
        types = {e["type"] for _, e in entries}
        assert len(types) == 1, f"{inf_name} has mixed slot types: {types}"
        slot_type = entries[0][0]  # dummy
        slot_type = entries[0][1]["type"]

        # Build expected full tensor
        # For each entry (rank, slot), it covers some region of the fused
        # inference tensor. Compute that region; place slot.weight there.
        any_entry = entries[0][1]
        # Pick any inference worker that has this tensor to get expected shape/dtype.
        ref_worker = None
        for gr, w in inference_dumps.items():
            t = _fetch(w, inf_name)
            if t is not None:
                ref_worker = w
                ref_tensor = t
                break
        if ref_worker is None:
            print(f"  SKIP {inf_name}: no inference worker has this tensor")
            continue
        expected = torch.empty_like(ref_tensor)
        expected.fill_(0)  # Will be overwritten by slots; any un-touched bytes are a red flag.
        filled = torch.zeros(expected.shape[0], dtype=torch.bool)

        for rank, entry in entries:
            w = entry["weight"]
            if slot_type == "GatheredSlot":
                # Full tensor at offset_rows..offset_rows+rows. Only rank 0's
                # copy is used (all ranks should have identical content; they
                # round-robin on the wire).
                if rank != 0:
                    continue
                ofs = entry["offset_rows"]
                rows = w.shape[0]
                expected[ofs:ofs + rows] = w
                filled[ofs:ofs + rows] = True
            elif slot_type == "ShardedSlot":
                # rank r writes its shard at rows [ofs + r*S, ofs + (r+1)*S)
                # where S is the per-shard size from entry.
                S = w.shape[0]
                my_rank = entry["my_rank"]
                ofs = entry["offset_rows"]
                start = ofs + my_rank * S
                expected[start:start + S] = w
                filled[start:start + S] = True
            else:
                print(f"  SKIP {inf_name}: unexpected type {slot_type}")
                break

        if not filled.all():
            holes = (~filled).sum().item()
            print(f"  HOLE {inf_name}: {holes} rows not written by any trainer slot")
            mismatches += 1
            continue

        # Now compare against every inference worker.
        mismatch_in_this = False
        for gr, worker in inference_dumps.items():
            actual = _fetch(worker, inf_name)
            if actual is None:
                continue
            total += 1
            ok, msg = compare_tensors(expected, actual, inf_name)
            if not ok:
                mismatches += 1
                mismatch_in_this = True
                print(f"  DIFF {inf_name} (inf rank {gr}): {msg}")

        if not mismatch_in_this:
            print(f"  OK   {inf_name} ({len(entries)} contributing trainer slots, {len(inference_dumps)} inf workers)")

    # --- Scale tensors for FP8 fused specs (if present) ------------------- #
    for inf_scale, entries in sorted(scales_by_inf_name.items()):
        # Same logic as above but for scale tensors.
        ref_tensor = None
        for gr, w in inference_dumps.items():
            t = _fetch(w, inf_scale)
            if t is not None:
                ref_tensor = t
                break
        if ref_tensor is None:
            continue
        expected = torch.empty_like(ref_tensor)
        expected.fill_(-999.0)
        filled = torch.zeros(expected.shape[0], dtype=torch.bool)
        for rank, entry in entries:
            sc = entry.get("scale")
            if sc is None:
                continue
            slot_type = entry["type"]
            if slot_type == "GatheredSlot":
                if rank != 0:
                    continue
                ofs = entry["scale_offset_rows"]
                rows = sc.shape[0]
                expected[ofs:ofs + rows] = sc
                filled[ofs:ofs + rows] = True
            elif slot_type == "ShardedSlot":
                S = sc.shape[0]
                my_rank = entry["my_rank"]
                ofs = entry["scale_offset_rows"]
                start = ofs + my_rank * S
                expected[start:start + S] = sc
                filled[start:start + S] = True
        if not filled.all():
            holes = (~filled).sum().item()
            print(f"  SCALE HOLE {inf_scale}: {holes} rows not written")
            mismatches += 1
            continue
        for gr, worker in inference_dumps.items():
            actual = _fetch(worker, inf_scale)
            if actual is None:
                continue
            total += 1
            ok, msg = compare_tensors(expected, actual, inf_scale)
            if not ok:
                mismatches += 1
                print(f"  DIFF SCALE {inf_scale} (inf rank {gr}): {msg}")
            else:
                pass  # silent OK

    # --- ExpertSlot: per-global-expert comparison ------------------------- #
    # Build per-expert trainer content keyed by (moe_prefix, global_expert).
    expert_by_global: dict[tuple[str, int], tuple[int, int, dict]] = {}
    for rank, entry in expert_slots:
        moe = entry["moe_prefix"]
        owned = entry.get("owned_global_experts", [])
        for local_idx, global_id in enumerate(owned):
            expert_by_global[(moe, global_id)] = (rank, local_idx, entry)

    # For each inference worker, look up its expert_map, and compare each
    # local expert against the matching trainer-side one.
    for gr, worker in inference_dumps.items():
        for moe, inf_owned in worker["expert_map"].items():
            # Find the weight tensor names
            w13 = f"{moe}.w13_weight"
            w2 = f"{moe}.w2_weight"
            for w_name in (w13, w2):
                t = _fetch(worker, w_name)
                if t is None:
                    continue
                for local_idx, global_id in enumerate(inf_owned):
                    key = (moe, global_id)
                    if key not in expert_by_global:
                        print(f"  ORPHAN inf expert {w_name}[local={local_idx}] global={global_id}: no trainer has it")
                        mismatches += 1
                        continue
                    t_rank, t_local, t_entry = expert_by_global[key]
                    # match slot_key's dst (w13_weight vs w2_weight)
                    if "w13" in w_name and "w13" not in t_entry["slot_key"]:
                        continue
                    if "w2" in w_name and "w2" not in t_entry["slot_key"]:
                        continue
                    expected = t_entry["weight"][t_local]
                    actual = t[local_idx]
                    total += 1
                    ok, msg = compare_tensors(expected, actual, f"{w_name}[E{global_id}]")
                    if not ok:
                        mismatches += 1
                        print(f"  DIFF EXPERT {w_name}[E{global_id}] inf_rank={gr} trainer_rank={t_rank}: {msg}")

                    # Scale for FP8 experts
                    sc_name = w_name.replace("_weight", "_weight_scale_inv")
                    sc_t = _fetch(worker, sc_name)
                    if sc_t is not None and t_entry.get("scale") is not None:
                        expected_sc = t_entry["scale"][t_local]
                        actual_sc = sc_t[local_idx]
                        total += 1
                        ok, msg = compare_tensors(expected_sc, actual_sc, f"{sc_name}[E{global_id}]")
                        if not ok:
                            mismatches += 1
                            print(f"  DIFF SCALE EXPERT {sc_name}[E{global_id}] inf_rank={gr} trainer_rank={t_rank}: {msg}")

    print()
    print(f"=== {total} comparisons, {mismatches} mismatches ===")


def main():
    if len(sys.argv) != 2:
        print("usage: nixl_diff.py <dump_dir>")
        sys.exit(1)
    dump_dir = Path(sys.argv[1])
    trainer, inference = load_dumps(dump_dir)
    if not trainer or not inference:
        print("missing dumps")
        sys.exit(1)
    report_mismatches(trainer, inference)


if __name__ == "__main__":
    main()
