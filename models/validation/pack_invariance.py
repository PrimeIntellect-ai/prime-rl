"""Pack-position invariance gate at 65k (single GPU).

v3.1 trains single-stage at 65k with packed rows; correctness requires that a
document's loss is independent of its pack neighbors: block-diagonal attention
(flash-attn varlen via cu_seqlens) AND Mamba-2 state reset at document
boundaries (seq_idx in mamba_chunk_scan_combined).

Three packs over a tiny random NemotronVL model (real kernels), all padded to
--seq-len, each containing the same probe document D (image + supervised span):

  P1: [F1, D, F2]  — D at offset `off`
  P2: [F3, D, F4]  — D at the SAME offset, different neighbor content.
      Any difference in D's token NLLs is cross-document leakage: with correct
      masking + state resets the computation over D is identical bit-for-bit.
  P3: [D, F5]      — D at offset 0. Kernel tiling differs (mamba chunk
      alignment, attention tile order), so bf16-level noise is expected; large
      differences again mean leakage.

PASS: P1-vs-P2 max|dNLL| == 0 (bitwise) and P1-vs-P3 mean-NLL rel diff < 1e-3.
Run: uv run torchrun --nproc_per_node=1 models/validation/pack_invariance.py --seq-len 65536
"""

import argparse
import sys
from pathlib import Path

import torch

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_vl import NemotronVLForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cp_equivalence_mm import IMG_TOKEN_ID, TILE_H, TILE_W, tiny_config  # noqa: E402

IMG_RUN = 6  # tokens per tile (4x6 patch grid / pixel shuffle)


def make_doc(g: torch.Generator, length: int, with_image: bool, img_at: int = 40) -> dict:
    input_ids = torch.randint(32, 200, (length,), generator=g)
    if with_image:
        input_ids[img_at : img_at + IMG_RUN] = IMG_TOKEN_ID
    target_ids = torch.randint(32, 200, (length,), generator=g)
    loss_mask = torch.zeros(length, dtype=torch.bool)
    loss_mask[length // 2 :] = True
    # Each doc owns its tile pixels so a doc keeps identical image content at any
    # pack position (a shared per-pack generator would tie pixels to tile order).
    tile = torch.randn(1, 3, TILE_H, TILE_W, generator=g) if with_image else None
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "position_ids": torch.arange(length),
        "n_images": 1 if with_image else 0,
        "tile": tile,
    }


def pack(docs: list[dict], seq_len: int, g: torch.Generator) -> dict:
    total = sum(len(d["input_ids"]) for d in docs)
    assert total <= seq_len
    if total < seq_len:
        docs = docs + [make_doc(g, seq_len - total, with_image=False)]
    batch = {
        "input_ids": torch.cat([d["input_ids"] for d in docs]).unsqueeze(0),
        "target_ids": torch.cat([d["target_ids"] for d in docs]).unsqueeze(0),
        "loss_mask": torch.cat([d["loss_mask"] for d in docs]).unsqueeze(0),
        "position_ids": torch.cat([d["position_ids"] for d in docs]).unsqueeze(0),
        "seq_lens": torch.tensor([len(d["input_ids"]) for d in docs]),
    }
    tiles = [d["tile"] for d in docs if d["tile"] is not None]
    batch["pixel_values"] = torch.cat(tiles, dim=0) if tiles else torch.zeros(0, 3, TILE_H, TILE_W)
    return batch


@torch.no_grad()
def probe_nlls(model, batch: dict, probe_slice: slice) -> torch.Tensor:
    batch = {k: v.cuda() for k, v in batch.items()}
    out = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        labels=batch["target_ids"],
        temperature=torch.ones_like(batch["target_ids"], dtype=torch.float32),
        pixel_values=batch["pixel_values"].to(torch.bfloat16),
        seq_lens=batch["seq_lens"],
    )
    return (-out["logprobs"][0, probe_slice]).float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65536)
    args = parser.parse_args()

    torch.manual_seed(0)
    config = tiny_config()
    config.text_config.max_position_embeddings = args.seq_len
    model = NemotronVLForCausalLM(config).to(torch.bfloat16).cuda()
    inject_prime_lm_head(model, chunk_size=1024)

    g = torch.Generator().manual_seed(42)
    doc_len = 2048
    off = 3072 + 128  # not chunk-aligned: crosses mamba chunk + attention tile boundaries
    probe = make_doc(g, doc_len, with_image=True)
    fillers = [
        make_doc(g, ln, with_image=(i % 2 == 0))
        for i, ln in ((0, off), (1, args.seq_len - off - doc_len), (2, off), (3, args.seq_len - off - doc_len))
    ]

    p1 = pack([fillers[0], probe, fillers[1]], args.seq_len, g)
    p2 = pack([fillers[2], probe, fillers[3]], args.seq_len, g)
    p3 = pack([probe], args.seq_len, g)

    # The probe's tiles sit at different offsets of pixel_values across packs; give
    # every pack identical tile pixels (pack() seeds them identically) so the image
    # content of D is the same everywhere.
    s_mid = slice(off, off + doc_len)
    s_front = slice(0, doc_len)

    n1 = probe_nlls(model, p1, s_mid)
    n2 = probe_nlls(model, p2, s_mid)
    n3 = probe_nlls(model, p3, s_front)

    mask = probe["loss_mask"]
    d12 = (n1[mask] - n2[mask]).abs().max().item()
    m1, m3 = n1[mask].mean().item(), n3[mask].mean().item()
    rel13 = abs(m1 - m3) / abs(m1)
    d13 = (n1[mask] - n3[mask]).abs().max().item()

    print(f"P1 vs P2 (same offset, different neighbors): max|dNLL| = {d12:.3e}  (leak detector)")
    print(f"P1 vs P3 (offset {off} vs 0): mean NLL {m1:.6f} vs {m3:.6f}, rel = {rel13:.3e}, max|d| = {d13:.3e}")

    ok = d12 == 0.0 and rel13 < 1e-3
    print(f"{'PASS' if ok else 'FAIL'}: neighbors bitwise-isolated={d12 == 0.0}, position-shift rel={rel13:.1e}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
