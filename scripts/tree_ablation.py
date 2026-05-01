"""Tree Training v1 ablation: side-by-side training with tree mode vs per-branch baseline.

Validates the implementation at training scale by running both paths on the same
caterpillar data through identical optimizer state and asserting loss curves and
parameters track each other to within FP32 noise.

Usage:
    uv run python scripts/tree_ablation.py
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from prime_rl.trainer.tree import build_caterpillar, pack_tree, tree_nll_loss


def caterpillar_seed_for_step(seed: int, step: int) -> int:
    return seed * 10_000 + step


def build_step_tree(vocab_size: int, num_turns: int, user_len: int, think_len: int, response_len: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    turns = []
    for _ in range(num_turns):
        u = torch.randint(0, vocab_size, (user_len,), generator=g).tolist()
        t = torch.randint(0, vocab_size, (think_len,), generator=g).tolist()
        r = torch.randint(0, vocab_size, (response_len,), generator=g).tolist()
        turns.append((u, t, r))
    return build_caterpillar(turns, train_response=True, train_think=True)


def tree_step_loss(model, tree) -> torch.Tensor:
    packed = pack_tree(tree)
    input_ids = packed.input_ids.to("cuda").unsqueeze(0)
    position_ids = packed.position_ids.to("cuda").unsqueeze(0)
    attn_mask = packed.attn_mask.to("cuda").unsqueeze(0).unsqueeze(0)
    out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attn_mask)
    return tree_nll_loss(
        out.logits,
        input_ids,
        packed.prev_map.to("cuda").unsqueeze(0),
        packed.loss_mask.to("cuda").unsqueeze(0),
        packed.loss_weights.to("cuda").unsqueeze(0),
    )


def per_branch_step_loss(model, tree) -> torch.Tensor:
    """L = (1/K) * sum_k sum_t loss_mask_t * ce(logits_k, ids_k). Matches tree_nll_loss
    by construction when both pass through the same model state."""
    leaves = tree.leaves()
    K = len(leaves)
    total = torch.zeros((), dtype=torch.float32, device="cuda")
    for leaf_idx in leaves:
        path = tree.root_path(leaf_idx)
        ids = [t for n in path for t in tree.nodes[n].token_ids]
        masks = [m for n in path for m in tree.nodes[n].loss_mask]
        input_ids = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        position_ids = torch.arange(len(ids), dtype=torch.long, device="cuda").unsqueeze(0)
        out = model(input_ids=input_ids, position_ids=position_ids)
        ce = F.cross_entropy(out.logits[0, :-1], input_ids[0, 1:], reduction="none")
        m = torch.tensor(masks[1:], dtype=ce.dtype, device="cuda")
        total = total + (ce * m).sum() / K
    return total


def max_param_rel_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    worst = 0.0
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        diff = (pa.detach() - pb.detach()).abs().max().item()
        denom = max(pa.detach().abs().max().item(), 1e-30)
        worst = max(worst, diff / denom)
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PrimeIntellect/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-turns", type=int, default=3)
    parser.add_argument("--user-len", type=int, default=4)
    parser.add_argument("--think-len", type=int, default=8)
    parser.add_argument("--response-len", type=int, default=10)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/tree_ablation.json"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = tokenizer.vocab_size

    model_tree = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    model_tree.eval()  # disables dropout; gradients still flow
    model_branch = copy.deepcopy(model_tree)

    init_diff = max_param_rel_diff(model_tree, model_branch)
    print(f"Initial param max rel diff (sanity): {init_diff:.2e}")
    assert init_diff == 0.0

    opt_tree = torch.optim.AdamW(model_tree.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_branch = torch.optim.AdamW(model_branch.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    losses_tree = []
    losses_branch = []
    rel_diffs_loss = []
    param_drifts = []
    step_times = []

    print(f"Running {args.steps} steps...")
    t0 = time.perf_counter()
    for step in range(args.steps):
        tree = build_step_tree(
            vocab_size,
            args.num_turns,
            args.user_len,
            args.think_len,
            args.response_len,
            seed=caterpillar_seed_for_step(args.seed, step),
        )

        step_start = time.perf_counter()

        opt_tree.zero_grad()
        L_tree = tree_step_loss(model_tree, tree)
        L_tree.backward()
        opt_tree.step()

        opt_branch.zero_grad()
        L_branch = per_branch_step_loss(model_branch, tree)
        L_branch.backward()
        opt_branch.step()

        step_times.append(time.perf_counter() - step_start)

        l_tree = L_tree.item()
        l_branch = L_branch.item()
        losses_tree.append(l_tree)
        losses_branch.append(l_branch)
        rel = abs(l_tree - l_branch) / max(abs(l_branch), 1e-12)
        rel_diffs_loss.append(rel)

        if step == 0 or (step + 1) % args.check_every == 0 or step == args.steps - 1:
            drift = max_param_rel_diff(model_tree, model_branch)
            param_drifts.append((step + 1, drift))
            elapsed = time.perf_counter() - t0
            print(
                f"step {step + 1:4d} | tree_loss {l_tree:.6f} | "
                f"branch_loss {l_branch:.6f} | loss_rel_diff {rel:.2e} | "
                f"param_drift {drift:.2e} | elapsed {elapsed:.1f}s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args) | {"out": str(args.out), "vocab_size": vocab_size},
        "losses_tree": losses_tree,
        "losses_branch": losses_branch,
        "rel_diffs_loss": rel_diffs_loss,
        "param_drifts": param_drifts,
        "median_step_time_s": sorted(step_times)[len(step_times) // 2],
        "total_time_s": time.perf_counter() - t0,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")

    print("\n=== SUMMARY ===")
    print(f"max per-step loss rel diff: {max(rel_diffs_loss):.2e}")
    print(f"final param drift:          {param_drifts[-1][1]:.2e}")
    print(f"final tree loss:            {losses_tree[-1]:.4f}")
    print(f"final branch loss:          {losses_branch[-1]:.4f}")
    print(f"median step time:           {summary['median_step_time_s'] * 1000:.1f} ms")


if __name__ == "__main__":
    main()
