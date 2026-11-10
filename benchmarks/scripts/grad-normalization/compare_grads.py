"""Compare two post-scale parameter-gradient dumps written by the gn_probes grad dump (GN_GRAD_DUMP_STEP)."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_a", help="grads_step<N>.pt of arm A")
    parser.add_argument("dump_b", help="grads_step<N>.pt of arm B")
    parser.add_argument("--top", type=int, default=10, help="tensors to list, worst cosine first")
    args = parser.parse_args()

    a = torch.load(args.dump_a, weights_only=False)
    b = torch.load(args.dump_b, weights_only=False)
    print(f"A meta: {a['meta']}")
    print(f"B meta: {b['meta']}")
    names_a, names_b = set(a["grads"]), set(b["grads"])
    if names_a != names_b:
        print(f"only in A: {sorted(names_a - names_b)}")
        print(f"only in B: {sorted(names_b - names_a)}")

    rows = []
    total_diff_sq = total_a_sq = 0.0
    for name in sorted(names_a & names_b):
        ga, gb = a["grads"][name].double(), b["grads"][name].double()
        diff = ga - gb
        norm_a, norm_b = ga.norm().item(), gb.norm().item()
        cosine = (ga.flatten() @ gb.flatten()).item() / (norm_a * norm_b) if norm_a and norm_b else float("nan")
        rows.append(
            (
                cosine,
                name,
                diff.abs().max().item(),
                diff.norm().item() / norm_a if norm_a else float("nan"),
                norm_b / norm_a if norm_a else float("nan"),
            )
        )
        total_diff_sq += diff.square().sum().item()
        total_a_sq += ga.square().sum().item()

    cosines = torch.tensor([row[0] for row in rows], dtype=torch.float64)
    print(f"tensors: {len(rows)}")
    print(f"global relative L2 diff ||A-B||/||A||: {(total_diff_sq / total_a_sq) ** 0.5:.3e}")
    print(f"cosine min / median: {cosines.nanquantile(0.0).item():.6f} / {cosines.nanmedian().item():.6f}")
    print(f"max abs diff over all tensors: {max(row[2] for row in rows):.3e}")
    print("\n| tensor | cosine | max abs diff | rel L2 diff | norm B / norm A |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for cosine, name, max_abs, rel, ratio in sorted(rows, key=lambda row: row[0])[: args.top]:
        print(f"| {name} | {cosine:.6f} | {max_abs:.3e} | {rel:.3e} | {ratio:.6f} |")


if __name__ == "__main__":
    main()
