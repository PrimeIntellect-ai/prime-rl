"""Plot loss curves and equivalence diagnostics from scripts/tree_ablation.py output."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    data = json.loads(Path("outputs/tree_ablation.json").read_text())
    losses_tree = data["losses_tree"]
    losses_branch = data["losses_branch"]
    rel_diffs = data["rel_diffs_loss"]
    drifts = data["param_drifts"]
    steps = list(range(1, len(losses_tree) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].plot(steps, losses_tree, label="tree_nll_loss", linewidth=2)
    axes[0].plot(steps, losses_branch, label="per-branch baseline", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (sum)")
    axes[0].set_title("Loss curves: tree vs per-branch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(steps, [max(r, 1e-12) for r in rel_diffs], color="tab:red")
    axes[1].axhline(1e-5, color="black", linestyle=":", alpha=0.4, label="1e-5 (FP32 target)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("|tree − branch| / |branch|")
    axes[1].set_title("Per-step loss relative diff (log)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    drift_steps = [d[0] for d in drifts]
    drift_vals = [d[1] for d in drifts]
    axes[2].semilogy(drift_steps, drift_vals, marker="o")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("max param rel diff")
    axes[2].set_title("Parameter drift between model copies")
    axes[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = Path("outputs/tree_ablation.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
