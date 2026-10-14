"""Per-family weight-shift statistics of trained fp32 masters against their bf16 start.

For each serving mode (UE8M0 or amax/448 block scales, round-to-nearest or stochastic rounding over seeds):
P(fp8 Δ), survival P(fp8 Δ | bf16 Δ), spurious P(fp8 Δ | no bf16 Δ) and served-vs-served retention
``sum((served_final - served_start) * delta) / sum(delta^2)`` with ``delta = master - start``. Also P(fp32 Δ),
P(bf16 Δ) and a histogram of |delta|/bin on the start's grid over elements with bf16 Δ.
"""

import re
from collections import defaultdict

import torch
from bin_stats import block_scales, e4m3_bin

FAMILY = re.compile(r"model\.layers\.\d+\.(?:self_attn\.([qkvo])_proj|mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight$")
GRIDS = (("ue8m0", True), ("amax", False))
SR_SEEDS = (0, 1, 2)
HISTOGRAM_EDGES = (0.1, 0.25, 0.5, 1.0, 2.0)


def family_of(name: str) -> str | None:
    match = FAMILY.match(name)
    if match is None:
        return None
    return f"attn.{match.group(1)}" if match.group(1) else f"expert.{match.group(2)}"


def serve_rtn(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (w / scale).to(torch.float8_e4m3fn).float() * scale


def serve_sr(w: torch.Tensor, scale: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    spacing = e4m3_bin(w, scale)
    lower = torch.floor(w / spacing) * spacing
    round_up = torch.rand(w.shape, device=w.device, generator=generator) < (w - lower) / spacing
    return serve_rtn(lower + round_up * spacing, scale)


class WeightShiftStats:
    def __init__(self):
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, family: str, start: torch.Tensor, master: torch.Tensor, seed_offset: int) -> None:
        start, master = start.float(), master.float()
        delta = master - start
        final = master.to(torch.bfloat16).float()
        changed = final != start
        row = self.sums[family]
        row["elements"] += start.numel()
        row["fp32_changed"] += int((delta != 0).sum())
        row["changed"] += int(changed.sum())
        row["sum_sq_delta"] += float(delta.square().sum())
        for grid, ue8m0 in GRIDS:
            start_scale, final_scale = block_scales(start, ue8m0), block_scales(final, ue8m0)
            ratio = (delta.abs() / e4m3_bin(start, start_scale))[changed]
            edges = torch.tensor((0.0, *HISTOGRAM_EDGES, float("inf")), device=ratio.device)
            for i in range(len(edges) - 1):
                row[f"hist_{grid}_{i}"] += int(((ratio >= edges[i]) & (ratio < edges[i + 1])).sum())
            row[f"scale_changed_{grid}"] += int((start_scale != final_scale).sum()) / (128 * 128)
            modes = [("rtn", serve_rtn(final, final_scale), serve_rtn(start, start_scale))]
            for seed in SR_SEEDS:
                generator = torch.Generator(device=start.device).manual_seed(1000 * seed + seed_offset)
                served_final = serve_sr(final, final_scale, generator)
                modes.append((f"sr{seed}", served_final, serve_sr(start, start_scale, generator)))
            for mode, served_final, served_start in modes:
                key = f"{grid}_{mode}"
                fp8_changed = served_final != served_start
                row[f"fp8_changed_{key}"] += int(fp8_changed.sum())
                row[f"survived_{key}"] += int((fp8_changed & changed).sum())
                row[f"ret_num_{key}"] += float(((served_final - served_start) * delta).sum())

    def merge(self, other: "WeightShiftStats") -> None:
        for family, row in other.sums.items():
            for key, value in row.items():
                self.sums[family][key] += value


def summarize(row: dict[str, float]) -> dict[str, float]:
    n, changed = row["elements"], row["changed"]
    out = {
        "elements": int(n),
        "P(fp32 Δ)": row["fp32_changed"] / n,
        "P(bf16 Δ)": changed / n,
        "rms_delta": (row["sum_sq_delta"] / n) ** 0.5,
    }
    for grid, _ in GRIDS:
        out[f"P(block scale Δ) {grid}"] = row[f"scale_changed_{grid}"] / (n / (128 * 128))
        for mode in ["rtn", *(f"sr{seed}" for seed in SR_SEEDS)]:
            key = f"{grid}_{mode}"
            survived = row[f"survived_{key}"]
            out[f"P(fp8 Δ) {key}"] = row[f"fp8_changed_{key}"] / n
            out[f"P(fp8 Δ|bf16 Δ) {key}"] = survived / max(changed, 1)
            out[f"P(fp8 Δ|no bf16 Δ) {key}"] = (row[f"fp8_changed_{key}"] - survived) / max(n - changed, 1)
            out[f"ret {key}"] = row[f"ret_num_{key}"] / row["sum_sq_delta"]
        bins = [row[f"hist_{grid}_{i}"] for i in range(len(HISTOGRAM_EDGES) + 1)]
        labels = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", ">2"]
        out.update({f"hist {grid} {label}": count / max(changed, 1) for label, count in zip(labels, bins)})
    return out


def report(stats: WeightShiftStats) -> dict[str, dict[str, float]]:
    total: dict[str, float] = defaultdict(float)
    for row in stats.sums.values():
        for key, value in row.items():
            total[key] += value
    families = {family: summarize(row) for family, row in sorted(stats.sums.items())}
    return {**families, "ALL": summarize(total)}
