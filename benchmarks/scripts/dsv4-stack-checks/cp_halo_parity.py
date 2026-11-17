"""CP parity for DeepSeek V4 attention: the cp halo path against a cp 1 reference, one layer of each type.

torchrun --nproc-per-node <cp> benchmarks/scripts/dsv4-stack-checks/cp_halo_parity.py [--tokens T] [--backends cudnn torch] [--modes eager compile] [--dims small|flash]

Every rank runs the cp 1 reference on the whole row itself, then the cp halo path on its shard, forward and
backward. It compares compressor entries (attention and indexer compressors), indexer and HCA picks, attention
outputs, input gradients and parameter gradients (summed over ranks). Rows cover long rows of documents crossing
the shard edges and short rows: below `cp * halo_rows`, exactly `cp * halo_rows`, the cp-token minimum, many 1-
and 2-token documents, and an all-padding row whose loss mask zeroes the output gradient. Exit code 1 if any case
fails.

The reference runs every token-wise linear layer in shard-sized blocks of the row, as each CP rank does, so both
use the same GEMM kernels: cuBLAS picks a different kernel at a different token count, and that rounding alone
flips near-tied indexer picks (the indexer weights projection at 2048 vs 8192 tokens, cp 4 with Flash dims).

Verdict per tensor, with `rel` the max abs difference over the reference's max abs:
- eager forward (outputs, entries, picks): bitwise equal to the reference.
- otherwise: `rel(halo, ref) <= 2 * spread + (2 * cp + 1) * 2**-8`. `spread` is the largest `rel` of the reference
  against itself rerun eagerly (the backward's atomics) and, in compile mode, against the compiled cp 1 module.
  `2**-8` is bf16's unit roundoff, and `2 * cp + 1` counts the bf16 roundings on halo's gradient path that the
  reference does not share: a halo token's input gradient sums bf16 contributions from up to `cp` ranks, a
  parameter gradient sums `cp` per-rank partials each rounded to bf16, and the reference rounds its result once.
The reference against the other compressor backend is printed but not bounded on: its compressor entries differ by
about 1e-2, which flips most indexer picks and so moves the outputs and gradients by up to about 0.2.
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch._inductor.config
import torch.distributed as dist

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4 import attention as A
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.utils.cp import CPContext
from prime_rl.utils.utils import default_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=8192)
parser.add_argument("--backends", nargs="+", default=["cudnn", "torch"])
parser.add_argument("--modes", nargs="+", default=["eager", "compile"])
parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--rows", nargs="+", default=None)
parser.add_argument("--dims", default="small", choices=["small", "flash"])
parser.add_argument("--kv-precision", default="fp8")
args = parser.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
group = dist.group.WORLD
torch._inductor.config.emulate_precision_casts = True
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 4096

LAYER_TYPES = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
RATES = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
SMALL = dict(
    hidden_size=256,
    num_attention_heads=32,
    head_dim=128,
    partial_rotary_factor=0.5,
    q_lora_rank=128,
    o_groups=2,
    o_lora_rank=64,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=128,
    dsa_backend="tilelang",
)
BF16_UNIT_ROUNDOFF = 2.0**-8
OTHER_BACKEND = {"cudnn": "torch", "torch": "cudnn"}


def make_config(backend: str) -> DeepseekV4Config:
    dims = SMALL if args.dims == "small" else dict(dsa_backend="cudnn_flashmla")
    config = DeepseekV4Config(
        num_hidden_layers=3,
        layer_types=LAYER_TYPES,
        compress_rates=RATES,
        sliding_window=128,
        max_position_embeddings=max(65536, args.tokens),
        kv_precision=args.kv_precision,
        compressor_backend=backend,
        **dims,
    )
    return config


def halo_rows_of(config: DeepseekV4Config) -> int:
    return max(config.sliding_window - 1, *(2 * rate - 1 for rate in config.compress_rates.values()))


def rows_for(cp: int, halo_rows: int) -> dict[str, tuple[list[int], bool]]:
    """{row name: (doc lengths, loss-masked)}; long rows are `--tokens` long and placed on the shard edges `r * n`."""
    T = args.tokens
    n = T // cp

    def from_cuts(cuts, total):
        bounds = [0, *sorted(c for c in set(cuts) if 0 < c < total), total]
        return [b - a for a, b in zip(bounds, bounds[1:])]

    def crossing(seed, offsets):
        g = torch.Generator().manual_seed(seed)
        cuts = [r * n + offsets[(r - 1) % len(offsets)] for r in range(1, cp)]
        pos = 0
        while True:
            pos += int(torch.randint(max(n // 7, 1), max(n // 2, 2), (1,), generator=g))
            if pos >= T:
                break
            cuts.append(pos)
        return from_cuts(cuts, T)

    def cycled(cycle, total):
        lens, i = [], 0
        while sum(lens) < total:
            lens.append(cycle[i % len(cycle)])
            i += 1
        lens[-1] -= sum(lens) - total
        return [length for length in lens if length > 0]

    below = cp * (halo_rows - 1)
    rows = {
        "cross_a": (crossing(11, [1, -1, 3, -4, 5, 127, -128]), False),
        "cross_b": (crossing(12, [129, -255, 256, 2, -7, 130, -3]), False),
        "short": (cycled([1, 2, 3, 4, 5, 7, 8, 9, 127, 128, 129, 200, 254, 60, 33, 250], T), False),
        "edge": ([length for _ in range(cp) for length in (100, n - 229, 129)], False),
        "tiny_edges": ([length for _ in range(cp) for length in (1, n - 2, 1)], False),
        "single": ([T], False),
        "ones_twos": (cycled([1, 2, 2, 1, 1], T), False),
        "at_halo": (cycled([3, 130, 1, 300], cp * halo_rows), False),
        "below_halo": (cycled([3, 130, 1, 300], below), False),
        "below_halo_single": ([below], False),
        "quarter_halo": (cycled([5, 60, 2, 17], cp * (halo_rows // 4)), False),
        "min_split": ([cp], False),
        "min_split_docs": ([1] * cp, False),
        "short_ones_twos": (cycled([1, 2, 2, 1, 1], cp * (halo_rows // 2)), False),
        "short_all_pad": ([cp * (halo_rows // 2)], True),
    }
    assert all(sum(v) % cp == 0 and min(v) > 0 for v, _ in rows.values())
    assert all(sum(v) == T for name, (v, _) in rows.items() if name in ("cross_a", "cross_b", "edge", "tiny_edges"))
    if args.rows is not None:
        rows = {k: v for k, v in rows.items() if k in args.rows}
    return rows


def randomize(module):
    g = torch.Generator(device="cuda").manual_seed(1234)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("norm.weight"):
                p.uniform_(0.5, 1.5, generator=g)
            elif name.endswith("sinks") or name.endswith("position_bias"):
                p.normal_(0.0, 1.0, generator=g)
            else:
                p.normal_(0.0, 0.05, generator=g)


recorded: dict[str, torch.Tensor] = {}
roles: dict[int, str] = {}
recording = {"on": False}
_compress = A.DeepseekV4Compressor.compress
_csa_forward = A.DeepseekV4CSACompressor.forward
_hca_forward = A.DeepseekV4HCACompressor.forward


def compress(self, *a, **k):
    out = _compress(self, *a, **k)
    if recording["on"]:
        recorded[f"entries/{roles[id(self)]}"] = out.detach().clone()
    return out


def csa_forward(self, *a, **k):
    out = _csa_forward(self, *a, **k)
    if recording["on"]:
        recorded["picks/indexer"] = out[1].detach().clone()
    return out


def hca_forward(self, *a, **k):
    out = _hca_forward(self, *a, **k)
    if recording["on"]:
        recorded["picks/hca"] = out[1].detach().clone()
    return out


A.DeepseekV4Compressor.compress = compress
A.DeepseekV4CSACompressor.forward = csa_forward
A.DeepseekV4HCACompressor.forward = hca_forward
_linear_forward = torch.nn.Linear.forward
blocked = {"row": 0, "shard": 0}


def linear_in_shard_blocks(self, x):
    if x.dim() != 3 or x.shape[1] != blocked["row"] or blocked["shard"] == blocked["row"]:
        return _linear_forward(self, x)
    return torch.cat([_linear_forward(self, block) for block in x.split(blocked["shard"], dim=1)], dim=1)


torch.nn.Linear.forward = linear_in_shard_blocks


def build_attn(config, layer_idx):
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        rotary = DeepseekV4RotaryEmbedding(config)
        attn = DeepseekV4Attention(config, layer_idx, rotary)
    if attn.compressor is not None:
        roles[id(attn.compressor)] = "attn"
        if hasattr(attn.compressor, "indexer"):
            roles[id(attn.compressor.indexer.compressor)] = "indexer"
    return attn, rotary


def build_packed(rotary, doc_lens, cp):
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(doc_lens),
        device=torch.device("cuda"),
        cp_rank=rank if cp else 0,
        cp_world_size=world if cp else 1,
    )


def run(attn, fn, x_global, dy_global, packed, cp, record):
    recorded.clear()
    recording["on"] = record
    attn.zero_grad(set_to_none=True)
    n = x_global.shape[1] // world
    blocked.update(row=0 if cp else x_global.shape[1], shard=n)
    if cp:
        attn.cp_context = CPContext(group, rank, world)
        x = x_global[:, rank * n : (rank + 1) * n].detach().clone().requires_grad_()
        dy = dy_global[:, rank * n : (rank + 1) * n]
    else:
        attn.cp_context = CPContext()
        x = x_global.detach().clone().requires_grad_()
        dy = dy_global
    out = fn(x, packed)[0]
    (out.float() * dy.float()).sum().backward()
    recording["on"] = False
    grads = {name: p.grad.detach().float().clone() for name, p in attn.named_parameters() if p.grad is not None}
    if cp:
        for g in grads.values():
            dist.all_reduce(g)
    torch.cuda.synchronize()
    return dict(out=out.detach()[0], dx=x.grad.detach()[0], grads=grads, rec=dict(recorded))


def local_ref(ref):
    """The reference sliced to this rank's queries, for the per-query tensors."""
    n = ref["out"].shape[0] // world
    sl = slice(rank * n, (rank + 1) * n)
    rec = {k: (v[:, sl] if k.startswith("picks/") else v) for k, v in ref["rec"].items()}
    return dict(out=ref["out"][sl], dx=ref["dx"][sl], grads=ref["grads"], rec=rec)


def err(a, b):
    """(max abs diff, max abs of b, bitwise) for one rank's tensors; inf on a shape mismatch."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf"), 0.0, False
    a, b = a.float(), b.float()
    if a.numel() == 0:
        return 0.0, 0.0, True
    return (a - b).abs().max().item(), b.abs().max().item(), torch.equal(a, b)


def pick_mismatch(a, b):
    """Fraction of query rows whose pick sets differ, inf on a shape mismatch."""
    if a.shape != b.shape:
        return float("inf")
    a, b = a.sort(dim=-1).values, b.sort(dim=-1).values
    return (a != b).any(-1).float().mean().item() if a.numel() else 0.0


def compare(a, b):
    """Metrics of `a` against `b`, reduced over ranks: {name: (max abs diff, rel, bitwise)}."""
    out = {}
    names = ["out", "dx"] + [f"grad/{k}" for k in b["grads"]] + sorted(b["rec"])
    for name in names:
        if name.startswith("picks/"):
            m = pick_mismatch(a["rec"][name], b["rec"][name]) if name in a["rec"] else float("inf")
            t = torch.tensor([m, 0.0, float(m == 0)], device="cuda", dtype=torch.float64)
        else:
            if name.startswith("grad/"):
                x, y = a["grads"].get(name[5:]), b["grads"][name[5:]]
            elif name.startswith("entries/"):
                x, y = a["rec"].get(name), b["rec"][name]
            else:
                x, y = a[name], b[name]
            d, s, eq = err(x, y)
            t = torch.tensor([d, s, float(eq)], device="cuda", dtype=torch.float64)
        bitwise = t[2:].clone()
        dist.all_reduce(t[:2], op=dist.ReduceOp.MAX)
        dist.all_reduce(bitwise, op=dist.ReduceOp.MIN)
        d, s = t[0].item(), t[1].item()
        rel = d if name.startswith("picks/") else d / max(s, 1e-30)
        out[name] = (d, rel, bool(bitwise.item()))
    return out


def tolerance(name, spread, mode):
    """None for a bitwise check, else the largest `rel` halo may show against the reference."""
    forward = name == "out" or name.startswith("entries/") or name.startswith("picks/")
    if forward and mode == "eager":
        return None
    same_backend_spread = max(spread.get("rerun", 0.0), spread.get("compile", 0.0))
    return 2.0 * same_backend_spread + (2 * world + 1) * BF16_UNIT_ROUNDOFF


failures = []
lines = []
for backend in args.backends:
    config, other_config = make_config(backend), make_config(OTHER_BACKEND[backend])
    rows = rows_for(world, halo_rows_of(config))
    for mode in args.modes:
        for layer_idx in args.layers:
            torch._dynamo.reset()
            roles.clear()
            attn, rotary = build_attn(config, layer_idx)
            randomize(attn)
            other, other_rotary = build_attn(other_config, layer_idx)
            other.load_state_dict(attn.state_dict())
            fn = torch.compile(attn, fullgraph=True) if mode == "compile" else attn
            record = mode == "eager"
            for row_name, (doc_lens, masked) in rows.items():
                T = sum(doc_lens)
                g = torch.Generator(device="cuda").manual_seed(99 + layer_idx)
                x_global = torch.randn(1, T, config.hidden_size, device="cuda", generator=g).to(torch.bfloat16)
                dy_global = (torch.randn(1, T, config.hidden_size, device="cuda", generator=g) * 0.01).to(
                    torch.bfloat16
                )
                if masked:
                    dy_global = torch.zeros_like(dy_global)
                ref_packed = build_packed(rotary, doc_lens, False)
                ref = local_ref(run(attn, attn, x_global, dy_global, ref_packed, False, record))
                variants = {
                    "rerun": run(attn, attn, x_global, dy_global, ref_packed, False, record),
                    "backend": run(
                        other, other, x_global, dy_global, build_packed(other_rotary, doc_lens, False), False, record
                    ),
                }
                if mode == "compile":
                    variants["compile"] = run(attn, fn, x_global, dy_global, ref_packed, False, False)
                halo = run(attn, fn, x_global, dy_global, build_packed(rotary, doc_lens, True), True, record)
                hr = compare(halo, ref)
                spreads = {k: compare(local_ref(v), ref) for k, v in variants.items()}
                case = f"cp{world} {backend} {mode} L{layer_idx}:{config.layer_types[layer_idx][:7]} {row_name}({T} tok, {len(doc_lens)} docs)"
                for name in hr:
                    spread = {k: v[name][1] for k, v in spreads.items() if name in v}
                    tol = tolerance(name, spread, mode)
                    ok = hr[name][2] if tol is None else hr[name][1] <= tol
                    lines.append(
                        f"{'PASS' if ok else 'FAIL'} {case} {name}: halo-vs-ref rel={hr[name][1]:.2e} "
                        f"bitwise={hr[name][2]} tol={'bitwise' if tol is None else f'{tol:.2e}'} "
                        + " ".join(f"{k}={v:.2e}" for k, v in spread.items())
                    )
                    if not ok:
                        failures.append(f"{case} {name}")
            del attn, other, fn

if rank == 0:
    print("\n".join(lines))
    by_case = defaultdict(int)
    for f in failures:
        by_case[f.rsplit(" ", 1)[0]] += 1
    print(f"SUMMARY cp{world}: {len(lines) - len(failures)} pass, {len(failures)} fail, failing cases: {len(by_case)}")
    for case, count in by_case.items():
        print(f"  FAILCASE {case}: {count} tensors")
    sys.stdout.flush()
dist.barrier()
dist.destroy_process_group()
sys.exit(1 if failures else 0)
