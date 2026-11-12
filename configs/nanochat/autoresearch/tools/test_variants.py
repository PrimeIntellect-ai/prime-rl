"""CPU unit tests for residual variants: forward/backward on a tiny nanochat, init equivalence, FLOPs/param accounting."""
import json, sys, torch
import cpu_import as c

VARIANTS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
    {"type": "plain"},
    {"type": "gated", "streams": 4, "low_rank": 64, "read_scale": 2.0},
    {"type": "hyper", "streams": 2},
    {"type": "hyper", "streams": 4},
    {"type": "hyper", "streams": 4, "dynamic": True},
    {"type": "hyper", "streams": 4, "mix": "sinkhorn"},
    {"type": "hyper", "streams": 4, "mix": "none"},
]

def tiny(res):
    m = c.build(4, res, device="cpu", seq_len=64, vocab=512)
    m.initialize_weights()
    return m

torch.manual_seed(0)
ids = torch.randint(0, 512, (1, 64))
pos = torch.cat([torch.arange(40), torch.arange(24)])[None]
seq_lens = torch.tensor([40, 24])

def run(m, perturb=False):
    if perturb:
        with torch.no_grad():
            for b in m.model.layers:
                b.attn.c_proj.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(1))
                b.mlp.c_proj.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(2))
    h = m.model(input_ids=ids, position_ids=pos, seq_lens=seq_lens).last_hidden_state
    return h

base = run(tiny({"type": "plain"}), perturb=True)
for res in VARIANTS:
    m = tiny(res)
    h = run(m, perturb=True)
    diff = (h - base).abs().max().item()
    logits = h @ m.lm_head.weight.t()
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    loss.backward()
    nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    big = c.build(12, res)  # meta, d12 accounting
    cnt = c.counts(big)
    plain = c.counts(c.build(12))
    extra_params = cnt["total"] - plain["total"]
    print(json.dumps(res), f"| init max|h-plain|={diff:.2e} loss={loss.item():.4f} nograd={nograd} nonfinite={nonfinite}"
          f" | d12 fpt={cnt['flops_per_token']/1e6:.1f}M (+{100*(cnt['flops_per_token']/plain['flops_per_token']-1):.2f}%)"
          f" extra_params={extra_params}")
