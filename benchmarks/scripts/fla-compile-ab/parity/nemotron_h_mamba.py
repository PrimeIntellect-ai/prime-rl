import sys

import torch
from old_mamba import NemotronHMamba2 as OldMamba

from prime_rl.trainer.models.layers.fla_ops import prepare_chunk_indices
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.mamba import NemotronHMamba2 as NewMamba

limit = (0.0, float("inf")) if sys.argv[1] == "inf" else (0.001, 0.1)
torch.manual_seed(0)
config = NemotronHConfig(
    hidden_size=512,
    hybrid_override_pattern="M",
    mamba_num_heads=16,
    mamba_head_dim=64,
    n_groups=2,
    ssm_state_size=64,
    conv_kernel=4,
    chunk_size=128,
    use_conv_bias=True,
    time_step_limit=limit,
)
new = NewMamba(config).cuda().to(torch.bfloat16)
with torch.no_grad():
    new.conv1d.bias.normal_()
old = OldMamba(config).cuda().to(torch.bfloat16)
old.load_state_dict(new.state_dict())
seq = [700, 37, 1290, 21, 1]
cu = torch.tensor([0, *torch.tensor(seq).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
x = torch.randn(1, sum(seq), 512, device="cuda", dtype=torch.bfloat16)
g = torch.randn_like(x)


def run(module, fn):
    module.zero_grad()
    xi = x.clone().requires_grad_()
    out = fn(xi)
    out.backward(g)
    return out, xi.grad, {n: p.grad for n, p in module.named_parameters()}


ref = run(old, lambda xi: old(xi, cu))
ref2 = run(old, lambda xi: old(xi, cu))
ci = prepare_chunk_indices(cu)
eager = run(new, lambda xi: new(xi, cu, ci))
compiled_fn = torch.compile(new, fullgraph=True)
compiled = run(new, lambda xi: compiled_fn(xi, cu, ci))


def report(name, a, b):
    rel = lambda t, r: ((t.float() - r.float()).norm() / r.float().norm()).item()
    grads = {n: f"{rel(a[2][n], b[2][n]):.2e}" for n in b[2]}
    print(name, "out", f"{rel(a[0], b[0]):.2e}", "dx", f"{rel(a[1], b[1]):.2e}", grads)


print("dt_limit", limit)
report("old vs old (noise floor)", ref2, ref)
report("new eager vs old", eager, ref)
report("new compiled vs old", compiled, ref)
print("grad norms", {n: f"{p.norm().item():.3e}" for n, p in ref[2].items()})
