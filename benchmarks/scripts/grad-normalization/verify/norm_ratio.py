import sys

import torch

base = sys.argv[1]
for pair in sys.argv[2:]:
    a_name, b_name = pair.split(":")
    a = torch.load(f"{base}/{a_name}.gn/grads_step1.pt", weights_only=False)["grads"]
    b = torch.load(f"{base}/{b_name}.gn/grads_step1.pt", weights_only=False)["grads"]
    na = sum(a[k].double().square().sum().item() for k in a) ** 0.5
    nb = sum(b[k].double().square().sum().item() for k in b) ** 0.5
    dot = sum((a[k].double() * b[k].double()).sum().item() for k in a)
    diff = sum((a[k].double() - b[k].double()).square().sum().item() for k in a) ** 0.5
    print(
        f"{a_name} vs {b_name}: ||A||={na:.6e} ||B||/||A||={nb / na:.6f} global cos={dot / na / nb:.8f} rel L2={diff / na:.3e}"
    )
