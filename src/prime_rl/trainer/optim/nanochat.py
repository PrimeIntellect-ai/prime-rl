"""Nanochat's combined optimizer (https://github.com/karpathy/nanochat, nanochat/optim.py @ ff46300), adapted from
modded-nanogpt. Muon for the block matrices (polar-express orthogonalization, NorMuon variance reduction, cautious
weight decay), AdamW for embeddings, unembedding and scalars.

Weights must be fully replicated on every GPU (``model.dp_replicate`` equal to the number of GPUs): gradients arrive
already reduced over the replicas and every rank applies the same update to its full copy.
"""

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from prime_rl.configs.pretrain import NanochatMuonAdamWConfig


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


# Coefficients for polar express orthogonalization, num_iters=5, safety_factor=2e-2, cushion=2
# (https://arxiv.org/pdf/2505.16932).
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


def _local(tensor: Tensor) -> Tensor:
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        assert local.shape == tensor.shape, "The nanochat optimizer needs fully replicated weights"
        return local
    return tensor


class NanochatMuonAdamW(Optimizer):
    """``param_groups`` entries have a ``kind``, ``muon`` or ``adamw``. Muon groups hold same-shape matrices, stacked for
    the fused update. ``total_steps`` and ``weight_decay`` set the decay horizon and its value at step 0. Momentum warms
    up from 0.85 to 0.95 over the first 300 steps, as in nanochat's training loop."""

    def __init__(self, param_groups: list[dict], total_steps: int, weight_decay: float, momentum_warmup: int = 300):
        super().__init__(param_groups, defaults={})
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.momentum_warmup = momentum_warmup
        self.num_steps = 0
        # 0-D CPU tensors avoid recompiling the fused kernels when these values change.
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def state_dict(self):
        state = super().state_dict()
        state["num_steps"] = self.num_steps
        return state

    def load_state_dict(self, state_dict):
        self.num_steps = state_dict["num_steps"]
        super().load_state_dict({k: v for k, v in state_dict.items() if k != "num_steps"})

    def _step_adamw(self, group: dict) -> None:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            p, grad = _local(parameter), _local(parameter.grad)
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad.to(p.dtype),
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )
            if group.get("round_to_bf16", False):
                # nanochat stores embeddings in bf16.
                p.copy_(p.bfloat16().to(p.dtype))

    def _step_muon(self, group: dict) -> None:
        params = [_local(p) for p in group["params"]]
        state = self.state[group["params"][0]]
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([_local(parameter.grad).to(dtype) for parameter in group["params"]])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        fraction = min(self.num_steps / self.momentum_warmup, 1)
        momentum = (1 - fraction) * 0.85 + fraction * 0.95
        weight_decay = self.weight_decay * (1 - self.num_steps / self.total_steps)
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            else:
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay
                self._step_muon(group)
        self.num_steps += 1


def create_nanochat_optimizer(config: NanochatMuonAdamWConfig, named_params) -> NanochatMuonAdamW:
    betas = tuple(config.adam_betas)
    adamw = dict(kind="adamw", eps=1e-10, weight_decay=0.0)
    named_params = [(name, parameter) for name, parameter in named_params if parameter.requires_grad]

    def select(predicate) -> list[torch.nn.Parameter]:
        return [parameter for name, parameter in named_params if predicate(name, parameter)]

    def is_embedding(name, parameter):
        return name.endswith("embed_tokens.weight") or name.endswith("value_embed.weight")

    def is_scalar(name, parameter):
        return name.endswith("resid_lambda") or name.endswith("x0_lambda")

    def is_connection_matrix(name, parameter):
        return parameter.ndim >= 2 and ("_connection." in name or name.startswith("model.norm."))

    param_groups = [
        dict(params=select(lambda n, p: n.startswith("lm_head")), lr=config.unembedding_lr, betas=betas, **adamw),
        dict(params=select(is_embedding), lr=config.embedding_lr, betas=betas, round_to_bf16=True, **adamw),
        dict(params=select(lambda n, p: n.endswith("resid_lambda")), lr=config.scalar_lr * 0.01, betas=betas, **adamw),
        dict(
            params=select(lambda n, p: n.endswith("x0_lambda")),
            lr=config.scalar_lr,
            betas=(config.x0_beta1, betas[1]),
            **adamw,
        ),
    ]

    def is_matrix(name, parameter):
        return parameter.ndim >= 2 and not name.startswith("lm_head") and not is_embedding(name, parameter)

    if config.connection_matrix_optimizer == "adamw":
        connection_matrices = select(is_connection_matrix)
        if connection_matrices:
            param_groups.append(dict(params=connection_matrices, lr=config.connection_lr, betas=betas, **adamw))

    extra_scalars = select(lambda n, p: p.ndim < 2 and not is_scalar(n, p))
    if extra_scalars:
        param_groups.append(dict(params=extra_scalars, lr=config.extra_scalar_lr, betas=betas, **adamw))

    matrices = select(
        lambda n, p: (
            is_matrix(n, p) and (config.connection_matrix_optimizer == "muon" or not is_connection_matrix(n, p))
        )
    )
    for shape in sorted({parameter.shape for parameter in matrices}):
        param_groups.append(
            dict(
                params=[p for p in matrices if p.shape == shape],
                kind="muon",
                lr=config.matrix_lr,
                momentum=0.85,
                ns_steps=5,
                beta2=0.95,
                weight_decay=config.weight_decay,
            )
        )
    grouped = sum(len(group["params"]) for group in param_groups)
    assert grouped == len(named_params), f"{grouped} of {len(named_params)} parameters ended up in a group"
    return NanochatMuonAdamW(param_groups, total_steps=config.total_steps, weight_decay=config.weight_decay)
