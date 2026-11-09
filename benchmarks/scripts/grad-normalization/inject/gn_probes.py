"""Backward probes, simulated MXFP8 gradient rounding and parameter-gradient statistics for the RL trainer (debug only)."""

import copy
import json
import math
import os
import types
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.expert_compute import GroupedGemmExpertCompute
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.vlm import get_language_model

BLOCK = 32
E4M3_MAX = 448.0
SEED_QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_SAMPLES = 1 << 18


def quantize_blocks(blocks: torch.Tensor, e_min: int) -> torch.Tensor:
    """Round fp32 (num_blocks, 32) values through E4M3 with scale 2^max(ceil(log2(amax/448)), e_min)."""
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(min=e_min))
    return (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * scale


def fake_quantize(grad: torch.Tensor, e_min: int) -> torch.Tensor:
    """Fake-quantize ``grad`` in 1x32 blocks along its last (hidden) axis."""
    blocks = grad.float().reshape(-1, BLOCK)
    return quantize_blocks(blocks, e_min).reshape(grad.shape).to(grad.dtype)


def measure(grad: torch.Tensor, scan: list[int]) -> list[float]:
    """Return [numel, nonzero, sum of squares, max, p50, p99, nonzero blocks, then per e_min: zeroed, blocks at the minimum scale].

    p50 and p99 come from an odd-strided subsample of at most QUANTILE_SAMPLES entries.
    """
    blocks = grad.detach().float().reshape(-1, BLOCK)
    magnitudes = blocks.abs()
    values = magnitudes.flatten()
    numel = values.numel()
    nonzero = blocks != 0
    amax = magnitudes.amax(dim=-1)
    natural_exponent = torch.ceil(torch.log2(amax / E4M3_MAX))
    live_blocks = amax > 0
    sample = values[:: (numel // QUANTILE_SAMPLES) | 1]
    parts = [
        torch.tensor(float(numel), device=blocks.device),
        nonzero.sum(),
        blocks.double().square().sum(),
        values.max(),
        sample.kthvalue(max(1, round(0.5 * sample.numel()))).values,
        sample.kthvalue(max(1, math.ceil(0.99 * sample.numel()))).values,
        live_blocks.sum(),
    ]
    for e_min in scan:
        parts.append((nonzero & (quantize_blocks(blocks, e_min) == 0)).sum())
        parts.append((live_blocks & (natural_exponent < e_min)).sum())
    return torch.stack([part.double() for part in parts]).tolist()


def materialize(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Return the full tensor behind a DTensor (collective when sharded), or None for a sharded CPU tensor."""
    if tensor is None or not isinstance(tensor, DTensor):
        return tensor
    if tensor.device_mesh.size() == 1:
        return tensor.to_local()
    if tensor.device.type != "cuda":
        return None
    return tensor.full_tensor()


class Tap(torch.autograd.Function):
    """Identity in forward, probe (and optional simulated MXFP8 rounding) on the incoming gradient in backward."""

    @staticmethod
    def forward(ctx, x, key, valid_rows):
        ctx.key = key
        ctx.valid_rows = valid_rows
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return STATE.on_activation_grad(ctx.key, grad, ctx.valid_rows), None, None


class ProbeState:
    def __init__(self):
        self.out_dir = Path(os.environ["GN_OUT"])
        self.e_min = int(os.environ.get("GN_EMIN", "-127"))
        self.mx_round = os.environ.get("GN_MX_ROUND", "0") == "1"
        self.mx_round_sites = set(os.environ.get("GN_MX_ROUND_SITES", "dy,dz").split(","))
        scan = [int(value) for value in os.environ.get("GN_EMIN_SCAN", "").split(",") if value.strip()]
        self.scan = sorted(set(scan) | {self.e_min})
        dump_step = os.environ.get("GN_GRAD_DUMP_STEP")
        self.grad_dump_step = int(dump_step) if dump_step else None
        self.model = None
        self.optimizer = None
        self.manager = None
        self.parallel_dims = None
        self.layers: dict[int, dict[str, list[torch.nn.Parameter]]] = {}
        self.step = 1
        self.reset_step()

    def reset_step(self):
        self.acts: dict[tuple[int, str], list[float]] = {}
        self.micro_batches = 0
        self.local_tokens = 0.0
        self.seed_ratios: list[torch.Tensor] = []
        self.normalization: dict | None = None
        self.grad_norm: float | None = None
        self.grads_ready = False
        self.param_stats: dict[tuple[int, str], dict] = {}
        self.before_step: dict[tuple[int, str], list[torch.Tensor]] = {}
        self.global_tokens: float | None = None

    @property
    def is_master(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def attach_model(self, model):
        """Tap each MoE layer's routed (dy, dy_rows, dz), shared expert (sdy, sdz) and attention (ao, aqkv) sites."""
        self.model = model
        for name, layer in get_language_model(model).layers.named_children():
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, MoE):
                continue
            index = int(name)
            experts = moe.experts
            if not isinstance(experts.compute, GroupedGemmExpertCompute) or experts.gate_up_proj is None:
                raise RuntimeError("gn_probes needs grouped-GEMM expert compute with the fused gate_up projection")

            prepare_output = moe.prepare_expert_output
            moe.prepare_expert_output = partial(self._tap_output, prepare_output, (index, "dy"))

            probed_compute = copy.copy(experts.compute)
            probed_compute.gemm = partial(
                self._tap_gemm, experts.compute.gemm, index, 2 * experts.hidden_dim, experts.down_proj.shape[1]
            )
            experts.compute = probed_compute

            if moe.shared_expert is not None:
                shared = moe.shared_expert
                shared.forward = partial(self._shared_forward, shared, index)
                shared.register_forward_hook(partial(self._tap_module_output, (index, "sdy")))
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                attn.o_proj.register_forward_hook(partial(self._tap_module_output, (index, "ao")))
                projections = [attn.qkv_proj] if attn.qkv_proj is not None else [attn.q_proj, attn.k_proj, attn.v_proj]
                for projection in projections:
                    projection.register_forward_hook(partial(self._tap_module_output, (index, "aqkv")))

            groups = {
                "routed_gate_up": [experts.gate_up_proj],
                "routed_down": [experts.down_proj],
                "shared": list(moe.shared_expert.parameters()) if moe.shared_expert is not None else [],
                "attn": list(layer.self_attn.parameters()) if hasattr(layer, "self_attn") else [],
            }
            self.layers[index] = {group: params for group, params in groups.items() if params}

    @staticmethod
    def _tap_output(prepare_output, key, x):
        return Tap.apply(prepare_output(x), key, None)

    @staticmethod
    def _tap_module_output(key, module, args, output):
        return Tap.apply(output, key, None)

    @staticmethod
    def _shared_forward(shared, index, x, routed_experts=None):
        """Shared expert forward with its gate and up outputs tapped jointly as the fused gate_up output (sdz)."""
        gate_up = Tap.apply(torch.cat([shared.gate_proj(x), shared.up_proj(x)], dim=-1), (index, "sdz"), None)
        gate, up = gate_up.chunk(2, dim=-1)
        return shared.down_proj(shared.activation.apply(gate, up))

    @staticmethod
    def _tap_gemm(gemm, index, gate_up_width, dim, x, weight, offsets):
        out = gemm(x, weight, offsets)
        if out.shape[-1] == gate_up_width:
            return Tap.apply(out, (index, "dz"), offsets[-1])
        if out.shape[-1] == dim:
            return Tap.apply(out, (index, "dy_rows"), offsets[-1])
        return out

    def attach_optimizer(self, optimizer, manager):
        self.optimizer = optimizer
        self.manager = manager
        original_step = optimizer.step

        def step(_optimizer, *args, **kwargs):
            self.on_grads_ready(manager, self.model)
            self.snapshot_params()
            result = original_step(*args, **kwargs)
            self.after_optimizer_step()
            return result

        optimizer.step = types.MethodType(step, optimizer)

    def on_activation_grad(self, key, grad, valid_rows):
        rows = grad if valid_rows is None else grad[: int(valid_rows)]
        if rows.numel() > 0:
            values = measure(rows, self.scan)
            totals = self.acts.setdefault(key, [0.0] * (len(values) + 2))
            totals[0] += 1
            for i, value in enumerate(values):
                if i == 3:
                    totals[i + 1] = max(totals[i + 1], value)
                else:
                    totals[i + 1] += value
        if self.mx_round and key[1] in self.mx_round_sites:
            return fake_quantize(grad, self.e_min)
        return grad

    def on_compute_loss(self, kwargs):
        self.micro_batches += 1
        rl_weights = kwargs.get("rl_weights")
        for i, (logprobs, mask, advantages) in enumerate(
            zip(kwargs["trainer_logprobs"], kwargs["loss_mask"], kwargs["advantages"])
        ):
            weights = rl_weights[i] if rl_weights is not None else None
            self.local_tokens += float(weights[mask].sum() if weights is not None else mask.sum())
            if logprobs.requires_grad:
                weighted_advantage = (advantages * weights if weights is not None else advantages).abs().float()
                selected = mask & (weighted_advantage > 0)
                logprobs.register_hook(partial(self._on_seed_grad, selected, weighted_advantage))

    def _on_seed_grad(self, selected, weighted_advantage, grad):
        self.seed_ratios.append(grad.detach().float().abs()[selected] / weighted_advantage[selected])

    def on_normalization(self, norm, args, kwargs):
        counts = args[0] if args else kwargs.get("counts")
        self.normalization = {
            "counts": dict(counts),
            "placement": getattr(norm, "placement", None),
            "grad_scale": getattr(norm, "grad_scale", None),
            "reference": getattr(norm, "reference", None),
        }

    def on_grad_norm(self, grad_norm):
        self.grad_norm = float(grad_norm)

    @property
    def cp(self) -> int:
        return self.parallel_dims.cp if self.parallel_dims is not None else 1

    def on_grads_ready(self, manager, model):
        """Runs once per step after the post-backward scale and before clipping."""
        if self.grads_ready:
            return
        self.grads_ready = True
        tokens = torch.tensor([self.local_tokens], dtype=torch.float64, device="cuda")
        if dist.is_initialized():
            dist.all_reduce(tokens)
        self.global_tokens = tokens.item() / self.cp
        if manager is not None:
            return
        for index, groups in self.layers.items():
            for group, params in groups.items():
                grads = [materialize(param.grad) for param in params]
                grads = [grad for grad in grads if grad is not None]
                if not grads:
                    continue
                sumsq = sum(float(grad.double().square().sum()) for grad in grads)
                numel = sum(grad.numel() for grad in grads)
                zeros = sum(int((grad == 0).sum()) for grad in grads)
                self.param_stats[(index, group)] = {
                    "grad_frob": math.sqrt(sumsq),
                    "grad_rms": math.sqrt(sumsq / numel),
                    "grad_zero_frac": zeros / numel,
                    "numel": numel,
                }
        if self.grad_dump_step == self.step:
            self.dump_grads(model)

    def dump_grads(self, model):
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = materialize(param.grad)
            if self.is_master and grad is not None:
                grads[name] = grad.detach().float().cpu()
        if self.is_master:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            payload = {"step": self.step, "global_tokens": self.global_tokens, "normalization": self.normalization}
            torch.save({"meta": payload, "grads": grads}, self.out_dir / f"grads_step{self.step}.pt")

    def snapshot_params(self):
        """Keep MoE parameters for the update RMS; skipped under full offload, which updates during backward."""
        if self.manager is not None:
            return
        for index, groups in self.layers.items():
            for group in ("routed_gate_up", "routed_down", "shared"):
                if group in groups:
                    self.before_step[(index, group)] = [
                        materialize(param.data).detach().float().clone() for param in groups[group]
                    ]

    def optimizer_state(self, param):
        for optimizer in (self.optimizer, getattr(self.optimizer, "optimizer", None)):
            state = getattr(optimizer, "state", None)
            if state is not None and param in state:
                return state[param]
        return None

    def learning_rate(self):
        for optimizer in (self.optimizer, getattr(self.optimizer, "optimizer", None)):
            groups = getattr(optimizer, "param_groups", None)
            if groups:
                return groups[0]["lr"]
        return None

    def after_optimizer_step(self):
        lr = self.learning_rate()
        for (index, group), before in self.before_step.items():
            params = self.layers[index][group]
            after = [materialize(param.data).detach().float() for param in params]
            update_sumsq = sum(float((a - b).double().square().sum()) for a, b in zip(after, before))
            numel = sum(b.numel() for b in before)
            stats = self.param_stats.setdefault((index, group), {"numel": numel})
            stats["update_rms_over_lr"] = math.sqrt(update_sumsq / numel) / lr if lr else None
            exp_avgs = []
            for param in params:
                state = self.optimizer_state(param)
                exp_avgs.append(materialize(state["exp_avg"]) if state is not None and "exp_avg" in state else None)
            if exp_avgs and all(exp_avg is not None for exp_avg in exp_avgs):
                exp_avg_sumsq = sum(float(exp_avg.double().square().sum()) for exp_avg in exp_avgs)
                stats["exp_avg_rms"] = math.sqrt(exp_avg_sumsq / numel)
        if self.is_master:
            self.write_record(lr)
        self.step += 1
        self.reset_step()

    def seed_summary(self):
        if not self.seed_ratios:
            return None
        all_ratios = torch.cat(self.seed_ratios).double() * self.cp
        ratios = all_ratios[all_ratios > 0]
        if ratios.numel() == 0:
            return {"tokens": all_ratios.numel(), "nonzero_frac": 0.0}
        quantiles = torch.quantile(ratios.cpu(), torch.tensor(SEED_QUANTILES, dtype=torch.float64)).tolist()
        tokens = self.global_tokens
        return {
            "tokens": all_ratios.numel(),
            "nonzero_frac": ratios.numel() / all_ratios.numel(),
            "ratio_quantiles": dict(zip(map(str, SEED_QUANTILES), quantiles)),
            "ratio_times_n_quantiles": dict(zip(map(str, SEED_QUANTILES), [q * tokens for q in quantiles])),
        }

    def write_record(self, lr):
        acts = []
        for (index, site), totals in sorted(self.acts.items()):
            calls, numel, nonzero, sumsq, max_abs, p50_sum, p99_sum, live_blocks = totals[:8]
            per_emin = totals[8:]
            acts.append(
                {
                    "layer": index,
                    "site": site,
                    "calls": int(calls),
                    "rms": math.sqrt(sumsq / numel),
                    "p50": p50_sum / calls,
                    "p99": p99_sum / calls,
                    "max": max_abs,
                    "nonzero_frac": nonzero / numel,
                    "zeroed_frac": {
                        str(e_min): per_emin[2 * i] / nonzero if nonzero else 0.0 for i, e_min in enumerate(self.scan)
                    },
                    "min_scale_block_frac": {
                        str(e_min): per_emin[2 * i + 1] / live_blocks if live_blocks else 0.0
                        for i, e_min in enumerate(self.scan)
                    },
                }
            )
        params = [
            {"layer": index, "group": group, **stats} for (index, group), stats in sorted(self.param_stats.items())
        ]
        record = {
            "step": self.step,
            "micro_batches": self.micro_batches,
            "global_tokens": self.global_tokens,
            "cp": self.cp,
            "normalization": self.normalization,
            "mx_round": self.mx_round,
            "mx_round_sites": sorted(self.mx_round_sites),
            "e_min": self.e_min,
            "emin_scan": self.scan,
            "grad_norm": self.grad_norm,
            "lr": lr,
            "gradient_manager": self.manager is not None,
            "seed": self.seed_summary(),
            "acts": acts,
            "params": params,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "probes.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")


STATE = ProbeState()
