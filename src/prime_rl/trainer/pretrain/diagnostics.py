from functools import cache

import torch
from torch import Tensor, nn

from prime_rl.trainer.model import forward
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.pretrain.metrics import group_norms, param_group, reduce_stats
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_language_model

MIN_VISIBLE_KEYS = 64
"""Attention entropy is averaged over queries that see at least this many keys."""

COLLAPSED_ENTROPY = 0.5
"""A head whose mean attention entropy (nats) is below this counts as collapsed."""


def probe_batch(micro_batch: dict[str, Tensor], num_tokens: int) -> dict[str, Tensor]:
    """The first ``num_tokens`` tokens of a training micro-batch, on the GPU: the fixed batch of the diagnostic
    forward."""
    lengths, total = [], 0
    for length in micro_batch["seq_lens"].tolist():
        lengths.append(min(length, num_tokens - total))
        total += lengths[-1]
        if total == num_tokens:
            break
    batch = {key: micro_batch[key][:, :total].to("cuda") for key in ("input_ids", "position_ids", "target_ids")}
    batch["seq_lens"] = torch.tensor(lengths, device="cuda")
    return batch


def attention_stats(q: Tensor, k: Tensor, cu_seqlens: Tensor, scale: float) -> tuple[Tensor, Tensor, Tensor]:
    """Per head: max attention logit (q·k times the softmax scale, after RoPE and QK-norm, over causal pairs
    within a document), summed attention entropy (nats) over the queries that see at least `MIN_VISIBLE_KEYS`
    keys, and the number of those queries. ``q`` is `[tokens, heads, head_dim]`, ``k`` `[tokens, kv_heads, head_dim]`."""
    num_tokens, num_heads, _ = q.shape
    positions = torch.arange(num_tokens, device=q.device)
    document = torch.searchsorted(cu_seqlens[1:].long(), positions, right=True)
    mask = (positions[None, :] <= positions[:, None]) & (document[None, :] == document[:, None])
    valid = mask.sum(dim=1) >= MIN_VISIBLE_KEYS
    groups = num_heads // k.shape[1]
    max_logits, entropy_sums = [], []
    for head in range(num_heads):
        scores = (q[:, head].float() @ k[:, head // groups].float().T * scale).masked_fill(~mask, float("-inf"))
        max_logits.append(scores.amax())
        entropy = torch.special.entr(scores.softmax(dim=-1)).sum(dim=-1)
        entropy_sums.append(entropy[valid].sum())
    return torch.stack(max_logits), torch.stack(entropy_sums), valid.sum()


@cache
def warn_no_attention_stats() -> None:
    get_logger().warning(
        "The diagnostic probe records attention statistics only for FlashAttention layers; none found, skipping them"
    )


@torch.no_grad()
def run_probe(model: nn.Module, batch: dict[str, Tensor], group) -> dict[str, float]:
    """Forward the probe batch with hooks: attention max logit and entropy per layer and head (FlashAttention
    layers), residual-stream RMS and max per layer, and the probe loss. Compiled layers run eagerly for the probe,
    so the hooks see every call. Every rank probes its own batch; the statistics are reduced over ``group``."""
    layers = get_language_model(model).layers
    attentions = [module for module in layers.modules() if isinstance(module, FlashAttention)]
    if not attentions:
        warn_no_attention_stats()
    attention, activations = [], []

    def capture(module: FlashAttention):
        compute = module._compute_attention

        def compute_and_record(q, k, v, cu_seqlens, max_seqlen):
            attention.append(attention_stats(q, k, cu_seqlens, module.scaling))
            return compute(q, k, v, cu_seqlens, max_seqlen)

        return compute_and_record

    def record_activations(module, args, output):
        hidden = (output[0] if isinstance(output, tuple) else output).float()
        activations.append(
            (hidden.square().sum(), torch.tensor(hidden.numel(), device=hidden.device), hidden.abs().amax())
        )

    for module in attentions:
        module._compute_attention = capture(module)
    hooks = [layer.register_forward_hook(record_activations) for layer in layers]
    labels = batch["target_ids"]
    with torch.compiler.set_stance("force_eager"):
        out = forward(model, batch["input_ids"], batch["position_ids"], seq_lens=batch["seq_lens"], labels=labels)
    for module in attentions:
        del module._compute_attention
    for hook in hooks:
        hook.remove()

    square_sum, count, absmax = (torch.stack(values) for values in zip(*activations))
    sums = {
        "square_sum": square_sum,
        "count": count.double(),
        "loss": out["loss"].double(),
        "tokens": (labels != IGNORE_INDEX).sum().double(),
    }
    maxes = {"absmax": absmax}
    if attention:
        max_logit, entropy_sum, queries = (torch.stack(values) for values in zip(*attention))
        sums |= {"entropy_sum": entropy_sum, "queries": queries.double()}
        maxes["max_logit"] = max_logit
    stats = reduce_stats(sums, maxes, group)

    rms = (stats["square_sum"] / stats["count"]).sqrt()
    metrics = {
        "act/rms/max": rms.max().item(),
        "act/absmax/max": stats["absmax"].max().item(),
        "probe/loss": (stats["loss"] / stats["tokens"]).item(),
    }
    for layer in range(len(rms)):
        metrics[f"act/rms/layer_{layer}"] = rms[layer].item()
        metrics[f"act/absmax/layer_{layer}"] = stats["absmax"][layer].item()
    if attention:
        head_entropy = stats["entropy_sum"] / stats["queries"][:, None]  # [layers, heads]
        layer_max_logit = stats["max_logit"].amax(dim=1)
        layer_entropy = head_entropy.mean(dim=1)
        metrics |= {
            "attn/max_logit/max": layer_max_logit.max().item(),
            "attn/entropy/min": layer_entropy.min().item(),
            "attn/entropy_min_head": head_entropy.min().item(),
            "attn/collapsed_heads": int((head_entropy < COLLAPSED_ENTROPY).sum()),
        }
        for layer in range(len(layer_max_logit)):
            metrics[f"attn/max_logit/layer_{layer}"] = layer_max_logit[layer].item()
            metrics[f"attn/entropy/layer_{layer}"] = layer_entropy[layer].item()
    return metrics


def norm_role(name: str) -> str:
    """Normalization layers by role (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `norm`),
    everything else by `param_group`."""
    module = name.split(".")[-2]
    return f"norm_gain/{module}" if "norm" in module else param_group(name)


class UpdateTracker:
    """Size of one optimizer update relative to the weights, per parameter group: ||W_after - W_before|| /
    ||W_before||, plus the weight RMS per group and of the norm gains by role."""

    def before_step(self, model: nn.Module) -> None:
        self.before = {name: param.detach().clone() for name, param in model.named_parameters()}

    def after_step(self, model: nn.Module) -> dict[str, float]:
        params = dict(model.named_parameters())
        weight = group_norms(((name, self.before[name]) for name in params), norm_role)
        update = group_norms(((name, params[name].detach() - self.before[name]) for name in params), norm_role)
        numel = {}
        for name, param in params.items():
            numel[norm_role(name)] = numel.get(norm_role(name), 0) + param.numel()
        del self.before
        metrics = {}
        for group, norm in weight.items():
            metrics[f"param/rms/{group}"] = (norm / numel[group] ** 0.5).item()
            metrics[f"update/rms_ratio/{group}"] = (update[group] / norm).item()
        return metrics
