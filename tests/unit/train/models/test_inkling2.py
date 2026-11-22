import json
import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.models.inkling2 import (
    Inkling2Config,
    Inkling2LoopConfig,
    Inkling2LoopForCausalLM,
    Inkling2LoopTextConfig,
    modeling_inkling2,
)
from prime_rl.trainer.models.layers.moe import TokenChoiceTopKRouter

# Blocks span 2 layers and layers 3 and 7 attend globally. The clip thresholds are sized for 16-dim heads, so
# scores stay O(1) and the softmax does not amplify rounding differences.
TEXT_CONFIG = dict(
    vocab_size=160,
    unpadded_vocab_size=150,
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    sliding_window_size=8,
    local_layer_ids=[0, 1, 2, 4, 5, 6],
    d_rel=4,
    rel_extent=12,
    log_scaling_n_floor=4,
    qk_l2_clip_threshold=8.0,
    rel_l2_clip_threshold=4.0,
    n_layers_per_attn_res_block=2,
    n_routed_experts=8,
    num_experts_per_tok=2,
    intermediate_size=32,
)
# The core covers two blocks, so a later pass reads two sources: the block sealed at layer 4 and the last partial.
# It holds a sliding layer that starts a block (2, 4), one that does not (5) and a global layer (3).
LOOP_CONFIG = dict(core_start=2, core_end=6, num_passes=3, gate_rank=4)
NUM_LOOP_VISITS = 16


class ExactExpertCompute:
    """Per-expert matmuls in the parameter dtype. The shared expert kernel rounds to BF16, which a float64
    comparison cannot see through."""

    token_group_alignment = 8

    def validate(self, experts) -> None:
        pass

    def __call__(self, experts, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        outputs, start = [], 0
        for expert, count in enumerate(num_tokens_per_expert.tolist()):
            rows = x[start : start + count]
            hidden = experts.activation.apply(rows @ experts.gate_proj[expert].T, rows @ experts.up_proj[expert].T)
            outputs.append(hidden @ experts.down_proj[expert].T)
            start += count
        outputs.append(x.new_zeros(x.shape[0] - start, experts.down_proj.shape[1]))
        return torch.cat(outputs)


def dense_flex_attention(query, key, value, score_mod, block_mask, scale, enable_gqa, return_lse):
    """`flex_attention` over the full score matrix: FlexAttention has no CPU backward."""
    batch, heads, seq_len, _ = query.shape
    if enable_gqa:
        key = key.repeat_interleave(heads // key.shape[1], dim=1)
        value = value.repeat_interleave(heads // value.shape[1], dim=1)
    b = torch.arange(batch)[:, None, None, None]
    h = torch.arange(heads)[None, :, None, None]
    q_idx = torch.arange(seq_len)[None, None, :, None]
    kv_idx = torch.arange(seq_len)[None, None, None, :]
    scores = score_mod(query @ key.transpose(-1, -2) * scale, b, h, q_idx, kv_idx)
    scores = scores.masked_fill(~block_mask.mask_mod(b, h, q_idx, kv_idx), float("-inf"))
    return scores.softmax(-1) @ value, scores.logsumexp(-1)


@pytest.fixture(autouse=True)
def dense_attention(monkeypatch):
    monkeypatch.setattr(modeling_inkling2, "flex_attention", dense_flex_attention)


def build_model(loop_config: dict | None = LOOP_CONFIG):
    if loop_config is None:
        config = Inkling2Config(text_config=TEXT_CONFIG)
    else:
        config = Inkling2LoopConfig(text_config=TEXT_CONFIG | {"loop_config": loop_config})
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith(("tokens_per_expert", "routing_confidence_sum")):
                continue
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            if "norm.weight" in name or name.endswith("l2_clip.scale"):
                tensor.copy_(1 + 0.3 * noise.abs())
            elif "attn_res." in name or "middle_cycle" in name or name.endswith("embed_tokens.weight"):
                tensor.copy_(noise)
            else:
                tensor.copy_(0.3 * noise)
    for layer in model.model.layers:
        layer.mlp.experts.set_compute(ExactExpertCompute())
    return model.eval()


def checkpoint_weights(model) -> dict[str, torch.Tensor]:
    """The model's weights under their checkpoint names, still attached to the parameters."""
    return model.convert_to_hf(dict(model.state_dict(keep_vars=True)))


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    out = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    return out if weight is None else out * weight


def mix(sources: list[torch.Tensor], queries: torch.Tensor) -> torch.Tensor:
    values = torch.stack(sources)
    inv_rms = torch.rsqrt(values.square().mean(-1) + 1e-6)
    logits = torch.einsum("nth,qh->qtn", values, queries.reshape(-1, values.shape[-1])) * inv_rms.T
    probs = (logits / math.sqrt(values.shape[-1])).softmax(-1)
    return torch.einsum("qtn,nth->qth", probs, values * inv_rms[..., None])


def short_conv(x: torch.Tensor, taps: torch.Tensor, scale: float) -> torch.Tensor:
    """`taps[..., lag]` weighs `x[t - lag]`."""
    width = taps.shape[-1]
    padded = torch.cat([x.new_zeros(width - 1, x.shape[1]), x])
    lagged = torch.stack([padded[width - 1 - lag : padded.shape[0] - lag] for lag in range(width)], dim=-1)
    return ((lagged * taps).sum(-1) + x) * scale


def clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    x = x if scale is None else x * scale
    return x * (threshold / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)


def gate_message(w: dict, controller: str, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    z = torch.cat([rms_norm(x), rms_norm(message)], dim=-1) @ w[controller + "down_weight"].T
    forget, gain = torch.tanh(z @ w[controller + "forget_weight"].T), torch.tanh(z @ w[controller + "gain_weight"].T)
    return (1 - forget) * x + gain * message


def reference_layer(config, w: dict, i: int, blocks: list, partial: torch.Tensor, messages: torch.Tensor | None):
    """One naive layer visit of one document, read from the checkpoint layout. `messages` holds the Q, K, V and
    MoE messages of a loop core layer after the first pass."""
    num_tokens, width = partial.shape[0], config.sconv_kernel_size
    static_scale, dynamic_scale = (width + 1) ** -0.5, (2 * width + 1) ** -0.5
    distance = torch.arange(num_tokens)[:, None] - torch.arange(num_tokens)[None, :]
    layer, attn, local = f"layers.{i}.", f"layers.{i}.attn.", i in config.local_layer_ids
    starts_block = i % config.n_layers_per_attn_res_block == 0
    heads = config.swa_num_attention_heads if local else config.num_attention_heads
    kv_heads = config.swa_num_key_value_heads if local else config.num_key_value_heads
    dim = config.swa_head_dim if local else config.head_dim

    streams = [partial] * 3 if i == 0 else list(mix([*blocks, partial], w[f"attn_res.attn_queries.{i - 1}"]))
    if local:
        streams = streams[:1] * 3
    if messages is not None:
        streams = [
            gate_message(w, f"{layer}controllers.{name}.", stream, message)
            for name, stream, message in zip(("attn", "k", "v"), streams, messages)
        ]
    norms = ["attn_norm"] * 3 if local else ["q_input_norm", "k_input_norm", "v_input_norm"]
    xq, xk, xv = (rms_norm(stream, w[f"{layer}{norm}.weight"]) for stream, norm in zip(streams, norms))

    q = (xq @ w[attn + "wq_du.weight"].T).view(num_tokens, heads, dim)
    r = (xq @ w[attn + "wr_du.weight"].T).view(num_tokens, heads, config.d_rel)
    gate = (xq @ w[attn + "gate_attn.weight"].T).view(num_tokens, heads, dim)
    sink = xq @ w[attn + "attn_sink_proj.weight"].T
    k, v = xk @ w[attn + "wk_dv.weight"].T, xv @ w[attn + "wv_dv.weight"].T
    if local:
        k = short_conv(k, w[attn + "k_sconv.weight"][:, 0].flip(-1), static_scale)
        v = short_conv(v, w[attn + "v_sconv.weight"][:, 0].flip(-1), static_scale)
        tau = 1.0
    else:
        k_taps, v_taps = (
            w[f"{attn}{name}_sconv.static_w"].T
            + (source @ w[f"{attn}{name}_sconv.weight_proj"].T)
            .view(num_tokens, kv_heads, width)
            .repeat_interleave(dim, dim=1)
            for name, source in (("k", xk), ("v", xv))
        )
        k, v = short_conv(k, k_taps, dynamic_scale), short_conv(v, v_taps, dynamic_scale)
        positions = torch.arange(num_tokens) + 1
        tau = 1 + config.log_scaling_alpha * torch.log((positions / config.log_scaling_n_floor).clamp(min=1.0))
        tau = tau[:, None, None]
    q = clip(q, config.qk_l2_clip_threshold, w[attn + "q_l2_clip.scale"]) * tau
    k = clip(k.view(num_tokens, kv_heads, dim), config.qk_l2_clip_threshold).repeat_interleave(heads // kv_heads, 1)
    v = rms_norm(v.view(num_tokens, kv_heads, dim)).repeat_interleave(heads // kv_heads, 1)
    rel = clip(r, config.rel_l2_clip_threshold, w[attn + "rel_l2_clip.scale"]) * tau
    rel = rel @ w[attn + "rel_logits_proj.proj"]
    extent = rel.shape[-1]
    bias = rel.transpose(0, 1).gather(-1, distance.clamp(0, extent - 1).expand(heads, -1, -1)) * (distance < extent)
    visible = (distance >= 0) & (distance < config.sliding_window_size) if local else distance >= 0
    scores = (torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(dim) + bias).masked_fill(~visible, float("-inf"))
    out = torch.einsum("hij,jhd->ihd", scores.softmax(-1), v)
    out = out * gate.sigmoid() * (scores.logsumexp(-1).T - sink).sigmoid()[..., None]
    attn_output = short_conv(
        out.flatten(1) @ w[attn + "wo_ud.weight"].T, w[layer + "attn_sconv.weight"][:, 0].flip(-1), static_scale
    )

    sources = [*blocks, partial, attn_output] if starts_block else [*blocks, partial + attn_output]
    x = mix(sources, w[f"attn_res.mlp_queries.{i}"])[0]
    if messages is not None:
        x = gate_message(w, f"{layer}controllers.mlp.", x, messages[3])
    x = rms_norm(x, w[layer + "mlp_norm.weight"])
    logits = x @ w[layer + "mlp.gate.weight"].T
    selected = (logits.sigmoid() + w[layer + "mlp.gate.bias"]).topk(config.num_experts_per_tok).indices
    mlp_rows = []
    for token, expert_ids in enumerate(selected):
        route = logits[token, expert_ids].sigmoid()
        row = 0
        for expert, weight in zip(expert_ids.tolist(), route / route.sum() * config.route_scale):
            gate_up = x[token] @ w[layer + "mlp.experts.w13_weight"][expert].T
            gate_branch, up_branch = gate_up[0::2], gate_up[1::2]
            hidden = config.mlp_gate_softcap * torch.tanh(gate_branch / config.mlp_gate_softcap)
            hidden = hidden * gate_branch.sigmoid()
            hidden = hidden * config.mlp_up_softcap * torch.tanh(up_branch / config.mlp_up_softcap)
            row = row + weight * (hidden @ w[layer + "mlp.experts.w2_weight"][expert].T)
        mlp_rows.append(row)
    mlp_output = short_conv(torch.stack(mlp_rows), w[layer + "mlp_sconv.weight"][:, 0].flip(-1), static_scale)

    if starts_block:
        return [*blocks, partial], attn_output + mlp_output
    return blocks, partial + attn_output + mlp_output


def reference_forward(config, weights: dict, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Naive forward of one document: the final hidden states and the partial block after every layer visit."""
    w = {name.removeprefix("model.llm."): tensor.double() for name, tensor in weights.items()}
    core, visits = config.loop_core_layers, []
    blocks, partial = [], rms_norm(w["embed.weight"][input_ids], w["embed_norm.weight"])

    def run(layers, blocks, partial, bank=None):
        for i in layers:
            messages = None
            if bank is not None:
                j = i - core.start
                queries, kv_queries = w["middle_cycle_cross_pass_queries"], w["middle_cycle_cross_pass_kv_queries"]
                messages = mix(
                    bank, torch.stack([queries[2 * j], kv_queries[2 * j], kv_queries[2 * j + 1], queries[2 * j + 1]])
                )
            blocks, partial = reference_layer(config, w, i, blocks, partial, messages)
            visits.append(partial)
        return blocks, partial

    blocks, partial = run(range(core.start), blocks, partial)
    restart, bank = (blocks, partial), None
    for _ in range(getattr(config, "loop_num_passes", 0)):
        blocks, partial = run(core, *restart, bank)
        bank = [*blocks[len(restart[0]) + 1 :], partial]
    blocks, partial = run(range(core.stop, config.num_hidden_layers), blocks, partial)
    return rms_norm(mix([*blocks, partial], w["attn_res.output_query"])[0], w["norm.weight"]), visits


def assert_matches(actual: torch.Tensor, expected: torch.Tensor, what: str, tolerance: float = 3e-4) -> None:
    """The model rounds to fp32 where serving does, and the error grows with depth: compare relative to the
    tensor's scale. A wiring mistake is off by the scale itself."""
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=tolerance * scale, msg=lambda m: f"{what}: {m}")


def model_forward(model, input_ids: torch.Tensor, seq_lens: list[int], routed_experts: torch.Tensor | None = None):
    """The model's final hidden states, the partial block after every layer visit and the experts every visit picked."""
    visits, routes = [], []
    hooks = [
        layer.register_forward_hook(lambda module, args, output: visits.append(output[0][0]))
        for layer in model.model.layers
    ]
    hooks += [
        module.register_forward_hook(lambda module, args, output: routes.append(output[1]))
        for module in model.modules()
        if isinstance(module, TokenChoiceTopKRouter)
    ]
    hidden_states = model.model(
        input_ids=input_ids,
        position_ids=torch.cat([torch.arange(seq_len) for seq_len in seq_lens]).unsqueeze(0),
        seq_lens=torch.tensor(seq_lens),
        routed_experts=routed_experts,
    ).last_hidden_state[0]
    for hook in hooks:
        hook.remove()
    return hidden_states, visits, torch.stack(routes, dim=1).unsqueeze(0)


@pytest.mark.parametrize("loop_config", [LOOP_CONFIG, None], ids=["loop", "no_loop"])
@pytest.mark.parametrize("seq_lens", [[19], [9, 1, 3, 12]], ids=["one_document", "packed_documents"])
def test_forward_and_gradients_match_reference(seq_lens, loop_config):
    model = build_model(loop_config)
    config = model.config
    input_ids = torch.randint(0, 150, (1, sum(seq_lens)), generator=torch.Generator().manual_seed(1))
    hidden_states, visits, _ = model_forward(model, input_ids, seq_lens)
    assert len(visits) == config.num_layer_visits == (NUM_LOOP_VISITS if loop_config else 8)

    weights = checkpoint_weights(model)
    reference, start = [], 0
    for seq_len in seq_lens:
        document, document_visits = reference_forward(config, weights, input_ids[0, start : start + seq_len])
        for visit, (ours, theirs) in enumerate(zip(visits, document_visits, strict=True)):
            assert_matches(ours[start : start + seq_len], theirs, f"visit {visit}")
        reference.append(document)
        start += seq_len
    reference = torch.cat(reference)
    assert_matches(hidden_states, reference, "final hidden states")

    cotangent = torch.randn(reference.shape, generator=torch.Generator().manual_seed(2), dtype=reference.dtype)
    params = {name: param for name, param in model.named_parameters() if not name.startswith("lm_head.")}
    grads = torch.autograd.grad((hidden_states * cotangent).sum(), list(params.values()))
    reference_grads = torch.autograd.grad((reference * cotangent).sum(), list(params.values()))
    for name, grad, reference_grad in zip(params, grads, reference_grads):
        assert_matches(grad, reference_grad, f"gradient of {name}", tolerance=3e-3)
        if "controllers." in name or "middle_cycle" in name:
            assert reference_grad.abs().max() > 0, f"{name} does not reach the output"


def test_loop_wiring():
    model = build_model()
    seq_lens = [9, 14]
    input_ids = torch.randint(0, 150, (1, sum(seq_lens)), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        hidden_states, _, routes = model_forward(model, input_ids, seq_lens)
        assert routes.shape == (1, sum(seq_lens), NUM_LOOP_VISITS, 2)

        # Replaying the routes of each visit reproduces the forward; core layer 3 routes differently in its passes.
        replayed, _, _ = model_forward(model, input_ids, seq_lens, routed_experts=routes)
        torch.testing.assert_close(replayed, hidden_states)
        swapped = routes.clone()
        swapped[:, :, [3, 11]] = routes[:, :, [11, 3]]
        assert not torch.equal(routes[:, :, 3], routes[:, :, 11])
        assert not torch.allclose(model_forward(model, input_ids, seq_lens, routed_experts=swapped)[0], hidden_states)
        with pytest.raises(ValueError, match="once per layer visit"):
            model_forward(model, input_ids, seq_lens, routed_experts=routes[:, :, :8])

        # Each of the four messages of a core layer has its own query row.
        for table in (model.model.middle_cycle_cross_pass_queries, model.model.middle_cycle_cross_pass_kv_queries):
            original = table.clone()
            table[[0, 1]] = original[[1, 0]]
            assert not torch.allclose(model_forward(model, input_ids, seq_lens)[0], hidden_states)
            table.copy_(original)

    fewer_passes = build_model(LOOP_CONFIG | {"num_passes": 2})
    with torch.no_grad():
        assert not torch.allclose(model_forward(fewer_passes, input_ids, seq_lens)[0], hidden_states)


def test_release_layout():
    """The full-size model names and shapes its weights like the Inkling 2 Mini release."""
    config = Inkling2LoopConfig(unpadded_vocab_size=200058)
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(config)
    # The weights as the trainer broadcasts them: bf16, except what the release keeps in fp32.
    params = {
        name: param.to(torch.float32 if model.keep_in_fp32_for_weight_transfer(name) else torch.bfloat16)
        for name, param in model.named_parameters()
    }
    checkpoint = model.convert_to_hf({name: params.get(name, tensor) for name, tensor in model.state_dict().items()})
    shapes = {name.removeprefix("model.llm."): tuple(tensor.shape) for name, tensor in checkpoint.items()}
    assert all(name.startswith("model.llm.") for name in checkpoint)

    hidden, kv, inner, rank = 2304, 1024, 1152, 64
    globals_ = {5, 11, 17, 23}
    expected = {
        "embed.weight": (201024, hidden),
        "embed_norm.weight": (hidden,),
        "norm.weight": (hidden,),
        "unembed.weight": (201024, hidden),
        "attn_res.output_query": (hidden,),
        "middle_cycle_cross_pass_queries": (24, hidden),
        "middle_cycle_cross_pass_kv_queries": (24, hidden),
    }
    for i in range(24):
        layer = f"layers.{i}."
        if i < 23:
            expected[f"attn_res.attn_queries.{i}"] = (3, hidden) if i + 1 in globals_ else (hidden,)
        expected[f"attn_res.mlp_queries.{i}"] = (hidden,)
        expected |= {
            layer + "attn.wq_du.weight": (3072, hidden),
            layer + "attn.wk_dv.weight": (kv, hidden),
            layer + "attn.wv_dv.weight": (kv, hidden),
            layer + "attn.wo_ud.weight": (hidden, 3072),
            layer + "attn.wr_du.weight": (384, hidden),
            layer + "attn.gate_attn.weight": (3072, hidden),
            layer + "attn.attn_sink_proj.weight": (24, hidden),
            layer + "attn.q_l2_clip.scale": (24, 128),
            layer + "attn.rel_l2_clip.scale": (24, 16),
            layer + "attn.rel_logits_proj.proj": (16, 1024),
            layer + "attn_sconv.weight": (hidden, 1, 4),
            layer + "mlp_sconv.weight": (hidden, 1, 4),
            layer + "mlp_norm.weight": (hidden,),
            layer + "mlp.gate.weight": (128, hidden),
            layer + "mlp.gate.bias": (128,),
            layer + "mlp.experts.w13_weight": (128, 2 * inner, hidden),
            layer + "mlp.experts.w2_weight": (128, hidden, inner),
        }
        if i in globals_:
            expected |= {f"{layer}{name}_input_norm.weight": (hidden,) for name in "qkv"}
            expected |= {f"{layer}attn.{name}_sconv.static_w": (4, kv) for name in "kv"}
            expected |= {f"{layer}attn.{name}_sconv.weight_proj": (32, hidden) for name in "kv"}
        else:
            expected[layer + "attn_norm.weight"] = (hidden,)
            expected |= {f"{layer}attn.{name}_sconv.weight": (kv, 1, 4) for name in "kv"}
        if 6 <= i < 18:
            for name in ("attn", "k", "v", "mlp"):
                expected[f"{layer}controllers.{name}.down_weight"] = (rank, 2 * hidden)
                expected[f"{layer}controllers.{name}.gain_weight"] = (hidden, rank)
                expected[f"{layer}controllers.{name}.forget_weight"] = (hidden, rank)
    assert shapes == expected

    fp32 = {name for name, tensor in checkpoint.items() if tensor.dtype == torch.float32}
    assert fp32 == {
        "model.llm." + name
        for name in expected
        if name.startswith(("attn_res.", "middle_cycle_")) or name.endswith("mlp.gate.bias")
    }
    restored = model.convert_to_prime(dict(checkpoint))
    assert restored.keys() == model.state_dict().keys()


@pytest.mark.parametrize(
    "loop_config",
    [
        {"core_start": 1, "core_end": 6},
        {"core_start": 2, "core_end": 5},
        {"core_start": 0, "core_end": 4},
        {"core_start": 2, "core_end": 10},
        {"num_passes": 0},
        {"gate_rank": 0},
        {"is_varloop": True},
        {"shared_cache": True},
    ],
    ids=str,
)
def test_invalid_loops_are_rejected(loop_config):
    with pytest.raises(ValueError):
        Inkling2LoopConfig(text_config=TEXT_CONFIG | {"loop_config": LOOP_CONFIG | loop_config})


def test_loop_config_is_not_dropped():
    release_style = {"model_type": "inkling2_loop", "text_config": TEXT_CONFIG | {"loop_config": LOOP_CONFIG}}
    config = AutoConfig.for_model(**release_style)
    assert isinstance(config, Inkling2LoopConfig)
    assert config.loop_core_layers == range(2, 6) and config.loop_num_passes == 3 and config.loop_gate_rank == 4
    assert config.num_layer_visits == NUM_LOOP_VISITS
    restored = Inkling2LoopConfig.from_dict(config.to_dict())
    assert restored.loop_core_layers == range(2, 6) and restored.num_layer_visits == NUM_LOOP_VISITS

    # A looped checkpoint read as the non-looped model would load without its loop.
    with pytest.raises(ValueError, match="inkling2_loop"):
        Inkling2Config(text_config=TEXT_CONFIG | {"loop_config": LOOP_CONFIG})
    # The trainer can drop layers after the config is built.
    config.num_hidden_layers = 4
    with pytest.raises(ValueError, match="loop core"):
        AutoModelForCausalLMPrimeRL.from_config(config)
    assert Inkling2LoopTextConfig().num_layer_visits == 48


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def write_quantized_release(release: Path) -> dict[str, torch.Tensor]:
    """A two-shard release with an NVFP4 layer and an MXFP8 layer; returns the bf16 weights it encodes."""
    generator = torch.Generator().manual_seed(0)
    experts, rows, width = 3, 4, 64

    def e4m3(shape, low=0, high=0x7F):
        # Both signs, subnormals included, without the NaN code 0x7F.
        codes = torch.randint(low, high, shape, generator=generator, dtype=torch.uint8)
        return codes | (torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8) << 7)

    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    first, second = shards.values()
    expected = {}
    for name, sidecar_shard in (("w13", first), ("w2", second)):
        stem = f"model.llm.layers.0.mlp.experts.{name}_weight"
        packed = torch.randint(0, 256, (experts, rows, width // 2), generator=generator, dtype=torch.uint8)
        scale = e4m3((experts, rows, width // 16))
        amax = torch.rand(experts, generator=generator) + 0.5
        denominator = torch.tensor([448.0])
        first[stem] = packed
        # An expert weight's scales can live in another shard.
        sidecar_shard[stem + "_scale"] = scale.view(torch.float8_e4m3fn)
        sidecar_shard[stem + "_amax"] = amax
        sidecar_shard[stem + "_outer_scale_denominator"] = denominator
        values = E2M1[torch.stack((packed & 0xF, packed >> 4), dim=-1).long()].view(experts, rows, -1, 16)
        values = values * scale.view(torch.float8_e4m3fn).float()[..., None] * (amax / denominator)[:, None, None, None]
        expected[stem] = values.flatten(-2).to(torch.bfloat16)

        stem = f"model.llm.layers.1.mlp.experts.{name}_weight"
        weight = e4m3((experts, rows, width)).view(torch.float8_e4m3fn)
        scale = torch.randint(120, 134, (experts, rows, width // 32), generator=generator, dtype=torch.uint8)
        second[stem] = weight
        second[stem + "_scale"] = scale
        values = weight.float().view(experts, rows, -1, 32) * torch.exp2(scale.float() - 127)[..., None]
        expected[stem] = values.flatten(-2).to(torch.bfloat16)
    first["model.llm.norm.weight"] = expected["model.llm.norm.weight"] = torch.randn(8).to(torch.bfloat16)
    second["model.llm.attn_res.output_query"] = expected["model.llm.attn_res.output_query"] = torch.randn(8)

    for shard, tensors in shards.items():
        save_file(tensors, release / shard, metadata={"format": "pt"})
    weight_map = {name: shard for shard, tensors in shards.items() for name in tensors}
    (release / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = {
        "model_type": "inkling2_loop",
        "text_config": {"loop_config": LOOP_CONFIG},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }
    (release / "config.json").write_text(json.dumps(config))
    groups = [
        {"compute_dtype": "float32", "tensors": ["model.llm.attn_res.output_query"]},
        {"compute_dtype": "bfloat16", "tensors": ["model.llm.norm.weight"]},
        {"compute_dtype": "nvfp4", "tensors": [name for name in expected if ".layers.0." in name]},
        {"compute_dtype": "mxfp8", "tensors": [name for name in expected if ".layers.1." in name]},
    ]
    quant_config = {"quant_method": "inkling2", "model": {"parameter_groups": groups}}
    (release / "hf_quant_config.json").write_text(json.dumps(quant_config))
    save_file({"fuse_weight": torch.randn(4)}, release / "mtp.safetensors")
    (release / "tokenizer.json").write_text("{}")
    return expected


def test_convert_release_to_bf16(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "tools"))
    import convert_inkling2_to_bf16

    def convert(*args: Path):
        monkeypatch.setattr(sys, "argv", ["convert_inkling2_to_bf16.py", *map(str, args)])
        convert_inkling2_to_bf16.main()

    release, output = tmp_path / "release", tmp_path / "release-BF16"
    release.mkdir()
    expected = write_quantized_release(release)
    convert(release)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"].keys() == expected.keys()
    converted = {}
    for shard in set(index["weight_map"].values()):
        converted |= load_file(output / shard)
    assert converted.keys() == expected.keys()
    for name, tensor in expected.items():
        assert converted[name].dtype == tensor.dtype and converted[name].shape == tensor.shape, name
        # Bit for bit, signed zeros included.
        assert converted[name].contiguous().view(-1).view(torch.uint8).equal(tensor.view(-1).view(torch.uint8)), name

    config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in config and config["text_config"]["loop_config"] == LOOP_CONFIG
    # vLLM would read a quantization method from the release's precision metadata.
    assert not (output / "hf_quant_config.json").exists()
    for side_file in ("mtp.safetensors", "tokenizer.json"):
        assert (output / side_file).read_bytes() == (release / side_file).read_bytes()

    # A rerun checks the shards it finds: complete ones are kept, a truncated one is an error.
    convert(release, output)
    with pytest.raises(SystemExit):
        convert(release, release)
    shard = output / "model-00001-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-2])
    with pytest.raises(ValueError, match="file size"):
        convert(release, output)

    # The precision metadata must agree with the tensor dtypes.
    quant_config = json.loads((release / "hf_quant_config.json").read_text())
    quant_config["model"]["parameter_groups"][2]["tensors"].pop()
    (release / "hf_quant_config.json").write_text(json.dumps(quant_config))
    with pytest.raises(ValueError, match="disagree"):
        convert(release, tmp_path / "other")
