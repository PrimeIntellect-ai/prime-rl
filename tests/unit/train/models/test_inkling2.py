import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, supports_custom_impl
from prime_rl.trainer.models.inkling2 import Inkling2Config, Inkling2TextConfig, modeling_inkling2
from prime_rl.trainer.models.layers.activations import SoftcappedSwiglu
from prime_rl.trainer.vlm import get_language_model, get_layer_prefix

# Layers 2 and 5 attend globally, blocks span 3 layers. The clip thresholds are sized for 16-dim heads: every
# vector is clipped, and scores stay O(1) so the softmax does not amplify rounding differences.
TEXT_CONFIG = dict(
    vocab_size=160,
    unpadded_vocab_size=150,
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=6,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=6,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    sliding_window_size=8,
    local_layer_ids=[0, 1, 3, 4, 6, 7],
    d_rel=4,
    rel_extent=12,
    log_scaling_n_floor=4,
    qk_l2_clip_threshold=2.0,
    rel_l2_clip_threshold=1.0,
    n_layers_per_attn_res_block=3,
    n_routed_experts=8,
    num_experts_per_tok=2,
    intermediate_size=32,
    mlp_up_softcap=2.0,
)


class ExactExpertCompute:
    """Expert matmuls in the parameter dtype. The shared expert kernel rounds to BF16, which a float64
    comparison cannot see through. Every expert runs on every row and each row keeps its own, so the shapes
    do not depend on the routing and a compiled layer traces it without a break."""

    token_group_alignment = 8

    def validate(self, experts) -> None:
        pass

    def __call__(self, experts, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(x.shape[0], device=x.device)
        # Rows are grouped by expert; the rows after the last group are padding.
        expert = torch.searchsorted(num_tokens_per_expert.cumsum(0), rows, right=True)
        hidden = experts.activation.apply(x @ experts.gate_proj.mT, x @ experts.up_proj.mT)
        outputs = (hidden @ experts.down_proj.mT)[expert.clamp(max=experts.num_experts - 1), rows]
        return outputs * (expert < experts.num_experts).unsqueeze(-1)


def build_model(dtype: torch.dtype = torch.float64, **text_overrides):
    config = Inkling2Config(text_config=TEXT_CONFIG | text_overrides)
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=dtype)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            if "norm.weight" in name or name.endswith(("q_scale", "rel_scale")):
                tensor.copy_(1 + 0.3 * noise.abs())
            elif name.endswith(("query", "embed_tokens.weight")):
                tensor.copy_(noise)
            else:
                tensor.copy_(0.3 * noise)
    for layer in get_language_model(model).layers:
        layer.mlp.experts.set_compute(ExactExpertCompute())
    return model.eval()


@pytest.fixture
def model():
    return build_model()


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


def reference_hidden_states(config: Inkling2TextConfig, weights: dict, input_ids: torch.Tensor) -> torch.Tensor:
    """Naive forward of one document, read from the checkpoint layout."""
    w = {name.removeprefix("model.llm."): tensor.double() for name, tensor in weights.items()}
    num_tokens, width = input_ids.numel(), config.sconv_kernel_size
    static_scale, dynamic_scale = (width + 1) ** -0.5, (2 * width + 1) ** -0.5
    distance = torch.arange(num_tokens)[:, None] - torch.arange(num_tokens)[None, :]
    partial, blocks = rms_norm(w["embed.weight"][input_ids], w["embed_norm.weight"]), []
    for i in range(config.num_hidden_layers):
        layer, attn, local = f"layers.{i}.", f"layers.{i}.attn.", config.is_local_layer(i)
        starts_block = i % config.n_layers_per_attn_res_block == 0
        heads = config.swa_num_attention_heads if local else config.num_attention_heads
        kv_heads = config.swa_num_key_value_heads if local else config.num_key_value_heads
        dim = config.swa_head_dim if local else config.head_dim

        streams = [partial] * 3 if i == 0 else mix([*blocks, partial], w[f"attn_res.attn_queries.{i - 1}"])
        if local:
            xq = xk = xv = rms_norm(streams[0], w[layer + "attn_norm.weight"])
        else:
            xq, xk, xv = (rms_norm(streams[j], w[f"{layer}{name}_input_norm.weight"]) for j, name in enumerate("qkv"))
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
            for name, source in (("k", xk), ("v", xv)):
                dynamic = (source @ w[f"{attn}{name}_sconv.weight_proj"].T).view(num_tokens, kv_heads, width)
                taps = w[f"{attn}{name}_sconv.static_w"].T + dynamic.repeat_interleave(dim, dim=1)
                if name == "k":
                    k = short_conv(k, taps, dynamic_scale)
                else:
                    v = short_conv(v, taps, dynamic_scale)
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
        x = rms_norm(mix(sources, w[f"attn_res.mlp_queries.{i}"])[0], w[layer + "mlp_norm.weight"])
        logits = x @ w[layer + "mlp.gate.weight"].T
        selected = (logits.sigmoid() + w[layer + "mlp.gate.bias"]).topk(config.num_experts_per_tok).indices
        mlp_output = torch.zeros_like(x)
        for token, expert_ids in enumerate(selected):
            route = logits[token, expert_ids].sigmoid()
            for expert, weight in zip(expert_ids.tolist(), route / route.sum() * config.route_scale):
                gate_up = x[token] @ w[layer + "mlp.experts.w13_weight"][expert].T
                gate_branch, up_branch = gate_up[0::2], gate_up[1::2]
                hidden = config.mlp_gate_softcap * torch.tanh(gate_branch / config.mlp_gate_softcap)
                hidden = hidden * gate_branch.sigmoid()
                hidden = hidden * config.mlp_up_softcap * torch.tanh(up_branch / config.mlp_up_softcap)
                mlp_output[token] += weight * (hidden @ w[layer + "mlp.experts.w2_weight"][expert].T)
        mlp_output = short_conv(mlp_output, w[layer + "mlp_sconv.weight"][:, 0].flip(-1), static_scale)

        if starts_block:
            blocks.append(partial)
            partial = attn_output + mlp_output
        else:
            partial = partial + attn_output + mlp_output
    return rms_norm(mix([*blocks, partial], w["attn_res.output_query"])[0], w["norm.weight"])


def checkpoint_weights(model) -> dict[str, torch.Tensor]:
    return model.convert_to_hf({name: tensor.clone() for name, tensor in model.state_dict().items()})


def test_config_matches_checkpoint_layout():
    text = Inkling2TextConfig()
    assert [idx for idx in range(text.num_hidden_layers) if not text.is_local_layer(idx)] == list(range(5, 48, 6))
    config = Inkling2Config(text_config=text.to_dict(), vision_config={"hidden_size": 32}, mtp_config={"block_size": 8})
    restored = Inkling2Config.from_dict(config.to_dict())
    assert isinstance(restored.text_config, Inkling2TextConfig)
    assert restored.vision_config == {"hidden_size": 32}
    assert restored.mtp_config == {"block_size": 8}
    assert supports_custom_impl(restored)
    with pytest.raises(ValueError, match="n_shared_experts"):
        Inkling2TextConfig(n_shared_experts=1)


def test_state_dict_roundtrips_through_checkpoint_names(model):
    assert get_layer_prefix(model) == "model.language_model.layers."
    prime = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    checkpoint = checkpoint_weights(model)

    assert {name for name in checkpoint if ".layers." not in name} == {
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
        "model.llm.unembed.weight",
        "model.llm.attn_res.output_query",
        *(f"model.llm.attn_res.attn_queries.{idx}" for idx in range(7)),
        *(f"model.llm.attn_res.mlp_queries.{idx}" for idx in range(8)),
    }
    layer = "model.llm.layers.2."
    assert {name.removeprefix(layer) for name in checkpoint if name.startswith(layer)} == {
        "q_input_norm.weight",
        "k_input_norm.weight",
        "v_input_norm.weight",
        "mlp_norm.weight",
        "attn_sconv.weight",
        "mlp_sconv.weight",
        "attn.wq_du.weight",
        "attn.wk_dv.weight",
        "attn.wv_dv.weight",
        "attn.wr_du.weight",
        "attn.gate_attn.weight",
        "attn.attn_sink_proj.weight",
        "attn.wo_ud.weight",
        "attn.q_l2_clip.scale",
        "attn.rel_l2_clip.scale",
        "attn.rel_logits_proj.proj",
        "attn.k_sconv.static_w",
        "attn.k_sconv.weight_proj",
        "attn.v_sconv.static_w",
        "attn.v_sconv.weight_proj",
        "mlp.gate.weight",
        "mlp.gate.bias",
        "mlp.experts.w13_weight",
        "mlp.experts.w2_weight",
    }
    # Full-attention layers mix separate query, key and value inputs; the unembedding keeps its padding rows.
    assert checkpoint["model.llm.attn_res.attn_queries.1"].shape == (3, 64)
    assert checkpoint["model.llm.attn_res.attn_queries.0"].shape == (64,)
    assert checkpoint["model.llm.unembed.weight"].shape == (160, 64)
    gate_up = checkpoint[layer + "mlp.experts.w13_weight"]
    assert torch.equal(gate_up[:, 0::2], prime["model.language_model.layers.2.mlp.experts.gate_proj"])
    assert torch.equal(gate_up[:, 1::2], prime["model.language_model.layers.2.mlp.experts.up_proj"])

    restored = model.convert_to_prime({name: tensor.clone() for name, tensor in checkpoint.items()})
    assert restored.keys() == prime.keys()
    assert all(torch.equal(restored[name], prime[name]) for name in prime)


def test_quantized_experts_are_rejected(model):
    checkpoint = checkpoint_weights(model)
    name = "model.llm.layers.0.mlp.experts.w13_weight"
    checkpoint[name] = torch.zeros(8, 64, 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match="convert_inkling2_to_bf16"):
        model.convert_to_prime(checkpoint)


# A clipped key or value forgets the scale of its short convolution, and a clip that always binds never
# takes its pass-through branch: also run with clips that bind for some vectors, and for none.
@pytest.mark.parametrize(
    "clip_thresholds", [(2.0, 1.0), (8.0, 4.0), (1e3, 1e3)], ids=["all_clipped", "some_clipped", "none_clipped"]
)
@pytest.mark.parametrize("seq_lens", [[19], [9, 1, 3, 12]], ids=["one_document", "packed_documents"])
def test_forward_matches_reference(seq_lens, clip_thresholds):
    qk_threshold, rel_threshold = clip_thresholds
    model = build_model(qk_l2_clip_threshold=qk_threshold, rel_l2_clip_threshold=rel_threshold)
    text_config = model.config.text_config
    input_ids = torch.randint(0, 150, (1, sum(seq_lens)), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        hidden_states = model.model(input_ids=input_ids, seq_lens=torch.tensor(seq_lens)).last_hidden_state[0]
        logits = model(input_ids=input_ids, seq_lens=torch.tensor(seq_lens))["logits"][0]

    weights = checkpoint_weights(model)
    start = 0
    for seq_len in seq_lens:
        reference = reference_hidden_states(text_config, weights, input_ids[0, start : start + seq_len])
        torch.testing.assert_close(hidden_states[start : start + seq_len], reference, rtol=1e-3, atol=1e-4)
        start += seq_len

    # The head covers the real vocabulary only, and its logits are soft-capped.
    cap = text_config.final_logit_softcapping
    unembed = weights["model.llm.unembed.weight"][: text_config.unpadded_vocab_size]
    torch.testing.assert_close(logits, cap * torch.tanh(hidden_states @ unembed.T / cap))


def packed_row(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Two documents, each longer than the sliding window and the relative extent."""
    input_ids = torch.randint(0, 150, (1, 150), generator=torch.Generator().manual_seed(2)).to(device)
    return input_ids, torch.tensor([70, 80])


def dense_routing(text_model, input_ids: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """The experts the dense path routes every token to, as `routed_experts [1, tokens, layers, top_k]`. Both
    paths replay them: a near-tie that fell the other way would stand in for an attention difference."""
    routes = []
    hooks = [
        layer.mlp.router.register_forward_hook(lambda module, args, output: routes.append(output[1]))
        for layer in text_model.layers
    ]
    text_model.attention_backend = "dense"
    with torch.no_grad():
        text_model(input_ids=input_ids, seq_lens=seq_lens)
    for hook in hooks:
        hook.remove()
    return torch.stack(routes, dim=1).unsqueeze(0)


def assert_attention_paths_match(dtype: torch.dtype, device: str) -> None:
    """`flex_relative_attention` against `dense_relative_attention` on a packed row, over a window and over
    whole documents. Both compute in fp32 whatever the inputs' dtype."""
    generator = torch.Generator().manual_seed(3)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(device, dtype)

    _, doc_ids = modeling_inkling2.document_layout(torch.tensor([0, 70, 150], device=device))
    padded_doc_ids = F.pad(doc_ids, (0, modeling_inkling2.flex_length(150) - 150), value=-1)
    q, k, v = draw(150, 6, 16), draw(150, 2, 16), draw(150, 2, 16)
    for window, extent in ((8, 8), (None, 12)):
        rel = draw(150, 6, extent)
        block_mask = modeling_inkling2.relative_block_mask(padded_doc_ids, window)
        flex = modeling_inkling2.flex_relative_attention(q, k, v, rel, block_mask)
        dense = modeling_inkling2.dense_relative_attention(q, k, v, rel, doc_ids, window)
        for flex_part, dense_part in zip(flex, dense):
            torch.testing.assert_close(flex_part, dense_part, rtol=1e-4, atol=1e-4)


def test_flex_attention_matches_dense(model, monkeypatch):
    # Uncompiled, so this checks the score and mask logic, not the fused kernel.
    monkeypatch.setattr(modeling_inkling2, "compiled_flex_attention", lambda: flex_attention)
    for dtype in (torch.bfloat16, torch.float32):
        assert_attention_paths_match(dtype, "cpu")

    text_model = get_language_model(model)
    input_ids, seq_lens = packed_row("cpu")
    routed_experts = dense_routing(text_model, input_ids, seq_lens)
    hidden_states = {}
    for backend in ("dense", "flex"):
        text_model.attention_backend = backend
        with torch.no_grad():
            hidden_states[backend] = text_model(
                input_ids=input_ids, seq_lens=seq_lens, routed_experts=routed_experts
            ).last_hidden_state
    # Both paths compute their scores in fp32; a wrong mask or bias would be off by O(1).
    torch.testing.assert_close(hidden_states["flex"], hidden_states["dense"], rtol=3e-4, atol=3e-4)


@pytest.fixture
def trainer_kernels(monkeypatch):
    """Kernels built as in a training run, but with IEEE fp32 products so that tight tolerances hold: the
    trainer's entrypoints pick this assembler (`prime_rl._compat`), and its default matmul precision builds
    the attention kernel with TF32, which these tests do not cover."""
    import triton.knobs

    monkeypatch.setenv("TRITON_PTXAS_PATH", triton.knobs.nvidia.ptxas_blackwell.path)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_compiled_flex_attention_matches_dense(dtype, trainer_kernels):
    # The fused kernel, which packed rows take on CUDA, on inputs of the dtypes training computes in.
    assert_attention_paths_match(dtype, "cuda")


@pytest.mark.gpu
def test_compiled_layers_backward_through_flex_attention(trainer_kernels):
    # The kernel's forward and backward inside compiled layers, as the trainer runs them, against the dense
    # path's gradients without compilation.
    model = build_model(torch.float32).to("cuda").train()
    text_model = get_language_model(model)
    input_ids, seq_lens = packed_row("cuda")
    routed_experts = dense_routing(text_model, input_ids, seq_lens)
    gradients = {}
    for backend in ("dense", "flex"):
        text_model.attention_backend = backend
        if backend == "flex":
            for layer in text_model.layers:
                layer.compile()
        model.zero_grad()
        logits = model(input_ids=input_ids, seq_lens=seq_lens, routed_experts=routed_experts)["logits"][0]
        F.cross_entropy(logits[:-1], input_ids[0, 1:]).backward()
        gradients[backend] = {name: param.grad.clone() for name, param in model.named_parameters()}

    for name, dense in gradients["dense"].items():
        difference = (gradients["flex"][name] - dense).abs().max()
        assert difference <= 1e-3 * dense.abs().max(), name


def test_each_layer_type_compiles_once():
    # The trainer compiles every layer. The eight layers are of four types (the first, one that starts a
    # block, a local and a full-attention one), and all but the first come behind one, two or three completed
    # blocks. Three slots also tell a placeholder shared between slots from one per slot.
    torch._dynamo.reset()
    model = AutoModelForCausalLMPrimeRL.from_config(Inkling2Config(text_config=TEXT_CONFIG))
    model.init_buffers_post_meta()

    def run_uncompiled(graph_module, example_inputs):
        return graph_module.forward

    for layer in get_language_model(model).layers:
        layer.compile(backend=run_uncompiled)
    model.model(input_ids=torch.zeros(1, 24, dtype=torch.long), seq_lens=torch.tensor([24]))
    compiled = torch._dynamo.eval_frame._debug_get_cache_entry_list(modeling_inkling2.Inkling2DecoderLayer.forward)
    torch._dynamo.reset()
    assert len(compiled) == 4


def test_softcap_of_zero_is_no_cap():
    # The serving kernels read a cap of 0 as "no cap"; dividing by it would give zeros and NaN.
    gate = torch.tensor([-2.0, 0.0, 0.5, 3.0])
    up = torch.tensor([1.0, -1.0, 0.0, 2.0])
    swiglu = gate * torch.sigmoid(gate) * up
    torch.testing.assert_close(SoftcappedSwiglu(0.0, 0.0).apply(gate, up), swiglu)
    torch.testing.assert_close(
        SoftcappedSwiglu(4.0, 0.0).apply(gate, up), 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * up
    )


def test_init_buffers_post_meta_zeroes_checkpoint_buffers(model):
    # A model initialized from scratch never loads them, and they are exported with the checkpoint.
    text_model = get_language_model(model)
    selection_biases = [layer.mlp.router.selection_bias for layer in text_model.layers]
    with torch.no_grad():
        for buffer in (text_model.unembed_padding, *selection_biases):
            buffer.fill_(7.0)

    model.init_buffers_post_meta()

    for buffer in (text_model.unembed_padding, *selection_biases):
        assert buffer.count_nonzero() == 0
