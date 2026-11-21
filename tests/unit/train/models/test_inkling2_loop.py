import copy
import inspect

import pytest
import torch
from safetensors.torch import load_file

from prime_rl.trainer.models import get_custom_causal_lm_cls, get_custom_vlm_cls, supports_custom_impl
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopConfig, Inkling2LoopForCausalLM, Inkling2LoopTextConfig
from prime_rl.trainer.models.inkling2_loop import attention as inkling2_attention
from prime_rl.trainer.models.inkling2_loop.attention import rel_attention
from prime_rl.trainer.models.inkling2_loop.dequantize import (
    E2M1_VALUES,
    dequantize_state_dict_,
    nvfp4_requantization_mismatches,
)
from prime_rl.trainer.models.inkling2_loop.layers import (
    DynamicShortConv,
    Inkling2FinalNorm,
    Inkling2RMSNorm,
    MessageGate,
    StaticShortConv,
    depth_mix,
    l2_clip,
)
from prime_rl.trainer.models.inkling2_loop.modeling_inkling2_loop import Inkling2DecoderLayer, softcapped_swiglu
from prime_rl.trainer.models.layers.activations import SoftcappedSwiglu
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE
from prime_rl.trainer.vlm import get_language_model, get_layer_prefix
from tests.unit.train.models.fixtures.make_inkling2_loop_golden import (
    GOLDEN,
    TEXT_CONFIG,
    checkpoint_digest,
    checkpoint_tensor,
)

SEQ_LENS = (23, 40)


@pytest.fixture(autouse=True, scope="module")
def mkl_kernels_chosen():
    """torch's MKL picks its vector-math kernels on the first call of the process. When several OpenMP threads make
    that call together, one of them can run a low-accuracy kernel (exp off by 3e-9 in FP64, 1.5e-4 in FP32), which
    moves the first forward of a process: a test of this file run alone then fails a few times in a hundred. A call
    on one element runs on one thread and settles the choice."""
    torch.exp(torch.zeros(1, dtype=torch.float64))


def _text_config(**overrides) -> Inkling2LoopTextConfig:
    """Eight layers in blocks of two: sliding layers 0, 2, 4, 6 and full layers 1, 3, 5, 7, with layers 2..5 looped
    three times. Sliding and full layers get different head counts so a mix-up changes shapes."""
    values = dict(
        vocab_size=320,
        unpadded_vocab_size=300,
        hidden_size=64,
        num_hidden_layers=8,
        n_layers_per_attn_res_block=2,
        loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=8, is_varloop=False),
        local_layer_ids=[0, 2, 4, 6],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=6,
        swa_num_key_value_heads=3,
        swa_head_dim=8,
        sliding_window_size=8,
        rel_extent=12,
        d_rel=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=32,
        route_scale=2.0,
        log_scaling_n_floor=4,
        qk_l2_clip_threshold=2.0,
        rel_l2_clip_threshold=1.0,
        eos_token_id=1,
    )
    values.update(overrides)
    return Inkling2LoopTextConfig(**values)


class _ExactExpertCompute:
    """Expert compute in the input's own dtype, so a CPU forward is exact instead of BF16-rounded."""

    token_group_alignment = 8

    def validate(self, experts: GroupedExperts) -> None:
        return None

    def __call__(self, experts: GroupedExperts, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        outputs = []
        start = 0
        for expert, count in enumerate(num_tokens_per_expert.tolist()):
            rows = x[start : start + count]
            hidden = experts.activation.apply(rows @ experts.gate_proj[expert].T, rows @ experts.up_proj[expert].T)
            outputs.append(hidden @ experts.down_proj[expert].T)
            start += count
        return torch.cat(outputs + [x.new_zeros((x.shape[0] - start, x.shape[1]))])


def _randomized_model(config: Inkling2LoopTextConfig, seed: int = 0) -> Inkling2LoopForCausalLM:
    """A model whose every path matters: the default init has zero mixing queries and near-zero activations."""
    model = Inkling2LoopForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            if name.endswith(("layernorm.weight", "norm.weight", "q_scale", "rel_scale")):
                param.copy_(1.0 + 0.3 * noise)
            elif name.endswith("query") or name.endswith("message_queries"):
                param.copy_(noise)
            elif name.endswith(("conv.weight", "static_weight")):
                param.copy_(0.3 * noise)
            else:
                param.copy_(1.5 * noise / param.shape[-1] ** 0.5)
        for module in model.modules():
            if isinstance(module, MoE):
                module.router.selection_bias.copy_(
                    0.1 * torch.randn(module.router.selection_bias.shape, generator=generator)
                )
                module.experts.set_compute(_ExactExpertCompute())
    return model


def _logits(
    model: Inkling2LoopForCausalLM,
    input_ids: torch.Tensor,
    seq_lens: list[int],
    routed_experts: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        output = model(input_ids=input_ids.unsqueeze(0), seq_lens=torch.tensor(seq_lens), routed_experts=routed_experts)
    return output["logits"][0]


def test_release_config_selects_the_text_model():
    config = Inkling2LoopConfig(
        text_config=_text_config().to_dict(),
        vision_config={"hidden_size": 32},
        quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    restored = Inkling2LoopConfig.from_dict(config.to_dict())
    assert isinstance(restored.text_config, Inkling2LoopTextConfig)
    assert restored.text_config.layer_types == ["sliding_attention", "full_attention"] * 4
    assert restored.text_config.loop_core == range(2, 6)
    assert restored.vision_config == {"hidden_size": 32}
    assert not hasattr(restored, "quantization_config")
    assert supports_custom_impl(restored)
    assert get_custom_causal_lm_cls(restored) is Inkling2LoopForCausalLM
    # Text only, as Inkling 2 Small: not a VLM entry, so `get_model` rejects `model.vlm` for it.
    assert get_custom_vlm_cls(restored) is None

    defaults = Inkling2LoopTextConfig()
    assert [idx for idx, kind in enumerate(defaults.layer_types) if kind == "full_attention"] == [5, 11, 17, 23]
    assert defaults.loop_core == range(6, 18)
    assert defaults.unsupported_variant() is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(n_shared_experts=1),
        dict(use_gate_bias=False),
        dict(loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=8, is_varloop=True)),
        dict(loop_config=dict(core_start=2, core_end=6, num_passes=1, gate_rank=8)),
        dict(loop_config=dict(core_start=1, core_end=6, num_passes=3, gate_rank=8)),
        dict(loop_config=dict(core_start=2, core_end=5, num_passes=3, gate_rank=8)),
        dict(unpadded_vocab_size=321),
    ],
)
def test_unsupported_checkpoint_variants_are_rejected(overrides):
    with pytest.raises(NotImplementedError):
        Inkling2LoopForCausalLM(_text_config(**overrides))


def test_checkpoint_conversion_roundtrip_and_layout():
    config = _text_config()
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(config)
    assert get_layer_prefix(model) == "model.layers."
    assert len(get_language_model(model).layers) == 8

    prime = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    expected = {name: tensor.clone() for name, tensor in prime.items()}
    checkpoint = model.convert_to_hf(dict(prime))
    assert model.is_hf_state_dict(checkpoint)
    assert not model.is_prime_state_dict(checkpoint)
    assert all(name.startswith("model.llm.") for name in checkpoint)

    experts = "model.layers.3.mlp.experts"
    w13 = checkpoint["model.llm.layers.3.mlp.experts.w13_weight"]
    torch.testing.assert_close(w13[:, 0::2], expected[f"{experts}.gate_proj"])
    torch.testing.assert_close(w13[:, 1::2], expected[f"{experts}.up_proj"])
    torch.testing.assert_close(checkpoint["model.llm.layers.3.mlp.experts.w2_weight"], expected[f"{experts}.down_proj"])
    # The checkpoint's attention queries start at layer 1; full layers hold one per stream.
    assert "model.llm.attn_res.attn_queries.7" not in checkpoint
    torch.testing.assert_close(
        checkpoint["model.llm.attn_res.attn_queries.2"], expected["model.layers.3.attention_input.query"]
    )
    assert checkpoint["model.llm.attn_res.attn_queries.2"].shape == (3, 64)
    assert checkpoint["model.llm.attn_res.attn_queries.1"].shape == (64,)
    # (query, mlp) and (key, value) pairs per core layer.
    message_queries = expected["model.message_queries"]
    torch.testing.assert_close(
        checkpoint["model.llm.middle_cycle_cross_pass_queries"], message_queries[:, [0, 3]].flatten(0, 1)
    )
    torch.testing.assert_close(
        checkpoint["model.llm.middle_cycle_cross_pass_kv_queries"], message_queries[:, [1, 2]].flatten(0, 1)
    )
    unembed = checkpoint["model.llm.unembed.weight"]
    assert unembed.shape == (320, 64)
    assert unembed[300:].abs().max() == 0
    torch.testing.assert_close(unembed[:300], expected["lm_head.weight"])

    # The draft head and the vision tower ship next to the language model.
    checkpoint["layers.0.mlp.down_proj.weight"] = torch.zeros(1)
    checkpoint["model.visual.proj_out.weight"] = torch.zeros(1)
    roundtrip = model.convert_to_prime(checkpoint)
    assert roundtrip.keys() == expected.keys()
    for name, tensor in roundtrip.items():
        torch.testing.assert_close(tensor, expected[name])


def test_truncated_model_loops_whole_blocks_and_loads_the_checkpoint():
    """`debug.num_layers` cuts the stack. A cut inside the loop core keeps the core's whole blocks and runs the layers
    after them once. The message queries keep every checkpoint row, so a conversion of the full model loads too."""
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(_text_config())
        truncated = Inkling2LoopForCausalLM(_text_config(num_hidden_layers=5))
        without_core = Inkling2LoopForCausalLM(_text_config(num_hidden_layers=3))
    assert truncated.model.loop_core == range(2, 4)
    assert truncated.model.num_layer_visits == 9
    assert len(without_core.model.loop_core) == 0 and without_core.model.message_queries is None

    state_dict = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    checkpoint = model.convert_to_hf(dict(state_dict))
    converted = truncated.convert_to_prime(dict(checkpoint))
    assert {name: converted[name].shape for name in truncated.state_dict()} == {
        name: tensor.shape for name, tensor in truncated.state_dict().items()
    }
    torch.testing.assert_close(converted["model.message_queries"], state_dict["model.message_queries"])
    full_conversion = model.convert_to_prime(dict(checkpoint))
    for name in truncated.state_dict():
        torch.testing.assert_close(full_conversion[name], converted[name])
    assert "model.message_queries" not in without_core.convert_to_prime(dict(checkpoint))

    truncated = _randomized_model(_text_config(num_hidden_layers=5))
    assert _logits(truncated, torch.arange(2, 2 + SEQ_LENS[0]), [SEQ_LENS[0]]).isfinite().all()


def test_each_weight_transfer_group_is_recognized_and_converts_alone():
    config = _text_config()
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(config)
    state_dict = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    whole = model.convert_to_hf(dict(state_dict))
    groups = [{k: v for k, v in state_dict.items() if not k.startswith("model.layers.")}]
    groups += [{k: v for k, v in state_dict.items() if k.startswith(f"model.layers.{i}.")} for i in range(8)]
    converted = {}
    for layer_idx, group in enumerate(groups, start=-1):
        assert model.is_prime_state_dict(group)
        converted.update(model.convert_layer_to_hf(dict(group), layer_idx))
    assert converted.keys() == whole.keys()
    for name, tensor in converted.items():
        torch.testing.assert_close(tensor, whole[name])

    fp32 = {name for name in state_dict if model.keep_in_fp32_for_weight_transfer(name)}
    assert "model.layers.3.attention_input.query" in fp32
    assert "model.layers.0.mlp.router.selection_bias" in fp32
    assert {"model.norm.query", "model.message_queries"} <= fp32
    assert not any(name.endswith(".weight") for name in fp32)


def test_expert_activation_is_the_shared_one_with_the_config_caps():
    """The released softcaps select the shared `SoftcappedSwiglu` itself. Its caps are class attributes, so a config
    with other caps must not get them silently: it selects a subclass that applies its own."""
    assert Inkling2DecoderLayer(_text_config(), 1).mlp.experts.activation is SoftcappedSwiglu
    config = _text_config(mlp_gate_softcap=0.5, mlp_up_softcap=0.75)
    activation = Inkling2DecoderLayer(config, 1).mlp.experts.activation
    assert issubclass(activation, SoftcappedSwiglu) and activation is not SoftcappedSwiglu
    # Every layer holds the same class: a compiled layer is guarded on it, and its graph serves the layers alike.
    assert Inkling2DecoderLayer(config, 2).mlp.experts.activation is activation

    torch.manual_seed(0)
    gate, up = 3 * torch.randn(2, 5, 8)
    expected = 0.5 * torch.tanh(gate / 0.5) * torch.sigmoid(gate) * (0.75 * torch.tanh(up / 0.75))
    torch.testing.assert_close(activation.apply(gate, up), expected)
    assert (SoftcappedSwiglu.apply(gate, up) - expected).abs().max() > 1


def _pack_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def test_dequantize_nvfp4_and_mxfp8_experts():
    """Adapted from the test of the reference that ``dequantize.py`` credits (SkyRL, Apache-2.0,
    ``tests/backends/skyrl_train/patches/megatron/inkling2/test_quant.py`` at the same commit): its sizes, its
    packing of two codes into a byte and its pair of assertions."""
    torch.manual_seed(0)
    num_experts, rows, columns = 3, 4, 64
    prefix = "model.llm.layers.0.mlp.experts"
    codes = torch.randint(0, 16, (num_experts, rows, columns))
    nvfp4_scale = torch.rand(num_experts, rows, columns // 16).mul(4).add(0.25).to(torch.float8_e4m3fn)
    amax = torch.rand(num_experts) + 0.1
    denominator = torch.tensor([2688.0])
    mxfp8_weight = torch.randn(num_experts, rows, columns).to(torch.float8_e4m3fn)
    mxfp8_scale = torch.randint(120, 130, (num_experts, rows, columns // 32), dtype=torch.uint8)
    state_dict = {
        f"{prefix}.w13_weight": _pack_nvfp4(codes),
        f"{prefix}.w13_weight_scale": nvfp4_scale,
        f"{prefix}.w13_weight_amax": amax,
        f"{prefix}.w13_weight_outer_scale_denominator": denominator,
        f"{prefix}.w2_weight": mxfp8_weight,
        f"{prefix}.w2_weight_scale": mxfp8_scale,
        "model.llm.layers.0.mlp.gate.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    packed = {name: tensor.clone() for name, tensor in state_dict.items()}
    dequantize_state_dict_(state_dict)
    assert set(state_dict) == {f"{prefix}.w13_weight", f"{prefix}.w2_weight", "model.llm.layers.0.mlp.gate.weight"}

    # Bit 3 of a code is the sign; bits 0-2 index the magnitude.
    magnitudes = torch.tensor(E2M1_VALUES)
    values = magnitudes[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    expected = values * nvfp4_scale.float().repeat_interleave(16, dim=-1) * (amax / denominator).view(-1, 1, 1)
    w13 = state_dict[f"{prefix}.w13_weight"]
    assert w13.dtype == torch.bfloat16
    torch.testing.assert_close(w13, expected.bfloat16(), rtol=0, atol=0)
    quantized = (packed[f"{prefix}.w13_weight"], nvfp4_scale, amax, denominator)
    assert nvfp4_requantization_mismatches(w13, *quantized) == 0
    # A weight with every pair of values swapped is not the dequantization of these codes.
    assert nvfp4_requantization_mismatches(w13.unflatten(-1, (-1, 2)).flip(-1).flatten(-2), *quantized) > 0

    expected = mxfp8_weight.float() * torch.exp2(mxfp8_scale.float() - 127).repeat_interleave(32, dim=-1)
    torch.testing.assert_close(state_dict[f"{prefix}.w2_weight"], expected.bfloat16(), rtol=0, atol=0)

    # A bf16 export has nothing to dequantize.
    bf16 = {f"{prefix}.w13_weight": torch.ones(2, 4, 4, dtype=torch.bfloat16)}
    dequantize_state_dict_(bf16)
    assert set(bf16) == {f"{prefix}.w13_weight"}

    del packed[f"{prefix}.w13_weight_amax"]
    with pytest.raises(KeyError):
        dequantize_state_dict_(packed)


def _dense_rel_attention(q, k, v, rel, first_key, scale):
    """Whole-matrix autograd reference for ``rel_attention``."""
    num_tokens, num_heads, _ = q.shape
    group = num_heads // k.shape[1]
    extent = rel.shape[-1]
    keys, values = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    token = torch.arange(num_tokens)
    distance = token[:, None] - token[None, :]
    bias = rel.permute(1, 0, 2).gather(-1, distance.clamp(0, extent - 1).expand(num_heads, -1, -1))
    scores = torch.einsum("ihd,jhd->hij", q, keys) * scale + bias * ((distance >= 0) & (distance < extent))
    visible = (distance >= 0) & (token[None, :] >= first_key[:, None])
    scores = scores.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    return torch.einsum("hij,jhd->ihd", torch.exp(scores - lse[..., None]), values), lse.t()


@pytest.mark.parametrize("window", [None, 7])
@pytest.mark.parametrize("score_elements_per_chunk", [1 << 26, 240])
def test_rel_attention_matches_dense_reference(monkeypatch, window, score_elements_per_chunk):
    """Forward and every gradient, over three packed documents, also when the queries split into several chunks."""
    monkeypatch.setattr(inkling2_attention, "SCORE_ELEMENTS_PER_CHUNK", score_elements_per_chunk)
    torch.manual_seed(0)
    num_tokens, num_heads, num_kv_heads, head_dim, extent = 37, 6, 3, 8, 12
    shapes = dict(
        q=(num_heads, head_dim), k=(num_kv_heads, head_dim), v=(num_kv_heads, head_dim), rel=(num_heads, extent)
    )
    inputs = [torch.randn(num_tokens, *shape, dtype=torch.float64, requires_grad=True) for shape in shapes.values()]
    first_key = torch.repeat_interleave(torch.tensor([0, 10, 15]), torch.tensor([10, 5, 22]))
    if window is not None:
        first_key = torch.maximum(first_key, torch.arange(num_tokens) - (window - 1))

    out, lse = rel_attention(*inputs, first_key, 0.3)
    expected_out, expected_lse = _dense_rel_attention(*inputs, first_key, 0.3)
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(lse, expected_lse)

    out_weight, lse_weight = torch.randn_like(out), torch.randn_like(lse)
    grads = torch.autograd.grad((out * out_weight).sum() + (lse * lse_weight).sum(), inputs)
    expected_grads = torch.autograd.grad((expected_out * out_weight).sum() + (expected_lse * lse_weight).sum(), inputs)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad)


def test_packed_documents_do_not_see_each_other():
    """A packed row gives every document the logits it gets alone: attention, every convolution and the
    position-dependent scales all stop at document boundaries."""
    model = _randomized_model(_text_config())
    torch.manual_seed(1)
    documents = [torch.randint(2, 300, (length,)) for length in SEQ_LENS]
    packed = _logits(model, torch.cat(documents), list(SEQ_LENS))
    start = 0
    for document in documents:
        alone = _logits(model, document, [len(document)])
        torch.testing.assert_close(packed[start : start + len(document)], alone, rtol=1e-3, atol=1e-3)
        start += len(document)
    # Another first token of the second document changes every later position of that document, and nothing in
    # the first one.
    other = documents[1].clone()
    other[0] = 2 + (other[0] - 1) % 298
    changed = _logits(model, torch.cat([documents[0], other]), list(SEQ_LENS))
    torch.testing.assert_close(changed[: SEQ_LENS[0]], packed[: SEQ_LENS[0]], rtol=1e-3, atol=1e-3)
    later = slice(SEQ_LENS[0] + 1, None)
    assert (changed[later] - packed[later]).abs().amax(dim=-1).min() > 1e-2


def test_loop_passes_and_messages_change_the_output():
    """Two passes, three passes, and three passes with zeroed message queries are three different functions."""
    config = _text_config()
    model = _randomized_model(config)
    torch.manual_seed(2)
    input_ids = torch.randint(2, 300, (SEQ_LENS[0],))
    three_passes = _logits(model, input_ids, [SEQ_LENS[0]])

    two_passes = _randomized_model(_text_config(loop_config=dict(config.loop_config, num_passes=2)))
    two_passes.load_state_dict(model.state_dict())
    assert (_logits(two_passes, input_ids, [SEQ_LENS[0]]) - three_passes).abs().max() > 1e-2

    with torch.no_grad():
        model.model.message_queries.zero_()
    assert (_logits(model, input_ids, [SEQ_LENS[0]]) - three_passes).abs().max() > 1e-3


def test_logits_match_the_transformers_reference():
    """The fixture holds the logits of an FP64 forward of transformers' `Inkling2LoopForCausalLM` (PR 51050 at commit
    6173a2f, version 5.24.0.dev0, its FP32 islands widened by the generator), rounded to FP32, on a checkpoint in the
    release layout whose every tensor is seeded by its name.
    `fixtures/make_inkling2_loop_golden.py` writes it and holds the recipe. Loading that checkpoint through the
    conversion chain must reproduce the logits. Sliding and full layers, the loop with messages, clips, softcaps,
    the selection bias, sinks and both kinds of convolution are all active in it (the generator prints by how much),
    and no routing decision is near a tie."""
    model = Inkling2LoopForCausalLM(Inkling2LoopTextConfig(**TEXT_CONFIG)).eval()
    checkpoint_shapes = model.convert_to_hf(dict(model.state_dict()))
    checkpoint = {name: checkpoint_tensor(name, tensor.shape) for name, tensor in checkpoint_shapes.items()}
    golden = load_file(GOLDEN)
    # The reference loaded this very checkpoint: the same names, and the same bits from the recipe.
    assert torch.equal(checkpoint_digest(checkpoint), golden["checkpoint_sha256"])
    # As in the release, the unembedding's rows past the logits are not zero: the conversion has to drop them.
    assert checkpoint["model.llm.unembed.weight"][model.config.num_logits :].any()
    model.load_state_dict(model.convert_to_prime(checkpoint))

    documents = [golden["input_ids_0"], golden["input_ids_1"]]
    expected = [golden["logits_0"], golden["logits_1"]]
    packed, seq_lens = torch.cat(documents), [len(document) for document in documents]
    # The experts as the model is built run BF16 GEMMs, whose rounding moves these logits by 8e-3.
    torch.testing.assert_close(_logits(model, packed, seq_lens), torch.cat(expected), rtol=0, atol=5e-2)

    # With exact experts, an FP32 forward is within 3e-6 of the reference under AVX-512, AVX2 and SSE kernels alike.
    for module in model.modules():
        if isinstance(module, MoE):
            module.experts.set_compute(_ExactExpertCompute())
    for document, logits in zip(documents, expected):
        torch.testing.assert_close(_logits(model, document, [len(document)]), logits, rtol=0, atol=1e-4)
    # Packed into one row, the two documents still get the reference's logits.
    torch.testing.assert_close(_logits(model, packed, seq_lens), torch.cat(expected), rtol=0, atol=1e-4)


def test_router_replay_takes_one_row_per_layer_visit():
    model = _randomized_model(_text_config())
    decoder = model.model
    assert decoder.num_layer_visits == 16
    assert [decoder.layer_visit_ids(layer_idx) for layer_idx in (1, 2, 5, 6)] == [[1], [2, 6, 10], [5, 9, 13], [14]]

    # A routing the model would not choose: two distinct random experts per token and layer visit.
    torch.manual_seed(4)
    input_ids = torch.randint(2, 300, (SEQ_LENS[0],))
    routed_experts = torch.rand(1, SEQ_LENS[0], 16, 8).argsort(dim=-1)[..., :2]

    used = []
    for layer in decoder.layers:
        layer.mlp.router.register_forward_hook(lambda module, args, output: used.append(output[1]))
    _logits(model, input_ids, [SEQ_LENS[0]], routed_experts)
    # The routers ran once per visit, in visit order, each on its own row.
    used = torch.stack(used, dim=1)
    assert used.shape == (SEQ_LENS[0], 16, 2)
    assert torch.equal(used, routed_experts[0])

    with pytest.raises(ValueError, match="visits"):
        _logits(model, input_ids, [SEQ_LENS[0]], routed_experts[:, :, :8])


def test_bf16_inputs_get_the_fp32_result_rounded_once():
    """The trainer computes in BF16, where only GEMMs may run. Norms, clips, convolutions, depth mixes, message gates
    and the attention kernel compute in FP32: BF16 inputs and parameters give what their FP32 upcasts give, rounded
    once where the output is BF16."""
    torch.manual_seed(5)
    num_tokens, hidden_size, num_heads, bf16 = 11, 32, 4, torch.bfloat16
    positions = torch.arange(num_tokens) % 7
    state, source = torch.randn(2, 1, num_tokens, hidden_size, dtype=bf16)
    heads = torch.randn(num_tokens, num_heads, hidden_size // num_heads, dtype=bf16)

    def in_bf16_and_fp32(module: torch.nn.Module, *gemms: str) -> tuple[torch.nn.Module, torch.nn.Module]:
        """A seeded module in BF16, and its twin with the same values in FP32, except in the GEMMs it names."""
        for param in module.parameters():
            torch.nn.init.normal_(param, std=0.2)
        twin = copy.deepcopy(module.to(bf16)).float()
        for gemm in gemms:
            twin.get_submodule(gemm).to(bf16)
        return module, twin

    def identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    norm, fp32_norm = in_bf16_and_fp32(Inkling2RMSNorm(hidden_size, 1e-6))
    identical(norm(state, bf16), fp32_norm(state.float(), torch.float32).to(bf16))
    conv, fp32_conv = in_bf16_and_fp32(StaticShortConv(hidden_size, 4))
    identical(conv(state, None, positions), fp32_conv(state.float(), None, positions).to(bf16))
    conv, fp32_conv = in_bf16_and_fp32(DynamicShortConv(hidden_size, hidden_size, num_heads, 4), "weight_proj")
    identical(conv(state, source, positions), fp32_conv(state.float(), source, positions).to(bf16))
    scale = torch.randn(num_heads, hidden_size // num_heads, dtype=bf16)
    identical(l2_clip(heads, 2.5, scale), l2_clip(heads.float(), 2.5, scale.float()).to(bf16))
    identical(l2_clip(heads, 2.5), l2_clip(heads.float(), 2.5).to(bf16))
    blocks, block_mask = torch.randn(1, num_tokens, 3, hidden_size, dtype=bf16), torch.tensor([True, True, False])
    final_norm, fp32_final_norm = in_bf16_and_fp32(Inkling2FinalNorm(hidden_size, 1e-6))
    identical(
        final_norm(blocks, block_mask, state), fp32_final_norm(blocks.float(), block_mask, state.float()).to(bf16)
    )
    activation = softcapped_swiglu(0.5, 0.75)
    identical(activation.apply(state, source), activation.apply(state.float(), source.float()).to(bf16))

    # Depth mixes, message gates and the attention kernel return FP32.
    sources, queries = torch.randn(1, num_tokens, 3, hidden_size, dtype=bf16), torch.randn(2, hidden_size, dtype=bf16)
    mixed = depth_mix(sources, queries)
    gate, fp32_gate = in_bf16_and_fp32(MessageGate(hidden_size, 4, 1e-6), "down_proj")
    gated = gate(state, source, bf16)
    k, v = torch.randn(2, num_tokens, num_heads // 2, hidden_size // num_heads, dtype=bf16)
    operands = [heads, k, v, torch.randn(num_tokens, num_heads, 6, dtype=bf16)]
    operands = [operand.requires_grad_() for operand in operands]
    first_key = torch.arange(num_tokens) - positions
    out, lse = rel_attention(*operands, first_key, 0.35)
    assert mixed.dtype == gated.dtype == out.dtype == lse.dtype == torch.float32
    identical(mixed, depth_mix(sources.float(), queries.float()))
    identical(gated, fp32_gate(state.float(), source.float(), bf16))
    fp32_operands = [operand.detach().float().requires_grad_() for operand in operands]
    expected_out, expected_lse = rel_attention(*fp32_operands, first_key, 0.35)
    identical(out, expected_out)
    identical(lse, expected_lse)
    # The kernel's backward is FP32 too, and rounds each gradient once to its operand's dtype.
    out_weight, lse_weight = torch.randn_like(out), torch.randn_like(lse)
    grads = torch.autograd.grad((out * out_weight).sum() + (lse * lse_weight).sum(), operands)
    expected_grads = torch.autograd.grad(
        (expected_out * out_weight).sum() + (expected_lse * lse_weight).sum(), fp32_operands
    )
    for grad, expected_grad in zip(grads, expected_grads):
        identical(grad, expected_grad.to(bf16))


def test_dynamic_layer_args_are_layer_inputs():
    """torch.compile leaves these args' row length dynamic by their names: a renamed layer input would silently
    bring the recompile at every new row length back."""
    layer_inputs = inspect.signature(Inkling2DecoderLayer.forward).parameters
    assert {name for name, _ in Inkling2LoopForCausalLM.dynamic_layer_arg_dims()} <= layer_inputs.keys()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_reaches_every_parameter(dtype):
    """Also in BF16, the trainer's compute dtype, which nothing else runs through the whole model on CPU."""
    model = _randomized_model(_text_config()).to(dtype)
    torch.manual_seed(3)
    input_ids = torch.randint(2, 300, (1, sum(SEQ_LENS)))
    labels = torch.randint(2, 300, (1, sum(SEQ_LENS)))
    loss = model(input_ids=input_ids, labels=labels, seq_lens=torch.tensor(SEQ_LENS))["loss"]
    loss.backward()
    assert loss.isfinite()
    without_grad = [
        name
        for name, param in model.named_parameters()
        if param.grad is None or not param.grad.any() or not param.grad.isfinite().all()
    ]
    assert without_grad == []
