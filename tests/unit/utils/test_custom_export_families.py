import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from prime_rl.utils.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.weight_conversion import conversion_chain


def _tensor(offset=0, shape=(2, 3)):
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) + offset


def _roundtrip(config, hf_state, expected_prime):
    operations = conversion_chain(config)
    original = {name: value.clone() for name, value in hf_state.items()}
    prime = apply_hf_to_prime(dict(hf_state), operations)
    for name, expected in expected_prime.items():
        assert name in prime
        torch.testing.assert_close(prime[name], expected, rtol=0, atol=0)
    restored = apply_prime_to_hf(prime, operations)
    assert restored.keys() == original.keys()
    for name, expected in original.items():
        torch.testing.assert_close(restored[name], expected, rtol=0, atol=0)
    return restored


@pytest.mark.parametrize(
    ("model_type", "hf_container", "hf_projections", "hf_router", "hf_bias"),
    [
        ("afmoe", "mlp", ("gate_proj", "down_proj", "up_proj"), None, "mlp.expert_bias"),
        (
            "glm4_moe",
            "mlp",
            ("gate_proj", "down_proj", "up_proj"),
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
        ),
        (
            "glm_moe_dsa",
            "mlp",
            ("gate_proj", "down_proj", "up_proj"),
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
        ),
        (
            "laguna",
            "mlp",
            ("gate_proj", "down_proj", "up_proj"),
            "mlp.gate.weight",
            "mlp.experts.e_score_correction_bias",
        ),
        (
            "minimax_m2",
            "block_sparse_moe",
            ("w1", "w2", "w3"),
            "block_sparse_moe.gate.weight",
            "block_sparse_moe.e_score_correction_bias",
        ),
        ("qwen3_moe", "mlp", ("gate_proj", "down_proj", "up_proj"), "mlp.gate.weight", None),
    ],
)
def test_custom_expert_family_roundtrip(model_type, hf_container, hf_projections, hf_router, hf_bias):
    prefix = "model.layers.0"
    hf = {f"{prefix}.self_attn.q_proj.weight": _tensor()}
    expected = {}
    for index, (prime_projection, hf_projection) in enumerate(
        zip(("gate_proj", "down_proj", "up_proj"), hf_projections)
    ):
        experts = [_tensor(index * 20 + expert * 6) for expert in range(2)]
        for expert, tensor in enumerate(experts):
            hf[f"{prefix}.{hf_container}.experts.{expert}.{hf_projection}.weight"] = tensor
        expected[f"{prefix}.mlp.experts.{prime_projection}"] = torch.stack(experts)
    if hf_router:
        hf[f"{prefix}.{hf_router}"] = _tensor(100)
        expected[f"{prefix}.mlp.router.gate.weight"] = hf[f"{prefix}.{hf_router}"]
    if hf_bias:
        hf[f"{prefix}.{hf_bias}"] = torch.tensor([0.1234567, -0.7654321])
        expected[f"{prefix}.mlp.router.selection_bias"] = hf[f"{prefix}.{hf_bias}"]
    _roundtrip({"model_type": model_type, "num_hidden_layers": 1}, hf, expected)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_custom_dense_family_identity(model_type):
    hf = {
        "model.embed_tokens.weight": _tensor(),
        "model.layers.0.self_attn.q_proj.weight": _tensor(10),
        "model.layers.0.mlp.gate_proj.weight": _tensor(20),
        "lm_head.weight": _tensor(30),
    }
    _roundtrip({"model_type": model_type, "num_hidden_layers": 1}, hf, hf)


@pytest.mark.parametrize(
    "layer_type", ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
)
def test_deepseek_v4_published_names_roundtrip(layer_type):
    hf = {
        "embed.weight": _tensor(),
        "norm.weight": torch.ones(3),
        "head.weight": _tensor(10),
        "hc_head_base": _tensor(20),
        "layers.0.attn.wq_a.weight": _tensor(30),
        "layers.0.hc_attn_fn": _tensor(40),
        "layers.0.ffn.gate.bias": torch.tensor([0.2, -0.3]),
    }
    expected = {
        "model.embed_tokens.weight": hf["embed.weight"],
        "lm_head.weight": hf["head.weight"],
        "model.hc_head.hc_base": hf["hc_head_base"],
        "model.layers.0.self_attn.q_a_proj.weight": hf["layers.0.attn.wq_a.weight"],
        "model.layers.0.attn_hc.fn": hf["layers.0.hc_attn_fn"],
        "model.layers.0.mlp.router.selection_bias": hf["layers.0.ffn.gate.bias"],
    }
    for prime_projection, hf_projection in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
        experts = [_tensor(50 + expert * 6) for expert in range(2)]
        for expert, tensor in enumerate(experts):
            hf[f"layers.0.ffn.experts.{expert}.{hf_projection}.weight"] = tensor
        expected[f"model.layers.0.mlp.experts.{prime_projection}"] = torch.stack(experts)
    if layer_type != "sliding_attention":
        hf["layers.0.attn.compressor.wkv.weight"] = _tensor(70)
        expected["model.layers.0.self_attn.compressor.kv_proj.weight"] = _tensor(70)
    if layer_type == "compressed_sparse_attention":
        hf["layers.0.attn.indexer.compressor.ape"] = _tensor(80)
        expected["model.layers.0.self_attn.compressor.indexer.compressor.position_bias"] = _tensor(80)
    _roundtrip(
        {"model_type": "deepseek_v4", "num_hidden_layers": 1, "num_hash_layers": 0, "layer_types": [layer_type]},
        hf,
        expected,
    )


def test_gpt_oss_interleaved_experts_roundtrip():
    prefix = "model.layers.0.mlp.experts"
    gate_up = _tensor(shape=(2, 3, 4))
    bias = _tensor(40, shape=(2, 4))
    down = _tensor(60, shape=(2, 2, 3))
    hf = {
        f"{prefix}.gate_up_proj": gate_up,
        f"{prefix}.gate_up_proj_bias": bias,
        f"{prefix}.down_proj": down,
        f"{prefix}.down_proj_bias": _tensor(80, shape=(2, 3)),
        "model.layers.0.mlp.router.weight": _tensor(100),
        "model.layers.0.mlp.router.bias": torch.tensor([0.1, -0.2]),
    }
    expected = {
        f"{prefix}.gate_proj": gate_up[..., ::2].transpose(-2, -1),
        f"{prefix}.up_proj": gate_up[..., 1::2].transpose(-2, -1),
        f"{prefix}.gate_proj_bias": bias[..., ::2],
        f"{prefix}.up_proj_bias": bias[..., 1::2],
        f"{prefix}.down_proj": down.transpose(-2, -1),
        "model.layers.0.mlp.router.gate.weight": _tensor(100),
    }
    _roundtrip({"model_type": "gpt_oss", "num_hidden_layers": 1}, hf, expected)


def test_nemotron_h_prefix_and_hybrid_layers_roundtrip():
    hf = {
        "backbone.embeddings.weight": _tensor(),
        "backbone.norm_f.weight": torch.ones(3),
        "backbone.layers.0.mixer.A_log": torch.tensor([0.1, 0.2]),
        "backbone.layers.1.mixer.q_proj.weight": _tensor(10),
        "backbone.layers.2.mixer.gate.weight": _tensor(20),
        "backbone.layers.2.mixer.shared_experts.up_proj.weight": _tensor(30),
    }
    expected = {
        "model.embed_tokens.weight": _tensor(),
        "model.norm.weight": torch.ones(3),
        "model.layers.0.mamba.A_log": torch.tensor([0.1, 0.2]),
        "model.layers.1.self_attn.q_proj.weight": _tensor(10),
        "model.layers.2.mlp.router.gate.weight": _tensor(20),
        "model.layers.2.mlp.shared_expert.up_proj.weight": _tensor(30),
    }
    for projection in ("up_proj", "down_proj"):
        experts = [_tensor(40 + expert * 6) for expert in range(2)]
        for expert, tensor in enumerate(experts):
            hf[f"backbone.layers.2.mixer.experts.{expert}.{projection}.weight"] = tensor
        expected[f"model.layers.2.mlp.experts.{projection}"] = torch.stack(experts)
    _roundtrip(
        {
            "model_type": "nemotron_h",
            "num_hidden_layers": 3,
            "layer_types": ["linear_attention", "full_attention", "moe"],
        },
        hf,
        expected,
    )


@pytest.mark.parametrize("vision", [False, True], ids=["text", "vlm"])
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_qwen3_5_variants_roundtrip(vision, moe):
    text_type = "qwen3_5_moe_text" if moe else "qwen3_5_text"
    text_config = {"model_type": text_type, "num_hidden_layers": 1}
    config = dict(text_config)
    if vision:
        config = {
            "model_type": "qwen3_5_moe" if moe else "qwen3_5",
            "text_config": text_config,
            "vision_config": {},
        }
    prefix = "model.language_model.layers.0" if vision else "model.layers.0"
    hf = {f"{prefix}.linear_attn.A_log": torch.tensor([0.1, 0.2])}
    expected = dict(hf)
    if moe:
        combined = _tensor(shape=(2, 4, 3))
        hf[f"{prefix}.mlp.experts.gate_up_proj"] = combined
        hf[f"{prefix}.mlp.gate.weight"] = _tensor(40)
        hf[f"{prefix}.mlp.shared_expert_gate.weight"] = _tensor(50)
        expected.update(
            {
                f"{prefix}.mlp.experts.gate_proj": combined[:, :2],
                f"{prefix}.mlp.experts.up_proj": combined[:, 2:],
                f"{prefix}.mlp.router.gate.weight": _tensor(40),
                f"{prefix}.mlp.shared_expert.output_gate.weight": _tensor(50),
            }
        )
    else:
        hf[f"{prefix}.mlp.gate_proj.weight"] = _tensor(20)
        expected[f"{prefix}.mlp.gate_proj.weight"] = _tensor(20)
    _roundtrip(config, hf, expected)


@pytest.mark.parametrize("nested", [False, True], ids=["text", "nested-text"])
def test_qwen3_8_flash_next_ngram_and_fused_experts_roundtrip(nested):
    text = {"model_type": "qwen4_exp_text", "num_hidden_layers": 1, "ple_layer_ids": [1], "split_ngram_parts": 2}
    config = {"model_type": "qwen4_exp", "text_config": text} if nested else text
    layer = "model.language_model.layers.0" if nested else "model.layers.0"
    ngram = f"{layer}.ple.ple_embedding.ngram_embedding"
    combined = _tensor(shape=(2, 4, 3))
    hf = {
        f"{ngram}.shard_0.weight": _tensor(shape=(3, 2)),
        f"{ngram}.shard_1.weight": _tensor(6, shape=(2, 2)),
        f"{layer}.mlp.experts.gate_up_proj": combined,
        f"{layer}.mlp.gate.weight": _tensor(40),
    }
    expected = {
        f"{ngram}.weight": _tensor(shape=(5, 2)),
        f"{layer}.mlp.experts.gate_proj": combined[:, :2],
        f"{layer}.mlp.experts.up_proj": combined[:, 2:],
        f"{layer}.mlp.router.gate.weight": _tensor(40),
    }
    _roundtrip(config, hf, expected)


def _streaming_case(family):
    config = {"model_type": family, "num_hidden_layers": 1, "tie_word_embeddings": False}
    nested = family == "qwen3_5"
    prefix = "model.language_model" if nested else "model"
    hf = {
        f"{prefix}.embed_tokens.weight": _tensor(shape=(8, 4)),
        f"{prefix}.norm.weight": torch.ones(4),
        "lm_head.weight": _tensor(shape=(8, 4)),
    }
    layer = f"{prefix}.layers.0"
    if family in {"llama", "qwen3", "glm4_moe", "glm_moe_dsa"}:
        config.update(hidden_size=4, intermediate_size=6, num_attention_heads=2, num_key_value_heads=2, vocab_size=8)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            hf[f"{layer}.{norm}.weight"] = torch.ones(4)
        attention = ("q_proj", "k_proj", "v_proj", "o_proj")
        if family == "glm_moe_dsa":
            attention = (
                "q_a_proj",
                "q_a_layernorm",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_a_layernorm",
                "kv_b_proj",
                "o_proj",
            )
        for projection in attention:
            hf[f"{layer}.self_attn.{projection}.weight"] = _tensor(shape=(4, 4))
        if family == "qwen3":
            for norm in ("q_norm", "k_norm"):
                hf[f"{layer}.self_attn.{norm}.weight"] = torch.ones(2)
    if family in {"llama", "qwen3"}:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (4, 6) if projection == "down_proj" else (6, 4)
            hf[f"{layer}.mlp.{projection}.weight"] = _tensor(shape=shape)
    elif family == "gpt_oss":
        hf.update(
            {
                f"{layer}.mlp.experts.gate_up_proj": _tensor(shape=(2, 4, 4)),
                f"{layer}.mlp.experts.gate_up_proj_bias": _tensor(shape=(2, 4)),
                f"{layer}.mlp.experts.down_proj": _tensor(shape=(2, 2, 4)),
                f"{layer}.mlp.experts.down_proj_bias": _tensor(shape=(2, 4)),
            }
        )
    elif family == "nemotron_h":
        config["layer_types"] = ["linear_attention"]
        hf = {
            name.replace("model.embed_tokens", "backbone.embeddings").replace("model.norm", "backbone.norm_f"): tensor
            for name, tensor in hf.items()
        }
        hf["backbone.layers.0.mixer.A_log"] = torch.tensor([0.1234567, -0.7654321])
    elif family in {"qwen3_5", "qwen3_8_flash_next"}:
        text = {"model_type": "qwen3_5_moe_text", "num_hidden_layers": 1, "tie_word_embeddings": False}
        if family == "qwen3_5":
            config = {"model_type": "qwen3_5_moe", "text_config": text, "vision_config": {}}
            hf[f"{layer}.linear_attn.A_log"] = torch.tensor([0.1234567, -0.7654321])
            hf[f"{layer}.linear_attn.norm.weight"] = torch.tensor([0.1234567, -0.7654321])
        else:
            hf.pop(f"{prefix}.norm.weight")
            config = {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 1,
                "ple_layer_ids": [1],
                "split_ngram_parts": 2,
                "tie_word_embeddings": False,
                "eos_token_id": 2,
            }
            ngram = f"{layer}.ple.ple_embedding.ngram_embedding"
            hf[f"{ngram}.shard_0.weight"] = _tensor(shape=(3, 2))
            hf[f"{ngram}.shard_1.weight"] = _tensor(6, shape=(2, 2))
        hf[f"{layer}.mlp.experts.gate_up_proj"] = _tensor(shape=(2, 4, 4))
        hf[f"{layer}.mlp.gate.weight"] = _tensor(shape=(2, 4))
    else:
        container = "block_sparse_moe" if family == "minimax_m2" else "mlp"
        projections = (
            ("w1", "w2", "w3") if family in {"minimax_m2", "deepseek_v4"} else ("gate_proj", "down_proj", "up_proj")
        )
        if family == "deepseek_v4":
            config.update(layer_types=["sliding_attention"], num_hash_layers=0)
            hf = {
                "embed.weight": hf.pop("model.embed_tokens.weight"),
                "norm.weight": hf.pop("model.norm.weight"),
                "head.weight": hf.pop("lm_head.weight"),
            }
            layer, container = "layers.0", "ffn"
        for projection in projections:
            for expert in range(2):
                hf[f"{layer}.{container}.experts.{expert}.{projection}.weight"] = _tensor(expert * 20, shape=(4, 4))
        router = f"{layer}.{container}.gate.weight"
        bias = None
        if family in {"glm4_moe", "glm_moe_dsa"}:
            config.update(first_k_dense_replace=0, n_shared_experts=0, n_routed_experts=2)
            bias = f"{layer}.mlp.gate.e_score_correction_bias"
        elif family == "laguna":
            bias = f"{layer}.mlp.experts.e_score_correction_bias"
        elif family == "afmoe":
            bias = f"{layer}.mlp.expert_bias"
        elif family == "minimax_m2":
            bias = f"{layer}.block_sparse_moe.e_score_correction_bias"
        elif family == "deepseek_v4":
            bias = f"{layer}.ffn.gate.bias"
        hf[router] = _tensor(shape=(2, 4))
        if bias:
            hf[bias] = torch.tensor([0.1234567, -0.7654321])
    return config, hf


@pytest.mark.parametrize(
    "family",
    [
        "afmoe",
        "deepseek_v4",
        "glm4_moe",
        "glm_moe_dsa",
        "gpt_oss",
        "laguna",
        "llama",
        "minimax_m2",
        "nemotron_h",
        "qwen3",
        "qwen3_5",
        "qwen3_8_flash_next",
        "qwen3_moe",
    ],
)
def test_custom_family_streaming_cpu_dcp_export(tmp_path, family):
    config, hf = _streaming_case(family)
    prime = apply_hf_to_prime({name: tensor.clone() for name, tensor in hf.items()}, conversion_chain(config))
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": prime, "optimizer": {"exp_avg": torch.ones(3)}}}, checkpoint_id=checkpoint, no_dist=True)
    output = tmp_path / "hf"
    write_hf_shards(iter_cpu_hf_groups(checkpoint, config), output, max_shard_size=64)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    exported = {
        name: tensor
        for filename in set(index["weight_map"].values())
        for name, tensor in load_file(output / filename).items()
    }
    assert exported.keys() == hf.keys()
    for name, tensor in hf.items():
        fp32 = (
            (family in {"glm4_moe", "glm_moe_dsa"} and name.endswith("e_score_correction_bias"))
            or (family == "nemotron_h" and name.endswith("A_log"))
            or (family == "deepseek_v4" and name in {"norm.weight", "layers.0.ffn.gate.bias"})
            or (family == "qwen3_5" and name.endswith(("linear_attn.A_log", "linear_attn.norm.weight")))
        )
        dtype = torch.float32 if fp32 else torch.bfloat16
        assert exported[name].dtype == dtype
        torch.testing.assert_close(exported[name], tensor.to(dtype), rtol=0, atol=0)


def test_generic_hf_gpt2_streaming_export_loads_and_runs(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(vocab_size=16, n_positions=8, n_embd=4, n_layer=1, n_head=1, tie_word_embeddings=False)
    model = GPT2LMHeadModel(config).eval()
    state = model.state_dict()
    assert sum(t.numel() * t.element_size() for t in state.values()) < 10_000
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    output = tmp_path / "hf"
    write_hf_shards(iter_cpu_hf_groups(checkpoint, config.to_dict()), output, max_shard_size=256)
    config.save_pretrained(output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"].keys() == state.keys()
    restored = GPT2LMHeadModel.from_pretrained(output, dtype=torch.bfloat16).eval()
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, state[name].bfloat16(), rtol=0, atol=0)
    with torch.no_grad():
        result = restored(torch.tensor([[1, 2]]))
    assert result.logits.shape == (1, 2, 16)
    assert torch.isfinite(result.logits).all()
