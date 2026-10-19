import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_conversion import conversion_chain
from prime_rl.utils.utils import default_dtype


def _real_prime_state(model_cls, config):
    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = model_cls._from_config(config)
    from prime_rl.trainer.models.layers.moe import GroupedExperts

    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.init_weights(config.initializer_range)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_real_llama_cpu_dcp_export_matches_state(tmp_path):
    from transformers.models.llama.configuration_llama import LlamaConfig

    from prime_rl.trainer.models.llama import LlamaForCausalLM as PrimeRLLlamaForCausalLM

    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
    )
    state = _real_prime_state(PrimeRLLlamaForCausalLM, config)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    output = tmp_path / "hf"
    write_hf_shards(iter_cpu_hf_groups(checkpoint, config.to_dict()), output, max_shard_size=256)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(state)
    exported = {
        name: tensor
        for filename in set(index["weight_map"].values())
        for name, tensor in load_file(output / filename).items()
    }
    assert set(exported) == set(state)
    for name, tensor in state.items():
        assert exported[name].dtype == torch.bfloat16
        torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0)


def test_real_qwen3_moe_cpu_dcp_export_matches_converted_state(tmp_path):
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeForCausalLM as PrimeRLQwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        tie_word_embeddings=False,
        attention_bias=True,
    )
    state = _real_prime_state(PrimeRLQwen3MoeForCausalLM, config)
    config_dict = config.to_dict()
    expected = apply_prime_to_hf(dict(state), conversion_chain(config_dict))
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    output = {name: tensor for group in iter_cpu_hf_groups(checkpoint, config_dict) for name, tensor in group.items()}
    assert set(output) == set(expected)
    for name, tensor in expected.items():
        assert output[name].dtype == tensor.dtype
        torch.testing.assert_close(output[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("drop", "wrong_shape"),
    [
        ("model.layers.0.self_attn.q_proj.weight", None),
        ("model.layers.0.mlp.experts.gate_proj", None),
        (None, "model.layers.0.mlp.experts.gate_proj"),
    ],
    ids=["missing-attention", "missing-experts", "wrong-shape"],
)
def test_real_qwen3_moe_rejects_malformed_checkpoint(tmp_path, drop, wrong_shape):
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeForCausalLM as PrimeRLQwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        tie_word_embeddings=False,
        attention_bias=True,
    )
    state = _real_prime_state(PrimeRLQwen3MoeForCausalLM, config)
    if drop is not None:
        state.pop(drop)
    else:
        name = wrong_shape
        state[name] = state[name].reshape(-1, 1)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, config.to_dict()))


def test_real_glm4_moe_preserves_selection_bias_dtype(tmp_path):
    from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
    from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM

    config = Glm4MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        moe_intermediate_size=4,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        num_hidden_layers=1,
        rope_theta=10000.0,
        first_k_dense_replace=0,
        partial_rotary_factor=0.5,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    state = _real_prime_state(PrimeRLGlm4MoeForCausalLM, config)
    (prime_bias,) = (name for name in state if name.endswith("selection_bias"))
    state[prime_bias] = state[prime_bias].to(torch.float32)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    output = {
        name: tensor for group in iter_cpu_hf_groups(checkpoint, config.to_dict()) for name, tensor in group.items()
    }
    (exported_bias,) = (name for name in output if name.endswith("e_score_correction_bias"))
    assert output[exported_bias].dtype == torch.float32
    torch.testing.assert_close(output[exported_bias], state[prime_bias], rtol=0, atol=0)


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
