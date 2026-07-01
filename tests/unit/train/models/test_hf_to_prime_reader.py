import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM
from transformers import Qwen3_5MoeForCausalLM as HFQwen3_5MoeForCausalLM
from transformers import Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM

from prime_rl.trainer.hf_to_prime import HFToPrimeStorageReader, materialize_hf_to_prime
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeForCausalLM
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeForCausalLM as PrimeRLQwen3MoeForCausalLM


def _glm4_moe_config():
    return Glm4MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=128,
        moe_intermediate_size=16,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hidden_layers=3,
        rope_theta=1000000.0,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
        use_grouped_mm=False,
    )


def _qwen3_moe_config():
    return Qwen3MoeConfig(
        vocab_size=128,
        head_dim=8,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=128,
        max_window_layers=4,
        moe_intermediate_size=16,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        rope_theta=1000000.0,
        use_qk_norm=True,
        mlp_only_layers=[1],
        use_grouped_mm=False,
    )


def _qwen3_5_moe_config():
    return Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        use_grouped_mm=False,
    )


def _check_hf_to_prime_reader_matches_state_dict_conversion(tmp_path, config, hf_cls, prime_cls):
    config._attn_implementation = "sdpa"
    with torch.device("cpu"):
        hf_model = hf_cls._from_config(config)
        prime_model = prime_cls._from_config(config)

    hf_model.save_pretrained(tmp_path, safe_serialization=True)

    expected = hf_model.state_dict()
    prime_cls.convert_to_prime(expected)

    loaded = prime_model.state_dict()
    dcp_load(
        loaded,
        storage_reader=HFToPrimeStorageReader(tmp_path.as_posix(), prime_cls.convert_layer_to_prime),
        no_dist=True,
    )
    assert not (tmp_path / "prime").exists()

    assert expected.keys() <= loaded.keys()
    for key, expected_tensor in expected.items():
        assert torch.equal(expected_tensor.detach().cpu(), loaded[key].detach().cpu()), key


def test_hf_to_prime_reader_glm4_moe(tmp_path):
    _check_hf_to_prime_reader_matches_state_dict_conversion(
        tmp_path,
        _glm4_moe_config(),
        HFGlm4MoeForCausalLM,
        PrimeRLGlm4MoeForCausalLM,
    )


def test_hf_to_prime_reader_qwen3_moe(tmp_path):
    _check_hf_to_prime_reader_matches_state_dict_conversion(
        tmp_path,
        _qwen3_moe_config(),
        HFQwen3MoeForCausalLM,
        PrimeRLQwen3MoeForCausalLM,
    )


def test_materialize_hf_to_prime_qwen3_moe(tmp_path):
    config = _qwen3_moe_config()
    config._attn_implementation = "sdpa"
    with torch.device("cpu"):
        hf_model = HFQwen3MoeForCausalLM._from_config(config)
        prime_model = PrimeRLQwen3MoeForCausalLM._from_config(config)

    hf_model.save_pretrained(tmp_path, safe_serialization=True)

    expected = hf_model.state_dict()
    PrimeRLQwen3MoeForCausalLM.convert_to_prime(expected)

    materialize_hf_to_prime(
        path=tmp_path,
        output_path=tmp_path / "prime",
        convert_layer_to_prime=PrimeRLQwen3MoeForCausalLM.convert_layer_to_prime,
    )

    loaded = prime_model.state_dict()
    dcp_load(
        loaded,
        storage_reader=HuggingFaceStorageReader((tmp_path / "prime").as_posix()),
        no_dist=True,
    )

    assert (tmp_path / "prime").exists()
    assert expected.keys() <= loaded.keys()
    for key, expected_tensor in expected.items():
        assert torch.equal(expected_tensor.detach().cpu(), loaded[key].detach().cpu()), key


def test_hf_to_prime_reader_qwen3_5_moe(tmp_path):
    _check_hf_to_prime_reader_matches_state_dict_conversion(
        tmp_path,
        _qwen3_5_moe_config(),
        HFQwen3_5MoeForCausalLM,
        PrimeRLQwen3_5MoeForCausalLM,
    )


def test_hf_to_prime_reader_streams_layer_conversion_with_cat(tmp_path):
    source = {
        "model.layers.0.left.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "model.layers.0.right.weight": torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
    }
    target = {
        "model.layers.0.cat.weight": torch.empty(4, 3),
    }
    save_file(source, tmp_path / "model.safetensors")

    class CatConverter:
        @staticmethod
        def convert_layer_to_prime(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
            assert layer_idx == 0
            state_dict["model.layers.0.cat.weight"] = torch.cat(
                [
                    state_dict.pop("model.layers.0.left.weight"),
                    state_dict.pop("model.layers.0.right.weight"),
                ],
                dim=0,
            )
            return state_dict

    dcp_load(
        target,
        storage_reader=HFToPrimeStorageReader(tmp_path.as_posix(), CatConverter.convert_layer_to_prime),
        no_dist=True,
    )

    assert not (tmp_path / "prime").exists()
    assert torch.equal(target["model.layers.0.cat.weight"], torch.cat(list(source.values()), dim=0))
