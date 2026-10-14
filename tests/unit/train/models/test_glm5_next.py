import pytest
import torch

from prime_rl.trainer.model import get_full_offload_dtype_policy
from prime_rl.trainer.models.conversion_ops import Drop, apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.glm5_next import Glm5NextConfig, Glm5NextForCausalLM, Glm5NextTextConfig, sparse_attention
from prime_rl.trainer.vlm import get_language_model


@pytest.fixture(params=[False, True], ids=["text", "composite"])
def config(request):
    text = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        q_lora_rank=8,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hc_mult=2,
        pad_token_id=0,
        topk_method="noaux_tc",
    )
    return Glm5NextConfig(text_config=text) if request.param else text


def test_config_and_checkpoint_roundtrip(config):
    restored = type(config).from_dict(config.to_dict())
    # vLLM reads checkpoint keys the trainer does not model (e.g. topk_method) off an export's config.
    assert getattr(restored, "text_config", restored).topk_method == "noaux_tc"
    model = Glm5NextForCausalLM(restored)
    model.init_buffers_post_meta()
    assert len(get_language_model(model).layers) == 2
    original = {
        name: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for name, value in model.state_dict().items()
    }
    converted = dict(original)
    operations = model.conversion_chain(restored)
    apply_prime_to_hf(converted, operations)
    assert model.is_hf_state_dict(converted) and not model.is_prime_state_dict(converted)
    assert any(name.endswith("layers.0.hc_attn_fn") for name in converted)
    assert any(name.endswith("layers.0.self_attn.A_log") for name in converted)
    assert any(name.endswith("layers.1.mlp.experts.3.down_proj.weight") for name in converted)
    apply_hf_to_prime(converted, operations)
    assert converted.keys() == original.keys()
    for name, value in original.items():
        torch.testing.assert_close(converted[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_kpool_indexer_ignores_selection_order(monkeypatch, reverse):
    def fake_fp8_indexer(q, k, w, ks, ke, topk):
        rows = []
        for start, end in zip(ks.tolist(), ke.tolist()):
            row = list(range(max(start, end - topk), end)) + [k.shape[0]] * topk
            rows.append(row[:topk][::-1] if reverse else row[:topk])
        return torch.tensor(rows, dtype=torch.int32)

    monkeypatch.setattr(sparse_attention, "fp8_indexer", fake_fp8_indexer)
    torch.manual_seed(0)
    indexer = sparse_attention.KPoolIndexer(
        hidden_size=16, q_lora_rank=8, num_heads=2, head_dim=128, token_budget=16, pool_size=4
    )
    cu_seqlens = torch.tensor([0, 10, 40])
    layout = sparse_attention.PoolLayout.build(cu_seqlens, local_tokens=40, cp_rank=0, cp_world_size=1, pool_size=4)
    indices = indexer(torch.randn(1, 40, 16), torch.randn(1, 40, 8), layout)[0, :, 0]

    expected = []
    for token in range(40):
        document_start = 0 if token < 10 else 10
        complete_end = document_start + (token + 1 - document_start) // 4 * 4
        selected = list(range(max(document_start, complete_end - 16), token + 1))
        expected.append(selected + [40] * (indexer.output_width - len(selected)))
    assert indices.sort(dim=1).values.tolist() == expected


def test_fp32_modules_stay_fp32_at_rest_and_on_the_wire(config):
    with torch.device("meta"):
        model = Glm5NextForCausalLM(config)
    fp32_params = {id(param) for module in model.fp32_modules() for param in module.parameters()}
    assert fp32_params
    policy = get_full_offload_dtype_policy(model)
    for name, param in model.named_parameters():
        dtype = torch.float32 if id(param) in fp32_params else torch.bfloat16
        assert policy[id(param)] == (dtype, dtype), name
        if id(param) in fp32_params:
            assert model.keep_in_fp32_for_weight_transfer(name), name


def test_debug_cut_drops_the_checkpoint_mtp_layer(config):
    text = getattr(config, "text_config", config)
    layers_prefix = "model.language_model.layers" if hasattr(config, "text_config") else "model.layers"
    mtp_layer = f"{layers_prefix}.{text.num_hidden_layers}."

    def dropped_layers() -> list[str]:
        drops = [op for op in Glm5NextForCausalLM.conversion_chain(config) if isinstance(op, Drop) and op.is_prefix]
        return [op.name for op in drops if op.name.startswith(layers_prefix)]

    assert dropped_layers() == [mtp_layer]
    text.num_hidden_layers = 1  # what get_model does for `model.debug.num_layers = 1`
    assert dropped_layers() == [mtp_layer]
