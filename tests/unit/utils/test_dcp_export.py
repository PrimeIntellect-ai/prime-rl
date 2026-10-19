import json
import pickle
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_conversion import conversion_chain
from prime_rl.utils.utils import default_dtype


def _model_state(layout):
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.models.llama.configuration_llama import LlamaConfig

    from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
    from prime_rl.trainer.models.layers.moe import GroupedExperts
    from prime_rl.trainer.models.llama import LlamaForCausalLM
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    kwargs = dict(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    if layout == "dense":
        config, model_cls = LlamaConfig(**kwargs), LlamaForCausalLM
    elif layout == "moe":
        config = Qwen3MoeConfig(**kwargs, moe_intermediate_size=4, num_experts=2, num_experts_per_tok=1)
        model_cls = Qwen3MoeForCausalLM
    elif layout == "custom":
        config = Glm4MoeConfig(
            **kwargs,
            moe_intermediate_size=4,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
            first_k_dense_replace=0,
            norm_topk_prob=True,
            partial_rotary_factor=0.5,
        )
        model_cls = Glm4MoeForCausalLM
    else:
        config = GPT2Config(vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2, tie_word_embeddings=False)
        model_cls = GPT2LMHeadModel
    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = model_cls._from_config(config)
    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.init_weights(config.initializer_range)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    gathered = {}
    for name, tensor in state.items():
        if tensor.is_floating_point() and layout != "hf-native":
            dtype = torch.float32 if model.keep_in_fp32_for_weight_transfer(name) else torch.bfloat16
            tensor = tensor.to(dtype)
        gathered[name] = tensor
    return config, state, apply_prime_to_hf(gathered, conversion_chain(config.to_dict()))


@pytest.mark.parametrize("layout", ["dense", "moe", "custom", "hf-native"])
def test_export_model_layouts(tmp_path, layout):
    import torch.distributed.checkpoint as dcp

    config, state, expected = _model_state(layout)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    output = tmp_path / "hf"
    write_hf_shards(iter_cpu_hf_groups(checkpoint, config.to_dict()), output, max_shard_size=256)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    exported = {
        name: tensor
        for filename in set(index["weight_map"].values())
        for name, tensor in load_file(output / filename).items()
    }
    assert exported.keys() == expected.keys() == index["weight_map"].keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0)
    config.save_pretrained(output)
    if layout == "hf-native":
        from transformers import AutoModelForCausalLM

        restored = AutoModelForCausalLM.from_pretrained(output)
        assert torch.isfinite(restored(torch.tensor([[0, 1]])).logits).all()


@pytest.mark.parametrize("broken", ["missing-attention", "missing-experts", "wrong-shape"])
def test_rejects_broken_checkpoint(tmp_path, broken):
    import torch.distributed.checkpoint as dcp

    config, state, _ = _model_state("moe")
    if broken == "missing-attention":
        del state["model.layers.0.self_attn.q_proj.weight"]
    elif broken == "missing-experts":
        del state["model.layers.0.mlp.experts.gate_proj"]
    else:
        state["model.layers.0.mlp.experts.gate_proj"] = torch.ones(1)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, config.to_dict()))


@pytest.mark.parametrize("broken", ["gap", "overlap", "out-of-bounds"])
def test_rejects_bad_checkpoint_chunks(tiny_run, broken):
    _, step, _, _ = tiny_run
    checkpoint = step / "trainer"
    path = checkpoint / ".metadata"
    metadata = pickle.loads(path.read_bytes())
    entry = metadata.state_dict_metadata["app.model.model.embed_tokens.weight"]
    chunk = entry.chunks[0]
    if broken == "gap":
        chunk.sizes = torch.Size([chunk.sizes[0] - 1, chunk.sizes[1]])
    elif broken == "overlap":
        entry.chunks.append(deepcopy(chunk))
    else:
        chunk.offsets = torch.Size([1, 0])
    path.write_bytes(pickle.dumps(metadata))
    config = json.loads((step.parents[1] / "configs/latest/resolved/trainer.json").read_text())
    config = json.loads((Path(config["model"]["name"]) / "config.json").read_text())
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, config))
