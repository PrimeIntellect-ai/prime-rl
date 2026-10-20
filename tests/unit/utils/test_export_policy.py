import importlib
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from transformers import GenerationConfig, PretrainedConfig

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards


def test_tied_qwen3_5_moe_export_reties_headless_weights(tmp_path, tiny_run):
    """Real Qwen 3.5 MoE tie: the stale head must not ship; native HF reties the actual embed."""
    from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        tie_word_embeddings=True,
    )
    with torch.device("cpu"):
        model = Qwen3_5ForCausalLM._from_config(config)
    state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
    state["model.embed_tokens.weight"].fill_(1)
    state["lm_head.weight"].fill_(99)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)

    from convert_bf16_to_fp8 import quantize_state_dict

    for group in iter_cpu_hf_groups(checkpoint, config.to_dict()):
        quantized, _ = quantize_state_dict(group, 2, "cpu")
        assert "lm_head.weight" not in quantized
    exported = {
        name: tensor for group in iter_cpu_hf_groups(checkpoint, config.to_dict()) for name, tensor in group.items()
    }
    assert "lm_head.weight" not in exported
    assert exported["model.embed_tokens.weight"].flatten()[0].item() == 1

    export = tmp_path / "hf"
    write_hf_shards(iter([exported]), export)
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as NativeConfig

    native = NativeConfig(**config.to_dict())
    native.save_pretrained(export)
    restored, info = AutoModelForCausalLM.from_pretrained(export, config=native, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert restored.get_input_embeddings().weight.flatten()[0].item() == 1
    assert restored.get_output_embeddings().weight.flatten()[0].item() == 1


def test_gpu_export_drops_tied_head_and_stale_metadata(tiny_run, monkeypatch):
    _, step, _, _ = tiny_run
    tool = importlib.import_module("convert_dcp_to_bf16_gpu")
    config = PretrainedConfig(quantization_config={"quant_method": "fp8"})
    config.text_config = PretrainedConfig(tie_word_embeddings=True, quantization_config={"quant_method": "fp8"})
    model = SimpleNamespace(config=config, generation_config=GenerationConfig(use_cache=False), _tied_weights_keys=None)
    gathered = {"model.embed_tokens.weight": torch.ones(2, 2), "lm_head.weight": torch.full((2, 2), 99.0)}
    monkeypatch.setattr(tool, "setup_single_process_env", lambda: None)
    monkeypatch.setattr(tool, "setup_torch_distributed", lambda: None)
    monkeypatch.setattr(tool, "resolve_ep", lambda config: None)
    monkeypatch.setattr(tool, "get_parallel_dims", lambda config: None)
    monkeypatch.setattr(tool, "setup_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(tool, "AppState", lambda *args: None)
    monkeypatch.setattr(tool, "dcp_load", lambda **kwargs: None)
    monkeypatch.setattr(tool, "gather_weights_parallel", lambda *args, **kwargs: gathered)
    monkeypatch.setattr(tool, "convert_state_dict_to_hf", lambda model, state: state)
    _, model_config, tokenizer_config, exported, _ = tool.load_and_convert(step)
    assert set(exported) == {"model.embed_tokens.weight"}
    output = step / "gpu-assets"
    output.mkdir()
    tool.save_model_assets(model, model_config, tokenizer_config, output, strip_quantization_config=True)
    restored = PretrainedConfig.from_pretrained(output)
    assert not hasattr(restored, "quantization_config")
    assert "quantization_config" not in restored.text_config
    assert model.config.quantization_config == {"quant_method": "fp8"}
    assert model.config.text_config.quantization_config == {"quant_method": "fp8"}
