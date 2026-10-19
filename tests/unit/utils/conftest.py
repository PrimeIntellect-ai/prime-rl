import importlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def local_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    source = tmp_path / "base-model"
    source.mkdir()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(source)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 1,
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 3,
        "eos_token_id": 1,
        "pad_token_id": None,
        "use_cache": False,
        "dtype": "float32",
        "torch_dtype": "float16",
        "custom_export_attribute": {"preserve": True},
    }
    (source / "config.json").write_text(json.dumps(config))
    return source, config


@pytest.fixture
def tiny_run(tmp_path, local_assets, monkeypatch):
    source, config = local_assets
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    run = tmp_path / "run"
    step = run / "checkpoints" / "step_1"
    checkpoint = step / "trainer"
    configs = run / "configs" / "latest" / "resolved"
    configs.mkdir(parents=True)
    (configs / "trainer.json").write_text(
        json.dumps({"model": {"name": str(source)}, "tokenizer": {"chat_template": "{{ messages[0]['content'] }}"}})
    )
    state = {
        "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "model.norm.weight": torch.ones(2),
        "lm_head.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }
    prefix = "model.layers.0"
    for name in ("input_layernorm", "post_attention_layernorm"):
        state[f"{prefix}.{name}.weight"] = torch.ones(2)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[f"{prefix}.self_attn.{name}.weight"] = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    for name in ("gate_proj", "up_proj", "down_proj"):
        state[f"{prefix}.mlp.{name}.weight"] = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dcp.save(
        {"app": {"model": state, "optimizer": {"exp_avg": torch.full((2, 2), 99.0)}, "step": 1}},
        checkpoint_id=checkpoint,
        no_dist=True,
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    cli = importlib.import_module("convert_dcp_to_bf16")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    return cli, step, state, before
