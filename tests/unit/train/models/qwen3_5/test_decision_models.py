"""d3 and Clef-Flash, the decision models prime-rl builds from their releases as vLLM 0.33 serves them."""

import ast
import importlib
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerFast
from transformers.models.qwen3_5 import configuration_qwen3_5 as hf_configuration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_modeling

import prime_rl.trainer.parallel_dims as parallel_dims_module
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import (
    get_model,
    load_dcp_from_hf,
    overrides_init_missing_weights,
    refuse_weight_broadcast,
    setup_fsdp,
)
from prime_rl.trainer.models import get_decision_model_cls
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from prime_rl.trainer.models.qwen3_5.clef_joint_schema import JointSchemaHead
from prime_rl.trainer.models.qwen3_5.modeling_clef_flash import SCALARS, ClefFlashForJointSchema
from prime_rl.trainer.models.qwen3_5.modeling_d3 import D3ForSequenceClassification
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.loss import LossInputs, LossOutputs, run_loss_fn
from prime_rl.trainer.sft.system_one.loss import question_logits
from prime_rl.utils.weights import group_by_layer, load_state_dict, load_state_dict_keys

HIDDEN = 64
CODES = ["A", "B", "C", "D", "E"]
# d3-nano's decision_config.json, with 5 of its 255 codes and another temperature.
DECISION_CONFIG = {
    "format_version": 1,
    "format_id": "d3-code-readout-v1",
    "prompt": "d3",
    "codes": CODES,
    "token_ids": [32, 33, 34, 35, 36],
    "temperature": 1.5,
    "attention_mode": "noncausal_full_attention",
    "pooling": "last",
    "max_length": None,
    "readout_dtype": "float32",
}
JOINT_HEAD_CONFIG = {
    "hidden_size": HIDDEN,
    "width": 32,
    "routing_layers": 1,
    "layers": 1,
    "heads": 4,
    "feedforward": 48,
}

# vLLM 0.33's weight mappers (WeightsMapper.orig_to_new_prefix): D3ForSequenceClassification's (models/d3.py:66-71),
# which replaces Qwen3.5's, and ClefFlashForJointSchema's, Qwen3.5's (models/qwen3_vl.py:1969-1975 and
# models/qwen3_5.py:476-479). A name no prefix maps keeps its name, and vLLM raises for one that names no parameter
# (models/utils.py:440-448).
D3_MAPPER = {"language_model.": "language_model.model.", "mtp.": None}
CLEF_MAPPER = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
    "mtp.": None,
}


def vllm_name(name: str, mapper: dict[str, str | None]) -> str | None:
    for prefix, new in mapper.items():
        if name.startswith(prefix):
            if new is None:
                return None
            name = name.replace(prefix, new, 1)
    return name


def vllm_route(directory: Path) -> str:
    """The class vLLM 0.33 serves a Qwen3.5 checkpoint with (transformers_utils/model_arch_config_convertor.py:629-650)."""
    architectures = json.loads((directory / "config.json").read_text())["architectures"]
    if architectures == ["Qwen3_5ForConditionalGeneration"] and (directory / "joint_head_config.json").exists():
        return "ClefFlashForJointSchema"
    if architectures == ["Qwen3_5Model"] and (directory / "decision_config.json").exists():
        return "D3ForSequenceClassification"
    return architectures[0]


def tiny_config(architecture: str, tied: bool) -> Qwen3_5Config:
    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=tied,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=HIDDEN
    )
    return Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
        tie_word_embeddings=tied,
        architectures=[architecture],
    )


def release_backbone(config: Qwen3_5Config) -> dict[str, torch.Tensor]:
    """A release's backbone, under the names of transformers' class for its architecture."""
    torch.manual_seed(0)
    model = getattr(hf_modeling, config.architectures[0])(hf_configuration.Qwen3_5Config(**config.to_dict()))
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def write_tokenizer(directory: Path) -> None:
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "a": 1}, unk_token="<eos>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>").save_pretrained(directory)


def write_d3_release(directory: Path, decision_config: dict = DECISION_CONFIG) -> dict[str, torch.Tensor]:
    """d3-nano's layout: config.json with the bare backbone's architecture and a tied embedding, decision_config.json,
    the backbone in one model.safetensors without an index or MTP weights, and readout.safetensors {"weight": F32}."""
    config = tiny_config("Qwen3_5Model", tied=True)
    config.save_pretrained(directory)
    (directory / "decision_config.json").write_text(json.dumps(decision_config))
    weights = release_backbone(config)
    save_file(weights, directory / "model.safetensors")
    readout = {"weight": torch.randn(len(decision_config["codes"]), HIDDEN)}
    save_file(readout, directory / "readout.safetensors")
    write_tokenizer(directory)
    return weights | readout


def clef_head_weights() -> dict[str, torch.Tensor]:
    """A head as joint_head.safetensors stores it: vLLM's names without a prefix, 0-dimensional scalars."""
    torch.manual_seed(1)
    head = {name: tensor.detach().clone() for name, tensor in JointSchemaHead(**JOINT_HEAD_CONFIG).state_dict().items()}
    for scalar in SCALARS:
        head[scalar] = torch.randn(())
    return head


def write_clef_release(directory: Path) -> dict[str, torch.Tensor]:
    """Clef-Flash's layout: the Qwen3.5 checkpoint with an untied LM head, its MTP weights and an index, plus
    joint_head_config.json and joint_head.safetensors."""
    config = tiny_config("Qwen3_5ForConditionalGeneration", tied=False)
    config.save_pretrained(directory)
    (directory / "joint_head_config.json").write_text(json.dumps(JOINT_HEAD_CONFIG))
    backbone = release_backbone(config) | {"mtp.fc.weight": torch.randn(HIDDEN, 2 * HIDDEN)}
    save_file(backbone, directory / "model-00001-of-00001.safetensors")
    index = {"metadata": {}, "weight_map": dict.fromkeys(backbone, "model-00001-of-00001.safetensors")}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    head = clef_head_weights()
    save_file(head, directory / "joint_head.safetensors")
    write_tokenizer(directory)
    return backbone | head


def build(directory: Path, device: str = "meta") -> PreTrainedModelPrimeRL:
    config = AutoConfig.from_pretrained(directory)
    model_cls = get_decision_model_cls(config, str(directory))
    with torch.device(device):
        return model_cls._from_config(config, attn_implementation="flash_attention_2", dtype=torch.float32)


def model_config(directory: Path, implementation: str | None = None) -> ModelConfig:
    implementation = None if implementation is None else {"import_path": implementation}
    return ModelConfig(name=str(directory), attn="flash_attention_2", implementation=implementation)


class Decider(D3ForSequenceClassification):
    FORMAT_VERSIONS = (1, 2)


def test_the_release_selects_the_class_vllm_serves_it_with(tmp_path):
    d3, clef, decider, qwen = (tmp_path / name for name in ("d3", "clef", "decider", "qwen"))
    write_d3_release(d3)
    write_clef_release(clef)
    write_d3_release(decider, DECISION_CONFIG | {"format_version": 2})
    tiny_config("Qwen3_5ForConditionalGeneration", tied=False).save_pretrained(qwen)

    for directory, expected in [(d3, D3ForSequenceClassification), (clef, ClefFlashForJointSchema)]:
        model = get_model(model_config(directory), device=torch.device("meta"))
        assert type(model) is expected and vllm_route(directory) == expected.__name__
        assert model.config.architectures == json.loads((directory / "config.json").read_text())["architectures"]
    assert get_model(model_config(d3), device=torch.device("meta")).config.decision_config == DECISION_CONFIG
    assert get_model(model_config(clef), device=torch.device("meta")).config.joint_head_config == JOINT_HEAD_CONFIG
    assert get_decision_model_cls(AutoConfig.from_pretrained(qwen), str(qwen)) is None

    # vLLM routes Decider to d3 too and refuses its format_version 2; a subclass that reads it extends the d3 class.
    assert vllm_route(decider) == "D3ForSequenceClassification"
    with pytest.raises(ValueError, match="unsupported decision_config.json format_version 2"):
        get_model(model_config(decider), device=torch.device("meta"))
    model = get_model(model_config(decider, f"{__name__}.Decider"), device=torch.device("meta"))
    assert type(model) is Decider and model.config.decision_config["format_version"] == 2
    with pytest.raises(TypeError, match="must extend D3ForSequenceClassification"):
        get_model(model_config(d3, "prime_rl.trainer.models.qwen3_5.Qwen3_5ForCausalLM"), device=torch.device("meta"))


def test_d3_checks_its_decision_config(tmp_path):
    for index, (change, error) in enumerate(
        [
            ({"pooling": "mean"}, "pooling 'mean'"),
            ({"attention_mode": "bidirectional"}, "attention_mode 'bidirectional'"),
            ({"token_ids": [32]}, "one token id per answer code"),
        ]
    ):
        write_d3_release(tmp_path / str(index), DECISION_CONFIG | change)
        with pytest.raises(ValueError, match=error):
            build(tmp_path / str(index))

    write_d3_release(tmp_path / "d3")
    model = build(tmp_path / "d3")
    full_attention = [layer.self_attn for layer in model.model.language_model.layers if hasattr(layer, "self_attn")]
    assert full_attention and not any(attention.is_causal for attention in full_attention)
    assert model.score.weight.dtype == torch.float32 and model.score.weight.shape == (len(CODES), HIDDEN)
    assert model.cp_blocker(model.config, 2) is not None
    write_d3_release(tmp_path / "causal", DECISION_CONFIG | {"attention_mode": "causal"})
    assert all(layer.self_attn.is_causal for layer in build(tmp_path / "causal").model.language_model.layers[1:])


@pytest.mark.parametrize("kind", ["d3", "clef"])
def test_release_loads_and_converts_to_the_names_vllm_loads(tmp_path, kind):
    release = (write_d3_release if kind == "d3" else write_clef_release)(tmp_path)
    model = build(tmp_path, device="cpu")
    model_keys = dict.fromkeys(model.state_dict())

    # The loader's test (load_dcp_from_hf) for converting the release once.
    release_keys = dict.fromkeys(load_state_dict_keys(tmp_path))
    assert model.is_hf_state_dict(release_keys) and not model.is_prime_state_dict(release_keys)
    assert model.is_prime_state_dict(model_keys) and not model.is_hf_state_dict(model_keys)
    prime = model.convert_to_prime(load_state_dict(tmp_path))
    # Tied, d3's LM head loads through the embedding; Clef's lexical copy comes from init_missing_weights.
    added = "lm_head.weight" if kind == "d3" else "lexical_embedding.weight"
    assert model.load_state_dict(prime, strict=False) == ([added], [])
    assert overrides_init_missing_weights(model) == (kind == "clef")
    if kind == "clef":
        assert model.init_missing_weights([added]) == {added: model.lm_head.weight}
        assert model.init_missing_weights([]) == {}

    # The HF state dict of the model, as a weight broadcast sends it: every name vLLM's class loads, and only those.
    hf = model.convert_to_hf(dict(model.state_dict()))
    if kind == "d3":
        assert hf.keys() == {name for name in release if name != "weight"} | {"score.weight"}
        torch.testing.assert_close(hf["score.weight"], release["weight"])
        mapped = {vllm_name(name, D3_MAPPER) for name in hf}
        assert all(name == "score.weight" or name.startswith(("language_model.model.", "visual.")) for name in mapped)
        # An NCCL broadcast converts each layer's weights apart, so every group must read as prime.
        assert all(model.is_prime_state_dict(group) for group in group_by_layer(model.state_dict()).values())
    else:
        head = {f"pooler.head.{name}" for name in clef_head_weights()}
        assert hf.keys() == {name for name in release if not name.startswith("mtp.")} - set(clef_head_weights()) | head
        assert all(hf[f"pooler.head.{scalar}"].shape == () for scalar in SCALARS)
        mapped = {vllm_name(name, CLEF_MAPPER) for name in hf}
        namespaces = ("visual.", "language_model.model.", "language_model.lm_head.", "pooler.head.")
        assert all(name.startswith(namespaces) for name in mapped)
    for name, tensor in release.items():
        expected = hf.get("score.weight" if name == "weight" else name, hf.get(f"pooler.head.{name}"))
        if expected is not None:
            torch.testing.assert_close(expected, tensor)


@pytest.mark.parametrize("kind", [pytest.param("d3", marks=pytest.mark.gpu), "clef"])
def test_release_loads_through_the_conversion_cache(tmp_path, monkeypatch, kind):
    release = (write_d3_release if kind == "d3" else write_clef_release)(tmp_path)
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)
    config = model_config(tmp_path)
    model = get_model(config, device=torch.device("meta"), dtype=torch.float32)
    if kind == "clef":  # untied, so it can materialize on the CPU
        monkeypatch.setattr(model, "to_empty", lambda device: nn.Module.to_empty(model, device="cpu"))
    load_dcp_from_hf(model, config)

    assert (tmp_path / "prime" / ".prime-v1").is_file()
    expected = model.convert_to_prime(dict(release))
    loaded = model.state_dict()
    added = "lm_head.weight" if kind == "d3" else "lexical_embedding.weight"
    assert loaded.keys() - expected.keys() == {added}
    for name in loaded.keys() - {added}:
        torch.testing.assert_close(loaded[name].cpu(), expected[name])
    if kind == "clef":
        assert torch.equal(model.lexical_embedding.weight, model.lm_head.weight)


@pytest.mark.parametrize("temperature", [1.5, None])
def test_d3_logits_are_the_readout_at_the_read_tokens_over_the_temperature(tmp_path, temperature):
    decision_config = DECISION_CONFIG | {"temperature": temperature}
    if temperature is None:  # vLLM's d3 strategy then divides by 1
        del decision_config["temperature"]
    write_d3_release(tmp_path, decision_config)
    model = build(tmp_path, device="cpu")
    hidden_states = torch.randn(1, 7, HIDDEN, dtype=torch.bfloat16)
    loss_mask = torch.tensor([[False, False, True, False, False, False, True]])
    questions = [{"id": "q", "type": "noul", "options": ["false", "true"], "reads": [[2, 3]]}]
    questions += [{"id": "r", "type": "choice", "options": ["x", "y", "z"], "reads": [[3, 4]]}]
    batch = {"loss_mask": loss_mask, "sample_data": [{"id": 0, "num_tokens": 3, "questions": questions[:1]}]}
    batch["sample_data"].append({"id": 1, "num_tokens": 4, "questions": questions[1:]})
    logits = question_logits(model, hidden_states, batch)

    # The readout at fp32, divided by the temperature as vLLM's d3 strategy does.
    reads = hidden_states[loss_mask].float() @ model.score.weight.T
    assert [vector.dtype for vector in logits] == [torch.float32] * 2
    torch.testing.assert_close(logits[0], reads[0, :2] / (temperature or 1.0))
    torch.testing.assert_close(logits[1], reads[1, :3] / (temperature or 1.0))


def vllm_joint_schema() -> dict[str, Any]:
    """vLLM's joint schema head, its schema parsing and its index building, taken out of its source with ast so that
    nothing of vLLM is imported."""
    try:
        path = Path(
            importlib.metadata.distribution("vllm").locate_file("vllm/model_executor/models/clef_joint_schema.py")
        )
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("needs vLLM's source")
    if not path.is_file():
        pytest.skip(f"the installed vLLM has no {path.name}")
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("NUM_QUESTION_TYPES", "MAX_LOGIT_SCALE"):
            nodes.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in (
            "segment_mean",
            "SchemaIndices",
            "EvidenceRoutingLayer",
            "JointSchemaHead",
            "parse_joint_schema",
        ):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "JointSchemaPooler":
            (indices,) = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "_indices"]
            indices.decorator_list = []
            nodes.append(indices)
    namespace = {"math": math, "np": np, "torch": torch, "nn": nn, "F": F, "dataclass": dataclass, "Any": Any}
    namespace["async_tensor_h2d"] = lambda data, device, dtype=None: torch.as_tensor(data, device=device, dtype=dtype)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_clef_logits_are_vllms(tmp_path):
    vllm = vllm_joint_schema()
    write_clef_release(tmp_path)
    model = build(tmp_path, device="cpu")
    model.load_state_dict(model.convert_to_prime(load_state_dict(tmp_path)), strict=False)
    model.lexical_embedding.weight.data.copy_(model.lm_head.weight)
    vllm_head = vllm["JointSchemaHead"](**JOINT_HEAD_CONFIG)
    vllm_head.load_state_dict(clef_head_weights())
    assert vllm_head.state_dict().keys() == {
        name.removeprefix("pooler.head.") for name in model.pooler.head.state_dict()
    }

    # Two records packed in one row: noul and choice questions, then a score question.
    records = [
        (12, [("noul", [(1, 3), (4, 5), (6, 8)]), ("choice", [(8, 9), (9, 10), (10, 11), (11, 12)])]),
        (9, [("score", [(0, 2), (3, 4), (5, 6), (7, 9)])]),
    ]
    input_ids = torch.randint(0, 128, (1, 21))
    hidden_states = torch.randn(1, 21, HIDDEN)
    sample_data = [
        {
            "id": index,
            "num_tokens": num_tokens,
            "questions": [
                {"id": f"q{q}", "type": kind, "options": [f"o{o}" for o in range(len(spans) - 1)], "reads": spans}
                for q, (kind, spans) in enumerate(questions)
            ],
        }
        for index, (num_tokens, questions) in enumerate(records)
    ]
    logits = question_logits(model, hidden_states, {"input_ids": input_ids, "sample_data": sample_data})

    expected, start = [], 0
    for num_tokens, questions in records:
        # The request vLLM's Clef strategy sends for the record's prompt: spans count from the prompt's end.
        schema = {
            "question_types": [{"noul": 0, "choice": 1, "score": 2}[kind] for kind, _ in questions],
            "question_spans": [[spans[0][0] - num_tokens, spans[0][1] - num_tokens] for _, spans in questions],
            "option_spans": [[a - num_tokens, b - num_tokens] for _, spans in questions for a, b in spans[1:]],
            "option_counts": [len(spans) - 1 for _, spans in questions],
        }
        types, spans, counts = vllm["parse_joint_schema"]({"joint_schema": schema}, num_tokens)
        indices, lexical_ids, lexical_span_ids = vllm["_indices"](types, spans, counts, torch.device("cpu"))
        token_ids = input_ids[0, start : start + num_tokens]
        lexical = vllm["segment_mean"](
            F.embedding(token_ids[lexical_ids], model.lm_head.weight),
            lexical_span_ids,
            indices.span_lengths[len(types) :],
        )
        record_logits = vllm_head(hidden_states[0, start : start + num_tokens], lexical, indices).float()
        expected += record_logits.split(counts)
        start += num_tokens
    assert len(logits) == len(expected) == 3
    for actual, vector in zip(logits, expected):
        torch.testing.assert_close(actual, vector)


def test_decision_models_are_refused_with_weight_broadcasts(tmp_path):
    write_d3_release(tmp_path / "d3")
    write_clef_release(tmp_path / "clef")
    for name in ("d3", "clef"):
        with pytest.raises(ValueError, match="vLLM serves as a pooling model only"):
            refuse_weight_broadcast(build(tmp_path / name))
    tiny_config("Qwen3_5ForConditionalGeneration", tied=False).save_pretrained(tmp_path / "qwen")
    refuse_weight_broadcast(get_model(model_config(tmp_path / "qwen"), device=torch.device("meta")))


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "tools"))
    return importlib.import_module("convert_dcp_to_bf16")


@pytest.mark.parametrize("kind", ["d3", "clef"])
def test_export_is_the_release_layout(tmp_path, exporter, monkeypatch, kind):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    release_dir = tmp_path / "release"
    release = (write_d3_release if kind == "d3" else write_clef_release)(release_dir)
    step_dir = tmp_path / "run" / "checkpoints" / "step_1"
    config_dir = tmp_path / "run" / "configs" / "latest" / "resolved"
    config_dir.mkdir(parents=True)
    (config_dir / "sft.json").write_text(json.dumps({"model": {"name": str(release_dir)}, "tokenizer": {}}))
    model = get_model(model_config(release_dir), device=torch.device("meta"), dtype=torch.float32)
    state = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    dcp.save({"app": {"model": state}}, checkpoint_id=step_dir / "trainer", no_dist=True)

    output = exporter.convert(step_dir)
    assert vllm_route(output) == type(model).__name__
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == json.loads((release_dir / "config.json").read_text())["architectures"]
    assert not {"decision_config", "joint_head_config"} & config.keys() and "is_causal" not in config["text_config"]
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {name: tensor for file in set(index.values()) for name, tensor in load_file(output / file).items()}
    if kind == "d3":
        assert json.loads((output / "decision_config.json").read_text()) == DECISION_CONFIG
        readout = load_file(output / "readout.safetensors")
        assert readout.keys() == {"weight"} and readout["weight"].dtype == torch.float32
        assert torch.equal(readout["weight"], state["score.weight"])
        assert shards.keys() == release.keys() - {"weight"}
    else:
        assert json.loads((output / "joint_head_config.json").read_text()) == JOINT_HEAD_CONFIG
        head = load_file(output / "joint_head.safetensors")
        assert head.keys() == clef_head_weights().keys() and all(head[scalar].shape == () for scalar in SCALARS)
        assert torch.equal(head["residual_gate"].float(), state["pooler.head.residual_gate"].bfloat16().float()[0])
        assert shards.keys() == {name for name in release if not name.startswith("mtp.")} - head.keys()
    assert shards.keys() == index.keys()

    # The export's keys read as a release's, and it selects the same class.
    exported = dict.fromkeys(load_state_dict_keys(output))
    assert model.is_hf_state_dict(exported) and not model.is_prime_state_dict(exported)
    assert get_decision_model_cls(AutoConfig.from_pretrained(output), str(output)) is type(model)

    # Neither FP8 tool quantizes it: their quantization config does not cover the side files' weights.
    with pytest.raises(ValueError, match="FP8"):
        importlib.import_module("convert_bf16_to_fp8").convert(output)
    with pytest.raises(ValueError, match="FP8"):
        importlib.import_module("convert_dcp_to_fp8").convert(step_dir)


class ReadoutConfig(PretrainedConfig):
    model_type = "decision_readout_test"


class Backbone(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, position_ids, *, seq_lens, seq_lens_are_pre_shard=False):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = hidden_states + torch.tanh(layer(hidden_states))
        return SimpleNamespace(last_hidden_state=self.norm(hidden_states))


class ReadoutModel(PreTrainedModelPrimeRL):
    """A tied causal LM with d3's float32 readout."""

    config_class = ReadoutConfig
    fp32_modules = ("score",)

    def __init__(self, config, vocab: int = 32, hidden: int = 8):
        super().__init__(config)
        self.model = Backbone(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.score = nn.Linear(hidden, 5, bias=False, dtype=torch.float32)

    def init_buffers_post_meta(self):
        pass


def readout_model() -> ReadoutModel:
    torch.manual_seed(0)
    model = ReadoutModel(ReadoutConfig(tie_word_embeddings=True))
    inject_prime_lm_head(model)
    return model


def readout_loss(inputs: LossInputs) -> LossOutputs:
    logits = inputs.model.score(inputs.hidden_states[inputs.batch["loss_mask"]].float())
    return LossOutputs(loss=logits.logsumexp(-1).sum(), count=1, metrics={"logits": logits.detach().sum()})


def readout_batch(rank: int) -> dict:
    input_ids = torch.randint(0, 32, (1, 6), generator=torch.Generator().manual_seed(rank))
    loss_mask = torch.tensor([[False, False, True, False, False, True]])
    return {"input_ids": input_ids, "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]), "seq_lens": torch.tensor([3, 3]),
            "loss_mask": loss_mask}  # fmt: skip


def fp32_worker(rank: int, world: int, init_file: str, out_dir: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    parallel_dims_module.device_type = "cpu"  # the mesh is built on CUDA by default
    config = ModelConfig(name="readout", ep=1, compile=None)
    model = readout_model()
    setup_fsdp(model, config, get_parallel_dims(config))  # prime-rl's real FSDP setup

    dtypes = []
    model.score.register_forward_hook(lambda module, args, output: dtypes.append((module.weight.dtype, output.dtype)))
    outputs = run_loss_fn(readout_loss, model, readout_batch(rank))
    outputs.loss.backward()
    grad = model.score.weight.grad.full_tensor()
    torch.save(
        {"dtypes": dtypes, "grad": grad, "logits": outputs.metrics["logits"]}, os.path.join(out_dir, f"{rank}.pt")
    )
    dist.destroy_process_group()


def test_fp32_modules_compute_in_float32_under_fsdp(tmp_path):
    world = 2
    mp.spawn(fp32_worker, args=(world, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=world, join=True)

    # One process without FSDP: the backbone in bf16, as the root unit computes it, and the readout in float32.
    model = readout_model()
    model.model.to(torch.bfloat16)
    losses = [run_loss_fn(readout_loss, model, readout_batch(rank)) for rank in range(world)]
    sum(outputs.loss for outputs in losses).div(world).backward()
    for rank in range(world):
        result = torch.load(tmp_path / f"{rank}.pt")
        assert result["dtypes"] == [(torch.float32, torch.float32)]
        torch.testing.assert_close(result["logits"], losses[rank].metrics["logits"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result["grad"], model.score.weight.grad, rtol=1e-5, atol=1e-5)
