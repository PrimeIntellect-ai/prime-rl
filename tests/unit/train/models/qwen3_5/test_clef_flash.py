"""Clef-Flash: the model class, its batched joint schema head and its export, on a tiny release
(clef_flash_release.py)."""

import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, parse_safetensors_file_metadata
from safetensors.torch import load_file
from torch import nn
from torch.distributed.tensor import DTensor

import prime_rl.trainer.parallel_dims as parallel_dims_module
from prime_rl.configs.sft import SystemOneColumnsConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import forward, get_model, load_dcp_from_hf, setup_fsdp
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import clef_joint_schema as batched
from prime_rl.trainer.models.qwen3_5.clef_joint_schema import MAX_LOGIT_SCALE, NUM_QUESTION_TYPES, RecordSchema
from prime_rl.trainer.models.qwen3_5.modeling_clef_flash import SCALARS, ClefFlashForJointSchema, prime_name
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sft.system_one.clef_flash import ClefFlashStrategy
from prime_rl.trainer.sft.system_one.data import render_sample
from prime_rl.trainer.sft.system_one.loss import build_decision_loss_inputs
from prime_rl.trainer.sft.system_one.record import SystemOneRequest, parse_record
from tests.unit.train.models.qwen3_5.clef_flash_release import (
    JOINT_HEAD_CONFIG,
    clef_config,
    load_release,
    meta_model,
    tokenizer,
    write_release,
)


def async_tensor_h2d(data, device=None, dtype=None):
    """vLLM's host-to-device copy, on the CPU."""
    return torch.as_tensor(data, dtype=dtype, device=device)


# The reference: vLLM v0.34.0's per-prompt head, copied verbatim from vllm/model_executor/models/clef_joint_schema.py,
# lines 44-208 (segment_mean, SchemaIndices, EvidenceRoutingLayer, JointSchemaHead) and 307-358
# (JointSchemaPooler._indices). A copy, so the parity tests run without vLLM (e.g. on macOS).
# fmt: off
def segment_mean(
    values: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    sums = values.new_zeros(lengths.shape[0], values.shape[-1], dtype=torch.float32)
    sums.index_add_(0, segment_ids, values.float())
    return (sums / lengths[:, None]).to(values.dtype)


@dataclass
class SchemaIndices:
    """Where one prompt's questions and options are, on the model's device."""

    positions: torch.Tensor
    """Every token of every span, questions first."""
    span_ids: torch.Tensor
    """The span of each of ``positions``."""
    span_lengths: torch.Tensor
    """``[Q + O]``."""
    question_types: torch.Tensor
    """``[Q]``."""
    option_question: torch.Tensor
    """The question of each option, ``[O]``."""
    option_slots: torch.Tensor
    """Each option's flat index in a ``[Q, max_options]`` grid, ``[O]``."""
    option_mask: torch.Tensor
    """``[Q, max_options]``, true at the slots of options."""


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        # The Identity takes the place of the checkpoint's dropout.
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(feedforward, width),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Separate key and value tensors, like the reference, keep attention's
        # unpacked input projections.
        routed, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(NUM_QUESTION_TYPES, width)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(width, 1),
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        lexical: torch.Tensor,
        schema: SchemaIndices,
    ) -> torch.Tensor:
        """The ``[O]`` option logits of one prompt, from its ``[L, hidden]``
        final hidden states and each option's ``[O, hidden]`` mean output
        embedding."""
        num_questions = schema.question_types.shape[0]
        option_question = schema.option_question
        hidden_states = self.hidden_norm(hidden_states)
        memory = self.memory_projection(hidden_states).unsqueeze(0)
        global_vector = hidden_states[-1]
        means = segment_mean(
            hidden_states[schema.positions], schema.span_ids, schema.span_lengths
        )
        question_vectors, contexts = means[:num_questions], means[num_questions:]

        options = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_vectors)[option_question]
        ).unsqueeze(0)
        for layer in self.evidence_layers:
            options = layer(options, memory)
        options = options[0]

        # Each field summarizes its options, weighted by their match to it.
        base_fields = self.question_projection(question_vectors)
        width = options.shape[-1]
        grid = options.new_zeros(schema.option_mask.numel(), width)
        grid[schema.option_slots] = options
        grid = grid.view(*schema.option_mask.shape, width)
        routing = torch.einsum("qow,qw->qo", grid, base_fields) / math.sqrt(width)
        routing = routing.masked_fill(~schema.option_mask, float("-inf"))
        summaries = torch.einsum("qo,qow->qw", routing.softmax(dim=-1), grid)
        fields = (
            base_fields
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector).unsqueeze(0)
            + self.type_embedding(schema.question_types)
        ).unsqueeze(0)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[option_question]

        anchors = F.normalize(question_vectors + global_vector, dim=-1)
        prior_scale = self.prior_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        prior = prior_scale * (
            F.normalize(lexical, dim=-1) * anchors[option_question]
        ).sum(dim=-1)
        options = self.option_norm(options)
        cosine = F.cosine_similarity(fields, options, dim=-1)
        features = torch.cat(
            [fields, options, fields * options, torch.abs(fields - options)], dim=-1
        )
        residual = self.residual_scorer(features).squeeze(-1)
        joint_scale = self.joint_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        joint = joint_scale * cosine + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


class JointSchemaPooler:
    @staticmethod
    def _indices(
        types: list[int],
        spans: list[tuple[int, int]],
        counts: list[int],
        device: torch.device,
    ) -> tuple[SchemaIndices, torch.Tensor, torch.Tensor]:
        """The schema's indices, plus the option tokens' positions (on the CPU)
        and their option spans, for the lexical embeddings."""
        num_questions = len(types)
        lengths = np.array([end - start for start, end in spans], dtype=np.int64)
        span_ids = np.repeat(np.arange(len(spans)), lengths)
        positions = np.concatenate([np.arange(start, end) for start, end in spans])
        num_question_tokens = int(lengths[:num_questions].sum())
        counts_cpu = np.array(counts)
        max_options = max(counts)
        option_question = np.repeat(np.arange(num_questions), counts_cpu)
        option_slots = np.concatenate(
            [np.arange(count) + q * max_options for q, count in enumerate(counts)]
        )
        vectors = [
            positions,
            span_ids,
            lengths,
            np.array(types),
            option_question,
            option_slots,
            span_ids[num_question_tokens:] - num_questions,
        ]
        packed = async_tensor_h2d(np.concatenate(vectors), device, dtype=torch.long)
        (
            positions_d,
            span_ids_d,
            lengths_d,
            types_d,
            option_question_d,
            option_slots_d,
            lexical_span_ids,
        ) = packed.split([len(vector) for vector in vectors])
        schema = SchemaIndices(
            positions=positions_d,
            span_ids=span_ids_d,
            span_lengths=lengths_d,
            question_types=types_d,
            option_question=option_question_d,
            option_slots=option_slots_d,
            option_mask=async_tensor_h2d(
                np.arange(max_options) < counts_cpu[:, None], device
            ),
        )
        lexical_ids = torch.from_numpy(positions[num_question_tokens:])
        return schema, lexical_ids, lexical_span_ids
# fmt: on


# Records of a packed row: (length, each question's option count). One option alone, as a choice may have.
RECORDS = [(23, [2]), (41, [3, 1, 5]), (30, [4, 2])]
PADDING = 6


def packed_records(seed: int = 0) -> list[RecordSchema]:
    """The records back to back in a row, each question's instructions and options in spans of 1 to 3 tokens."""
    generator = np.random.default_rng(seed)
    records, start = [], 0
    for length, counts in RECORDS:
        question_spans, option_spans, cursor = [], [], start + 1
        for count in counts:
            for spans in [question_spans] + [option_spans] * count:
                width = int(generator.integers(1, 3))
                spans.append((cursor, cursor + width))
                cursor += width + 1
        assert cursor <= start + length
        types = [int(t) for t in generator.integers(0, NUM_QUESTION_TYPES, len(counts))]
        records.append(RecordSchema(start, start + length, types, question_spans + option_spans, counts))
        start += length
    return records


def random_heads(dtype: torch.dtype) -> tuple[batched.JointSchemaHead, JointSchemaHead]:
    """The batched head with random weights, and vLLM's with the same weights."""
    torch.manual_seed(0)
    head = batched.JointSchemaHead(**JOINT_HEAD_CONFIG)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
    reference = JointSchemaHead(**JOINT_HEAD_CONFIG)
    reference.load_state_dict(head.state_dict())
    return head.to(dtype), reference.to(dtype)


def run_both(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both heads on the same records: the batched head once on the packed row, vLLM's once per record. Returns their
    logits and the gradients of their sums with respect to the row's hidden states."""
    head, reference = random_heads(dtype)
    records = packed_records()
    seq = records[-1].end + PADDING
    torch.manual_seed(1)
    row = torch.randn(seq, JOINT_HEAD_CONFIG["hidden_size"]).to(dtype)
    # The lexical embedding of each row token
    token_lexical = torch.randn(seq, JOINT_HEAD_CONFIG["hidden_size"]).to(dtype)

    hidden = row.clone().requires_grad_()
    schema, lexical_positions, lexical_span_ids = batched.schema_indices(records, seq, torch.device("cpu"))
    num_questions = schema.question_types.shape[0]
    lexical = batched.segment_mean(
        token_lexical[lexical_positions], lexical_span_ids, schema.span_lengths[num_questions:]
    )
    logits = head(hidden, lexical, schema)
    logits.float().sum().backward()

    reference_hidden = row.clone().requires_grad_()
    reference_logits = []
    for record in records:
        spans = [(start - record.start, end - record.start) for start, end in record.spans]
        record_schema, lexical_ids, lexical_span_ids = JointSchemaPooler._indices(
            record.types, spans, record.counts, torch.device("cpu")
        )
        lexical = segment_mean(
            token_lexical[record.start : record.end][lexical_ids],
            lexical_span_ids,
            record_schema.span_lengths[len(record.types) :],
        )
        reference_logits.append(reference(reference_hidden[record.start : record.end], lexical, record_schema).float())
    reference_logits = torch.cat(reference_logits)
    reference_logits.sum().backward()
    return logits.float(), reference_logits, hidden.grad.float(), reference_hidden.grad.float()


def test_the_batched_head_equals_vllms_per_prompt_head_in_float32():
    logits, reference, grad, reference_grad = run_both(torch.float32)
    assert logits.shape == (sum(sum(counts) for _, counts in RECORDS),)
    torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad, reference_grad, rtol=1e-4, atol=1e-6)
    # Every record token reaches the logits, through the memory; the row's padding does not
    assert grad[:-PADDING].abs().sum(-1).gt(0).all() and grad[-PADDING:].eq(0).all()


def test_the_batched_head_is_close_to_vllms_per_prompt_head_in_bfloat16():
    logits, reference, grad, reference_grad = run_both(torch.bfloat16)
    torch.testing.assert_close(logits, reference, rtol=0.02, atol=0.05)
    # In bfloat16, both heads' gradients are a few percent off the float32 ones, about as far as from each other
    _, _, _, exact_grad = run_both(torch.float32)

    def error(tensor: torch.Tensor) -> float:
        return ((tensor - exact_grad).norm() / exact_grad.norm()).item()

    assert error(grad) < 2 * error(reference_grad) < 0.2


def class_name(name: str) -> str:
    """A release tensor's name in the class."""
    if name.startswith(("model.", "lm_head.")):
        return name
    return prime_name(name) if name in SCALARS else f"pooler.head.{name}"


def test_a_release_loads_as_clef_flash_with_its_lm_head_frozen(tmp_path, monkeypatch):
    weights = write_release(tmp_path / "release")
    model = load_release(tmp_path / "release", monkeypatch)

    assert type(model) is ClefFlashForJointSchema
    assert model.config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert model.config.joint_head_config == JOINT_HEAD_CONFIG
    state = model.state_dict()
    assert set(state) == {class_name(name) for name in weights}
    for name, tensor in weights.items():
        assert torch.equal(state[class_name(name)], tensor.reshape(state[class_name(name)].shape)), name
    assert {name for name, parameter in model.named_parameters() if not parameter.requires_grad} == {"lm_head.weight"}
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.pooler.head.parameters())


@pytest.mark.parametrize(("name", "hidden_size"), [("Cloudflare/clef-flash", 4096), ("Cloudflare/clef", 5120)])
def test_the_published_releases_build_as_the_class(name, hidden_size):
    """Built on the meta device from the config files on the Hub, with the tensors the release stores."""
    # First, as a cached weight index without the head's config file would mean a release without a head
    model = get_model(ModelConfig(name=name, attn="flash_attention_2"))
    assert type(model) is ClefFlashForJointSchema and model.config.text_config.hidden_size == hidden_size

    index = json.loads(Path(hf_hub_download(name, "model.safetensors.index.json")).read_text())
    head = parse_safetensors_file_metadata(name, "joint_head.safetensors").tensors
    state = model.state_dict()
    assert set(state) == {class_name(tensor) for tensor in [*index["weight_map"], *head]}
    for tensor, info in head.items():
        assert state[class_name(tensor)].numel() == math.prod(info.shape), tensor
        assert list(state[class_name(tensor)].shape) in (info.shape, [1]), tensor


@pytest.mark.parametrize(
    ("tied", "head", "error"),
    [
        (True, {}, "freezes its LM head, so it must be untied"),
        (False, {"hidden_size": 32}, "has hidden_size 32, but the backbone's is 64"),
    ],
)
def test_the_class_refuses_what_it_cannot_train(tied, head, error):
    with torch.device("meta"), pytest.raises(ValueError, match=error):
        ClefFlashForJointSchema(clef_config(tied, **head))


def test_the_class_builds_vllms_configs():
    with torch.device("meta"):
        ClefFlashForJointSchema(clef_config())
        ClefFlashForJointSchema(clef_config(layers=0, routing_layers=0))


def fsdp_worker(rank: int, world: int, release: str, init_file: str, out_dir: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    parallel_dims_module.device_type = "cpu"  # the mesh is built on CUDA by default
    config = ModelConfig(name=release, attn="flash_attention_2", ep=1, compile=None)
    # As setup_model does, with the weights on the CPU
    model = meta_model(release)
    inject_prime_lm_head(model)
    setup_fsdp(model, config, get_parallel_dims(config))
    model.to_empty = lambda device: nn.Module.to_empty(model, device="cpu")
    load_dcp_from_hf(model, config)
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    lm_head = model.lm_head.weight.full_tensor()

    # The vision encoder and the decoder layers need CUDA kernels: on the CPU the forward skips the one and the others
    # pass their input through
    model.model.visual = None
    for layer in model.model.language_model.layers:
        layer.forward = lambda hidden_states, *args, **kwargs: hidden_states
    seen = {}

    def read_lm_head(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        weight = model.lm_head.weight
        seen.update(sharded=isinstance(weight, DTensor), full=weight.shape == lm_head.shape)
        seen.update(equal=not seen["sharded"] and seen["full"] and torch.equal(weight, lm_head.to(weight.dtype)))
        return {"sum": hidden_states.sum()}

    input_ids = torch.arange(10)[None]
    forward(model, input_ids, input_ids, seq_lens=torch.tensor([10]), hidden_states_fn=read_lm_head)
    torch.save((frozen, seen), os.path.join(out_dir, f"{rank}.pt"))
    dist.destroy_process_group()


def test_under_fsdp_the_lm_head_is_frozen_and_gathered_where_the_head_reads_it(tmp_path):
    write_release(tmp_path / "release")
    world = 2
    args = (world, str(tmp_path / "release"), str(tmp_path / "rendezvous"), str(tmp_path))
    mp.spawn(fsdp_worker, args=args, nprocs=world, join=True)
    for rank in range(world):
        assert torch.load(tmp_path / f"{rank}.pt") == (
            {"lm_head.weight"},
            {"sharded": False, "full": True, "equal": True},
        )


def test_the_bf16_export_writes_the_release_layout_and_loads_back(tmp_path, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "tools"))
    exporter = importlib.import_module("convert_dcp_to_bf16")
    release = tmp_path / "release"
    write_release(release)
    config_dir = tmp_path / "run" / "configs" / "latest" / "resolved"
    config_dir.mkdir(parents=True)
    (config_dir / "sft.json").write_text(json.dumps({"model": {"name": str(release)}, "tokenizer": {}}))
    torch.manual_seed(0)
    state = {name: torch.randn(tensor.shape) for name, tensor in meta_model(release).state_dict().items()}
    step_dir = tmp_path / "run" / "checkpoints" / "step_1"
    dcp.save({"app": {"model": state}}, checkpoint_id=step_dir / "trainer", no_dist=True)

    output = exporter.convert(step_dir)

    head = load_file(output / "joint_head.safetensors")
    head_state = {name: state[class_name(name)] for name in head}
    assert {class_name(name) for name in head} == {name for name in state if name.startswith("pooler.head.")}
    assert head.keys() == head_state.keys()
    for name, tensor in head.items():
        assert tensor.dtype == torch.bfloat16 and tensor.shape == (() if name in SCALARS else head_state[name].shape)
        assert torch.equal(tensor, head_state[name].to(torch.bfloat16).reshape(tensor.shape)), name
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == {name for name in state if name.startswith(("model.", "lm_head."))}
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"] and "joint_head_config" not in config
    assert json.loads((output / "joint_head_config.json").read_text()) == JOINT_HEAD_CONFIG

    loaded = load_release(output, monkeypatch).state_dict()
    assert loaded.keys() == state.keys()
    for name, tensor in state.items():
        assert torch.equal(loaded[name], tensor.to(loaded[name].dtype)), name


QUESTIONS = [
    {
        "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "A service is down."}},
        "team": {
            "type": "choice",
            "instructions": "Who?",
            "criteria": {"technical": "Bugs", "billing": None, "x": None},
        },
    },
    {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]}},
]


def requests() -> list[SystemOneRequest]:
    rows = [
        {"id": f"r{i}", "state": json.dumps({"ticket": f"Checkout fails {i}."}), "questions": json.dumps(questions)}
        for i, questions in enumerate(QUESTIONS)
    ]
    return [parse_record(row, SystemOneColumnsConfig())[0] for row in rows]


def packed_row(strategy: ClefFlashStrategy) -> tuple[torch.Tensor, list[int], list[dict]]:
    """Two records' samples, packed into one row with padding: its input ids, sample lengths and sample data."""
    samples = [render_sample(strategy, request, 4096) for request in requests()]
    input_ids = torch.tensor([token for sample in samples for token in sample["input_ids"]] + [0] * PADDING)
    return input_ids, [len(sample["input_ids"]) for sample in samples], [sample["sample_data"] for sample in samples]


def test_option_logits_read_every_record_with_one_head_call(tmp_path, monkeypatch):
    write_release(tmp_path / "release")
    model = load_release(tmp_path / "release", monkeypatch).float()
    strategy = ClefFlashStrategy(tokenizer(), model.config)
    input_ids, sample_lens, sample_data = packed_row(strategy)
    torch.manual_seed(0)
    hidden_states = torch.randn(len(input_ids), JOINT_HEAD_CONFIG["hidden_size"])
    calls = []
    model.pooler.head.register_forward_hook(lambda *args: calls.append(1))

    questions = build_decision_loss_inputs(sample_lens, sample_data).questions
    logits = strategy.option_logits(model, hidden_states, input_ids, questions)
    assert len(calls) == 1 and logits.dtype == torch.float32 and logits.shape == (3 + 2 + 3,)

    alone, start = [], 0
    for sample_len, data in zip(sample_lens, sample_data):
        ids = input_ids[start : start + sample_len]
        record_questions = build_decision_loss_inputs([sample_len], [data]).questions
        alone.append(strategy.option_logits(model, hidden_states[start : start + sample_len], ids, record_questions))
        start += sample_len
    torch.testing.assert_close(logits, torch.cat(alone), rtol=1e-5, atol=1e-5)
