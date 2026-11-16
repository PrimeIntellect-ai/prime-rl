# Copied from vLLM 0.33.0, vllm/model_executor/models/clef_joint_schema.py (Apache-2.0, Copyright contributors to the
# vLLM project): segment_mean, SchemaIndices, EvidenceRoutingLayer and JointSchemaHead unchanged, so the parameter names
# are vLLM's; schema_indices adapted from JointSchemaPooler._indices.
"""Clef's joint schema head, which decides every question of a prompt together from the prompt's final hidden states."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NUM_QUESTION_TYPES = 3
MAX_LOGIT_SCALE = math.log(100.0)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def segment_mean(values: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
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
            EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)
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
        means = segment_mean(hidden_states[schema.positions], schema.span_ids, schema.span_lengths)
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
        prior = prior_scale * (F.normalize(lexical, dim=-1) * anchors[option_question]).sum(dim=-1)
        options = self.option_norm(options)
        cosine = F.cosine_similarity(fields, options, dim=-1)
        features = torch.cat([fields, options, fields * options, torch.abs(fields - options)], dim=-1)
        residual = self.residual_scorer(features).squeeze(-1)
        joint_scale = self.joint_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        joint = joint_scale * cosine + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


def schema_indices(
    types: list[int],
    spans: list[tuple[int, int]],
    counts: list[int],
    device: torch.device,
) -> tuple[SchemaIndices, torch.Tensor, torch.Tensor]:
    """The schema's indices for the ``[start, end)`` spans of the questions then of the options, in positions of the
    prompt's hidden states, plus the option tokens' positions and their option spans, for the lexical embeddings."""
    num_questions = len(types)
    lengths = np.array([end - start for start, end in spans], dtype=np.int64)
    span_ids = np.repeat(np.arange(len(spans)), lengths)
    positions = np.concatenate([np.arange(start, end) for start, end in spans])
    num_question_tokens = int(lengths[:num_questions].sum())
    counts_cpu = np.array(counts)
    max_options = max(counts)
    option_question = np.repeat(np.arange(num_questions), counts_cpu)
    option_slots = np.concatenate([np.arange(count) + q * max_options for q, count in enumerate(counts)])

    def indices(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    schema = SchemaIndices(
        positions=indices(positions),
        span_ids=indices(span_ids),
        span_lengths=indices(lengths),
        question_types=indices(types),
        option_question=indices(option_question),
        option_slots=indices(option_slots),
        option_mask=torch.as_tensor(np.arange(max_options) < counts_cpu[:, None], device=device),
    )
    return schema, indices(positions[num_question_tokens:]), indices(span_ids[num_question_tokens:] - num_questions)
