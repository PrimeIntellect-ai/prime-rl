"""Qwen3.5's two-block attention: the layout of its kernel call against the mask's rule, the model's hand-off of the
layout to that call, the kernel call against a dense masked attention, a compiled layer on rows of different sizes, and
the mask's effect on a tiny model through the trainer's forward.

The GPU tests have not been run."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._dynamo.utils import clear_compilation_metrics, get_compilation_metrics

from prime_rl.trainer.model import forward, mark_dynamic_args
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5 import Qwen3_5Model
from prime_rl.trainer.models.qwen3_5.attention import Qwen3_5Attention, TwoBlockLayout, prepare_two_block_layout
from prime_rl.trainer.vlm import get_language_model
from prime_rl.utils.cp import CPContext
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens
from tests.unit.train.models.qwen3_5.test_model import get_model, get_text_config, get_vlm_config
from tests.utils import assert_relative

# (seq_lens, prefix_lens) of packed rows
ROWS = [
    ([5], [2]),
    # One block
    ([4], [4]),
    # A prefix of one token, a segment of one block, and a second block of one token
    ([6, 3, 5], [1, 3, 4]),
    # Segments of one and two tokens
    ([3, 2, 7, 1], [2, 1, 3, 1]),
    # A padding segment, which is one block
    ([8, 8], [3, 8]),
]


def layout_of(seq_lens: list[int], prefix_lens: list[int], device: str = "cpu") -> TwoBlockLayout:
    cu_seqlens, _ = get_cu_seqlens_from_seq_lens(torch.tensor(seq_lens, device=device))
    return prepare_two_block_layout(cu_seqlens, torch.tensor(prefix_lens, device=device))


def allowed_by_rule(seq_lens: list[int], prefix_lens: list[int]) -> torch.Tensor:
    """``[tokens, tokens]``, true where the query (row) may attend the key (column): both are in one segment, and the
    key is at or before the query or in the query's block."""
    segment = torch.repeat_interleave(torch.arange(len(seq_lens)), torch.tensor(seq_lens))
    position = torch.cat([torch.arange(seq_len) for seq_len in seq_lens])
    in_prefix = position < torch.tensor(prefix_lens)[segment]
    same_segment = segment[:, None] == segment[None, :]
    at_or_before = position[None, :] <= position[:, None]
    same_block = in_prefix[:, None] == in_prefix[None, :]
    return same_segment & (at_or_before | same_block)


def allowed_by_layout(layout: TwoBlockLayout, total_tokens: int) -> torch.Tensor:
    """The same matrix, read from the batch elements of the kernel call."""
    allowed = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
    cu_seqlens_q, cu_seqlens_k = layout.cu_seqlens_q.tolist(), layout.cu_seqlens_k.tolist()
    for element in range(len(cu_seqlens_q) - 1):
        keys = layout.key_index[cu_seqlens_k[element] : cu_seqlens_k[element + 1]]
        # A key listed twice in one element would count twice in its softmax
        assert keys.unique().numel() == keys.numel()
        allowed[cu_seqlens_q[element] : cu_seqlens_q[element + 1], keys] = True
    return allowed


@pytest.mark.parametrize(("seq_lens", "prefix_lens"), ROWS)
def test_the_layout_allows_exactly_the_pairs_of_the_two_block_rule(seq_lens, prefix_lens):
    layout = layout_of(seq_lens, prefix_lens)
    query_lens, key_lens = torch.diff(layout.cu_seqlens_q), torch.diff(layout.cu_seqlens_k)

    assert layout.cu_seqlens_q.dtype == layout.cu_seqlens_k.dtype == torch.int32
    assert layout.key_index.dtype == torch.int64
    assert layout.cu_seqlens_q.shape == layout.cu_seqlens_k.shape
    # The queries are the row's tokens in order, and no element is empty
    assert layout.cu_seqlens_q[0] == 0 and layout.cu_seqlens_q[-1] == sum(seq_lens)
    assert layout.cu_seqlens_k[0] == 0 and layout.cu_seqlens_k[-1] == layout.key_index.numel()
    assert query_lens.min() > 0 and key_lens.min() > 0
    assert (layout.max_seqlen_q, layout.max_seqlen_k) == (query_lens.max().item(), key_lens.max().item())
    assert torch.equal(allowed_by_layout(layout, sum(seq_lens)), allowed_by_rule(seq_lens, prefix_lens))


def test_the_layout_of_a_row():
    layout = layout_of([6, 3, 5], [1, 3, 4])

    # Segment 0: its prefix, then its second block. Segment 1: one block. Segment 2: its prefix, then its second block
    assert layout.cu_seqlens_q.tolist() == [0, 1, 6, 9, 13, 14]
    assert layout.cu_seqlens_k.tolist() == [0, 1, 7, 10, 14, 19]
    assert layout.key_index.tolist() == [0, *range(0, 6), *range(6, 9), *range(9, 13), *range(9, 14)]
    assert (layout.max_seqlen_q, layout.max_seqlen_k) == (5, 6)


@pytest.mark.parametrize(
    ("prefix_lens", "error"),
    [
        ([2], "one entry per segment"),
        ([2, 3, 1], "one entry per segment"),
        ([[2, 3]], "one entry per segment"),
        ([2.0, 3.0], "must hold integers"),
        ([True, True], "must hold integers"),
        ([0, 3], "positive and at most seq_lens"),
        ([-1, 3], "positive and at most seq_lens"),
        ([2, 4], "positive and at most seq_lens"),
    ],
)
def test_the_layout_refuses_invalid_prefix_lens(prefix_lens, error):
    cu_seqlens, _ = get_cu_seqlens_from_seq_lens(torch.tensor([5, 3]))
    with pytest.raises(ValueError, match=error):
        prepare_two_block_layout(cu_seqlens, torch.tensor(prefix_lens))


def test_two_block_attention_refuses_a_sliding_window():
    attention = Qwen3_5Attention(get_text_config(), "flash_attention_2")
    attention.sliding_window = 4
    with pytest.raises(NotImplementedError, match="sliding-window"):
        attention._compute_two_block_attention(*[torch.zeros(5, 1, 8)] * 3, layout_of([5], [2]))


def test_the_model_makes_the_two_block_call_only_with_prefix_lens():
    """The forward of a model of one full-attention layer, up to its two kernel calls, which are recorded."""
    config = get_text_config()
    config.num_hidden_layers, config.layer_types = 1, ["full_attention"]
    model = Qwen3_5Model.from_config(config, attn_implementation="flash_attention_2", dtype=torch.float32)
    attention = model.layers[0].self_attn
    calls = []

    def record(call: str):
        def kernel_call(q, k, v, *layout):
            calls.append((call, (q, k, v), layout))
            return torch.zeros_like(q)

        return kernel_call

    attention._compute_attention = record("causal")
    attention._compute_two_block_attention = record("two_block")

    input_ids, position_ids = torch.arange(8)[None], torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
    seq_lens = torch.tensor([5, 3])
    with torch.no_grad():
        model(input_ids, position_ids, seq_lens=seq_lens)
        model(input_ids, position_ids, seq_lens=seq_lens, prefix_lens=torch.tensor([2, 3]))

    assert [call for call, _, _ in calls] == ["causal", "two_block"]
    (_, causal_states, (cu_seqlens, max_seqlen)), (_, states, (layout,)) = calls
    assert (cu_seqlens.tolist(), max_seqlen) == ([0, 5, 8], 5)
    assert layout.cu_seqlens_q.tolist() == [0, 2, 5, 8]
    assert layout.cu_seqlens_k.tolist() == [0, 2, 7, 10]
    assert layout.key_index.tolist() == [0, 1, *range(0, 5), *range(5, 8)]
    # Both calls take the row's queries, keys and values after RoPE
    assert [state.shape for state in states] == [(8, 4, 64), (8, 2, 64), (8, 2, 64)]
    for causal_state, state in zip(causal_states, states, strict=True):
        assert torch.equal(causal_state, state)


class Backbone(nn.Module):
    """A language model that records the ``prefix_lens`` it is called with."""

    def forward(self, input_ids, position_ids, *, seq_lens, seq_lens_are_pre_shard=False, prefix_lens="not passed"):
        self.prefix_lens = prefix_lens
        return SimpleNamespace(last_hidden_state=torch.zeros(1, input_ids.shape[1], 4))


class CausalBackbone(nn.Module):
    def forward(self, input_ids, position_ids, *, seq_lens, seq_lens_are_pre_shard=False):
        return SimpleNamespace(last_hidden_state=torch.zeros(1, input_ids.shape[1], 4))


class LanguageModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.model = backbone
        self.lm_head = nn.Linear(4, 8, bias=False)
        inject_prime_lm_head(self)


def read(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"hidden_states": hidden_states}


def test_the_trainer_forward_hands_prefix_lens_to_the_model_only_when_given():
    model = LanguageModel(Backbone())
    input_ids, position_ids = torch.zeros(1, 5, dtype=torch.long), torch.arange(5)[None]
    seq_lens, prefix_lens = torch.tensor([5]), torch.tensor([2])

    forward(model, input_ids, position_ids, seq_lens=seq_lens, hidden_states_fn=read)
    assert model.model.prefix_lens == "not passed"

    forward(model, input_ids, position_ids, seq_lens=seq_lens, prefix_lens=prefix_lens, hidden_states_fn=read)
    assert model.model.prefix_lens is prefix_lens


def test_the_trainer_forward_refuses_prefix_lens_for_a_model_without_two_block_attention():
    model = LanguageModel(CausalBackbone())
    with pytest.raises(AssertionError, match="CausalBackbone has no two-block attention"):
        forward(
            model,
            torch.zeros(1, 5, dtype=torch.long),
            torch.arange(5)[None],
            seq_lens=torch.tensor([5]),
            prefix_lens=torch.tensor([2]),
            hidden_states_fn=read,
        )


@pytest.mark.parametrize("language_model_only", [False, True], ids=["text", "language_model_only"])
def test_the_trainer_forward_refuses_prefix_lens_with_context_parallelism(language_model_only):
    config = get_text_config()
    if language_model_only:
        # The composite config of a text-only checkpoint: prefix_lens reaches the language model through its wrapper
        config = get_vlm_config(config)
        config.language_model_only = True
    model = get_model(config, device="meta", attn="flash_attention_2")
    get_language_model(model).cp_context = CPContext(None, 0, 2)
    with pytest.raises(ValueError, match="not supported with context parallelism"):
        forward(
            model,
            torch.zeros(1, 5, dtype=torch.long),
            torch.arange(5)[None],
            seq_lens=torch.tensor([5]),
            prefix_lens=torch.tensor([2]),
            hidden_states_fn=read,
        )


KERNEL_MODULES = {2: "flash_attn.flash_attn_interface", 3: "flash_attn_interface", 4: "flash_attn.cute.interface"}
OUTPUT_RTOL, GRAD_RTOL = 2e-2, 5e-2
"""bfloat16 kernels against a float32 reference on the same bfloat16 inputs."""


def two_block_attention(version: int) -> Qwen3_5Attention:
    """A Qwen3.5 attention on flash attention ``version``, or a skip where this GPU or environment has no such
    kernel."""
    capability = torch.cuda.get_device_capability()[0]
    if version == 3 and capability != 9:
        pytest.skip("FlashAttention 3 needs an SM90 GPU")
    if version == 4 and capability not in (9, 10, 11):
        pytest.skip("FlashAttention 4 needs an SM90 or SM100/SM110 GPU")
    pytest.importorskip(KERNEL_MODULES[version])
    return Qwen3_5Attention(get_text_config(), f"flash_attention_{version}")


def qkv(attention: Qwen3_5Attention, total_tokens: int) -> list[torch.Tensor]:
    """Random query, key and value states of a packed row, ``[tokens, heads, dim]`` each."""
    return [
        torch.randn(total_tokens, heads, attention.head_dim, device="cuda", dtype=torch.bfloat16)
        for heads in (attention.num_heads, attention.num_key_value_heads, attention.num_key_value_heads)
    ]


def leaves(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.clone().requires_grad_() for tensor in tensors]


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Softmax attention of ``[tokens, heads, dim]`` states in float32, each query over its ``allowed`` keys. Query
    head ``i`` reads key and value head ``i // groups``."""
    groups = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * q.shape[-1] ** -0.5
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v)


@pytest.mark.gpu
@pytest.mark.parametrize("version", [2, 3, 4])
@pytest.mark.parametrize(("seq_lens", "prefix_lens"), ROWS[2:])
def test_two_block_attention_equals_a_dense_masked_attention(version, seq_lens, prefix_lens):
    attention = two_block_attention(version)
    torch.manual_seed(0)
    inputs = qkv(attention, sum(seq_lens))

    actual_inputs = leaves(inputs)
    actual = attention._compute_two_block_attention(*actual_inputs, layout_of(seq_lens, prefix_lens, device="cuda"))
    actual.float().square().sum().backward()

    reference_inputs = leaves(inputs)
    reference = dense_attention(*reference_inputs, allowed_by_rule(seq_lens, prefix_lens).cuda())
    reference.square().sum().backward()

    assert actual.shape == reference.shape and actual.dtype == torch.bfloat16
    assert_relative(actual, reference, OUTPUT_RTOL, "output")
    for label, actual_input, reference_input in zip(("dq", "dk", "dv"), actual_inputs, reference_inputs, strict=True):
        assert_relative(actual_input.grad, reference_input.grad, GRAD_RTOL, label)


@pytest.mark.gpu
@pytest.mark.parametrize("version", [2, 3, 4])
def test_two_block_attention_traces_under_torch_compile(version):
    """``torch.compile(fullgraph=True)`` through the gather and the kernel's op, forward and backward, as the compiled
    decoder layers run them, on rows whose numbers of tokens, batch elements and gathered keys all differ: the second
    row makes those sizes dynamic, and the third runs with them dynamic."""
    attention = two_block_attention(version)
    compiled_attention = torch.compile(attention._compute_two_block_attention, fullgraph=True)
    torch.manual_seed(0)
    for seq_lens, prefix_lens in (([6, 3, 5], [1, 3, 4]), ([9, 4, 7, 12], [3, 4, 6, 1]), ([10, 6], [4, 6])):
        layout = layout_of(seq_lens, prefix_lens, device="cuda")
        inputs = qkv(attention, sum(seq_lens))
        eager_inputs, compiled_inputs = leaves(inputs), leaves(inputs)

        eager = attention._compute_two_block_attention(*eager_inputs, layout)
        # One weight for both losses, so that a difference belongs to the tracing
        weight = torch.randn_like(eager)
        (eager * weight).sum().backward()
        compiled = compiled_attention(*compiled_inputs, layout)
        (compiled * weight).sum().backward()

        torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
        for label, compiled_input, eager_input in zip(("dq", "dk", "dv"), compiled_inputs, eager_inputs, strict=True):
            assert_relative(compiled_input.grad, eager_input.grad, 1e-2, label)


@pytest.mark.gpu
def test_a_compiled_full_attention_layer_takes_new_rows_without_recompiling(monkeypatch):
    """The layout's tensors and ints are dynamic sources of the compiled decoder layers (``mark_dynamic_args``), so the
    graph of the first row also runs rows whose numbers of tokens, segments, batch elements and gathered keys differ.
    The model's forward calls the compiled layer by keyword, as in training, forward and backward."""
    # The sources are process-wide: restored after the test
    monkeypatch.setattr(torch.compiler.config, "dynamic_sources", torch.compiler.config.dynamic_sources)
    mark_dynamic_args()
    torch._dynamo.reset()
    clear_compilation_metrics()
    torch.manual_seed(0)
    model = get_model(get_text_config())
    model.model.layers[-1].compile(fullgraph=True)

    # The first and the last row hold a segment of one block
    for seq_lens, prefix_lens in (([40, 26], [28, 26]), ([33, 50, 21], [9, 20, 6]), ([70, 18], [70, 5])):
        out = forward(
            model,
            torch.randint(0, model.config.vocab_size, (1, sum(seq_lens)), device="cuda"),
            torch.cat([torch.arange(seq_len, device="cuda") for seq_len in seq_lens])[None],
            seq_lens=torch.tensor(seq_lens, device="cuda"),
            prefix_lens=torch.tensor(prefix_lens, device="cuda"),
            hidden_states_fn=read,
        )
        out["hidden_states"].float().sum().backward()

    # As test_compile.py counts the graphs of a model's decoder layers
    graphs = [
        metrics
        for metrics in get_compilation_metrics()
        if metrics.co_name is not None and metrics.is_forward is not False and metrics.graph_op_count
    ]
    assert len(graphs) == 1, [metrics.recompile_reason for metrics in graphs]


SEQ_LENS, PREFIX_LENS = [40, 24], [28, 24]
"""A row of a segment with a prefix of 28 tokens and a second block of 12, and a segment of one block."""
UNCHANGED = dict(atol=0.03, rtol=0.01)
"""The tolerance of the packing test of test_model.py. The final hidden states have a root mean square of 1."""
CHANGED = 0.1


def final_hidden_states(
    model: nn.Module, input_ids: torch.Tensor, prefix_lens: list[int] | None = None
) -> torch.Tensor:
    """The final hidden states of the row ``input_ids``, ``[tokens, hidden]``, through the trainer's forward."""
    position_ids = torch.cat([torch.arange(seq_len, device="cuda") for seq_len in SEQ_LENS])[None]
    with torch.no_grad():
        out = forward(
            model,
            input_ids,
            position_ids,
            seq_lens=torch.tensor(SEQ_LENS, device="cuda"),
            prefix_lens=None if prefix_lens is None else torch.tensor(prefix_lens, device="cuda"),
            hidden_states_fn=read,
        )
    return out["hidden_states"][0].float()


def model_and_row(vlm: bool = False) -> tuple[nn.Module, torch.Tensor]:
    """A tiny text model, or with ``vlm`` the model of the composite config, whose wrapper runs the language model
    on the embeddings it prepares; and a row of text tokens."""
    torch.manual_seed(0)
    config = get_text_config()
    model = get_model(get_vlm_config(config) if vlm else config)
    language_model = get_language_model(model)
    # The full attention of a random model adds little to its output. Raised, what a token attends shows in the
    # final hidden states
    with torch.no_grad():
        language_model.layers[-1].self_attn.o_proj.weight.mul_(8)
    return model, torch.randint(0, config.vocab_size, (1, sum(SEQ_LENS)), device="cuda")


def with_other_tokens(input_ids: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """``input_ids`` with every token of ``[start, end)`` replaced by another one."""
    changed = input_ids.clone()
    changed[0, start:end] = (changed[0, start:end] + 1) % 256
    return changed


@pytest.mark.gpu
@pytest.mark.parametrize("vlm", [False, True], ids=["text", "vlm"])
def test_the_prefix_does_not_see_the_second_block(vlm):
    model, input_ids = model_and_row(vlm)
    states = final_hidden_states(model, input_ids, PREFIX_LENS)
    changed = final_hidden_states(model, with_other_tokens(input_ids, 28, 40), PREFIX_LENS)

    # The prefix of the changed second block, and the other segment
    for unchanged in (slice(0, 28), slice(40, 64)):
        torch.testing.assert_close(changed[unchanged], states[unchanged], **UNCHANGED)
    assert (changed[28:40] - states[28:40]).abs().max() > CHANGED


@pytest.mark.gpu
def test_a_prefix_token_sees_the_later_prefix_tokens_which_a_causal_one_does_not():
    model, input_ids = model_and_row()
    later_prefix_changed = with_other_tokens(input_ids, 1, 28)

    states = final_hidden_states(model, input_ids, PREFIX_LENS)
    changed = final_hidden_states(model, later_prefix_changed, PREFIX_LENS)
    assert (changed[0] - states[0]).abs().max() > CHANGED

    causal_states = final_hidden_states(model, input_ids)
    causal_changed = final_hidden_states(model, later_prefix_changed)
    torch.testing.assert_close(causal_changed[0], causal_states[0], **UNCHANGED)


@pytest.mark.gpu
def test_without_prefix_lens_the_forward_is_the_causal_one():
    model, input_ids = model_and_row()
    states = final_hidden_states(model, input_ids)

    # As callers that know no prefix_lens run the model
    position_ids = torch.cat([torch.arange(seq_len, device="cuda") for seq_len in SEQ_LENS])[None]
    with torch.no_grad():
        direct = model(
            input_ids,
            position_ids=position_ids,
            seq_lens=torch.tensor(SEQ_LENS, device="cuda"),
            hidden_states_fn=read,
        )["hidden_states"][0].float()
    assert torch.equal(states, direct)
    # Causal: no token sees a later one
    changed = final_hidden_states(model, with_other_tokens(input_ids, 39, 40))
    torch.testing.assert_close(changed[:39], states[:39], **UNCHANGED)
    assert (final_hidden_states(model, input_ids, PREFIX_LENS) - states).abs().max() > CHANGED
