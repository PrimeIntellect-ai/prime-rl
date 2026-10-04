import math
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.worker.weight_layout import DenseVllmModelLayoutAdapter, LogicalSparseUpdate
from prime_rl.inference.vllm.worker.weight_transfer import load_sparse_delta_weights
from prime_rl.utils.delta import ModelDeltaManager


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...], *, tp_rank: int, tp_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(shape))
        self.weight.output_dim = 0
        self.tp_rank = tp_rank
        self.tp_size = tp_size


class RowParallelLinear(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...], *, tp_rank: int, tp_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(shape))
        self.weight.input_dim = 1
        self.tp_rank = tp_rank
        self.tp_size = tp_size


class QKVParallelLinear(torch.nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        tp_rank: int,
        tp_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        num_kv_head_replicas: int = 1,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(shape))
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_kv_head_replicas = num_kv_head_replicas


class MergedColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        output_sizes: tuple[int, ...],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(shape))
        self.output_sizes = output_sizes
        self.tp_rank = tp_rank
        self.tp_size = tp_size


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...], *, start: int, end: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(shape))
        self.num_embeddings_per_partition = shape[0]
        self.shard_indices = SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end)


def _model_with_parameter(path: str, module: torch.nn.Module) -> torch.nn.Module:
    root = torch.nn.Module()
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = torch.nn.Module()
        current.add_module(part, child)
        current = child
    current.add_module(parts[-1], module)
    return root


def _full_update(name: str, tensor: torch.Tensor, value_offset: int = 0) -> LogicalSparseUpdate:
    return LogicalSparseUpdate(
        name=name,
        shape=tuple(tensor.shape),
        indices=torch.arange(tensor.numel(), dtype=torch.int64),
        values=torch.arange(value_offset, value_offset + tensor.numel(), dtype=torch.float32),
    )


def _materialize(adapter: DenseVllmModelLayoutAdapter, updates: list[LogicalSparseUpdate], size: int) -> torch.Tensor:
    result = torch.zeros(size)
    for update in updates:
        local = adapter.to_local(update)
        result.index_add_(0, local.indices, local.values)
    return result


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_qkv_projection_matches_qwen_and_llama_fused_layout(tp_rank: int) -> None:
    qkv = QKVParallelLinear(
        (8, 2),
        tp_rank=tp_rank,
        tp_size=2,
        total_num_heads=4,
        total_num_kv_heads=2,
        num_heads=2,
        num_kv_heads=1,
        head_size=2,
    )
    model = _model_with_parameter("model.layers.0.self_attn.qkv_proj", qkv)
    adapter = DenseVllmModelLayoutAdapter(model)
    prefix = "model.layers.0.self_attn"
    q = torch.empty((8, 2))
    k = torch.empty((4, 2))
    v = torch.empty((4, 2))
    updates = [
        _full_update(f"{prefix}.q_proj.weight", q, 0),
        _full_update(f"{prefix}.k_proj.weight", k, 100),
        _full_update(f"{prefix}.v_proj.weight", v, 200),
    ]

    actual = _materialize(adapter, updates, qkv.weight.numel()).reshape_as(qkv.weight)

    q_values = updates[0].values.reshape_as(q)[tp_rank * 4 : (tp_rank + 1) * 4]
    k_values = updates[1].values.reshape_as(k)[tp_rank * 2 : (tp_rank + 1) * 2]
    v_values = updates[2].values.reshape_as(v)[tp_rank * 2 : (tp_rank + 1) * 2]
    assert torch.equal(actual, torch.cat((q_values, k_values, v_values)))


def test_qkv_projection_replicates_kv_heads_for_gqa() -> None:
    qkv = QKVParallelLinear(
        (6, 2),
        tp_rank=3,
        tp_size=4,
        total_num_heads=4,
        total_num_kv_heads=2,
        num_heads=1,
        num_kv_heads=1,
        head_size=2,
        num_kv_head_replicas=2,
    )
    model = _model_with_parameter("model.layers.0.self_attn.qkv_proj", qkv)
    adapter = DenseVllmModelLayoutAdapter(model)
    prefix = "model.layers.0.self_attn"
    q = _full_update(f"{prefix}.q_proj.weight", torch.empty((8, 2)), 0)
    k = _full_update(f"{prefix}.k_proj.weight", torch.empty((4, 2)), 100)
    v = _full_update(f"{prefix}.v_proj.weight", torch.empty((4, 2)), 200)

    actual = _materialize(adapter, [q, k, v], qkv.weight.numel()).reshape_as(qkv.weight)

    assert torch.equal(
        actual,
        torch.cat(
            (
                q.values.reshape(8, 2)[6:8],
                k.values.reshape(4, 2)[2:4],
                v.values.reshape(4, 2)[2:4],
            )
        ),
    )


def test_qkv_projection_preserves_wide_delta_values_for_exact_bfloat16_updates(tmp_path) -> None:
    qkv = QKVParallelLinear(
        (6, 2),
        tp_rank=0,
        tp_size=1,
        total_num_heads=1,
        total_num_kv_heads=1,
        num_heads=1,
        num_kv_heads=1,
        head_size=2,
    ).to(torch.bfloat16)
    with torch.no_grad():
        qkv.weight.fill_(1.0)
    model = _model_with_parameter("model.layers.0.self_attn.qkv_proj", qkv)
    name = "model.layers.0.self_attn.q_proj.weight"
    base = {name: torch.ones((2, 2), dtype=torch.bfloat16)}
    target = {name: base[name].clone()}
    target[name][0, 0] = -0.001
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)
    load_sparse_delta_weights(model, delta_path.as_posix())

    assert torch.equal(qkv.weight[:2], target[name])


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_gate_up_projection_matches_merged_column_layout(tp_rank: int) -> None:
    gate_up = MergedColumnParallelLinear(
        (6, 2),
        output_sizes=(6, 6),
        tp_rank=tp_rank,
        tp_size=2,
    )
    model = _model_with_parameter("model.layers.0.mlp.gate_up_proj", gate_up)
    adapter = DenseVllmModelLayoutAdapter(model)
    prefix = "model.layers.0.mlp"
    gate = _full_update(f"{prefix}.gate_proj.weight", torch.empty((6, 2)), 0)
    up = _full_update(f"{prefix}.up_proj.weight", torch.empty((6, 2)), 100)

    actual = _materialize(adapter, [gate, up], gate_up.weight.numel()).reshape_as(gate_up.weight)

    assert torch.equal(
        actual,
        torch.cat(
            (
                gate.values.reshape(6, 2)[tp_rank * 3 : (tp_rank + 1) * 3],
                up.values.reshape(6, 2)[tp_rank * 3 : (tp_rank + 1) * 3],
            )
        ),
    )


def test_row_and_column_parallel_projection() -> None:
    row = RowParallelLinear((4, 4), tp_rank=1, tp_size=2)
    row_model = _model_with_parameter("model.layers.0.mlp.down_proj", row)
    row_update = _full_update("model.layers.0.mlp.down_proj.weight", torch.empty((4, 8)))
    row_local = DenseVllmModelLayoutAdapter(row_model).to_local(row_update)
    assert torch.equal(row_update.values.reshape(4, 8)[:, 4:8].reshape(-1), row_local.values)
    row_result = torch.zeros(row.weight.numel()).index_add_(0, row_local.indices, row_local.values)
    assert torch.equal(row_result.reshape_as(row.weight), row_update.values.reshape(4, 8)[:, 4:8])

    column = ColumnParallelLinear((4, 4), tp_rank=1, tp_size=2)
    column_model = _model_with_parameter("model.layers.0.self_attn.o_proj", column)
    column_update = _full_update("model.layers.0.self_attn.o_proj.weight", torch.empty((8, 4)))
    column_local = DenseVllmModelLayoutAdapter(column_model).to_local(column_update)
    assert torch.equal(column_update.values.reshape(8, 4)[4:8].reshape(-1), column_local.values)
    assert torch.equal(column_local.indices, torch.arange(column.weight.numel()))


def test_vocab_and_replicated_projection() -> None:
    vocab = VocabParallelEmbedding((8, 3), start=5, end=10)
    vocab_model = _model_with_parameter("model.embed_tokens", vocab)
    vocab_update = _full_update("model.embed_tokens.weight", torch.empty((10, 3)))
    vocab_local = DenseVllmModelLayoutAdapter(vocab_model).to_local(vocab_update)
    expected = vocab_update.values.reshape(10, 3)[5:10]
    actual = torch.zeros_like(vocab.weight).reshape(-1).index_add_(0, vocab_local.indices, vocab_local.values)
    assert torch.equal(actual.reshape_as(vocab.weight)[:5], expected)
    assert torch.count_nonzero(actual.reshape_as(vocab.weight)[5:]) == 0

    norm = torch.nn.LayerNorm(4)
    norm_model = _model_with_parameter("model.norm", norm)
    norm_update = _full_update("model.norm.weight", torch.empty(4))
    norm_local = DenseVllmModelLayoutAdapter(norm_model).to_local(norm_update)
    assert torch.equal(norm_local.indices, norm_update.indices)
    assert torch.equal(norm_local.values, norm_update.values)


def test_quantized_parameter_is_rejected_explicitly() -> None:
    module = ColumnParallelLinear((2, 2), tp_rank=0, tp_size=1)
    module.quant_config = object()
    model = _model_with_parameter("model.proj", module)
    update = _full_update("model.proj.weight", torch.empty((2, 2)))

    with pytest.raises(ValueError, match="quantized inference layout"):
        DenseVllmModelLayoutAdapter(model).to_local(update)


def test_adapter_uses_installed_vllm_tp_parameter_metadata(monkeypatch) -> None:
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = root.model.layers[0]
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = linear.QKVParallelLinear(4, 2, 2, 1, bias=False)
    layer.self_attn.o_proj = linear.RowParallelLinear(4, 4, bias=False)
    adapter = DenseVllmModelLayoutAdapter(root)

    q = _full_update("model.layers.0.self_attn.q_proj.weight", torch.empty((4, 4)))
    q_local = adapter.to_local(q)
    o = _full_update("model.layers.0.self_attn.o_proj.weight", torch.empty((4, 4)))
    o_local = adapter.to_local(o)

    assert torch.equal(q_local.values, q.values.reshape(4, 4)[2:4].reshape(-1))
    assert torch.equal(o_local.values, o.values.reshape(4, 4)[:, 2:4].reshape(-1))


def _logical_dense_state(family: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    shapes = {
        "model.embed_tokens.weight": (10, 8),
        "model.layers.0.input_layernorm.weight": (8,),
        "model.layers.0.self_attn.q_proj.weight": (8, 8),
        "model.layers.0.self_attn.k_proj.weight": (4, 8),
        "model.layers.0.self_attn.v_proj.weight": (4, 8),
        "model.layers.0.self_attn.o_proj.weight": (8, 8),
        "model.layers.0.mlp.gate_proj.weight": (8, 8),
        "model.layers.0.mlp.up_proj.weight": (8, 8),
        "model.layers.0.mlp.down_proj.weight": (8, 8),
        "model.layers.0.post_attention_layernorm.weight": (8,),
        "model.norm.weight": (8,),
        "lm_head.weight": (10, 8),
    }
    if family == "qwen3":
        shapes |= {
            "model.layers.0.self_attn.q_norm.weight": (2,),
            "model.layers.0.self_attn.k_norm.weight": (2,),
        }
    base = {name: torch.zeros(shape) for name, shape in shapes.items()}
    target = {
        name: torch.arange(1, math.prod(shape) + 1, dtype=torch.float32).reshape(shape) + 1000 * position
        for position, (name, shape) in enumerate(shapes.items())
    }
    return base, target


class _FakeDenseModel(torch.nn.Module):
    def __init__(self, family: str, *, tp_rank: int, tp_size: int) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        vocab_partition = 12 // tp_size
        vocab_start = tp_rank * vocab_partition
        vocab_end = min(vocab_start + vocab_partition, 10)
        self.model.embed_tokens = VocabParallelEmbedding((vocab_partition, 8), start=vocab_start, end=vocab_end)
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.input_layernorm = torch.nn.RMSNorm(8)
        layer.post_attention_layernorm = torch.nn.RMSNorm(8)
        layer.self_attn = torch.nn.Module()
        q_size = 8 // tp_size
        num_kv_heads = max(1, 2 // tp_size)
        kv_size = num_kv_heads * 2
        kv_replicas = max(1, tp_size // 2)
        layer.self_attn.qkv_proj = QKVParallelLinear(
            (q_size + kv_size + kv_size, 8),
            tp_rank=tp_rank,
            tp_size=tp_size,
            total_num_heads=4,
            total_num_kv_heads=2,
            num_heads=4 // tp_size,
            num_kv_heads=num_kv_heads,
            head_size=2,
            num_kv_head_replicas=kv_replicas,
        )
        layer.self_attn.o_proj = RowParallelLinear((8, 8 // tp_size), tp_rank=tp_rank, tp_size=tp_size)
        if family == "qwen3":
            layer.self_attn.q_norm = torch.nn.RMSNorm(2)
            layer.self_attn.k_norm = torch.nn.RMSNorm(2)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = MergedColumnParallelLinear(
            (16 // tp_size, 8), output_sizes=(8, 8), tp_rank=tp_rank, tp_size=tp_size
        )
        layer.mlp.down_proj = RowParallelLinear((8, 8 // tp_size), tp_rank=tp_rank, tp_size=tp_size)
        self.model.norm = torch.nn.RMSNorm(8)
        self.lm_head = VocabParallelEmbedding((vocab_partition, 8), start=vocab_start, end=vocab_end)
        with torch.no_grad():
            for param in self.parameters():
                param.zero_()


def _expected_dense_parameters(
    model: _FakeDenseModel,
    target: dict[str, torch.Tensor],
    *,
    tp_rank: int,
    tp_size: int,
) -> dict[str, torch.Tensor]:
    expected = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    vocab_partition = 12 // tp_size
    vocab_start = tp_rank * vocab_partition
    vocab_end = min(vocab_start + vocab_partition, 10)
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        expected[name][: vocab_end - vocab_start] = target[name][vocab_start:vocab_end]

    layer = "model.layers.0"
    q_rows = 8 // tp_size
    q = target[f"{layer}.self_attn.q_proj.weight"][tp_rank * q_rows : (tp_rank + 1) * q_rows]
    kv_rows = 4 // tp_size if tp_size <= 2 else 2
    kv_source_rank = tp_rank if tp_size <= 2 else tp_rank // (tp_size // 2)
    kv_slice = slice(kv_source_rank * kv_rows, (kv_source_rank + 1) * kv_rows)
    k = target[f"{layer}.self_attn.k_proj.weight"][kv_slice]
    v = target[f"{layer}.self_attn.v_proj.weight"][kv_slice]
    expected[f"{layer}.self_attn.qkv_proj.weight"] = torch.cat((q, k, v))
    o_width = 8 // tp_size
    expected[f"{layer}.self_attn.o_proj.weight"] = target[f"{layer}.self_attn.o_proj.weight"][
        :, tp_rank * o_width : (tp_rank + 1) * o_width
    ]
    mlp_rows = 8 // tp_size
    gate = target[f"{layer}.mlp.gate_proj.weight"][tp_rank * mlp_rows : (tp_rank + 1) * mlp_rows]
    up = target[f"{layer}.mlp.up_proj.weight"][tp_rank * mlp_rows : (tp_rank + 1) * mlp_rows]
    expected[f"{layer}.mlp.gate_up_proj.weight"] = torch.cat((gate, up))
    expected[f"{layer}.mlp.down_proj.weight"] = target[f"{layer}.mlp.down_proj.weight"][
        :, tp_rank * mlp_rows : (tp_rank + 1) * mlp_rows
    ]
    replicated_names = {
        f"{layer}.input_layernorm.weight",
        f"{layer}.post_attention_layernorm.weight",
        f"{layer}.self_attn.q_norm.weight",
        f"{layer}.self_attn.k_norm.weight",
        "model.norm.weight",
    }
    for name in replicated_names & target.keys():
        expected[name] = target[name]
    return expected


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_logical_delta_applies_to_qwen3_and_llama_tp_layouts(tmp_path, family: str, tp_size: int) -> None:
    base, target = _logical_dense_state(family)
    delta_path = tmp_path / f"{family}-tp{tp_size}.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    for tp_rank in range(tp_size):
        model = _FakeDenseModel(family, tp_rank=tp_rank, tp_size=tp_size)
        load_sparse_delta_weights(model, delta_path.as_posix())
        expected = _expected_dense_parameters(model, target, tp_rank=tp_rank, tp_size=tp_size)
        actual = dict(model.named_parameters())
        assert actual.keys() == expected.keys()
        for name in actual:
            assert torch.equal(actual[name], expected[name]), name


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_logical_delta_preserves_qwen3_and_llama_logits(tmp_path, family: str) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

    if family == "qwen3":
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        model_type = Qwen3ForCausalLM
    else:
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        model_type = LlamaForCausalLM

    torch.manual_seed(0)
    base_model = model_type(config).eval()
    base_state = {name: tensor.clone() for name, tensor in base_model.state_dict().items()}
    parameter_names = set(dict(base_model.named_parameters()))
    target_state = {name: tensor.clone() for name, tensor in base_state.items()}
    for name in parameter_names:
        target_state[name].reshape(-1)[0] += 0.01
    target_model = model_type(config).eval()
    target_model.load_state_dict(target_state)
    delta_path = tmp_path / f"{family}-logits.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base_state, target_state, delta_path)

    load_sparse_delta_weights(base_model, delta_path.as_posix())

    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        actual = base_model(input_ids).logits
        expected = target_model(input_ids).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
