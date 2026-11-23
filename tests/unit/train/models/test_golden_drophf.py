"""Golden hashes of every model's structure, init, conversion and GPU step, written by `PRIME_RL_REGEN_GOLDENS=1`."""

import copy
import gzip
import hashlib
import json
import math
import os
import re
import sys
from functools import cache
from pathlib import Path

import pytest
import torch
from tiny_models import TINY_MODELS, TINY_PRIME_RL_MODELS
from triton.runtime.autotuner import Autotuner

from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_MODELS
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, cross_entropy_sum, inject_prime_lm_head
from prime_rl.trainer.vlm import get_layer_prefix
from prime_rl.transports.weights.nccl import (
    filter_state_dict_by_layers,
    get_max_layer_num,
    preprocess_layer_checkpoint,
)
from prime_rl.utils.weights import convert_state_dict_to_hf, get_chunked_conversion_ops

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REGEN = os.environ.get("PRIME_RL_REGEN_GOLDENS") == "1"
GOLDEN_DIR = Path(__file__).parent / "golden_drophf"
SEED = 1234
CPU_ATTN = "flash_attention_3"
GPU_SEQ_LENS = [96, 160, 57, 199]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
GPU_GRAD_RTOL = {"deepseek_v4": 1e-1, "glm_moe_dsa": 2e-3, "glm5_next": 7e-3, "laguna": 1.2e-4}
GRAD_STATS = re.compile(r"^(?P<layout>\S+ (?P<shape>\[[^\]]*\])) \S+ sum=(?P<sum>\S+) norm=(?P<norm>\S+)$")

MODEL_CLASSES = dict(_CUSTOM_CAUSAL_LM_MODELS)
CONFIG_CLASSES = {config_cls.model_type: config_cls for config_cls in MODEL_CLASSES}
GPU_CONFIGS = TINY_MODELS | TINY_PRIME_RL_MODELS

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=64,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    moe_shared_expert_intermediate_size=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    head_dim=64,
    q_lora_rank=16,
    o_lora_rank=16,
    o_groups=2,
    index_n_heads=2,
    indexer_n_heads=2,
    num_experts=8,
    n_routed_experts=8,
    num_local_experts=8,
    num_experts_per_tok=2,
    linear_num_heads=4,
    linear_head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    mamba_num_heads=8,
    mamba_head_dim=8,
    ssm_state_size=16,
    n_groups=2,
    hc_lowrank=16,
    ple_embed_dim=64,
    ngram_vocab_size_base=1024,
    depth=1,
    num_heads=4,
    out_hidden_size=64,
)
LAYER_KINDS = {
    "deepseek_v4": dict(
        num_hidden_layers=4,
        num_hash_layers=1,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
    ),
    "glm5_next_text": dict(num_hidden_layers=4, layer_types=["linear_attention", "deepseek_sparse_attention"] * 2),
}


def tiny_config(config_cls, tied: bool):
    """`test_scratch_init`'s tiny config, with `tie_word_embeddings` set on every level when `tied`."""
    default = config_cls()
    tie = {"tie_word_embeddings": True} if tied else {}

    def shrink(config):
        tiny = {key: value for key, value in TINY.items() if hasattr(config, key)}
        return tiny | LAYER_KINDS.get(config.model_type, {}) | tie

    sub_configs = {
        name: shrink(getattr(default, name)) for name in default.sub_configs if getattr(default, name, None) is not None
    }
    kwargs = sub_configs | tie if sub_configs else shrink(default)
    return config_cls(**kwargs, dsa_backend="tilelang")


TIED_BUILD_RAISES = {"inkling2", "inkling2_text"}
CASES = [
    pytest.param(config_cls, tied, id=config_cls.model_type + ("-tied" if tied else ""))
    for config_cls in MODEL_CLASSES
    for tied in (False, True)
    if not (tied and config_cls.model_type in TIED_BUILD_RAISES)
]


def case_name(config_cls, tied: bool) -> str:
    return config_cls.model_type + ("-tied" if tied else "")


def layout(tensor: torch.Tensor) -> str:
    return f"{str(tensor.dtype).removeprefix('torch.')} {list(tensor.shape)}"


def describe(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return f"{layout(tensor)} {hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]}"


def describe_with_stats(tensor: torch.Tensor) -> str:
    values = tensor.detach().double()
    return f"{describe(tensor)} sum={values.sum().item():.17g} norm={values.norm().item():.17g}"


def describe_all(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: describe(tensor) for name, tensor in tensors.items()}


def named_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {**dict(model.named_parameters(remove_duplicate=False)), **dict(model.named_buffers(remove_duplicate=False))}


def tie_groups(model: torch.nn.Module) -> list[list[str]]:
    names_by_param: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        names_by_param.setdefault(id(param), []).append(name)
    return sorted(sorted(names) for names in names_by_param.values() if len(names) > 1)


def attn_implementations(config) -> dict[str, str | None]:
    sub_configs = {
        name: getattr(config, name) for name in config.sub_configs if getattr(config, name, None) is not None
    }
    return {name: sub_config._attn_implementation for name, sub_config in ({"": config} | sub_configs).items()}


def clone(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state_dict.items()}


def build_on_meta(config_cls, tied: bool, dtype: torch.dtype) -> torch.nn.Module:
    with torch.device("meta"):
        return MODEL_CLASSES[config_cls].from_config(
            tiny_config(config_cls, tied), attn_implementation=CPU_ATTN, dtype=dtype
        )


@torch.no_grad()
def scratch_init(model: torch.nn.Module, device: str) -> None:
    """The trainer's `initialize_from_scratch`, re-tying the weights that FSDP keeps tied through `to_empty`."""
    model.to_empty(device=device)
    model.tie_weights()
    for tensor in [*model.parameters(), *model.buffers()]:
        tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
    model.init_buffers_post_meta()
    torch.manual_seed(SEED)
    model.initialize_weights()


def read_goldens(arm: str) -> dict:
    path = GOLDEN_DIR / f"{arm}.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}


load_goldens = cache(read_goldens)


def flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        value = dict.fromkeys(value, "present")
    elif isinstance(value, list):
        value = {str(index): item for index, item in enumerate(value)}
    if isinstance(value, dict):
        return {key: leaf for name, item in value.items() for key, leaf in flatten(item, f"{prefix}/{name}").items()}
    return {prefix: value}


def golden_and_actual(arm: str, case: str, record: dict) -> tuple[dict, dict]:
    """The golden record of `case` and `record` in its JSON form; with regeneration on, `record` becomes the golden."""
    actual = json.loads(json.dumps(record))
    if REGEN:
        goldens = read_goldens(arm)
        goldens[case] = actual
        GOLDEN_DIR.mkdir(exist_ok=True)
        text = json.dumps(goldens, indent=1, sort_keys=True) + "\n"
        (GOLDEN_DIR / f"{arm}.json.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
        return copy.deepcopy(actual), actual
    goldens = load_goldens(arm)
    assert case in goldens, f"no golden for {arm}/{case}; regenerate with PRIME_RL_REGEN_GOLDENS=1"
    return goldens[case], actual


def assert_same(label: str, expected: dict, actual: dict) -> None:
    expected, actual = flatten(expected), flatten(actual)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    report = [f"missing {key}" for key in missing] + [f"extra {key}" for key in extra]
    report += [f"changed {key}: {expected[key]!r} -> {actual[key]!r}" for key in changed]
    assert not report, f"{label}: {len(report)} differences\n" + "\n".join(report[:40])


def check_golden(arm: str, case: str, record: dict) -> None:
    assert_same(f"{arm}/{case}", *golden_and_actual(arm, case, record))


def grad_errors(expected: str, actual: str) -> tuple[float, float]:
    """Relative error of the gradient's norm, and its sum's error over the largest sum change of that relative size."""
    want, got = GRAD_STATS.match(expected), GRAD_STATS.match(actual)
    if want["layout"] != got["layout"]:
        return math.inf, math.inf
    norm = float(want["norm"])
    if norm == 0:
        return (0.0, 0.0) if expected == actual else (math.inf, math.inf)
    sum_scale = norm * math.sqrt(math.prod(json.loads(want["shape"])))
    return abs(float(got["norm"]) - norm) / norm, abs(float(got["sum"]) - float(want["sum"])) / sum_scale


def assert_grads_close(label: str, expected: dict[str, str], actual: dict[str, str], rtol: float) -> None:
    assert expected.keys() == actual.keys(), f"{label}: gradient names differ"
    far = []
    for name, want in expected.items():
        got = actual[name]
        if want == got:
            continue
        if "none" in (want, got) or max(grad_errors(want, got)) > rtol:
            far.append(f"{name}: {want!r} -> {got!r}")
    assert not far, f"{label}: {len(far)} gradients beyond rtol {rtol}\n" + "\n".join(far[:40])


@pytest.mark.parametrize(("config_cls", "tied"), CASES)
def test_structure_is_unchanged(config_cls, tied):
    """Parameter and buffer names, shapes, dtypes, persistence and tying, as built and after `tie_weights`."""
    model = build_on_meta(config_cls, tied, torch.float32)
    state_dict_keys = sorted(model.state_dict())
    record = {
        "config": json.loads(model.config.to_json_string(use_diff=False)),
        "attn_implementation": attn_implementations(model.config),
        "modules": sorted(f"{name} {type(module).__name__}" for name, module in model.named_modules()),
        "parameters": {
            name: f"{layout(param)} requires_grad={param.requires_grad}"
            for name, param in model.named_parameters(remove_duplicate=False)
        },
        "buffers": {
            name: f"{layout(buffer)} persistent={name in state_dict_keys}"
            for name, buffer in model.named_buffers(remove_duplicate=False)
        },
        "state_dict": state_dict_keys,
        "tied_as_built": tie_groups(model),
        "num_active_params": model.num_active_params(),
    }
    model.to_empty(device="cpu")
    model.tie_weights()
    record["tied_after_to_empty_and_tie_weights"] = tie_groups(model)
    check_golden("structure", case_name(config_cls, tied), record)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("config_cls", "tied"), CASES)
def test_scratch_init_is_unchanged(config_cls, tied, dtype):
    """Bitwise seeded scratch init of a model built on meta, as the trainer builds it."""
    model = build_on_meta(config_cls, tied, DTYPES[dtype])
    scratch_init(model, "cpu")
    check_golden("init_meta", f"{case_name(config_cls, tied)}-{dtype}", describe_all(named_tensors(model)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("config_cls", "tied"), CASES)
def test_direct_cpu_init_is_unchanged(config_cls, tied, dtype):
    """Bitwise init of a model built straight on the CPU, which initializes its weights while it is built."""
    torch.manual_seed(SEED)
    with torch.device("cpu"):
        model = MODEL_CLASSES[config_cls].from_config(
            tiny_config(config_cls, tied), attn_implementation=CPU_ATTN, dtype=DTYPES[dtype]
        )
    check_golden("init_direct_cpu", f"{case_name(config_cls, tied)}-{dtype}", describe_all(named_tensors(model)))


def weight_broadcast_packets(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[dict]:
    """What `NCCLBroadcaster.send` puts on the wire for an unsharded model, packet by packet."""
    layer_prefix = get_layer_prefix(model)
    chunked_ops = get_chunked_conversion_ops(model, state_dict)
    num_layers = get_max_layer_num(state_dict, layer_prefix)
    packets = []
    for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
        chunked_weights = {key: layer_state_dict.pop(key) for key in chunked_ops if key in layer_state_dict}
        packets.append(preprocess_layer_checkpoint(model, layer_state_dict, layer_idx))
        for key, weight in chunked_weights.items():
            packets.extend(chunked_ops[key].iter_prime_to_hf(weight, lambda value, start, stop: value[start:stop]))
    return [
        {
            "keys": sorted(packet),
            "digest": hashlib.sha256(json.dumps(describe_all(packet), sort_keys=True).encode()).hexdigest()[:16],
        }
        for packet in packets
    ]


@pytest.mark.parametrize("config_cls", MODEL_CLASSES, ids=lambda config_cls: config_cls.model_type)
def test_conversion_is_unchanged(config_cls):
    """Every prime to HF conversion path gives the same tensors, and HF to prime inverts it."""
    model = build_on_meta(config_cls, False, torch.float32)
    scratch_init(model, "cpu")
    state_dict = model.state_dict()
    prime_keys = dict.fromkeys(state_dict)

    hf_state_dict = model.convert_to_hf(clone(state_dict))
    hf_keys = dict.fromkeys(hf_state_dict)
    round_trip = model.convert_to_prime(clone(hf_state_dict))
    assert round_trip.keys() <= state_dict.keys(), sorted(round_trip.keys() - state_dict.keys())
    unequal = [key for key, value in round_trip.items() if not torch.equal(value, state_dict[key].to(value.dtype))]
    assert unequal == []

    record = {
        "conversion_chain": [re.sub(r" at 0x[0-9a-f]+", "", repr(op)) for op in model.conversion_chain(model.config)],
        "is_hf_state_dict": {"prime": model.is_hf_state_dict(prime_keys), "hf": model.is_hf_state_dict(hf_keys)},
        "is_prime_state_dict": {
            "prime": model.is_prime_state_dict(prime_keys),
            "hf": model.is_prime_state_dict(hf_keys),
        },
        "keep_in_fp32_for_weight_transfer": sorted(
            key for key in state_dict if model.keep_in_fp32_for_weight_transfer(key)
        ),
        "convert_to_hf": describe_all(hf_state_dict),
        "round_trip_dropped": sorted(state_dict.keys() - round_trip.keys()),
        "round_trip_dtype_changes": {
            key: f"{layout(state_dict[key])} -> {layout(value)}"
            for key, value in round_trip.items()
            if value.dtype != state_dict[key].dtype
        },
        "convert_state_dict_to_hf": describe_all(convert_state_dict_to_hf(model, clone(state_dict))),
        "weight_broadcast": weight_broadcast_packets(model, clone(state_dict)),
    }
    check_golden("conversion", config_cls.model_type, record)


def packed_batch(vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A row of documents packed as training packs them, with each document's next tokens as labels."""
    generator = torch.Generator().manual_seed(SEED)
    seq_lens = torch.tensor(GPU_SEQ_LENS)
    input_ids = torch.randint(0, vocab_size, (1, int(seq_lens.sum())), generator=generator)
    position_ids = torch.cat([torch.arange(seq_len) for seq_len in GPU_SEQ_LENS]).unsqueeze(0)
    labels = input_ids.roll(-1, dims=1)
    labels[0, seq_lens.cumsum(0) - 1] = IGNORE_INDEX
    return input_ids.cuda(), position_ids.cuda(), labels.cuda(), seq_lens.cuda()


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def pinned_autotune(monkeypatch):
    """Runs every autotuned Triton kernel of prime-rl with its first config, whatever the timings of this process."""
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("prime_rl.") and module is not None:
            for value in vars(module).values():
                if isinstance(value, Autotuner):
                    monkeypatch.setattr(value, "configs", value.configs[:1])


def gpu_step_record(name: str, attn: str) -> dict:
    """Logits, loss and gradients of a seeded scratch-initialized model on a packed row."""
    source = GPU_CONFIGS[name]
    config_cls = CONFIG_CLASSES[source.model_type]
    config = copy.deepcopy(source) if type(source) is config_cls else config_cls.from_dict(source.to_dict())
    with torch.device("meta"):
        model = MODEL_CLASSES[config_cls].from_config(config, attn_implementation=attn, dtype=torch.bfloat16)
    inject_prime_lm_head(model, chunk_size=None)
    scratch_init(model, "cuda")
    model.train()

    input_ids, position_ids, labels, seq_lens = packed_batch(config.vocab_size)
    output = model(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens, seq_lens_are_pre_shard=False)
    logits = output["logits"]
    loss = cross_entropy_sum(logits, labels)
    loss.backward()
    return {
        "attn": attn,
        "logits": describe_with_stats(logits),
        "loss": describe_with_stats(loss),
        "grads": {
            param_name: describe_with_stats(param.grad) if param.grad is not None else "none"
            for param_name, param in model.named_parameters(remove_duplicate=False)
        },
    }


@pytest.mark.gpu
@pytest.mark.parametrize("name", GPU_CONFIGS)
def test_gpu_forward_backward_is_unchanged(name, deterministic, pinned_autotune):
    """Bitwise logits and loss on one GPU, and bitwise gradients except where the backward kernels are not."""
    major, minor = torch.cuda.get_device_capability()
    arm = f"gpu_sm{major}{minor}"
    if not REGEN and not load_goldens(arm):
        pytest.skip(f"no goldens for SM{major}{minor}")
    attn = {9: "flash_attention_3", 10: "flash_attention_4", 11: "flash_attention_4"}.get(major, "flash_attention_2")
    expected, actual = golden_and_actual(arm, name, gpu_step_record(name, attn))
    if name in GPU_GRAD_RTOL:
        assert_grads_close(f"{arm}/{name}", expected["grads"], actual["grads"], GPU_GRAD_RTOL[name])
        expected = {key: value for key, value in expected.items() if key != "grads"}
        actual = {key: value for key, value in actual.items() if key != "grads"}
    assert_same(f"{arm}/{name}", expected, actual)
