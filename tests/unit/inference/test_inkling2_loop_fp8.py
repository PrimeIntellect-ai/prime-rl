"""FP8 routed experts of the looped Inkling 2 (Inkling 2 Mini) server, as far as a CPU and the pinned vLLM go.

``Inkling2LoopForRL`` builds on vLLM's looped Inkling 2 model, which the vLLM release prime-rl pins does not have.
The ``serving`` fixture runs the serving module on stand-ins for those vLLM modules, so these tests cover the
module's own logic: which experts the constructor builds, and that ``load_weights`` rounds the expert tensors of
every layer and reports their scales. The FP8 experts have a module of their own, which needs none of those vLLM
modules and is imported as it is: where ``load_bf16_weight`` puts the rows of a checkpoint tensor. vLLM's
``per_block_cast_to_fp8`` is not replaced: it is plain torch code, and it runs here as written, without being
compiled, on weights that lie on the FP8 grid, so that the codes it must give are known.

Left to GPU checks: the forward of ``Inkling2FP8Experts`` (vLLM's FP8 kernels and its activation op exist only in
a process with a GPU), what the rounding gives on the serving GPU for trained weights and whether those bytes are
the trainer's, the server against the trainer on one layer and on the whole model, a server that starts with the
setting and takes a weight update, and a training run.
"""

import importlib.util
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference import patches
from prime_rl.inference.vllm.models.inkling2_loop.fp8_experts import Inkling2FP8Experts
from prime_rl.inference.vllm.worker import weight_transfer

ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
LOOPED_VLLM = "vllm.models.inkling2.nvidia"
HIDDEN, INTERMEDIATE, EXPERTS, LAYERS = 256, 384, 2, 2

# E4M3 bytes that are neither a zero (0x00, 0x80) nor a NaN (0x7F, 0xFF).
ORDINARY_CODES = torch.tensor([code for code in range(256) if code not in (0x00, 0x80, 0x7F, 0xFF)], dtype=torch.uint8)
# None is a power of two: with scales rounded up to powers of two the same weights get other codes.
BLOCK_SCALES = torch.tensor([0.75, 1.25, 1.5, 1.75]) / 64


class Attention(nn.Module):
    """Stands in for ``Inkling2Attention``. It has no ``rel_proj_shifted``, as before vLLM's post-load step, so
    ``load_weights`` leaves the derived tensors alone."""


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()
        self.mlp = nn.Identity()


class LoopedModel(nn.Module):
    """Stands in for vLLM's ``Inkling2LoopForConditionalGeneration``: what ``Inkling2LoopForRL`` reads of it."""

    _renames = (("model.llm.", "model."),)
    _shards: dict = {}

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.text_config = vllm_config.model_config.hf_config.text_config
        self.visual = None
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Layer() for _ in range(self.text_config.num_hidden_layers))


def not_on_a_cpu(*args, **kwargs):
    pytest.fail("called a vLLM kernel")


LOOPED_VLLM_STAND_INS = {
    "attention": {"Inkling2Attention": Attention},
    "loop": {"Inkling2LoopForConditionalGeneration": LoopedModel},
    "moe": {
        "Inkling2Gate": lambda config: nn.Identity(),
        "Inkling2MoE": type("Inkling2MoE", (nn.Module,), {}),
        "Inkling2RoutingCapture": lambda visit: nn.Identity(),
    },
    "ops.grouped_gemm": {"grouped_gemm": not_on_a_cpu},
    "ops.moe": {"combine": not_on_a_cpu},
    "ops.moe_sm90": {"select_experts": not_on_a_cpu, "swiglu": not_on_a_cpu},
    "ops.sconv": {"shift_rel_proj": not_on_a_cpu},
}


@pytest.fixture
def serving(monkeypatch):
    """The serving module, run with stand-ins for vLLM's looped Inkling 2 modules. It is not entered into
    ``sys.modules``: it is built on the stand-ins, and no other test should meet it."""
    for name, attributes in LOOPED_VLLM_STAND_INS.items():
        stand_in = ModuleType(f"{LOOPED_VLLM}.{name}")
        vars(stand_in).update(attributes)
        monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)
    spec = importlib.util.find_spec("prime_rl.inference.vllm.models.inkling2_loop.model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # vLLM's parallel state is set up by its workers.
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    # vLLM compiles `per_block_cast_to_fp8` at its first call. Here it runs as written.
    with torch.compiler.set_stance("force_eager"):
        yield module


def text_config(**overrides) -> SimpleNamespace:
    values = dict(
        num_hidden_layers=LAYERS,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=2,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        layer_visit_ids=lambda layer_id: (layer_id,),
    )
    return SimpleNamespace(**{**values, **overrides})


def vllm_config(expert_format: str | None = None) -> SimpleNamespace:
    """The part of vLLM's config that ``Inkling2LoopForRL`` reads, with the expert format where vLLM puts an HF
    override: on the HF config."""
    hf_config = SimpleNamespace(text_config=text_config())
    if expert_format is not None:
        setattr(hf_config, patches.INKLING2_LOOP_EXPERT_FORMAT_ATTRIBUTE, expert_format)
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))


def on_the_fp8_grid(rows: int, columns: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Codes ``[E, rows, columns]``, one scale per 128 x 128 block, and the BF16 weight they stand for.

    Every block holds the largest code (0x7E, the value 448), so its largest weight is 448 times its scale, and
    every weight is exact in BF16. Rounding such a weight per block gives back these codes, also when the scale
    arithmetic is off by a last bit."""
    generator = torch.Generator().manual_seed(seed)
    codes = ORDINARY_CODES[torch.randint(len(ORDINARY_CODES), (EXPERTS, rows, columns), generator=generator)]
    codes[:, ::128, ::128] = 0x7E
    # Different scales for neighbouring blocks, for the same block of two experts, and for the gate block and the up
    # block of the same rows.
    expert, block_row, block_column = torch.meshgrid(
        torch.arange(EXPERTS), torch.arange(rows // 128), torch.arange(columns // 128), indexing="ij"
    )
    scales = BLOCK_SCALES[(expert + block_row + 2 * block_column) % len(BLOCK_SCALES)]
    weight = codes.view(torch.float8_e4m3fn).float() * scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
    assert torch.equal(weight.to(torch.bfloat16).float(), weight)
    return codes, scales, weight.to(torch.bfloat16)


def interleaved(gate_then_up: torch.Tensor) -> torch.Tensor:
    """The checkpoint's layout of ``w13_weight``: gate row ``i`` in row ``2i``, up row ``i`` in row ``2i + 1``."""
    half = gate_then_up.shape[1] // 2
    checkpoint = torch.empty_like(gate_then_up)
    checkpoint[:, 0::2], checkpoint[:, 1::2] = gate_then_up[:, :half], gate_then_up[:, half:]
    return checkpoint


# --- the setting, from the server's arguments to the model ---


def test_quantization_goes_to_the_looped_model_as_an_hf_override(monkeypatch, tmp_path):
    PretrainedConfig(architectures=[ARCHITECTURE]).save_pretrained(tmp_path)
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: False)
    config = InferenceConfig(
        vllm={
            "model": str(tmp_path),
            "quantization": "fp8_per_block",
            # No parser is detected for a local folder, so the server's config names them.
            "reasoning_parser": "inkling",
            "tool_call_parser": "inkling",
        }
    )
    namespace = config.to_namespace()
    # The server's namespace already carries HF overrides: the dtypes of the router and of the LM head.
    overrides = dict(namespace.hf_overrides)
    assert overrides
    assert namespace.quantization == "fp8_per_block"

    patches.select_inkling2_loop_expert_format(namespace)

    # vLLM's own Inkling 2 model takes no quantization method but the release's, so vLLM must not see the setting.
    assert not hasattr(namespace, "quantization")
    assert namespace.hf_overrides == {**overrides, patches.INKLING2_LOOP_EXPERT_FORMAT_ATTRIBUTE: "fp8_per_block"}


@pytest.mark.parametrize(
    ("vllm_lacks_the_looped_model", "architectures"),
    [(True, [ARCHITECTURE]), (False, ["Qwen3ForCausalLM"]), (False, None)],
    ids=["a vLLM without the looped model", "another model", "a config that names no architecture"],
)
def test_quantization_stays_with_vllm(monkeypatch, tmp_path, vllm_lacks_the_looped_model, architectures):
    PretrainedConfig(architectures=architectures).save_pretrained(tmp_path)
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: vllm_lacks_the_looped_model)
    args = Namespace(model=str(tmp_path), quantization="fp8_per_block", hf_overrides={"head_dtype": "float32"})

    patches.select_inkling2_loop_expert_format(args)

    assert vars(args) == {
        "model": str(tmp_path),
        "quantization": "fp8_per_block",
        "hf_overrides": {"head_dtype": "float32"},
    }


def test_a_server_without_quantization_reads_no_checkpoint_config(monkeypatch):
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: False)
    args = Namespace(model="no/such-checkpoint")
    patches.select_inkling2_loop_expert_format(args)
    assert vars(args) == {"model": "no/such-checkpoint"}


def test_the_expert_format_selects_the_experts_of_every_layer(serving):
    bf16 = serving.Inkling2LoopForRL(vllm_config=vllm_config())
    fp8 = serving.Inkling2LoopForRL(vllm_config=vllm_config("fp8_per_block"))
    assert [type(layer.mlp.experts) for layer in bf16.model.layers] == [serving.Inkling2BF16Experts] * LAYERS
    assert [type(layer.mlp.experts) for layer in fp8.model.layers] == [Inkling2FP8Experts] * LAYERS


def test_neither_experts_class_takes_lora_adapters(serving):
    # vLLM's `_expert_lora_capable` reads this attribute and takes an experts module without it as a LoRA target.
    assert serving.Inkling2BF16Experts.supports_lora is False
    assert Inkling2FP8Experts.supports_lora is False


def test_an_unknown_expert_format_is_refused_before_the_model_is_built(serving, monkeypatch):
    monkeypatch.setattr(LoopedModel, "__init__", lambda *args, **kwargs: pytest.fail("built the model"))
    # The config accepts `mxfp8` for other models; this one rounds to `fp8_per_block` only.
    with pytest.raises(ValueError, match="mxfp8"):
        serving.Inkling2LoopForRL(vllm_config=vllm_config("mxfp8"))


# --- the experts ---


def test_fp8_experts_round_the_gate_rows_then_the_up_rows(serving):
    experts = Inkling2FP8Experts(text_config())
    served = experts.w13_weight.data_ptr(), experts.w13_weight_scale.data_ptr()
    codes, scales, gate_then_up = on_the_fp8_grid(2 * INTERMEDIATE, HIDDEN, seed=0)

    experts.load_bf16_weight("w13_weight", interleaved(gate_then_up))

    # A block of 128 rows of the served tensor lies in the gate projection or in the up projection, as the
    # trainer's blocks do. Rounded in the checkpoint's layout, a block would mix rows of both and get other codes.
    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert torch.equal(experts.w13_weight.view(torch.uint8), codes)
    torch.testing.assert_close(experts.w13_weight_scale.data, scales, rtol=1e-6, atol=0)
    # Written into the tensors the model already serves.
    assert (experts.w13_weight.data_ptr(), experts.w13_weight_scale.data_ptr()) == served


def test_fp8_experts_round_the_down_projection_in_the_checkpoints_layout(serving):
    experts = Inkling2FP8Experts(text_config())
    codes, scales, weight = on_the_fp8_grid(HIDDEN, INTERMEDIATE, seed=1)

    experts.load_bf16_weight("w2_weight", weight)

    assert torch.equal(experts.w2_weight.view(torch.uint8), codes)
    torch.testing.assert_close(experts.w2_weight_scale.data, scales, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "weight",
    [
        torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.float32),
        torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.float8_e4m3fn),
        torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
    ],
    ids=["fp32", "fp8", "half the rows"],
)
def test_fp8_experts_refuse_a_weight_that_is_not_the_checkpoints_bf16_tensor(serving, weight):
    experts = Inkling2FP8Experts(text_config())
    with pytest.raises(ValueError, match="w13_weight: expected BF16"):
        experts.load_bf16_weight("w13_weight", weight)


def test_fp8_experts_need_sizes_in_whole_blocks(serving):
    with pytest.raises(ValueError, match="multiples of"):
        Inkling2FP8Experts(text_config(intermediate_size=192))


# --- the model's loader ---


def test_load_weights_rounds_both_expert_tensors_of_every_layer_and_reports_their_scales(serving):
    model = serving.Inkling2LoopForRL(vllm_config=vllm_config("fp8_per_block"))
    checkpoint, codes = {}, {}
    for layer in range(LAYERS):
        experts = f"layers.{layer}.mlp.experts"
        codes[f"model.{experts}.w13_weight"], _, gate_then_up = on_the_fp8_grid(2 * INTERMEDIATE, HIDDEN, seed=layer)
        codes[f"model.{experts}.w2_weight"], _, down = on_the_fp8_grid(HIDDEN, INTERMEDIATE, seed=LAYERS + layer)
        checkpoint[f"model.llm.{experts}.w13_weight"] = interleaved(gate_then_up)
        checkpoint[f"model.llm.{experts}.w2_weight"] = down

    loaded = model.load_weights(iter(checkpoint.items()))

    # The scales come from no checkpoint tensor, and the loader still has to report every parameter as loaded.
    assert loaded == {
        f"model.layers.{layer}.mlp.experts.{name}"
        for layer in range(LAYERS)
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }
    for name, expected in codes.items():
        assert torch.equal(model.get_parameter(name).view(torch.uint8), expected), name


def test_an_fp8_server_takes_no_packed_experts(serving):
    model = serving.Inkling2LoopForRL(vllm_config=vllm_config("fp8_per_block"))
    packed = torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="RL serving needs them in BF16"):
        model.load_weights(iter([("model.llm.layers.0.mlp.experts.w13_weight", packed)]))


# --- the weight fingerprint ---


def test_count_nonzero_counts_fp8_values_by_their_bytes():
    # 0x00 is +0 and 0x80 is -0; 0x01 and 0x81 are the smallest values of either sign.
    codes = torch.tensor([0x00, 0x80, 0x01, 0x81, 0x38, 0xC2], dtype=torch.uint8)
    for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
        count = weight_transfer.count_nonzero(codes.view(fp8))
        assert type(count) is int
        assert count == 4 == codes.view(fp8).float().count_nonzero().item()
    # Other dtypes are counted as before: a byte with only its top bit set is a value of its own there.
    assert weight_transfer.count_nonzero(codes) == 5
    assert weight_transfer.count_nonzero(torch.tensor([0.0, -0.0, 1.5], dtype=torch.bfloat16)) == 1


def test_the_weight_fingerprint_covers_fp8_tensors():
    # +0, -0, the smallest value and 448, the largest.
    codes = torch.tensor([0x00, 0x80, 0x01, 0x7E], dtype=torch.uint8).view(torch.float8_e4m3fn)
    model = nn.Module()
    model.register_parameter("codes", nn.Parameter(codes, requires_grad=False))
    worker = weight_transfer.WeightRPCMixin()
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    fingerprint = worker.get_weight_fingerprint()

    assert fingerprint["codes"]["dtype"] == "torch.float8_e4m3fn"
    assert fingerprint["codes"]["nonzero"] == 2
