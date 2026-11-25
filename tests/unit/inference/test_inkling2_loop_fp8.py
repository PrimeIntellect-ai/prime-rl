"""FP8 routed experts of the looped Inkling 2 (Inkling 2 Mini) server, as far as a CPU and the pinned vLLM go.

Three parts of the serving code run here as they are, with no stand-in for a vLLM module: the function that hands
``vllm.quantization`` to the model, ``Inkling2FP8Experts`` (its module needs none of ``vllm.models.inkling2``, which
the vLLM release prime-rl pins does not have), and the count of non-zero FP8 values in the weight fingerprint.
vLLM's ``per_block_cast_to_fp8`` is not replaced either: it is plain torch code, and it runs here as written, without
being compiled, on weights that lie on the FP8 grid, so that the codes it must give are known.

Not covered here: the model class. ``Inkling2LoopForRL`` builds on vLLM's looped Inkling 2 model and imports only on
a vLLM that has it. So no case shows which experts its constructor builds for a format, that it refuses an unknown
format before the model is built, that ``load_weights`` sends both expert tensors of every layer to the rounding and
reports their scales as loaded, or that the BF16 experts say ``supports_lora = False``. These need a vLLM that has
the model (on a machine without a GPU it imports with ``CUDA_VISIBLE_DEVICES=""``).

Left to GPU checks: the forward of ``Inkling2FP8Experts`` (vLLM's FP8 kernels and its activation op exist only in
a process with a GPU), what the rounding gives on the serving GPU for trained weights and whether those bytes are
the trainer's, the server against the trainer on one layer and on the whole model, a server that starts with the
setting and takes a weight update, and a training run.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference import patches
from prime_rl.inference.vllm.models.inkling2_loop.fp8_experts import Inkling2FP8Experts
from prime_rl.inference.vllm.worker import weight_transfer

ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
HIDDEN, INTERMEDIATE, EXPERTS = 256, 384, 2

# E4M3 bytes that are neither a zero (0x00, 0x80) nor a NaN (0x7F, 0xFF).
ORDINARY_CODES = torch.tensor([code for code in range(256) if code not in (0x00, 0x80, 0x7F, 0xFF)], dtype=torch.uint8)
# None is a power of two: with scales rounded up to powers of two the same weights get other codes.
BLOCK_SCALES = torch.tensor([0.75, 1.25, 1.5, 1.75]) / 64


@pytest.fixture
def uncompiled():
    """vLLM compiles ``per_block_cast_to_fp8`` at its first call. Here it runs as written."""
    with torch.compiler.set_stance("force_eager"):
        yield


def text_config(**overrides) -> SimpleNamespace:
    """What ``Inkling2FP8Experts`` reads of the model's text config."""
    values = dict(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    return SimpleNamespace(**{**values, **overrides})


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


@pytest.mark.parametrize("quantization", ["fp8_per_block", "mxfp8"])
def test_quantization_goes_to_the_looped_model_as_an_hf_override(monkeypatch, tmp_path, quantization):
    PretrainedConfig(architectures=[ARCHITECTURE]).save_pretrained(tmp_path)
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: False)
    config = InferenceConfig(
        vllm={
            "model": str(tmp_path),
            "quantization": quantization,
            # No parser is detected for a local folder, so the server's config names them.
            "reasoning_parser": "inkling",
            "tool_call_parser": "inkling",
        }
    )
    namespace = config.to_namespace()
    # The server's namespace already carries HF overrides: the dtypes of the router and of the LM head.
    overrides = dict(namespace.hf_overrides)
    assert overrides
    assert namespace.quantization == quantization

    patches.select_inkling2_loop_expert_format(namespace)

    # vLLM's own Inkling 2 model takes no quantization method but the release's, so vLLM must not see the setting.
    assert not hasattr(namespace, "quantization")
    # The value goes over as it was set: the model is what refuses a format it does not serve, such as `mxfp8`.
    assert namespace.hf_overrides == {**overrides, patches.INKLING2_LOOP_EXPERT_FORMAT_ATTRIBUTE: quantization}


def test_the_architecture_is_read_from_the_config_vllm_builds_the_model_from(monkeypatch, tmp_path):
    # With `hf_config_path` set, vLLM reads the model's config from there, and the model's own folder may have none.
    weights, config = tmp_path / "weights", tmp_path / "config"
    weights.mkdir()
    PretrainedConfig(architectures=[ARCHITECTURE]).save_pretrained(config)
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: False)
    # A namespace without HF overrides, as the server's is with the FP32 router logits and LM head switched off.
    args = Namespace(model=str(weights), hf_config_path=str(config), quantization="fp8_per_block")

    patches.select_inkling2_loop_expert_format(args)

    assert vars(args) == {
        "model": str(weights),
        "hf_config_path": str(config),
        "hf_overrides": {patches.INKLING2_LOOP_EXPERT_FORMAT_ATTRIBUTE: "fp8_per_block"},
    }


@pytest.mark.parametrize(
    ("vllm_lacks_the_looped_model", "config"),
    [
        (True, {"architectures": [ARCHITECTURE]}),
        # The Inkling 2 without a loop: its servers take this setting through vLLM.
        (False, {"architectures": ["Inkling2ForConditionalGeneration"]}),
        (False, {}),
        (False, None),
    ],
    ids=["vllm_without_looped_model", "another_model", "no_architectures", "unresolvable_model"],
)
def test_quantization_stays_with_vllm(monkeypatch, tmp_path, vllm_lacks_the_looped_model, config):
    model = str(tmp_path / "model")
    # Without a config the folder is not made: transformers cannot resolve such a model, and vLLM is left to say so.
    if config is not None:
        PretrainedConfig(**config).save_pretrained(model)
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: vllm_lacks_the_looped_model)
    args = Namespace(model=model, quantization="fp8_per_block", hf_overrides={"head_dtype": "float32"})

    patches.select_inkling2_loop_expert_format(args)

    assert vars(args) == {"model": model, "quantization": "fp8_per_block", "hf_overrides": {"head_dtype": "float32"}}


def test_a_looped_model_without_quantization_is_left_alone(monkeypatch, tmp_path):
    # The default server of this model: its namespace has no `quantization`, and its experts are BF16.
    PretrainedConfig(architectures=[ARCHITECTURE]).save_pretrained(tmp_path)
    monkeypatch.setattr(patches, "_missing_in_vllm", lambda name: False)
    args = Namespace(model=str(tmp_path), hf_overrides={"head_dtype": "float32"})

    patches.select_inkling2_loop_expert_format(args)

    assert vars(args) == {"model": str(tmp_path), "hf_overrides": {"head_dtype": "float32"}}


# --- the experts ---


def test_fp8_experts_round_the_gate_rows_then_the_up_rows(uncompiled):
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


def test_fp8_experts_round_the_down_projection_in_the_checkpoints_layout(uncompiled):
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
    ids=["fp32", "fp8", "half_the_rows"],
)
def test_fp8_experts_refuse_a_weight_that_is_not_the_checkpoints_bf16_tensor(weight):
    experts = Inkling2FP8Experts(text_config())
    with pytest.raises(ValueError, match="w13_weight: expected BF16"):
        experts.load_bf16_weight("w13_weight", weight)


@pytest.mark.parametrize("size", ["hidden_size", "intermediate_size"])
def test_fp8_experts_need_sizes_in_whole_blocks(size):
    with pytest.raises(ValueError, match="multiples of"):
        Inkling2FP8Experts(text_config(**{size: 192}))


def test_fp8_experts_take_no_lora_adapters():
    # vLLM's `_expert_lora_capable` reads this attribute and takes an experts module without it as a LoRA target.
    assert Inkling2FP8Experts.supports_lora is False


# --- the weight fingerprint ---


def test_count_nonzero_counts_fp8_values_by_their_bytes():
    # 0x00 is +0 and 0x80 is -0; 0x01 and 0x81 are the smallest values of either sign.
    codes = torch.tensor([0x00, 0x80, 0x01, 0x81, 0x38, 0xC2], dtype=torch.uint8)
    for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
        count = weight_transfer.count_nonzero(codes.view(fp8))
        assert type(count) is int
        assert count == 4 == codes.view(fp8).float().count_nonzero().item()
    # Other dtypes go to torch's own count: a byte with only its top bit set is a value of its own there.
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
