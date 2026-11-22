import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

TOOL = Path(__file__).resolve().parents[3] / "tools" / "convert_inkling2_to_bf16.py"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
EXPERTS = "model.llm.layers.0.mlp.experts."
W13, W2 = EXPERTS + "w13_weight", EXPERTS + "w2_weight"
QUERY, EMBED, VISUAL = "model.llm.attn_res.mlp_queries.0", "model.llm.embed.weight", "model.visual.proj_in.weight"
NON_NAN_CODES = [code for code in range(256) if code & 0x7F != 0x7F]
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture
def converter(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))
    return importlib.import_module("convert_inkling2_to_bf16")


def e4m3(code: int) -> float:
    """The value of an E4M3 code, from its bits."""
    exponent, mantissa = (code >> 3) & 0xF, code & 0x7
    magnitude = mantissa / 8 * 2.0**-6 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return -magnitude if code & 0x80 else magnitude


E4M3_VALUES = torch.tensor([e4m3(code) for code in range(256)], dtype=torch.float64)


def decode_mxfp8(codes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """`e4m3(code) * 2 ** (scale - 127)` in float64, where every product is exact."""
    scales = torch.tensor([2.0 ** (scale - 127) for scale in range(256)], dtype=torch.float64)[scale_bytes.long()]
    return E4M3_VALUES[codes.long()] * scales.repeat_interleave(32, dim=-1)


def assert_same_numbers(dequantized: torch.Tensor, expected: torch.Tensor) -> None:
    assert dequantized.dtype == torch.bfloat16
    assert torch.equal(dequantized.double(), expected)
    assert torch.equal(dequantized.double().signbit(), expected.signbit())


def write_release(path: Path, nvfp4: bool = False) -> dict[str, torch.Tensor]:
    """A checkpoint in the release's layout: two shards, each with an expert weight whose scale is in the other."""
    generator = torch.Generator().manual_seed(0)

    def codes(*shape: int) -> torch.Tensor:
        return torch.tensor(NON_NAN_CODES, dtype=torch.uint8)[torch.randint(0, 254, shape, generator=generator)]

    def scale_bytes(*shape: int) -> torch.Tensor:
        return torch.randint(112, 119, shape, generator=generator, dtype=torch.uint8)

    tensors = {
        EMBED: torch.randn(8, 4, generator=generator).bfloat16(),
        QUERY: torch.randn(4, generator=generator),
        VISUAL: torch.randn(4, 4, generator=generator).bfloat16(),
    }
    if nvfp4:
        for name, rows in ((W13, 4), (W2, 2)):
            tensors[name] = torch.randint(0, 256, (2, rows, 32), generator=generator, dtype=torch.uint8)
            # Positive block scales between 0.5 and 4.
            tensors[name + "_scale"] = (48 + codes(2, rows, 4) % 24).view(torch.float8_e4m3fn)
            tensors[name + "_amax"] = torch.tensor([0.75, 1.5])
            tensors[name + "_outer_scale_denominator"] = torch.tensor(6.0)
    else:
        tensors |= {
            W13: codes(2, 4, 64).view(torch.float8_e4m3fn),
            W13 + "_scale": scale_bytes(2, 4, 2),
            W2: codes(2, 2, 32).view(torch.float8_e4m3fn),
            W2 + "_scale": scale_bytes(2, 2, 1),
        }
    in_second_shard = {W13 + "_scale", W2, VISUAL}
    weight_map = {name: SHARDS[name in in_second_shard] for name in tensors}

    path.mkdir(parents=True)
    for shard in SHARDS:
        save_file({name: tensors[name] for name in tensors if weight_map[name] == shard}, path / shard)
    save_file({"fuse_weight": torch.ones(2, dtype=torch.bfloat16)}, path / "mtp.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (path / "config.json").write_text(
        json.dumps({"model_type": "inkling2", "text_config": {"hidden_size": 4}, "quantization_config": {}})
    )
    groups = [
        {"compute_dtype": "float32", "tensors": [QUERY]},
        {"compute_dtype": "bfloat16", "tensors": [EMBED, VISUAL]},
        {"compute_dtype": "nvfp4" if nvfp4 else "mxfp8", "tensors": [W13, W2], "weight_block_shape": [32, 32]},
    ]
    (path / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": groups}, "mtp": {}})
    )
    for name in ("tiktoken/tokenizer.model", ".cache/huggingface/download/config.json.metadata"):
        (path / name).parent.mkdir(parents=True)
        (path / name).write_text(name)
    return tensors


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """An MXFP8 checkpoint in a Hub cache snapshot directory."""
    path = tmp_path / "models--thinkingmachines--tiny" / "snapshots" / "0123abc"
    return path, write_release(path)


def replace_tensor(path: Path, shard: str, name: str, tensor: torch.Tensor) -> None:
    save_file(load_file(path / shard) | {name: tensor}, path / shard)


def tree(path: Path) -> dict[str, bytes]:
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_mxfp8_is_exact_for_every_code_and_scale(converter):
    low, high = converter.MXFP8_SCALE_RANGE
    codes = torch.tensor(NON_NAN_CODES + [0, 0], dtype=torch.uint8).expand(1, 256, -1)
    scale_bytes = torch.arange(256, dtype=torch.uint8).view(1, -1, 1).expand(-1, -1, 8)
    dequantized = converter.dequantize_mxfp8(codes.contiguous().view(torch.float8_e4m3fn), scale_bytes)
    expected = decode_mxfp8(codes, scale_bytes)

    assert_same_numbers(dequantized[:, low : high + 1], expected[:, low : high + 1])
    # The range is all of it: one step below, the smallest code is lost; one step above, the largest overflows.
    assert not torch.equal(dequantized[:, low - 1].double(), expected[:, low - 1])
    assert dequantized[:, high + 1].isinf().any()


def test_converts_the_whole_tree(converter, release, tmp_path):
    source, tensors = release
    output = tmp_path / "bf16"
    converter.convert(source, output)

    converted = load_file(output / SHARDS[0]) | load_file(output / SHARDS[1])
    assert converted.keys() == {EMBED, QUERY, VISUAL, W13, W2}
    for name in (W13, W2):
        assert_same_numbers(converted[name], decode_mxfp8(tensors[name].view(torch.uint8), tensors[name + "_scale"]))
    for name in (EMBED, QUERY, VISUAL):
        assert converted[name].dtype == tensors[name].dtype
        assert torch.equal(converted[name], tensors[name])

    files = tree(output)
    index = json.loads(files.pop("model.safetensors.index.json"))
    assert index["weight_map"] == {EMBED: SHARDS[0], QUERY: SHARDS[0], W13: SHARDS[0], W2: SHARDS[1], VISUAL: SHARDS[1]}
    assert index["metadata"]["total_size"] == sum(tensor.nbytes for tensor in converted.values())
    assert json.loads(files.pop("config.json")) == {
        "model_type": "inkling2",
        "text_config": {"hidden_size": 4},
        "transformers_version": converter.TRANSFORMERS_VERSION,
    }
    groups = json.loads(files.pop("hf_quant_config.json"))["model"]["parameter_groups"]
    assert [group["compute_dtype"] for group in groups] == ["float32", "bfloat16", "bfloat16"]
    assert groups[-1]["tensors"] == [W13, W2]
    assert json.loads(files.pop(converter.RECORD)) == {
        "source": "thinkingmachines/tiny",
        "source_revision": "0123abc",
        "tool": "tools/convert_inkling2_to_bf16.py",
        "tool_commit": converter.tool_commit(),
        "expert_weights": {"mxfp8": 2},
        "exact": True,
    }
    # The rest is copied, subdirectories included, without the Hub client's download records.
    assert files.keys() == {*SHARDS, "mtp.safetensors", "tiktoken/tokenizer.model"}
    assert files["mtp.safetensors"] == (source / "mtp.safetensors").read_bytes()
    assert files["tiktoken/tokenizer.model"] == b"tiktoken/tokenizer.model"

    umask = os.umask(0)
    os.umask(umask)
    assert {file.stat().st_mode & 0o777 for file in output.rglob("*") if file.is_file()} == {0o666 & ~umask}


def test_a_rerun_continues_after_the_finished_files(converter, release, tmp_path, monkeypatch):
    source, _ = release
    output = tmp_path / "bf16"
    save_file = converter.save_file
    saved = []

    def interrupted_save(tensors, path, **kwargs):
        if saved:
            path.write_bytes(b"half a file")
            raise RuntimeError("interrupted")
        saved.append(path.name)
        save_file(tensors, path, **kwargs)

    monkeypatch.setattr(converter, "save_file", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        converter.convert(source, output)
    assert sorted(file.name for file in output.iterdir()) == sorted(
        [converter.RECORD, SHARDS[0], SHARDS[1] + ".partial"]
    )
    first_shard = (output / SHARDS[0]).stat()

    def recording_save(tensors, path, **kwargs):
        saved.append(path.name)
        save_file(tensors, path, **kwargs)

    monkeypatch.setattr(converter, "save_file", recording_save)
    converter.convert(source, output)
    assert saved == [SHARDS[0] + ".partial", SHARDS[1] + ".partial"]
    assert (output / SHARDS[0]).stat().st_mtime_ns == first_shard.st_mtime_ns

    converter.convert(source, tmp_path / "in-one-run")
    assert tree(output) == tree(tmp_path / "in-one-run")


def nan_code(source: Path, tensors: dict[str, torch.Tensor]) -> None:
    codes = tensors[W13].view(torch.uint8).clone()
    codes[1, 2, 40] = 0xFF
    replace_tensor(source, SHARDS[0], W13, codes.view(torch.float8_e4m3fn))


def scale_byte(value: int):
    def damage(source: Path, tensors: dict[str, torch.Tensor]) -> None:
        scales = tensors[W13 + "_scale"].clone()
        scales[0, 3, 1] = value
        replace_tensor(source, SHARDS[1], W13 + "_scale", scales)

    return damage


@pytest.mark.parametrize(
    ("damage", "message"),
    [(nan_code, "NaN codes"), (scale_byte(2), r"from 2 to \d+"), (scale_byte(247), r"from \d+ to 247")],
    ids=["nan_code", "scale_below_range", "scale_above_range"],
)
def test_refuses_weights_it_cannot_convert_exactly(converter, release, tmp_path, damage, message):
    source, tensors = release
    damage(source, tensors)
    with pytest.raises(ValueError, match=message):
        converter.convert(source, tmp_path / "bf16")
    assert [file.name for file in (tmp_path / "bf16").iterdir()] == [converter.RECORD]


def without_scale(source: Path) -> None:
    index = json.loads((source / "model.safetensors.index.json").read_text())
    del index["weight_map"][W2 + "_scale"]
    (source / "model.safetensors.index.json").write_text(json.dumps(index))


@pytest.mark.parametrize(
    ("damage", "error"),
    [
        (lambda source: (source / "hf_quant_config.json").unlink(), FileNotFoundError),
        (lambda source: (source / SHARDS[1]).unlink(), FileNotFoundError),
        (without_scale, ValueError),
    ],
    ids=["no_precision_metadata", "no_second_shard", "no_scale_tensor"],
)
def test_checks_the_input_before_it_writes(converter, release, tmp_path, damage, error):
    source, _ = release
    damage(source)
    with pytest.raises(error):
        converter.convert(source, tmp_path / "bf16")
    assert not (tmp_path / "bf16").exists()


def test_refuses_to_continue_another_conversion(converter, release, tmp_path):
    source, _ = release
    converter.convert(source, tmp_path / "bf16")
    with pytest.raises(ValueError, match="another source"):
        converter.convert(source, tmp_path / "bf16", source_revision="fedcba9")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / SHARDS[0]).write_bytes(b"")
    with pytest.raises(ValueError, match="not empty"):
        converter.convert(source, tmp_path / "other")


def test_command_fails_on_a_nan_code(release, tmp_path):
    source, tensors = release
    nan_code(source, tensors)
    result = subprocess.run([sys.executable, str(TOOL), str(source), str(tmp_path / "bf16")], capture_output=True)
    assert result.returncode != 0
    assert b"NaN codes" in result.stderr


def test_nvfp4_converts_with_a_warning(converter, tmp_path, capsys):
    source = tmp_path / "Inkling-2-Small"
    tensors = write_release(source, nvfp4=True)
    converter.convert(source, tmp_path / "bf16", source_revision="0123abc")

    assert "Convert thinkingmachines/Inkling-2-Small-MXFP8 instead" in capsys.readouterr().err
    record = json.loads((tmp_path / "bf16" / converter.RECORD).read_text())
    assert (record["source"], record["source_revision"], record["exact"]) == ("Inkling-2-Small", "0123abc", False)
    converted = load_file(tmp_path / "bf16" / SHARDS[0]) | load_file(tmp_path / "bf16" / SHARDS[1])
    assert converted.keys() == {EMBED, QUERY, VISUAL, W13, W2}
    magnitudes = torch.tensor(E2M1, dtype=torch.float64)
    for name in (W13, W2):
        # Two codes per byte, low nibble first; bit 3 is the sign.
        packed = tensors[name]
        nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
        values = magnitudes[nibbles & 7] * (1 - 2 * (nibbles >> 3))
        block_scales = E4M3_VALUES[tensors[name + "_scale"].view(torch.uint8).long()].repeat_interleave(16, dim=-1)
        expected = values * block_scales * (tensors[name + "_amax"].double() / 6.0).view(-1, 1, 1)
        torch.testing.assert_close(converted[name].double(), expected, rtol=2**-7, atol=0)
