import hashlib
import importlib
import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

TOOL = Path(__file__).resolve().parents[3] / "tools" / "convert_inkling2_to_bf16.py"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = "model.safetensors.index.json"
EXPERTS = "model.llm.layers.0.mlp.experts."
W13, W2 = EXPERTS + "w13_weight", EXPERTS + "w2_weight"
QUERY, EMBED, VISUAL = "model.llm.attn_res.mlp_queries.0", "model.llm.embed.weight", "model.visual.proj_in.weight"
# The tool dequantizes 16 experts at a time: three turns of that loop.
NUM_EXPERTS = 33
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


def write_release(path: Path, nvfp4: bool = False, w13_shape: tuple = (NUM_EXPERTS, 4, 64)) -> dict[str, torch.Tensor]:
    """A checkpoint in the release's layout: two shards, each with an expert weight whose scale is in the other.
    The second one holds the larger weight."""
    generator = torch.Generator().manual_seed(0)

    def codes(*shape: int) -> torch.Tensor:
        drawn = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        return drawn.masked_fill((drawn & 0x7F) == 0x7F, 0)

    def scale_bytes(*shape: int) -> torch.Tensor:
        return torch.randint(112, 119, shape, generator=generator, dtype=torch.uint8)

    tensors = {
        EMBED: torch.randn(8, 4, generator=generator).bfloat16(),
        QUERY: torch.randn(4, generator=generator),
        VISUAL: torch.randn(4, 4, generator=generator).bfloat16(),
    }
    for name, (experts, rows, inputs) in ((W13, w13_shape), (W2, (NUM_EXPERTS, 2, 32))):
        if nvfp4:
            tensors[name] = torch.randint(0, 256, (experts, rows, inputs // 2), generator=generator, dtype=torch.uint8)
            # Positive block scales between 0.5 and 4.
            tensors[name + "_scale"] = (48 + codes(experts, rows, inputs // 16) % 24).view(torch.float8_e4m3fn)
            tensors[name + "_amax"] = 0.5 + torch.rand(experts, generator=generator)
            tensors[name + "_outer_scale_denominator"] = torch.tensor(6.0)
        else:
            tensors[name] = codes(experts, rows, inputs).view(torch.float8_e4m3fn)
            tensors[name + "_scale"] = scale_bytes(experts, rows, inputs // 32)
    in_second_shard = {W13, W2 + "_scale", VISUAL}
    weight_map = {name: SHARDS[name in in_second_shard] for name in tensors}

    path.mkdir(parents=True)
    for shard in SHARDS:
        save_file({name: tensors[name] for name in tensors if weight_map[name] == shard}, path / shard)
    save_file({"fuse_weight": torch.ones(2, dtype=torch.bfloat16)}, path / "mtp.safetensors")
    (path / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
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
    # The last two are a download's records and a git clone's own copy of the files.
    for name in ("tiktoken/tokenizer.model", ".cache/huggingface/download/config.json.metadata", ".git/HEAD"):
        (path / name).parent.mkdir(parents=True)
        (path / name).write_text(name)
    return tensors


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """An MXFP8 checkpoint in a Hub cache snapshot directory."""
    path = tmp_path / "models--thinkingmachines--tiny" / "snapshots" / "0123abc"
    return path, write_release(path)


def replace_tensor(path: Path, name: str, tensor: torch.Tensor) -> None:
    shard = json.loads((path / INDEX).read_text())["weight_map"][name]
    save_file(load_file(path / shard) | {name: tensor}, path / shard)


def tree(path: Path) -> dict[str, bytes]:
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_mxfp8_is_exact_for_every_code_and_scale(converter):
    codes = torch.tensor([code for code in range(256) if code & 0x7F != 0x7F] + [0, 0x80], dtype=torch.uint8)
    codes = codes.expand(1, 256, -1)
    scale_bytes = torch.arange(256, dtype=torch.uint8).view(1, -1, 1).expand(-1, -1, 8)
    exact = decode_mxfp8(codes, scale_bytes)

    # The tool's range is all of it: at these scale bytes, and at no other, every code is a bf16 number.
    low, high = converter.MXFP8_SCALE_RANGE
    every_code_is_bf16 = (exact.to(torch.bfloat16).double() == exact).all(dim=-1)[0]
    assert every_code_is_bf16.nonzero().flatten().tolist() == list(range(low, high + 1))

    dequantized = converter.dequantize_mxfp8(codes.contiguous().view(torch.float8_e4m3fn), scale_bytes)
    assert_same_numbers(dequantized[:, low : high + 1], exact[:, low : high + 1])
    # A zero is a zero at any scale, with its sign.
    zeros = (codes[0, 0] & 0x7F) == 0
    assert_same_numbers(dequantized[..., zeros], exact[..., zeros])


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
    index = json.loads(files.pop(INDEX))
    assert index["weight_map"] == {EMBED: SHARDS[0], QUERY: SHARDS[0], W2: SHARDS[0], W13: SHARDS[1], VISUAL: SHARDS[1]}
    assert index["metadata"]["total_size"] == sum(tensor.nbytes for tensor in converted.values())
    assert json.loads(files.pop("config.json")) == {
        "model_type": "inkling2",
        "text_config": {"hidden_size": 4},
        "transformers_version": converter.TRANSFORMERS_VERSION,
    }
    groups = json.loads(files.pop("hf_quant_config.json"))["model"]["parameter_groups"]
    assert [group["compute_dtype"] for group in groups] == ["float32", "bfloat16", "bfloat16"]
    assert groups[-1]["tensors"] == [W13, W2]
    record = json.loads(files.pop(converter.RECORD))
    assert len(record.pop("input_fingerprint")) == 64
    assert record == {
        "source": "thinkingmachines/tiny",
        "source_revision": "0123abc",
        "tool": "tools/convert_inkling2_to_bf16.py",
        "tool_sha256": hashlib.sha256(TOOL.read_bytes()).hexdigest(),
        "tool_commit": converter.tool_commit(),
        "expert_weights": {"mxfp8": 2},
        "exact": True,
    }
    # The rest is copied, subdirectories included, without `.cache` and `.git`. No lock or temporary file stays.
    assert files.keys() == {*SHARDS, "mtp.safetensors", "tiktoken/tokenizer.model"}
    assert files["mtp.safetensors"] == (source / "mtp.safetensors").read_bytes()
    assert files["tiktoken/tokenizer.model"] == b"tiktoken/tokenizer.model"

    umask = os.umask(0)
    os.umask(umask)
    assert {file.stat().st_mode & 0o777 for file in output.rglob("*") if file.is_file()} == {0o666 & ~umask}


def test_a_rerun_continues_after_the_finished_files(converter, release, tmp_path, monkeypatch, capsys):
    source, _ = release
    output = tmp_path / "bf16"
    save_file = converter.save_file
    saved = []

    def stopped_save(tensors, path, **kwargs):
        if saved:
            path.write_bytes(b"half a file")
            raise RuntimeError("stopped")
        saved.append(path.name)
        save_file(tensors, path, **kwargs)

    monkeypatch.setattr(converter, "save_file", stopped_save)
    with pytest.raises(RuntimeError, match="stopped"):
        converter.convert(source, output, source_revision="0123abc")
    assert sorted(file.name for file in output.iterdir()) == sorted(
        [converter.RECORD, SHARDS[0], SHARDS[1] + ".partial"]
    )
    first_shard = (output / SHARDS[0]).stat()

    def recording_save(tensors, path, **kwargs):
        saved.append(path.name)
        save_file(tensors, path, **kwargs)

    # The rerun does not have to name the revision again.
    monkeypatch.setattr(converter, "save_file", recording_save)
    converter.convert(source, output)
    assert saved == [SHARDS[0] + ".partial", SHARDS[1] + ".partial"]
    assert (output / SHARDS[0]).stat().st_mtime_ns == first_shard.st_mtime_ns
    assert f"Kept {SHARDS[0]}" in capsys.readouterr().out

    converter.convert(source, tmp_path / "in-one-run")
    assert tree(output) == tree(tmp_path / "in-one-run")


def test_a_killed_conversion_continues(converter, tmp_path):
    # A second shard large enough to be killed while safetensors writes it, to its own hidden temporary file.
    source, output = tmp_path / "release", tmp_path / "bf16"
    write_release(source, w13_shape=(NUM_EXPERTS, 2048, 2048))
    command = [sys.executable, str(TOOL), str(source), str(output), "--source-revision", "0123abc"]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    while process.poll() is None:
        if (output / SHARDS[0]).exists() and any(output.glob(".tmp*")):
            process.kill()
    assert process.wait() == -signal.SIGKILL
    # Progress was flushed, the first file is complete, and the kill left a temporary file and the lock.
    assert f"Wrote {SHARDS[0]}" in process.stdout.read()
    assert any(output.glob(".tmp*")) and not (output / SHARDS[1]).exists()

    blocked = subprocess.run(command, capture_output=True, text=True)
    assert blocked.returncode != 0 and "Delete the file once that process is gone" in blocked.stderr
    (output / converter.LOCK).unlink()
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode == 0 and f"Kept {SHARDS[0]}" in rerun.stdout and f"Wrote {SHARDS[1]}" in rerun.stdout

    converter.convert(source, tmp_path / "in-one-run", source_revision="0123abc")
    assert tree(output) == tree(tmp_path / "in-one-run")


def nan_code(source: Path, tensors: dict[str, torch.Tensor]) -> None:
    codes = tensors[W13].view(torch.uint8).clone()
    codes[20, 2, 40] = 0xFF
    replace_tensor(source, W13, codes.view(torch.float8_e4m3fn))


def test_refuses_a_nan_code(converter, release, tmp_path):
    source, tensors = release
    nan_code(source, tensors)
    with pytest.raises(ValueError, match="NaN codes"):
        converter.convert(source, tmp_path / "bf16")
    # The shard before it is written, the one with the code is not.
    assert sorted(file.name for file in (tmp_path / "bf16").iterdir()) == sorted([converter.RECORD, SHARDS[0]])


def scale_byte(value: int):
    def damage(source: Path, tensors: dict[str, torch.Tensor]) -> None:
        scales = tensors[W13 + "_scale"].clone()
        scales[25, 3, 1] = value
        replace_tensor(source, W13 + "_scale", scales)

    return damage


def without_scale_in_index(source: Path, tensors: dict[str, torch.Tensor]) -> None:
    index = json.loads((source / INDEX).read_text())
    del index["weight_map"][W2 + "_scale"]
    (source / INDEX).write_text(json.dumps(index))


def cut_short(source: Path, tensors: dict[str, torch.Tensor]) -> None:
    with (source / SHARDS[1]).open("r+b") as file:
        file.truncate((source / SHARDS[1]).stat().st_size - 100)


def one_scale_per_row(source: Path, tensors: dict[str, torch.Tensor]) -> None:
    replace_tensor(source, W13 + "_scale", tensors[W13 + "_scale"][..., :1].clone())


def unlisted_expert(source: Path, tensors: dict[str, torch.Tensor]) -> None:
    metadata = json.loads((source / "hf_quant_config.json").read_text())
    metadata["model"]["parameter_groups"][-1]["tensors"] = [W13]
    (source / "hf_quant_config.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        (lambda source, tensors: (source / "hf_quant_config.json").unlink(), FileNotFoundError, "hf_quant_config"),
        (lambda source, tensors: (source / SHARDS[1]).unlink(), FileNotFoundError, "model-00002"),
        (without_scale_in_index, ValueError, "disagree on 1 tensors"),
        (cut_short, ValueError, "incomplete or damaged"),
        (one_scale_per_row, ValueError, "does not fit"),
        (unlisted_expert, ValueError, "lists other quantized tensors"),
        (scale_byte(2), ValueError, "1 scale bytes from 2 to 2"),
        (scale_byte(247), ValueError, "1 scale bytes from 247 to 247"),
        (lambda source, tensors: (source / "dangling").symlink_to("nowhere"), ValueError, "dangling"),
    ],
    ids=[
        "no_precision_metadata",
        "no_second_shard",
        "index_lacks_a_tensor",
        "short_weight_file",
        "scale_of_another_shape",
        "precision_metadata_lacks_a_weight",
        "scale_below_range",
        "scale_above_range",
        "dangling_link",
    ],
)
def test_checks_the_input_before_it_writes(converter, release, tmp_path, damage, error, message):
    source, tensors = release
    damage(source, tensors)
    with pytest.raises(error, match=message):
        converter.convert(source, tmp_path / "bf16")
    assert not (tmp_path / "bf16").exists()


def test_check_only_writes_nothing(converter, release, monkeypatch, capsys):
    source, tensors = release
    monkeypatch.setattr(sys, "argv", [str(TOOL), str(source), "--check"])
    converter.main()
    assert "Checked 2 weight files and the scales of {'mxfp8': 2} expert weights" in capsys.readouterr().out
    assert not source.with_name(source.name + "-BF16").exists()

    scale_byte(250)(source, tensors)
    with pytest.raises(ValueError, match="from 250 to 250"):
        converter.main()

    # Without --device the arithmetic runs on the CPU, also where a GPU is present.
    assert inspect.signature(converter.convert).parameters["device"].default == "cpu"
    calls = []
    monkeypatch.setattr(converter, "convert", lambda *arguments: calls.append(arguments))
    monkeypatch.setattr(sys, "argv", [str(TOOL), str(source), str(source.with_name("bf16"))])
    converter.main()
    assert calls == [(source.resolve(), source.with_name("bf16"), None, "cpu")]


def test_a_block_of_zeros_converts_at_any_scale(converter, release, tmp_path):
    source, tensors = release
    codes, scales = tensors[W13].view(torch.uint8).clone(), tensors[W13 + "_scale"].clone()
    codes[30, 1] = torch.tensor([0x00, 0x80] * 32, dtype=torch.uint8)
    scales[30, 1] = torch.tensor([0, 255], dtype=torch.uint8)
    replace_tensor(source, W13, codes.view(torch.float8_e4m3fn))
    replace_tensor(source, W13 + "_scale", scales)

    converter.convert(source, tmp_path / "bf16")
    expected = decode_mxfp8(codes, scales)
    assert expected[30, 1].abs().sum() == 0
    assert_same_numbers(load_file(tmp_path / "bf16" / SHARDS[1])[W13], expected)


def other_index(source: Path, output: Path) -> None:
    (source / INDEX).write_text((source / INDEX).read_text() + "\n")


def other_tool(source: Path, output: Path) -> None:
    record = json.loads((output / "bf16_conversion.json").read_text())
    (output / "bf16_conversion.json").write_text(json.dumps(record | {"tool_sha256": "0" * 64}))


def short_kept_file(source: Path, output: Path) -> None:
    with (output / SHARDS[0]).open("r+b") as file:
        file.truncate((output / SHARDS[0]).stat().st_size - 100)


def foreign_kept_file(source: Path, output: Path) -> None:
    save_file({"another.tensor": torch.ones(2)}, output / SHARDS[0])


@pytest.mark.parametrize(
    ("change", "revision", "message"),
    [
        (other_index, None, "input_fingerprint"),
        (other_tool, None, "tool_sha256"),
        (lambda source, output: None, "fedcba9", "source_revision"),
        (short_kept_file, None, "incomplete or damaged"),
        (foreign_kept_file, None, "does not hold the tensors"),
        (lambda source, output: (output / "bf16_conversion.json").unlink(), None, "not empty"),
        (
            lambda source, output: (output / "bf16_conversion.lock").write_text("process 1 on elsewhere\n"),
            None,
            "process 1",
        ),
    ],
    ids=["other_input", "other_tool", "other_revision", "short_file", "foreign_file", "no_record", "locked"],
)
def test_refuses_to_continue_what_it_did_not_start(converter, release, tmp_path, change, revision, message):
    source, _ = release
    output = tmp_path / "bf16"
    converter.convert(source, output)
    before = (output / SHARDS[1]).stat().st_mtime_ns
    change(source, output)
    with pytest.raises((ValueError, RuntimeError), match=message):
        converter.convert(source, output, source_revision=revision)
    assert (output / SHARDS[1]).stat().st_mtime_ns == before

    with pytest.raises(ValueError, match="inside the checkpoint"):
        converter.convert(source, source / "bf16")


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
