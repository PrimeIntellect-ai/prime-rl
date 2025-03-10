import subprocess
import pytest
import pyarrow.parquet as pq


def _test_torchrun(config, extra_args=[]):
    cmd = [
        "python",
        "src/zeroband/inference.py",
        f"@configs/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process  failed {result}")


@pytest.mark.parametrize("tp", [1, 2])
def test_inference(tmp_path, tp):
    _test_torchrun(config="inference/debug.toml", extra_args=["--output_path", str(tmp_path), "--tp", str(tp)])

    assert tmp_path.joinpath("step_0").exists()

    for file in tmp_path.joinpath("step_0").iterdir():
        if file.suffix == ".parquet":
            table = pq.read_table(file)
            assert set(table.schema.names) == {
                "input_tokens",
                "output_tokens",
                "input_logprobs",
                "output_logprobs",
                "advantages",
                "rewards",
                "proofs",
                "step",
            }

            # Check that proof lengths are correct
            proofs: list[bytes] = table.column("proofs").to_pylist()
            output_tokens: list[list[int]] = table.column("output_tokens").to_pylist()
            assert len(proofs) == len(output_tokens)
            for proof, output_token in zip(proofs, output_tokens):
                assert len(proof) % 258 == 0
                assert len(proof) // 258 == (len(output_token) + 31) // 32

            assert table.num_rows > 0
