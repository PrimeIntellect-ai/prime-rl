import subprocess

import pyarrow.parquet as pq
import pytest


def _test_torchrun(config, extra_args=[]):
    cmd = [
        "python",
        "src/zeroband/infer.py",
        f"@configs/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process  failed {result}")


@pytest.mark.parametrize("toploc    ", [True])  # , False])
@pytest.mark.parametrize("tp", [1])  # , 2])
def test_inference(tmp_path, tp, toploc):
    _test_torchrun(config="inference/debug.toml", extra_args=["--output_path", str(tmp_path), "--tp", str(tp), "--toploc", str(toploc)])

    assert tmp_path.joinpath("step_0").exists()

    for file in tmp_path.joinpath("step_0").iterdir():
        if file.suffix == ".parquet":
            table = pq.read_table(file)

            # Check that proof lengths are correct
            proofs: list[bytes] = table.column("proofs").to_pylist()
            output_tokens: list[list[int]] = table.column("output_tokens").to_pylist()
            assert len(proofs) == len(output_tokens)

            if toploc:
                for proof, output_token in zip(proofs, output_tokens):
                    assert len(proof) % 258 == 0
                    assert len(proof) // 258 == (len(output_token) + 31) // 32

            assert table.num_rows > 0
