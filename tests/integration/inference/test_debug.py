from pathlib import Path
from subprocess import Popen
from typing import Callable

import pyarrow.parquet as pq
import pytest

from zeroband.training.data import pa_schema

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "src/zeroband/infer.py", "@configs/inference/debug.toml"]


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_infer_debug")


@pytest.fixture(scope="module")
def process(output_path: Path, run_process: Callable[[list[str]], Popen]):
    return run_process(CMD + ["--output-path", str(output_path)])


def test_no_error(process: Popen):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"


def test_output_directories_exist(output_path: Path):
    assert output_path.joinpath("step_0").exists()
    assert output_path.joinpath("step_1").exists()
    assert output_path.joinpath("step_2").exists()
    assert output_path.joinpath("step_3").exists()
    assert not output_path.joinpath("step_4").exists()


def test_output_files_have_correct_schemas(output_path: Path):
    for file in output_path.rglob("*.parquet"):
        assert pq.read_schema(file).equals(pa_schema)


def test_toploc_proofs(output_path: Path):
    for file in output_path.rglob("*.parquet"):
        table = pq.read_table(file)

        # Assert number of proofs
        proofs: list[bytes] = table.column("proofs").to_pylist()
        output_tokens: list[list[int]] = table.column("output_tokens").to_pylist()
        assert len(proofs) == len(output_tokens)

        # Assert proof lengths
        for proof, output_token in zip(proofs, output_tokens):
            assert len(proof) % 258 == 0
            assert len(proof) // 258 == (len(output_token) + 31) // 32
