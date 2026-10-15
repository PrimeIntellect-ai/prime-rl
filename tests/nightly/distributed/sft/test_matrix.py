import pytest
from huggingface_hub import snapshot_download

from tests.nightly.distributed.sft.cases import CASES, assert_training_rows, load_metrics, sft_args
from tests.nightly.distributed.tiny_models import TOKENIZERS

pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

TRAIN_STEPS, RESUME_STEPS, TIMEOUT_SEC = 5, 3, 600


@pytest.mark.parametrize("case", [case.param() for case in CASES])
def test_sft(case, tiny_checkpoint, run_process, tmp_path):
    model_name = tiny_checkpoint(case.arch.name).as_posix()
    snapshot_download(TOKENIZERS[case.arch.name], allow_patterns=["*.json", "*.txt", "tokenizer*", "*.jinja"])

    def sft(*extra: str) -> None:
        cmd = ["uv", "run", "sft", *sft_args(case, model_name), "--output-dir", tmp_path.as_posix()]
        cmd += ["--run.name", case.id, *extra]
        assert run_process(cmd, timeout=TIMEOUT_SEC).returncode == 0

    if case.resume:
        sft("--max-steps", str(TRAIN_STEPS), "--ckpt")
        sft("--max-steps", str(TRAIN_STEPS + RESUME_STEPS), "--resume.step", str(TRAIN_STEPS))
        expected = range(1, TRAIN_STEPS + RESUME_STEPS + 1)
    else:
        sft("--max-steps", str(TRAIN_STEPS))
        expected = range(1, TRAIN_STEPS + 1)
    assert_training_rows(load_metrics(tmp_path / case.id), expected)
