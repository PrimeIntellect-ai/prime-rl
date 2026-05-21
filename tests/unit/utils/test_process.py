import subprocess

from prime_rl.utils import process


class FakeStdout:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePopen:
    def __init__(self, cmd, **kwargs) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = FakeStdout() if kwargs.get("stdout") is subprocess.PIPE else None


def test_start_tail_processes_uses_argv_without_shell(tmp_path, monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        popen = FakePopen(cmd, **kwargs)
        calls.append(popen)
        return popen

    monkeypatch.setattr(process, "Popen", fake_popen)

    log_path = tmp_path / "trainer '$(touch injected)'.log"

    started = process.start_tail_processes(log_path)

    assert started == calls[:1]
    assert calls[0].cmd == ["tail", "-F", str(log_path)]
    assert "shell" not in calls[0].kwargs


def test_start_tail_processes_can_strip_torchrun_prefix_without_shell(tmp_path, monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        popen = FakePopen(cmd, **kwargs)
        calls.append(popen)
        return popen

    monkeypatch.setattr(process, "Popen", fake_popen)

    log_path = tmp_path / "trainer.log"

    started = process.start_tail_processes(log_path, strip_torchrun_prefix=True)

    assert started == calls
    assert calls[0].cmd == ["tail", "-F", str(log_path)]
    assert calls[0].kwargs == {"stdout": subprocess.PIPE}
    assert calls[0].stdout is not None
    assert calls[0].stdout.closed is True
    assert calls[1].cmd == ["sed", "-u", process.TORCHRUN_LOG_PREFIX_PATTERN]
    assert calls[1].kwargs == {"stdin": calls[0].stdout}
    assert "shell" not in calls[1].kwargs
