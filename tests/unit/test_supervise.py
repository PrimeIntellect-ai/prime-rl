import socket
import sys
from pathlib import Path

import pytest

from prime_rl.entrypoints import supervise


def _supervise(monkeypatch, args: list[str]) -> int:
    monkeypatch.setattr(supervise, "RESTART_DELAY_S", 0)
    monkeypatch.setattr(sys, "argv", ["supervise", *args])
    with pytest.raises(SystemExit) as exit_info:
        supervise.main()
        raise SystemExit(0)
    return exit_info.value.code


def test_group_restarts_every_command_and_gives_up_after_its_budget(monkeypatch, tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    fail_once = f'echo $PRL_RESTART_COUNT >> {a}; [ "$PRL_RESTART_COUNT" -ge 1 ] || exit 3'
    sleeper = f'echo $PRL_RESTART_COUNT >> {b}; [ "$PRL_RESTART_COUNT" -ge 1 ] || sleep 300'
    assert _supervise(monkeypatch, ["--name", "g", "--restarts", "1", "--group", fail_once, sleeper]) == 0
    assert a.read_text().split() == b.read_text().split() == ["0", "1"]
    assert _supervise(monkeypatch, ["--name", "g", "--restarts", "1", "exit 7"]) == 7


def test_restarts_a_command_that_stops_answering(monkeypatch, tmp_path: Path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    count, server = tmp_path / "count", tmp_path / "server.py"
    server.write_text(
        "import http.server, os, sys, time\n"
        f"open({str(count)!r}, 'a').write(os.environ['PRL_RESTART_COUNT'] + '\\n')\n"
        "if os.environ['PRL_RESTART_COUNT'] != '0': sys.exit(0)\n"
        "start = time.monotonic()\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if time.monotonic() - start > 1.5: time.sleep(1000)  # answers, then hangs\n"
        "        self.send_response(200); self.end_headers()\n"
        f"http.server.ThreadingHTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
    )
    monkeypatch.setattr(supervise, "HEALTH_INTERVAL_S", 0.5)
    monkeypatch.setattr(supervise, "HEALTH_MISSES", 2)
    health = f"http://127.0.0.1:{port}/"
    assert (
        _supervise(monkeypatch, ["--name", "h", "--restarts", "1", "--health", health, f"{sys.executable} {server}"])
        == 0
    )
    assert count.read_text().split() == ["0", "1"]
