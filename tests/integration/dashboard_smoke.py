"""Browser smoke test for the run dashboard, run against real integration-run artifacts.

Launched at the end of each integration test so that any change to the on-disk
artifacts (metrics.jsonl rows, Episode records, config JSONs, log layout) that
breaks the dashboard fails the same CI run that produced them.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

DASHBOARD_STARTUP_TIMEOUT_S = 30
PAGE_SETTLE_MS = 4000


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _install_browser() -> None:
    subprocess.run(
        ["uv", "run", "playwright", "install", "--with-deps", "--only-shell", "chromium"],
        check=True,
        capture_output=True,
        timeout=600,
    )


def check_dashboard_smoke(output_dir: Path, run_name: str) -> None:
    """Serve ``output_dir`` and assert the dashboard renders the run end to end."""
    from playwright.sync_api import sync_playwright

    _install_browser()
    port = _free_port()
    server = subprocess.Popen(
        ["uv", "run", "dashboard", output_dir.as_posix(), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + DASHBOARD_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise AssertionError("dashboard did not start listening")

        errors: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"page: {e}"))
            base = f"http://127.0.0.1:{port}"

            # metrics: the run resolves and the overview renders charts with data
            page.goto(f"{base}/#run={run_name}&tab=metrics")
            page.wait_for_timeout(PAGE_SETTLE_MS)
            status = page.locator("#run-overview .badge").first.inner_text()
            assert status, "overview card did not render a status"
            fields = page.evaluate(
                """() => Object.fromEntries([...document.querySelectorAll('#run-overview .ov-field')]
                    .map(f => [f.querySelector('.lbl').innerText, f.querySelector('.val, .badge').innerText]))"""
            )
            run_type = fields.get("TYPE")
            assert run_type in ("RL", "SFT", "EVAL"), f"overview type field did not render: {fields!r}"
            if run_type == "EVAL":
                assert fields.get("EPISODES", "").isdigit(), f"overview episodes field did not render: {fields!r}"
                cards = page.locator(".stat-card").count()
                assert cards >= 5, f"expected >=5 stat cards, got {cards}"
            else:
                assert "/" in fields.get("STEP", ""), f"overview step field did not render: {fields!r}"
                charts = page.locator(".chart-card").count()
                assert charts >= 5, f"expected >=5 metric panels, got {charts}"
                with_data = page.evaluate("""() => [...document.querySelectorAll('.chart-card .u-wrap')].length""")
                assert with_data >= 5, f"expected >=5 mounted charts, got {with_data}"

            # config: the default view renders (launch TOML on new runs), and the
            # resolved concatenated document renders as a tree
            page.click("#tabs [data-tab=config]")
            page.wait_for_timeout(1500)
            assert page.eval_on_selector("#config-view", "e => e.innerText.length") > 100, "config view did not render"
            page.click("#config-format [data-fmt=json]")
            page.wait_for_timeout(1500)
            assert page.locator("#config-view .j-line").count() > 10, "resolved config tree did not render"

            # traces: when the run saved rollouts, episodes must render and open
            rollout_steps = page.evaluate(
                f"""fetch('{base}/api/runs/{run_name}/rollouts').then(r => r.json()).then(d => d.steps.length)"""
            )
            if rollout_steps:
                page.click("#tabs [data-tab=traces]")
                page.wait_for_timeout(3000)
                rows = page.locator("#episode-table tbody tr").count()
                assert rows > 0, "traces table rendered no episodes"
                page.click("#episode-table tbody tr >> nth=0")
                page.wait_for_timeout(3000)
                entries = page.locator("#tm-messages details.entry").count()
                assert entries > 0, "episode viewer rendered no messages"
                reward = page.locator(".tm-reward-big").first.inner_text()
                assert reward not in ("", "n/a"), f"episode reward did not render: {reward!r}"
                page.click("#tm-view [data-view=timeline]")
                page.wait_for_timeout(1000)
                lanes = page.locator("#tm-timeline .tl-lane").count()
                assert lanes > 0, "episode timeline rendered no agent lanes"
                lifecycle = page.locator("#tm-timeline .tl-span.lifecycle").count()
                assert lifecycle > 0, "episode timeline rendered no lifecycle spans"
                activity = page.locator("#tm-timeline .tl-span.activity")
                if activity.count():
                    activity.first.click()
                    page.wait_for_timeout(500)
                    assert page.locator("#tm-view [data-view=transcript].active").count(), (
                        "timeline activity did not return to the transcript"
                    )
                page.keyboard.press("Escape")

            # logs: the merged pane shows lines
            page.click("#tabs [data-tab=logs]")
            page.wait_for_timeout(PAGE_SETTLE_MS)
            log_lines = page.locator(".log-pane .ll").count()
            assert log_lines > 10, f"log pane rendered only {log_lines} lines"

            browser.close()

        assert not errors, f"dashboard raised browser errors: {errors[:5]}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    print(f"Dashboard smoke passed for {run_name} ({output_dir})", file=sys.stderr)


def make_dashboard_test(process_fixture: str, run_name: str):
    """The per-file dashboard smoke: `test_dashboard = make_dashboard_test("rl_process", RUN_NAME)`.
    Depends on the module's (module-scoped) training-process fixture, so it runs
    against the artifacts that run just produced."""
    import pytest

    @pytest.mark.usefixtures(process_fixture)
    def test_dashboard(output_dir: Path) -> None:
        check_dashboard_smoke(output_dir, run_name)

    return test_dashboard
