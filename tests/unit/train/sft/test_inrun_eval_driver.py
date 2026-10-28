"""Node 0's eval driver end to end on CPU, with fakes for what needs GPUs or the network: the engines (a stub),
the router (a tiny HTTP server answering /v1/models and /v1/chat/completions), the harness venv (a fake ``flow``
that writes a store's trace index for the tasks of its config) and the coordination store (in memory). Checks the
results the run leaves under <run dir>/eval/ and that the staging cleanup ran."""

import hashlib
import json
import os
import stat
import sys
import textwrap
import threading
from pathlib import Path

from prime_rl.configs.sft import SFTInRunEvalConfig
from prime_rl.trainer.sft.inrun_eval import harness, node, plan, tasks
from prime_rl.trainer.sft.inrun_eval.coord import MemoryStore

ROUTER = textwrap.dedent(
    """\
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port = int(sys.argv[sys.argv.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self.reply({"data": [{"id": "nemotron-3.5-super-tb4-sft"}]})
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.reply({"choices": [{"finish_reason": "stop", "message": {"content": "ready"}}]})
        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)

# A fake `flow run @ <config>`: one solver draw row per selected task and draw; the first task passes.
FLOW = textwrap.dedent(
    """\
    import json, re, sys, tomllib, pathlib
    config = tomllib.loads(pathlib.Path(sys.argv[sys.argv.index("@") + 1]).read_text())
    flow = config["flow"]
    store = pathlib.Path(flow["store"]); (store / "traces").mkdir(parents=True, exist_ok=True)
    assert flow["solver"]["client"]["base_url"].endswith("/v1"), flow["solver"]["client"]
    names = flow["select"]["include"]["names"]
    k = flow["pieces"]["rollouts"]["k"]
    assert flow["solver"]["harness"]["id"] == "tb4-mini-swe-agent-d5", flow["solver"]["harness"]
    assert flow["solver"]["harness"]["prepare_exclude_newer"] == "2026-10-24T07:23:00Z"
    with open(store / "traces" / "index.jsonl", "w") as f:
        for i, name in enumerate(names):
            if name.endswith("/slow"):
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
            for n in range(k):
                row = {"id": f"t{i}{n}", "labels": {"flow": {"job": f"{slug}-0123456789ab-qc",
                       "call": f"solver/draw{n}-abc", "seat": "solver"}}, "agent": "solver", "ok": True,
                       "error": None, "stop_condition": "agent_completed", "reward": 1.0 if i == 0 else 0.0}
                f.write(json.dumps(row) + "\\n")
    if any(name.endswith("/slow") for name in names):
        import time
        time.sleep(600)
    """
)

SOURCE = """[run]
name = "src"
output_dir = "/x"

[flow]
id = "sft-gen-task-qc"
store = "/x/src/store"
model = "teacher"

[flow.taskset]
id = "sft-gen-kernel-harbor"
dataset = "tb-science-vendor@85df9700aaf4"
repo = "PrimeIntellect-ai/nvidia-delivery@main"
registry_path = "registries/tb-science.json"

[flow.select.include]
names = []

[flow.pieces.rollouts]
on = true
k = 8

[flow.solver]
model = "teacher"

[flow.solver.harness]
id = "mini_swe_agent"
command_timeout = 1800

[flow.solver.client]
base_url = "https://api.pinference.ai/api/v1"
api_key_var = "PRIME_API_KEY"
type = "eval"

[flow.rollout_gate]
name = "glm"
max_concurrent = 2000
dir = "/x/gates"
"""


def executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def setup_fakes(tmp_path, monkeypatch) -> tuple[Path, MemoryStore]:
    bundle = tmp_path / "bundle"
    files = {
        "tasks.tsv": "terminal-bench/a\ttb-sci\tval\nterminal-bench/b\ttb-sci\ttrain\nterminal-bench/c\ttb-sci\tother\n",
        "smoke-tasks.txt": "terminal-bench/a\nterminal-bench/b\n",
        "harness/requirements.txt": "verifiers==0.4.1.dev340\n",
        "sft-gen/plugins/sft_gen_task_qc.py": "",
        "sft-gen/sft_gen/task_qc.py": "",
        "flow-configs/tb-sci.toml": SOURCE,
    }
    cache = tasks.harbor_cache_name(
        "tb-science-vendor@71124a84e6f3", "PrimeIntellect-ai/nvidia-delivery@main", "registries/tb-science.json"
    )
    files[f"harbor-cache/{cache}/a/task.toml"] = "x"
    files[f"harbor-cache/{cache}/a/tests/test.sh"] = "#!/bin/bash\n"
    for rel, text in files.items():
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_text(text)
        (bundle / rel).chmod(0o644)  # as the volume serves them: no exec bits
    modes = {rel: "0755" if rel.endswith("test.sh") else "0644" for rel in files}
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "files": {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()},
                "modes": modes,
            }
        )
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "vllm-router", ROUTER)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    executable(venv / "bin" / "flow", FLOW)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("PRIME_API_KEY", "key")
    monkeypatch.setenv("PRIME_TEAM_ID", "team")

    def fake_build_env(h, bundle_dir, work, log=print, run=None):
        work.mkdir(parents=True, exist_ok=True)
        cwd = work / "sft-gen"
        cwd.mkdir(exist_ok=True)
        env = harness.uv_env(dict(os.environ), work)
        return harness.HarnessEnv(venv=venv, cwd=cwd, env=env, info={"packages": {"verifiers": "0.4.1.dev340"}})

    monkeypatch.setattr(harness, "build_env", fake_build_env)
    probes = []
    monkeypatch.setattr(
        harness,
        "probe_credentials",
        lambda key, team, base: probes.append(team) or {"status": "ok", "checks": {}, "error": None},
    )
    monkeypatch.setattr(
        node.NodeServer, "serve", lambda self, ckpt, deadline: [f"http://127.0.0.1:{9000 + self.plan.node}"]
    )
    monkeypatch.setattr(node, "node_ip", lambda master: "127.0.0.1")
    store = MemoryStore()
    monkeypatch.setattr(node, "tcp_store", lambda *a, **k: store)
    monkeypatch.setattr(node, "BYE", 0)
    monkeypatch.setattr(harness, "FLOW_POLL", 0.5)
    return bundle, store


def harness_cfg(bundle: Path) -> dict:
    return {
        "bundle_dir": bundle,
        "sources": ["tb-sci"],
        "source_config_sha256": {"tb-sci": hashlib.sha256(SOURCE.encode()).hexdigest()},
        "tasks_file": bundle / "smoke-tasks.txt",
        "k": 2,
        "expected_tasks": 2,
        "expected_val_tasks": 1,
    }


def test_driver_end_to_end(tmp_path, monkeypatch):
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    cfg = SFTInRunEvalConfig(
        eval_only=True,
        model_dir=tmp_path / "model",
        archive_stores=True,
        serving={"router_port": 18431},
        harness=harness_cfg(bundle),
    )
    run_dir = tmp_path / "run"
    p = plan.build_plan(cfg, run_dir=run_dir, exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="127.0.0.1")
    messages = []
    driver = node.Driver(p, cfg, messages.append)
    driver.run()
    node.cleanup(p, cfg, messages.append)

    status = json.loads((run_dir / "eval" / "status.json").read_text())
    assert status["harness"]["ok"], status["harness"]
    entry = status["checkpoints"]["base"]
    assert entry["status"] == "done", (entry, messages)
    summary = json.loads((run_dir / "eval" / "base" / "summary.json").read_text())
    overall = summary["summary"]["overall"]
    assert overall["tasks"] == 2 and overall["rollouts"] == 4 and overall["pass@1"] == 0.5 and overall["pass@2"] == 0.5
    assert summary["summary"]["split"]["val"]["pass@1"] == 1.0 and summary["summary"]["split"]["train"]["pass@1"] == 0
    assert summary["complete"] and summary["rollout_gate"] == 24 and summary["tasks"] == {"tb-sci": 2}
    assert summary["cap"]["budget_hours"] is None and summary["coverage"]["fraction"] == 1.0
    assert status["credentials"] == {
        "prime_api_key_present": True,
        "prime_team_id": "team",
        "prime_team_id_source": "environment",
        "probe": {"status": "ok", "checks": {}, "error": None},
    }
    assert status["credentials"]["probe"]["status"] == "ok" and status["eval"]["status"] == "done"
    assert "key" not in json.dumps(status["credentials"]).replace("prime_api_key_present", "")
    # Liveness: node 0's heartbeat, with what landed.
    beat = status["heartbeat"]
    assert beat["phase"] == "finished" and beat["rollouts_done"] == 4 and beat["rows_landed"] == 4
    assert beat["rollouts_planned"] == 4 and beat["per_checkpoint"]["base"]["rollouts_done"] == 4
    assert beat["last_progress"] <= beat["time"] and beat["interval_seconds"] == node.HEARTBEAT
    assert summary["split_tasks"] == {"val": 1, "train": 1, "other": 0} and len(summary["tasks_tsv_sha256"]) == 64
    changed = summary["changed_config_keys"]["tb-sci"]
    assert "flow.taskset.dataset" in changed and "flow.pieces.rollouts.k" in changed
    assert "flow.solver.harness.id" in changed and summary["d5"]["exclude_newer"] == "2026-10-24T07:23:00Z"
    config = (run_dir / "eval" / "base" / "configs" / "tb-sci.toml").read_text()
    assert 'base_url = "http://127.0.0.1:18431/v1"' in config and "command_timeout = 1800" in config
    per_task = [json.loads(line) for line in (run_dir / "eval" / "base" / "per_task.jsonl").read_text().splitlines()]
    assert [(t["task"], t["c"]) for t in per_task] == [("terminal-bench/a", 2), ("terminal-bench/b", 0)]
    assert list((run_dir / "eval" / "base" / "stores").glob("tb-sci.tar.*"))
    top = json.loads((run_dir / "eval" / "summary.json").read_text())
    assert top["base"]["status"] == "done" and top["base"]["val"]["pass@1"] == 1.0
    assert json.loads(store.get("done"))["ok"] is True
    assert not Path(p.work_dir).exists()  # the work dir (venv, stores) never reaches the volume


def test_two_nodes_in_parallel(tmp_path, monkeypatch):
    """best on node 0 and last on node 1 at the same time: the follower serves when told, node 0 routes to it, both
    checkpoints are scored, the follower stops and leaves on 'done'; it keeps only its upload share of the exports."""
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    exports = {}
    for name, step in (("best", 80), ("last", 280)):
        directory = run_dir / "hf-export" / name
        directory.mkdir(parents=True)
        for file in ("a.safetensors", "b.safetensors", "config.json"):
            (directory / file).write_text(file)
        exports[name] = {"name": name, "ok": True, "dir": str(directory), "step": step, "verified": True}
    cfg = SFTInRunEvalConfig(
        schedule="parallel",
        budget_hours=1,
        reserve_minutes=5,
        archive_stores=False,
        serving={"router_port": 18441},
        harness=harness_cfg(bundle),
    )
    plans = [
        plan.build_plan(
            cfg,
            run_dir=run_dir,
            exports=exports,
            node=n,
            num_nodes=2,
            gpus_per_node=8,
            master_addr="127.0.0.1",
            node_files=[f"{'ab'[n]}.safetensors"],
            master_files=["config.json"],
        )
        for n in range(2)
    ]
    assert plans[0].rounds == [{"best": [0], "last": [1]}]
    messages = []
    follower = threading.Thread(target=node.follow, args=(plans[1], cfg, messages.append))
    follower.start()
    node.Driver(plans[0], cfg, messages.append).run()
    follower.join(timeout=60)
    assert not follower.is_alive()
    node.cleanup(plans[1], cfg, messages.append)

    assert json.loads(store.get("ready/0/1")) == {"ok": True, "urls": ["http://127.0.0.1:9001"]}
    assert store.check(["stop/0", "stopped/0/1", "done"])
    status = json.loads((run_dir / "eval" / "status.json").read_text())
    assert {name: entry["status"] for name, entry in status["checkpoints"].items()} == {"best": "done", "last": "done"}
    for name, port in (("best", 18441), ("last", 18442)):
        summary = json.loads((run_dir / "eval" / name / "summary.json").read_text())
        assert summary["step"] == exports[name]["step"] and summary["summary"]["overall"]["pass@1"] == 0.5
        assert (
            f'base_url = "http://127.0.0.1:{port}/v1"'
            in (run_dir / "eval" / name / "configs" / "tb-sci.toml").read_text()
        )
        assert f'name = "tb4-eval-{name}"' in (run_dir / "eval" / name / "configs" / "tb-sci.toml").read_text()
    # Node 1's staging keeps its upload share of each export, not the master's small files.
    assert sorted(p.name for p in (run_dir / "hf-export" / "last").iterdir()) == ["b.safetensors"]


def test_no_harness_fails_cleanly(tmp_path, monkeypatch):
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    (bundle / "tasks.tsv").write_text("tampered")  # the manifest check fails
    cfg = SFTInRunEvalConfig(
        eval_only=True, model_dir=tmp_path / "m", budget_hours=1, reserve_minutes=5, harness=harness_cfg(bundle)
    )
    p = plan.build_plan(
        cfg, run_dir=tmp_path / "run", exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x"
    )
    node.Driver(p, cfg, lambda m: None).run()
    status = json.loads((tmp_path / "run" / "eval" / "status.json").read_text())
    assert status["harness"]["ok"] is False and "tasks.tsv" in status["harness"]["error"]
    assert status["checkpoints"]["base"]["status"] == "failed"
    assert json.loads(store.get("done")) == {"ok": False}


def test_no_platform_credential_fails_the_eval_before_serving(tmp_path, monkeypatch):
    """Without the platform's PRIME_API_KEY nothing can run: the eval records it and never loads vLLM."""
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    monkeypatch.delenv("PRIME_API_KEY")
    monkeypatch.setenv("TB4_EVAL_PRIME_API_KEY", "not used")
    served = []
    monkeypatch.setattr(node.NodeServer, "serve", lambda self, ckpt, deadline: served.append(ckpt) or [])
    cfg = SFTInRunEvalConfig(eval_only=True, model_dir=tmp_path / "m", harness=harness_cfg(bundle))
    p = plan.build_plan(
        cfg, run_dir=tmp_path / "run", exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x"
    )
    node.Driver(p, cfg, lambda m: None).run()
    status = json.loads((tmp_path / "run" / "eval" / "status.json").read_text())
    assert status["eval"]["status"] == "no_credentials" and "PRIME_API_KEY is not set" in status["eval"]["error"]
    assert status["credentials"]["prime_api_key_present"] is False
    assert status["checkpoints"]["base"]["status"] == "failed"
    assert status["checkpoints"]["base"]["reason"] == "no_credentials"
    assert served == [] and json.loads(store.get("done")) == {"ok": False}
    top = json.loads((tmp_path / "run" / "eval" / "summary.json").read_text())
    assert top["base"]["reason"] == "no_credentials"


def test_refused_credentials_fail_the_eval_before_serving(tmp_path, monkeypatch):
    """The probe's verdict (the Prime API refuses the key for the team): nothing is served, the status says why."""
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    served = []
    monkeypatch.setattr(node.NodeServer, "serve", lambda self, ckpt, deadline: served.append(ckpt) or [])
    refusal = {"status": "forbidden", "checks": {"sandboxes": {"http": 403}}, "error": "HTTP 403 not a member"}
    monkeypatch.setattr(harness, "probe_credentials", lambda key, team, base: refusal)
    cfg = SFTInRunEvalConfig(eval_only=True, model_dir=tmp_path / "m", harness=harness_cfg(bundle))
    p = plan.build_plan(
        cfg, run_dir=tmp_path / "run", exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x"
    )
    node.Driver(p, cfg, lambda m: None).run()
    status = json.loads((tmp_path / "run" / "eval" / "status.json").read_text())
    assert status["eval"] == {"status": "forbidden", "error": "HTTP 403 not a member"}
    assert status["credentials"]["probe"] == refusal and status["checkpoints"]["base"]["reason"] == "forbidden"
    assert served == [] and json.loads(store.get("done")) == {"ok": False}


def test_harness_prep_timeout_is_recorded(tmp_path, monkeypatch):
    """A harness prep that outlasts its limit: recorded as such (no AttributeError), the checkpoint fails cleanly."""
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    slow = threading.Event()
    real = harness.build_env

    def slow_build(*a, **k):
        slow.wait(30)
        return real(*a, **k)

    monkeypatch.setattr(harness, "build_env", slow_build)
    monkeypatch.setattr(node, "HARNESS_PREP", 0.5)
    cfg = SFTInRunEvalConfig(
        eval_only=True, model_dir=tmp_path / "m", serving={"router_port": 18461}, harness=harness_cfg(bundle)
    )
    p = plan.build_plan(
        cfg, run_dir=tmp_path / "run", exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x"
    )
    try:
        node.Driver(p, cfg, lambda m: None).run()
    finally:
        slow.set()
    status = json.loads((tmp_path / "run" / "eval" / "status.json").read_text())
    assert status["eval"]["status"] == "harness_prep_timeout"
    assert "harness prep timed out" in status["eval"]["error"] and status["harness"]["ok"] is False
    entry = status["checkpoints"]["base"]
    assert entry["status"] == "failed" and entry["reason"] == "harness_prep_timeout"
    assert "harness prep timed out" in entry["error"] and "AttributeError" not in json.dumps(status)


def test_harness_staging_restores_task_modes(tmp_path, monkeypatch):
    """The real prepare_harness staging (build_env faked): the Harbor cache copy gets the manifest's exec bits."""
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    cfg = SFTInRunEvalConfig(eval_only=True, model_dir=tmp_path / "m", harness=harness_cfg(bundle))
    p = plan.build_plan(
        cfg, run_dir=tmp_path / "run", exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x"
    )
    driver = node.Driver(p, cfg, lambda m: None)
    driver.prepare_harness()
    assert driver.henv is not None, driver.status["harness"]
    home = Path(driver.henv.env["HOME"])
    (test_sh,) = (home / ".cache" / "harbor").rglob("test.sh")
    assert os.access(test_sh, os.X_OK) and not os.access(test_sh.parent.parent / "task.toml", os.X_OK)


def test_safety_cap_reports_cut_draws_apart(tmp_path, monkeypatch):
    """A configured cap that fires: the unfinished task's draws are 'cut' (not failures) and the coverage says so."""
    bundle, store = setup_fakes(tmp_path, monkeypatch)
    (bundle / "smoke-tasks.txt").write_text("terminal-bench/a\nterminal-bench/slow\n")
    tsv = (bundle / "tasks.tsv").read_text() + "terminal-bench/slow\ttb-sci\ttrain\n"
    (bundle / "tasks.tsv").write_text(tsv)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for rel in ("smoke-tasks.txt", "tasks.tsv"):
        manifest["files"][rel] = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    cfg = SFTInRunEvalConfig(
        eval_only=True,
        model_dir=tmp_path / "m",
        budget_hours=0.01,  # 36 s
        reserve_minutes=0.1,
        archive_stores=False,
        serving={"router_port": 18451},
        harness=harness_cfg(bundle),
    )
    p = plan.build_plan(
        cfg, run_dir=tmp_path / "run", exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x"
    )
    monkeypatch.setattr(harness, "FLOW_POLL", 0.2)
    import prime_rl.trainer.sft.inrun_eval.serving as serving_mod

    stop = serving_mod.stop
    monkeypatch.setattr(serving_mod, "stop", lambda procs, grace=300.0: stop(procs, grace=2.0))
    node.Driver(p, cfg, lambda m: None).run()
    summary = json.loads((tmp_path / "run" / "eval" / "base" / "summary.json").read_text())
    overall = summary["summary"]["overall"]
    assert summary["flow_runs"]["tb-sci"]["stopped"] and summary["cap"]["stopped_sources"] == ["tb-sci"]
    assert overall["cut"] == 2 and overall["missing"] == 0 and overall["errored"] == 0
    assert overall["pass@1"] == 1.0 and overall["pass@1_tasks"] == 1  # only 'a' has draws: both passed
    assert overall["coverage"]["fraction"] == 0.5 and not summary["complete"]
