"""A node's eval process (``python -m prime_rl.trainer.sft.inrun_eval.node <plan.json>``), started by the hand-off.

Every node serves the checkpoint node 0 assigns it per round (``serving``); node 0 also hosts the coordination store,
runs a router per checkpoint, the harness (three flow runs per checkpoint), scores and writes the results under
``<run dir>/eval/``. Failures are recorded, never raised: the process always exits 0 (the exports are already on the
volume and pushed; a failed pod could cost the run its post-run sync).
"""

import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from prime_rl.trainer.sft.inrun_eval import flowcfg, harness, scoring, serving
from prime_rl.trainer.sft.inrun_eval.coord import Coordinator, tcp_store
from prime_rl.trainer.sft.inrun_eval.plan import Checkpoint, Plan, round_deadline
from prime_rl.trainer.sft.inrun_eval.tasks import harbor_cache_name, parse_task_table, read_names, select_tasks

STORE_TIMEOUT = 1200.0  # a follower waits this long for node 0's store to come up
UNREACHABLE = 1800.0  # node 0's store unreachable this long: it is gone, give up
STOP_ACK = 900.0
BYE = 1800.0
GPU_FREE_MIB = 4096
HARNESS_PREP = 4 * 3600.0  # without a cap: the longest the harness setup (offline install, caches) may take
HEARTBEAT = 300.0  # how often node 0 refreshes status.json's heartbeat (liveness: time, phase, rollouts landed)


def node_ip(master_addr: str) -> str:
    """This node's address as the other nodes reach it (the interface towards MASTER_ADDR)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((master_addr, 9))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class NodeServer:
    """The vLLM engines of this node for one checkpoint at a time."""

    def __init__(self, plan: Plan, cfg, ip: str, log) -> None:
        self.plan, self.cfg, self.ip, self.log = plan, cfg, ip, log
        self.work = Path(plan.work_dir)
        self.procs: list[serving.Proc] = []
        self.inference_config: dict | None = None

    def model_dir(self, ckpt: Checkpoint) -> Path:
        """This node's copy of the checkpoint, downloaded from its repo when the local copy is incomplete."""
        problems = serving.check_model_dir(self.cfg.serving, Path(ckpt.model_dir))
        if not problems:
            return Path(ckpt.model_dir)
        self.log(f"{ckpt.name}: local copy at {ckpt.model_dir} is not servable: {problems[:3]}")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not (self.cfg.hf_fallback and ckpt.repo_id and token):
            raise RuntimeError(f"{ckpt.name}: no servable copy on node {self.plan.node} and no Hugging Face fallback")
        from huggingface_hub import snapshot_download

        target = self.work / "models" / ckpt.name
        self.log(f"{ckpt.name}: downloading {ckpt.repo_id} to {target}")
        snapshot_download(ckpt.repo_id, local_dir=target, token=token, max_workers=16)
        problems = serving.check_model_dir(self.cfg.serving, target)
        if problems:
            raise RuntimeError(f"{ckpt.name}: the download from {ckpt.repo_id} is not servable either: {problems[:3]}")
        return target

    def serve(self, ckpt: Checkpoint, deadline: float) -> list[str]:
        """Start the engines and wait until all answer. Returns their URLs; raises (engines stopped) on failure."""
        self.stop()
        serving.wait_gpus_free(GPU_FREE_MIB, 900, self.log)
        model_dir = self.model_dir(ckpt)
        sdir = self.work / "serve" / ckpt.name
        sdir.mkdir(parents=True, exist_ok=True)
        config = serving.inference_config(self.cfg.serving, model_dir, sdir)
        self.inference_config = config
        config_path = sdir / "inference.json"
        config_path.write_text(json.dumps(config, indent=2))
        specs = serving.engine_specs(self.cfg.serving, config_path, self.ip, sys.executable)
        env = serving.engine_env(self.cfg.serving, dict(os.environ))
        for spec in specs:
            if spec.overrides_path is not None:
                spec.overrides_path.write_text(json.dumps(spec.overrides))
            log = self.work / "logs" / f"vllm-{ckpt.name}-node{self.plan.node}-dp{spec.dp_rank}.log"
            self.procs.append(serving.start(f"vllm-dp{spec.dp_rank}", spec.cmd, {**env, **spec.env}, log))
        self.log(f"{ckpt.name}: started {len(specs)} vLLM engines from {model_dir}")
        urls = [f"http://{self.ip}:{spec.port}" for spec in specs]
        local = [f"http://127.0.0.1:{spec.port}" for spec in specs]
        start = time.time()
        while True:
            dead = [proc for proc in self.procs if not proc.alive()]
            if dead:
                tails = "\n".join(f"--- {p.name}:\n{serving.tail(p.log)}" for p in dead)
                self.stop()
                raise RuntimeError(f"{ckpt.name}: vLLM engine(s) {[p.name for p in dead]} exited\n{tails}")
            if all(serving.serves_model(url, self.cfg.serving.served_model_name) for url in local):
                self.log(f"{ckpt.name}: {len(urls)} engines ready after {time.time() - start:.0f}s")
                return urls
            if time.time() >= deadline:
                self.stop()
                raise RuntimeError(f"{ckpt.name}: engines not ready after {time.time() - start:.0f}s")
            time.sleep(15)

    def alive(self) -> bool:
        return bool(self.procs) and all(proc.alive() for proc in self.procs)

    def stop(self) -> None:
        if self.procs:
            serving.stop(self.procs)
            self.procs = []


def connect(plan: Plan, is_master: bool, log) -> Coordinator:
    end = time.time() + STORE_TIMEOUT
    while True:
        try:
            return Coordinator(tcp_store(plan.master_addr, plan.coord_port, is_master, timeout=STORE_TIMEOUT))
        except Exception as exc:
            if time.time() >= end:
                raise
            log(f"coordination store not reachable yet ({exc!r}), retrying")
            time.sleep(10)


def wait_key(coord: Coordinator, keys: list[str], deadline: float, log) -> tuple[str, object] | None:
    """``wait_any`` that gives up when node 0's store stays unreachable (node 0 is gone)."""
    failing_since: list[float | None] = [None]

    def abort() -> bool:
        try:
            coord.has("done")
            failing_since[0] = None
        except Exception:
            failing_since[0] = failing_since[0] or time.time()
        if failing_since[0] is not None and time.time() - failing_since[0] > UNREACHABLE:
            log("node 0's coordination store has been unreachable for 30 min: giving up")
            return True
        return False

    return coord.wait_any(keys, deadline, abort=abort, clock=time.time)


# Followers


def follow(plan: Plan, cfg, log) -> None:
    ip = node_ip(plan.master_addr)
    coord = connect(plan, is_master=False, log=log)
    coord.set(f"node/{plan.node}", {"ip": ip, "host": socket.gethostname()})
    server = NodeServer(plan, cfg, ip, log)
    ckpts = {c.name: c for c in plan.checkpoints}
    end = math.inf if plan.deadline is None else plan.deadline + 3600
    i = 0
    try:
        while True:
            got = wait_key(coord, [f"round/{i}", "done"], end, log)
            if got is None or got[0] == "done":
                return
            name = got[1]["assign"].get(str(plan.node))
            if name is not None:
                try:
                    urls = server.serve(ckpts[name], time.time() + cfg.serving.startup_timeout_seconds)
                    coord.set(f"ready/{i}/{plan.node}", {"ok": True, "urls": urls})
                except Exception as exc:
                    log(f"round {i}: serving {name} failed: {exc}")
                    coord.set(f"ready/{i}/{plan.node}", {"ok": False, "error": str(exc)[:2000]})
            got = wait_key(coord, [f"stop/{i}", "done"], end, log)
            server.stop()
            try:
                coord.set(f"stopped/{i}/{plan.node}", {})
            except Exception:
                pass
            if got is None or got[0] == "done":
                return
            i += 1
    finally:
        server.stop()


# Node 0


class Driver:
    def __init__(self, plan: Plan, cfg, log) -> None:
        self.plan, self.cfg, self.log = plan, cfg, log
        self.work = Path(plan.work_dir)
        self.eval_dir = Path(plan.eval_dir)
        self.bundle = Path(cfg.harness.bundle_dir)
        self.status: dict = {
            "started": plan.started,
            "deadline": plan.deadline,
            "cap_hours": cfg.budget_hours,
            "rounds": plan.rounds,
            "aliases": plan.aliases,
            "skipped": plan.skipped,
            "checkpoints": {},
            "harness": None,
            "heartbeat": None,
        }
        self.lock = threading.Lock()
        self.henv: harness.HarnessEnv | None = None
        """Set only once the harness is completely ready (venv, caches with their modes, kernel bundle)."""
        self.harness_error: str | None = None
        self.kernel_rewrites: dict[str, str] = {}
        # Heartbeat: what is running (checkpoint -> its flow stores), what landed, when anything last moved.
        self.phase = "starting"
        self.active_stores: dict[str, dict[str, Path]] = {}
        self.planned: dict[str, int] = {}
        self.progress: dict[str, dict[str, int]] = {}
        self.last_progress = time.time()
        self._last_counts: tuple | None = None
        self._beat_stop = threading.Event()
        self._beat_lock = threading.Lock()  # beats come from the heartbeat thread and the checkpoints' threads

    def write_status(self) -> None:
        """``<eval dir>/status.json``: where the eval is (node 0's staging; synced with the run's other files)."""
        try:
            with self.lock:
                text = json.dumps(self.status, indent=2, default=str)
                self.eval_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.eval_dir / "status.json.tmp"
                tmp.write_text(text)
                os.replace(tmp, self.eval_dir / "status.json")
        except Exception as exc:
            self.log(f"could not write status.json: {exc!r}")

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.last_progress = time.time()

    def beat(self) -> None:
        """Refresh ``status.json``'s heartbeat: now, the phase, the rollouts landed so far (per checkpoint and in
        all) and when anything last moved (a row landed, a phase changed)."""
        with self._beat_lock:
            self._beat()

    def _beat(self) -> None:
        now = time.time()
        with self.lock:
            active = {name: dict(stores) for name, stores in self.active_stores.items()}
        for name, stores in active.items():
            counts = {"rollouts_done": 0, "rows": 0}
            for store in stores.values():
                try:
                    got = scoring.progress(store)
                except Exception:
                    continue
                for key in counts:
                    counts[key] += got[key]
            self.progress[name] = {**counts, "rollouts_planned": self.planned.get(name)}
        done = sum(c["rollouts_done"] for c in self.progress.values())
        rows = sum(c["rows"] for c in self.progress.values())
        if self._last_counts != (done, rows):
            self._last_counts = (done, rows)
            if done or rows:
                self.last_progress = now
        self.status["heartbeat"] = {
            "time": now,
            "phase": self.phase,
            "last_progress": self.last_progress,
            "seconds_since_progress": round(now - self.last_progress),
            "rollouts_done": done,
            "rows_landed": rows,
            "rollouts_planned": sum(self.planned.values()) or None,
            "per_checkpoint": {name: dict(c) for name, c in self.progress.items()},
            "interval_seconds": HEARTBEAT,
        }
        self.write_status()
        # status.json only reaches the volume after the run; the log is visible live.
        hb = self.status["heartbeat"]
        self.log(
            f"heartbeat: phase {hb['phase']}, rollouts {hb['rollouts_done']}/{hb['rollouts_planned']}, "
            f"rows {hb['rows_landed']}, {hb['seconds_since_progress']} s since progress"
        )

    def heartbeat(self) -> None:
        while not self._beat_stop.wait(HEARTBEAT):
            try:
                self.beat()
            except Exception as exc:
                self.log(f"heartbeat failed: {exc!r}")

    def prepare_harness(self) -> None:
        """The harness venv, task caches and kernel bundle (in a thread, while the first engines load). ``henv`` is
        set only when all of it is ready."""
        try:
            problems = harness.check_bundle(self.cfg.harness, self.bundle)
            if problems:
                raise RuntimeError(f"eval bundle {self.bundle}: {problems[:10]}")
            henv = harness.build_env(self.cfg.harness, self.bundle, self.work / "harness", self.log)
            caches = []
            for label in self.cfg.harness.sources:
                fields = flowcfg.taskset_fields((self.bundle / "flow-configs" / f"{label}.toml").read_bytes())
                if label == "tb-sci" and self.cfg.harness.tb_sci_dataset:
                    fields["dataset"] = self.cfg.harness.tb_sci_dataset
                caches.append(harbor_cache_name(fields["dataset"], fields["repo"], fields["registry_path"]))
            problems = harness.stage_harbor_caches(self.bundle, Path(henv.env["HOME"]), sorted(set(caches)), self.log)
            if problems:
                raise RuntimeError(f"Harbor task caches: {problems}")
            bundles = {
                b
                for label in self.cfg.harness.sources
                for b in flowcfg.kernel_bundles((self.bundle / "flow-configs" / f"{label}.toml").read_bytes())
            }
            for target in sorted(bundles):
                self.kernel_rewrites.update(harness.stage_kernel_bundle(self.bundle, target, self.work, self.log))
            self.status["harness"] = {"ok": True, **henv.info, "kernel_rewrites": self.kernel_rewrites}
            self.henv = henv
            self.log(f"harness ready: {henv.info.get('packages')}")
        except Exception as exc:
            self.harness_error = f"{exc}"
            self.status["harness"] = {"ok": False, "error": str(exc)[:4000]}
            self.log(f"harness setup FAILED: {exc}\n{traceback.format_exc()}")

    def run(self) -> None:
        plan, cfg = self.plan, self.cfg
        ip = node_ip(plan.master_addr)
        coord = connect(plan, is_master=True, log=self.log)
        coord.set("node/0", {"ip": ip, "host": socket.gethostname()})
        for name in plan.skipped:
            self.status["checkpoints"][name] = {"status": "skipped", "reason": plan.skipped[name]}
        creds = harness.credentials(cfg.harness, dict(os.environ), plan.prime_team_id)
        self.status["credentials"] = {k: v for k, v in creds.items() if k != "problem"}
        if creds["problem"]:
            # Nothing can run without sandboxes: fail the eval phase now instead of loading vLLM for nothing.
            self.fail_before_serving(coord, "no_credentials", creds["problem"])
            return
        self.log(
            f"credentials: PRIME_API_KEY present, PRIME_TEAM_ID {creds['prime_team_id']} "
            f"({creds['prime_team_id_source']})"
        )
        if cfg.harness.credential_probe:
            self.set_phase("credential probe")
            probe = harness.probe_credentials(
                os.environ[harness.PRIME_API_KEY], creds["prime_team_id"], dict(os.environ)
            )
            self.status["credentials"]["probe"] = probe
            if probe["status"] in ("no_credentials", "forbidden"):
                self.fail_before_serving(coord, probe["status"], probe["error"])
                return
            if probe["status"] != "ok":
                self.log(f"credentials: {probe['error']}")
            else:
                self.log(f"credentials: the Prime API accepts PRIME_API_KEY for team {creds['prime_team_id']}")
        self.write_status()
        beat = threading.Thread(target=self.heartbeat, name="heartbeat", daemon=True)
        beat.start()
        self.set_phase("harness prep")
        prep = threading.Thread(target=self.prepare_harness, name="harness-prep", daemon=True)
        prep.start()
        server = NodeServer(plan, cfg, ip, self.log)
        ckpts = {c.name: c for c in plan.checkpoints}
        reserve = cfg.reserve_minutes * 60
        try:
            for i, assignment in enumerate(plan.rounds):
                now = time.time()
                end = round_deadline(now, plan.deadline, reserve, len(plan.rounds) - i)
                until = "no cap: until the flow runs finish" if end is None else f"capped at {time.ctime(end)}"
                self.log(f"round {i}: {assignment}, harness {until}")
                self.set_phase(f"round {i}: serving {sorted(assignment)}")
                by_node = {str(node): name for name, nodes in assignment.items() for node in nodes}
                coord.set(f"round/{i}", {"assign": by_node})
                ready: dict[str, dict] = {}
                if "0" in by_node:
                    try:
                        urls = server.serve(ckpts[by_node["0"]], time.time() + cfg.serving.startup_timeout_seconds)
                        ready["0"] = {"ok": True, "urls": urls}
                    except Exception as exc:
                        self.log(f"round {i}: serving {by_node['0']} on node 0 failed: {exc}")
                        ready["0"] = {"ok": False, "error": str(exc)[:2000]}
                others = [n for n in by_node if n != "0"]
                found = coord.wait_all(
                    [f"ready/{i}/{n}" for n in others],
                    time.time() + cfg.serving.startup_timeout_seconds + 1800,
                    clock=time.time,
                )
                for n in others:
                    ready[n] = found.get(f"ready/{i}/{n}") or {"ok": False, "error": "no answer"}
                limit = HARNESS_PREP if end is None else max(0.0, end - time.time())
                prep.join(timeout=limit)
                if self.henv is None:
                    # No harness: every checkpoint left fails the same way; stop serving instead of reloading.
                    if prep.is_alive():
                        self.harness_error = (
                            f"harness prep timed out: not ready after {(time.time() - plan.started) / 3600:.1f} h "
                            f"(limit {limit / 3600:.1f} h for this round)"
                        )
                        self.status["harness"] = {"ok": False, "error": self.harness_error}
                        self.log(f"eval phase FAILED: {self.harness_error}")
                        reason = "harness_prep_timeout"
                    else:
                        reason = "harness_failed"
                    self.status["eval"] = {"status": reason, "error": self.harness_error}
                    for later in plan.rounds[i:]:
                        for name in later:
                            self.status["checkpoints"][name] = {
                                "status": "failed",
                                "reason": reason,
                                "error": f"no harness: {self.harness_error}",
                            }
                    coord.set(f"stop/{i}", {})
                    break
                self.set_phase(f"round {i}: harness {sorted(assignment)}")
                threads = []
                for slot, (name, nodes) in enumerate(assignment.items()):
                    urls = [u for n in nodes for u in (ready[str(n)].get("urls") or []) if ready[str(n)].get("ok")]
                    failed = {str(n): ready[str(n)].get("error") for n in nodes if not ready[str(n)].get("ok")}
                    thread = threading.Thread(
                        target=self.evaluate, args=(ckpts[name], slot, urls, failed, end), name=f"eval-{name}"
                    )
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
                coord.set(f"stop/{i}", {})
                server.stop()
                coord.wait_all([f"stopped/{i}/{n}" for n in others], time.time() + STOP_ACK, clock=time.time)
        finally:
            server.stop()
            self.finish(coord)

    def fail_before_serving(self, coord: Coordinator, reason: str, error: str) -> None:
        """End the eval phase before anything is served (no credentials, refused credentials)."""
        self.log(f"eval phase FAILED before serving ({reason}): {error}")
        self.status["eval"] = {"status": reason, "error": error}
        for ckpt in self.plan.checkpoints:
            self.status["checkpoints"][ckpt.name] = {
                "status": "failed",
                "reason": reason,
                "error": error,
                "step": ckpt.step,
                "val_loss": ckpt.val_loss,
            }
        self.finish(coord)

    def evaluate(
        self, ckpt: Checkpoint, slot: int, urls: list[str], failed_nodes: dict, deadline: float | None
    ) -> None:
        """One checkpoint: router, harness, scoring, results. Records failures in the status."""
        entry = self.status["checkpoints"].setdefault(ckpt.name, {})
        entry.update(status="running", step=ckpt.step, val_loss=ckpt.val_loss, engines=len(urls))
        if failed_nodes:
            entry["failed_nodes"] = failed_nodes
        self.write_status()
        router = None
        try:
            if not urls:
                raise RuntimeError(f"no node serves {ckpt.name}: {failed_nodes}")
            if self.henv is None:
                raise RuntimeError(f"no harness: {self.harness_error or 'harness prep timed out'}")
            port = self.cfg.serving.router_port + slot
            router_bin = shutil.which("vllm-router") or str(Path(sys.executable).parent / "vllm-router")
            router = serving.start(
                f"router-{ckpt.name}",
                serving.router_cmd(self.cfg.serving, router_bin, port, urls),
                serving.serving_env(dict(os.environ)),
                self.work / "logs" / f"router-{ckpt.name}.log",
            )
            base_url = f"http://127.0.0.1:{port}"
            end = time.time() + 900
            while not serving.serves_model(base_url, self.cfg.serving.served_model_name):
                if not router.alive() or time.time() > end:
                    raise RuntimeError(f"router for {ckpt.name} not ready:\n{serving.tail(router.log)}")
                time.sleep(5)
            ok, detail = serving.probe_chat(base_url, self.cfg.serving.served_model_name)
            if not ok:
                raise RuntimeError(f"probe request through the router failed: {detail}")
            self.log(f"{ckpt.name}: router on :{port} over {len(urls)} engines, probe ok ({detail})")
            result = self.run_harness(ckpt, f"{base_url}/v1", len(urls), deadline)
            entry.update(status="done", **result)
        except Exception as exc:
            entry.update(status="failed", error=str(exc)[:4000])
            self.log(f"{ckpt.name}: eval FAILED: {exc}\n{traceback.format_exc()}")
        finally:
            if router is not None:
                serving.stop([router], grace=30)
            self.write_status()

    def run_harness(self, ckpt: Checkpoint, base_url: str, engines: int, deadline: float | None) -> dict:
        cfg, h = self.cfg, self.cfg.harness
        out = self.eval_dir / ckpt.name
        (out / "configs").mkdir(parents=True, exist_ok=True)
        table = parse_task_table((self.bundle / harness.TASKS).read_text())
        restrict = read_names(h.tasks_file) if h.tasks_file is not None else None
        selection = select_tasks(table, list(h.sources), restrict)
        total = sum(len(names) for names in selection.values())
        if h.expected_tasks is not None and total != h.expected_tasks:
            counts = {label: len(names) for label, names in selection.items()}
            raise RuntimeError(f"{total} tasks selected, expected {h.expected_tasks}: {counts}")
        gate_max = h.max_concurrent_rollouts or engines * h.rollouts_per_engine
        root = self.work / "runs" / ckpt.name
        env = harness.run_env(h, self.henv, dict(os.environ), self.plan.prime_team_id)
        procs, changed, stores, ks = {}, {}, {}, {}
        import tomli_w

        for label, names in selection.items():
            source = (self.bundle / "flow-configs" / f"{label}.toml").read_bytes()
            config, keys = flowcfg.derive(
                source,
                h.source_config_sha256[label],
                label=label,
                checkpoint=ckpt.name,
                model=cfg.serving.served_model_name,
                base_url=base_url,
                api_key_var=h.api_key_var,
                root=str(root),
                gate_name=f"tb4-eval-{ckpt.name}",
                gate_dir=str(self.work / "gates"),
                gate_max=gate_max,
                dataset=h.tb_sci_dataset if label == "tb-sci" else None,
                k=h.k,
                include_names=names if restrict is not None else None,
                path_rewrites=self.kernel_rewrites,
                d5_exclude_newer=h.d5_exclude_newer,
                in_box_uv_version=h.in_box_uv_version if h.d5_exclude_newer is not None else None,
            )
            path = self.work / "configs" / f"{ckpt.name}-{label}.toml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tomli_w.dumps(config))
            shutil.copyfile(path, out / "configs" / f"{label}.toml")
            changed[label] = keys
            stores[label] = Path(config["flow"]["store"])
            ks[label] = h.k or flowcfg.rollouts_k(source)
            self.planned[ckpt.name] = self.planned.get(ckpt.name, 0) + len(names) * ks[label]
            procs[label] = serving.start(
                f"flow-{label}",
                harness.flow_cmd(h, self.henv.venv, path),
                env,
                self.work / "logs" / f"flow-{ckpt.name}-{label}.log",
                cwd=self.henv.cwd,
            )
        self.log(
            f"{ckpt.name}: flow runs {list(procs)} started ({total} tasks, gate {gate_max} rollouts) against {base_url}"
        )
        started = time.time()
        with self.lock:
            self.active_stores[ckpt.name] = dict(stores)
        try:
            runs = harness.wait_flow_runs(procs, deadline, self.log)
        finally:
            finished = time.time()
            try:
                self.beat()  # the checkpoint's final counts
            except Exception as exc:
                self.log(f"heartbeat failed: {exc!r}")
            with self.lock:
                self.active_stores.pop(ckpt.name, None)

        scores: list[scoring.TaskScore] = []
        diagnostics = {}
        by_name = {row.name: row for row in table}
        for label, names in selection.items():
            rows = [by_name[name] for name in names]
            run = runs.get(label, {})
            label_scores, diag = scoring.score_store(
                stores[label],
                rows,
                label,
                ks[label],
                cfg.pass_threshold,
                cut=bool(run.get("stopped")),
                stopped_at=run.get("stopped_at"),
            )
            scores += label_scores
            diagnostics[label] = diag
        summary = scoring.summary_by_group(scores)
        with open(out / "per_task.jsonl", "w") as f:
            for score in scores:
                f.write(json.dumps(score.to_dict()) + "\n")
        (out / "logs").mkdir(exist_ok=True)
        for label in procs:
            log = self.work / "logs" / f"flow-{ckpt.name}-{label}.log"
            if log.is_file():
                shutil.copyfile(log, out / "logs" / log.name)
        result = {
            "checkpoint": ckpt.name,
            "step": ckpt.step,
            "val_loss": ckpt.val_loss,
            "model_dir": ckpt.model_dir,
            "served_model_name": cfg.serving.served_model_name,
            "engines": engines,
            "rollout_gate": gate_max,
            "tasks": {label: len(names) for label, names in selection.items()},
            "split_tasks": {
                split: sum(by_name[name].split == split for names in selection.values() for name in names)
                for split in ("val", "train", "other")
            },
            "tasks_tsv_sha256": harness.sha256_file(self.bundle / harness.TASKS),
            "summary": summary,
            "flow_runs": runs,
            "complete": not any(r.get("stopped") for r in runs.values()) and summary["overall"].get("missing", 0) == 0,
            "coverage": summary["overall"].get("coverage"),
            "cap": {
                "budget_hours": cfg.budget_hours,
                "round_deadline": deadline,
                "stopped_sources": sorted(label for label, r in runs.items() if r.get("stopped")),
            },
            "d5": {"exclude_newer": h.d5_exclude_newer, "in_box_uv_version": h.in_box_uv_version}
            if h.d5_exclude_newer is not None
            else None,
            "diagnostics": diagnostics,
            "changed_config_keys": changed,
            "harness": self.status.get("harness"),
            "serving": self.inference_config_record(),
            "timing": {"harness_started": started, "harness_finished": finished},
        }
        (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
        overall = summary["overall"]
        headline = {
            key: overall.get(key)
            for key in overall
            if (key.startswith("pass@") and not key.endswith("_tasks")) or key in ("tasks", "errored", "missing", "cut")
        }
        self.log(f"{ckpt.name}: {json.dumps(headline)} -> {out}")
        if cfg.archive_stores:
            self.archive(ckpt, stores, out)
        return {
            "summary": str(out / "summary.json"),
            "overall": overall,
            "val": summary["split"].get("val"),
            "train": summary["split"].get("train"),
            "complete": result["complete"],
            "coverage": result["coverage"],
        }

    def inference_config_record(self) -> dict:
        return {
            "vllm": "0.32.0 (image)",
            "tensor_parallel_size": self.cfg.serving.tensor_parallel_size,
            "data_parallel_size": self.cfg.serving.data_parallel_size,
            "config": self.cfg.serving.model_dump(mode="json"),
        }

    def archive(self, ckpt: Checkpoint, stores: dict[str, Path], out: Path) -> None:
        """Each store as a compressed tar next to the results, bounded by the time left of the budget."""
        (out / "stores").mkdir(exist_ok=True)
        zstd = shutil.which("zstd")
        for label, store in stores.items():
            left = None if self.plan.deadline is None else self.plan.deadline - time.time() - 600
            if (left is not None and left <= 0) or not store.is_dir():
                self.log(
                    f"{ckpt.name}/{label}: store not archived ({'missing' if not store.is_dir() else 'no time left'})"
                )
                continue
            target = out / "stores" / f"{label}.tar.{'zst' if zstd else 'gz'}"
            cmd = ["tar", "-C", str(store.parent), "-cf", str(target), store.name]
            cmd[3:3] = ["--use-compress-program", "zstd -T0 -3"] if zstd else ["-z"]
            try:
                subprocess.run(cmd, check=True, timeout=left, capture_output=True)
                self.log(f"{ckpt.name}/{label}: store archived to {target} ({target.stat().st_size / 2**30:.1f} GiB)")
            except Exception as exc:
                target.unlink(missing_ok=True)
                self.log(f"{ckpt.name}/{label}: archiving the store failed: {exc!r}")

    def finish(self, coord: Coordinator) -> None:
        self._beat_stop.set()
        self.phase = "finished"
        try:
            self.beat()
        except Exception as exc:
            self.log(f"heartbeat failed: {exc!r}")
        for alias, target in self.plan.aliases.items():
            entry = self.status["checkpoints"].get(target, {})
            self.status["checkpoints"][alias] = {"status": "alias", "same_as": target, "result": entry.get("status")}
            try:
                (self.eval_dir / alias).mkdir(parents=True, exist_ok=True)
                (self.eval_dir / alias / "SAME_AS.json").write_text(
                    json.dumps({"same_as": target, "results": str(self.eval_dir / target)}) + "\n"
                )
            except Exception:
                pass
        if "eval" not in self.status:
            ok = all(e.get("status") in ("done", "alias", "skipped") for e in self.status["checkpoints"].values())
            self.status["eval"] = {"status": "done" if ok else "failed", "error": None}
        self.status["finished"] = time.time()
        self.write_status()
        try:
            keys = (
                "status",
                "step",
                "val_loss",
                "overall",
                "val",
                "train",
                "complete",
                "coverage",
                "error",
                "same_as",
                "reason",
            )
            summary = {
                name: {k: v for k, v in entry.items() if k in keys}
                for name, entry in self.status["checkpoints"].items()
            }
            (self.eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        except Exception as exc:
            self.log(f"could not write summary.json: {exc!r}")
        try:
            ok = all(e.get("status") in ("done", "alias", "skipped") for e in self.status["checkpoints"].values())
            coord.set("done", {"ok": ok})
            others = [f"bye/{n}" for n in range(1, self.plan.num_nodes)]
            # Exit last: the other nodes clean their staging first.
            coord.wait_all(others, time.time() + BYE, clock=time.time)
        except Exception as exc:
            self.log(f"could not signal the other nodes: {exc!r}")


# Cleanup (every node)


def cleanup(plan: Plan, cfg, log) -> None:
    """Drop what must not reach the volume from this node's staging: the duplicate copies of the exports (keep this
    node's upload share; the master keeps the small files), verified exports when configured, the work dir."""
    for name, result in plan.exports.items():
        directory = result.get("dir")
        if not directory or not Path(directory).is_dir():
            continue
        directory = Path(directory)
        if plan.delete_pushed_exports and result.get("verified"):
            shutil.rmtree(directory, ignore_errors=True)
            log(f"deleted the verified export {directory} from this node's staging")
            continue
        if not plan.dedupe_exports:
            continue
        keep = set(plan.node_files)
        removed = 0
        for path in directory.iterdir():
            duplicate = path.suffix == ".safetensors" and path.name not in keep
            small_copy = plan.node != 0 and path.name in plan.master_files
            if duplicate or small_copy:
                path.unlink(missing_ok=True)
                removed += 1
        log(f"{directory}: removed {removed} files other nodes upload")
    if not cfg.keep_work_dir:
        shutil.rmtree(plan.work_dir, ignore_errors=True)


def main() -> None:
    from prime_rl.configs.sft import SFTInRunEvalConfig
    from prime_rl.utils.logger import setup_logger

    plan = Plan.from_json(Path(sys.argv[1]).read_text())
    cfg = SFTInRunEvalConfig.model_validate(plan.config)
    logger = setup_logger("info")

    def log(message: str) -> None:
        logger.info(f"[in-run eval, node {plan.node}] {message}")

    cap = "no cap" if plan.deadline is None else f"safety cap {time.ctime(plan.deadline)}"
    log(f"started (pid {os.getpid()}), {cap}")
    try:
        if plan.node == 0:
            driver = Driver(plan, cfg, log)
            driver.run()
        else:
            follow(plan, cfg, log)
    except BaseException as exc:
        log(f"eval process failed: {exc!r}\n{traceback.format_exc()}")
    finally:
        try:
            cleanup(plan, cfg, log)
        except Exception as exc:
            log(f"cleanup failed: {exc!r}")
        if plan.node != 0:
            try:
                Coordinator(tcp_store(plan.master_addr, plan.coord_port, False, timeout=60)).set(f"bye/{plan.node}", {})
            except Exception:
                pass
        log("done, exiting 0")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
