"""Run job-scoped Grafana/Prometheus for a multi-node Prime-RL Slurm run."""

import argparse
import json
import os
import re
import signal
import subprocess
import threading
from contextlib import ExitStack
from pathlib import Path

DEFAULT_BINARIES = Path.home() / ".local/share/glm-monitoring"
DASHBOARD = Path(__file__).resolve().parents[1] / "monitoring/inference-dashboard.json"


def output(*command):
    return subprocess.check_output(command, text=True, timeout=30).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def discover(run_dir, job_id):
    script_path = run_dir / "launcher/rl.sbatch"
    job = output("scontrol", "show", "job", "-o", str(job_id))
    fields = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", job))
    if fields.get("JobState") != "RUNNING":
        raise ValueError(f"Job {job_id} must be running; state={fields.get('JobState')}")
    if Path(fields["Command"]).resolve() != script_path:
        raise ValueError("The job's launch script does not match this run directory")
    allocated_hosts = set(output("scontrol", "show", "hostnames", fields["NodeList"]).splitlines())
    with Path(fields["StdOut"]).open() as log:
        startup_log = log.read(65536)
    match = re.search(r"^INFER_HOSTS=(.+)$", startup_log, re.M)
    if match is None:
        raise ValueError("Inference host assignment is not in the job log yet; retry after launch")
    # Slurm's displayed NodeList can differ from the launcher's rank order.
    hosts = match[1].split()
    settings = {key: int(value) for key, value in re.findall(r"^export (\w+)=(\d+)$", script_path.read_text(), re.M)}
    nodes = settings["NUM_INFER_NODES"]
    gpus = settings["GPUS_PER_NODE"]
    tp = settings["INFERENCE_TP"]
    if not nodes or gpus % tp:
        raise ValueError("Expected inference nodes and an integral number of local DP ranks")
    if len(hosts) != nodes or not set(hosts) <= allocated_hosts:
        raise ValueError("Logged inference hosts do not match the allocation/configuration")
    span = settings["NODES_PER_INFER_REPLICA"]
    roles = {}
    for index, host in enumerate(hosts[:nodes]):
        if "NUM_PREFILL_NODES" in settings:
            role = "prefill" if index % span < settings["NUM_PREFILL_NODES"] else "decode"
            port = settings[f"{role.upper()}_PORT"]
        else:
            role, port = "unified", settings["BACKEND_PORT"]
        entry = roles.setdefault(role, {"gpus": 0, "endpoints": []})
        entry["gpus"] += gpus
        entry["endpoints"].extend(f"{host}:{port + rank}" for rank in range(gpus // tp))
    return {
        "name": f"{run_dir.name}-{job_id}",
        "job_id": str(job_id),
        "roles": roles,
        "router_host": hosts[0],
        "router_port": settings["ROUTER_PORT"] + 21000,
    }


def prepare(root, spec, grafana_port, prometheus_port, router_port):
    for directory in (
        "logs",
        "dashboards",
        "data/grafana",
        "data/prometheus",
        "provisioning/datasources",
        "provisioning/dashboards",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    write_json(root / "deployment.json", spec)
    targets, rules = [], []
    for role, config in spec["roles"].items():
        labels = {"deployment": spec["name"], "role": role, "job_id": spec["job_id"]}
        for rank, address in enumerate(config["endpoints"]):
            targets.append(
                {"targets": [address], "labels": labels | {"rank": str(rank), "node": address.rsplit(":", 1)[0]}}
            )
        for metric, value in (("gpus", config["gpus"]), ("endpoints", len(config["endpoints"]))):
            rules.append({"record": f"inference_deployment_{metric}", "expr": f"vector({value})", "labels": labels})
    targets.append(
        {
            "targets": [f"127.0.0.1:{router_port}"],
            "labels": {
                "deployment": spec["name"],
                "role": "router",
                "job_id": spec["job_id"],
                "node": spec["router_host"],
            },
        }
    )
    write_json(root / "targets.json", targets)
    write_json(root / "inventory-rules.json", {"groups": [{"name": "allocation", "rules": rules}]})
    write_json(
        root / "prometheus.json",
        {
            "global": {"scrape_interval": "15s", "evaluation_interval": "15s", "scrape_timeout": "10s"},
            "rule_files": [str(root / "inventory-rules.json")],
            "scrape_configs": [
                {
                    "job_name": "inference",
                    "file_sd_configs": [{"files": [str(root / "targets.json")]}],
                    "metric_relabel_configs": [
                        {"source_labels": ["__name__"], "regex": ".*_created", "action": "drop"}
                    ],
                }
            ],
        },
    )
    write_json(
        root / "provisioning/datasources/prometheus.yaml",
        {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Inference Prometheus",
                    "uid": "inference-prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": f"http://127.0.0.1:{prometheus_port}",
                    "isDefault": True,
                    "editable": False,
                    "jsonData": {"timeInterval": "15s"},
                }
            ],
        },
    )
    write_json(
        root / "provisioning/dashboards/inference.yaml",
        {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "Inference",
                    "type": "file",
                    "updateIntervalSeconds": 15,
                    "options": {"path": str(root / "dashboards")},
                }
            ],
        },
    )
    (root / "dashboards/inference.json").write_bytes(DASHBOARD.read_bytes())
    (root / "grafana.ini").write_text(f"""[paths]
data = {root}/data/grafana
logs = {root}/logs
plugins = {root}/plugins
provisioning = {root}/provisioning
[server]
http_addr = 127.0.0.1
http_port = {grafana_port}
[auth.anonymous]
enabled = true
org_role = Viewer
[auth]
disable_login_form = true
[users]
allow_sign_up = false
[analytics]
reporting_enabled = false
check_for_updates = false
[news]
news_feed_enabled = false
[dashboards]
min_refresh_interval = 15s
default_home_dashboard_path = {root}/dashboards/inference.json
""")


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("job_id", type=int)
    parser.add_argument(
        "--binaries",
        type=Path,
        default=DEFAULT_BINARIES,
        help="Directory containing bin/prometheus, prometheus/promtool and grafana/bin/grafana",
    )
    parser.add_argument("--grafana-port", type=int, default=3000)
    parser.add_argument("--prometheus-port", type=int, default=19090)
    parser.add_argument("--router-port", type=int, default=19102, help="Local SSH tunnel port")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Discover targets and validate configs without starting services"
    )
    args = parser.parse_args()
    ports = (args.grafana_port, args.prometheus_port, args.router_port)
    if len(set(ports)) != 3 or any(not 1024 <= port <= 65535 for port in ports):
        parser.error("Use three distinct ports in 1024..65535")
    run_dir = args.run_dir.resolve()
    root = run_dir / "monitoring" / f"job_{args.job_id}"
    binaries = args.binaries.resolve()
    spec = discover(run_dir, args.job_id)
    prepare(root, spec, *ports)
    subprocess.run(
        [str(binaries / "prometheus/promtool"), "check", "config", str(root / "prometheus.json")], check=True
    )
    print(
        f"Discovered: {[(role, len(config['endpoints']), config['gpus']) for role, config in spec['roles'].items()]}",
        flush=True,
    )
    if args.prepare_only:
        print(f"Prepared {root}; no services started.")
        return
    commands = {
        "router-tunnel": [
            "ssh",
            "-NT",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:{args.router_port}:127.0.0.1:{spec['router_port']}",
            spec["router_host"],
        ],
        "prometheus": [
            str(binaries / "bin/prometheus"),
            f"--config.file={root}/prometheus.json",
            f"--storage.tsdb.path={root}/data/prometheus",
            "--storage.tsdb.retention.time=7d",
            "--storage.tsdb.retention.size=5GB",
            f"--web.listen-address=127.0.0.1:{args.prometheus_port}",
        ],
        "grafana": [
            str(binaries / "grafana/bin/grafana"),
            "server",
            f"--homepath={binaries}/grafana",
            f"--config={root}/grafana.ini",
        ],
    }
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    with ExitStack() as stack:
        processes = {}
        for name, command in commands.items():
            log = stack.enter_context((root / f"logs/{name}.log").open("a"))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stack.callback(stop, process)
            processes[name] = process
        print(f"Grafana: http://localhost:{args.grafana_port}/d/inference-live", flush=True)
        print(f"On your laptop: ssh -N -L {args.grafana_port}:127.0.0.1:{args.grafana_port} nebius", flush=True)
        print(f"Logs: {root}/logs. Stops with job {args.job_id} or Ctrl-C; does not control the RL job.", flush=True)
        while not stopped.wait(15):
            states = output("squeue", "--noheader", "--jobs", str(args.job_id), "--format=%T")
            if not states or not any(state in {"RUNNING", "COMPLETING", "SUSPENDED"} for state in states.splitlines()):
                print("Job ended; stopping monitoring.", flush=True)
                break
            for name, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited ({process.returncode}); see {root}/logs/{name}.log")


if __name__ == "__main__":
    main()
