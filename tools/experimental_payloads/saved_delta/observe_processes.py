"""Independent memory observer; excludes the fake environment sender."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    peak_rss = peak_pss = 0
    samples = 0
    with (args.output / "process.log").open("w") as log, (args.output / "memory.jsonl").open("w") as records:
        process = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT)
        parent = psutil.Process(process.pid)
        started = time.perf_counter()
        while process.poll() is None:
            members = []
            try:
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            for member in processes:
                try:
                    if any(Path(argument).name == "sender.py" for argument in member.cmdline()):
                        continue
                    memory = member.memory_full_info()
                    cpu = member.cpu_times()
                    members.append(
                        {
                            "pid": member.pid,
                            "rss_bytes": memory.rss,
                            "pss_bytes": memory.pss,
                            "cpu_seconds": cpu.user + cpu.system,
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            rss = sum(member["rss_bytes"] for member in members)
            pss = sum(member["pss_bytes"] for member in members)
            peak_rss, peak_pss = max(peak_rss, rss), max(peak_pss, pss)
            records.write(
                json.dumps(
                    {
                        "elapsed": time.perf_counter() - started,
                        "members": members,
                        "receiver_tree_rss_bytes": rss,
                        "receiver_tree_pss_bytes": pss,
                    }
                )
                + "\n"
            )
            records.flush()
            samples += 1
            time.sleep(0.25)
        status = process.wait()
    result = {
        "returncode": status,
        "elapsed": time.perf_counter() - started,
        "samples": samples,
        "receiver_tree_peak_rss_bytes": peak_rss,
        "receiver_tree_peak_pss_bytes": peak_pss,
        "limits": "250ms sampled process-tree maxima, including setup; fake sender excluded. "
        "Summed RSS double-counts shared mappings. Summed PSS apportions shared pages.",
    }
    (args.output / "observer.json").write_text(json.dumps(result, indent=2))
    raise SystemExit(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    main(parser.parse_args())
