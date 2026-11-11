"""ledger.jsonl: one JSON line per run, rewritten atomically on update (keyed by run_id)."""
import fcntl
import json
import os
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "ledger.jsonl"


def load():
    runs = {}
    if PATH.exists():
        for line in PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                runs[r["run_id"]] = r
    return runs


def upsert(rec):
    lock = open(str(PATH) + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        runs = load()
        runs.setdefault(rec["run_id"], {}).update(rec)
        tmp = str(PATH) + ".tmp"
        with open(tmp, "w") as f:
            for r in runs.values():
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, PATH)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
