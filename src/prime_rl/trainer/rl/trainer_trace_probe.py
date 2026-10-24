"""Opt-in trainer diagnostics; leaves numerical operations and configuration unchanged."""

from datetime import datetime, timezone
import faulthandler
from functools import wraps
import json
import os
import sys
import threading

import psutil


def run_with_diagnostics(train, config):
    rank = int(os.environ["RANK"])
    process = psutil.Process()
    stopped = threading.Event()

    def report(event, **fields):
        print(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                          "event": event, "rank": rank, **fields}), flush=True)

    def sample_memory():
        while not stopped.wait(60):
            memory = psutil.virtual_memory()
            report("diagnostic_memory", rss_gib=process.memory_info().rss / 1024**3,
                   host_available_gib=memory.available / 1024**3, host_used_pct=memory.percent)

    def traced(function, name):
        @wraps(function)
        def call(*args, **kwargs):
            report("diagnostic_phase_start", phase=name)
            result = function(*args, **kwargs)
            report("diagnostic_phase_end", phase=name)
            return result
        return call

    module = sys.modules[train.__module__]
    originals = {name: getattr(module, name) for name in
                 ("forward", "compute_loss", "begin_backward", "finish_backward")}
    for name, function in originals.items():
        setattr(module, name, traced(function, name))
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET,ENV"
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    observer = threading.Thread(target=sample_memory, daemon=True)
    observer.start()
    report("diagnostic_started", stack_dump_interval_seconds=120, memory_interval_seconds=60)
    try:
        return train(config)
    finally:
        faulthandler.cancel_dump_traceback_later()
        stopped.set()
        observer.join(timeout=2)
        for name, function in originals.items():
            setattr(module, name, function)
