"""The peaklog hook, plus a dump of each rank's NCCL flight recorder buffer to $FR_DUMP_DIR/rank<R>.pkl at exit."""

import os

exec(open(os.path.join(os.path.dirname(__file__), "..", "peaklog", "sitecustomize.py")).read())

if os.environ.get("FR_DUMP_DIR") and os.environ.get("RANK") is not None:
    import atexit

    import torch

    def _dump_flight_recorder():
        os.makedirs(os.environ["FR_DUMP_DIR"], exist_ok=True)
        with open(os.path.join(os.environ["FR_DUMP_DIR"], f"rank{os.environ['RANK']}.pkl"), "wb") as handle:
            handle.write(torch._C._distributed_c10d._dump_nccl_trace())

    atexit.register(_dump_flight_recorder)
