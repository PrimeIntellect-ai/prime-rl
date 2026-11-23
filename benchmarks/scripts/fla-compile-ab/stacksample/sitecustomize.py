"""The peaklog hook, plus a thread that appends the main thread's Python stack to $STACK_DIR/rank<R>-<pid>.txt every $STACK_INTERVAL_S seconds."""

import os

exec(open(os.path.join(os.path.dirname(__file__), "..", "peaklog", "sitecustomize.py")).read())

if os.environ.get("STACK_DIR") and os.environ.get("RANK") is not None:
    import sys
    import threading
    import time
    import traceback

    def _sample_main_stack():
        main_id = threading.main_thread().ident
        interval = float(os.environ.get("STACK_INTERVAL_S", "1"))
        os.makedirs(os.environ["STACK_DIR"], exist_ok=True)
        with open(os.path.join(os.environ["STACK_DIR"], f"rank{os.environ['RANK']}-{os.getpid()}.txt"), "a") as handle:
            while True:
                time.sleep(interval)
                frame = sys._current_frames().get(main_id)
                if frame is None:
                    return
                stack = traceback.extract_stack(frame)
                handle.write(f"@ {time.time():.2f}\n")
                for entry in stack[-40:]:
                    handle.write(f"  {entry.filename}:{entry.lineno} {entry.name}\n")
                handle.flush()

    threading.Thread(target=_sample_main_stack, daemon=True).start()
