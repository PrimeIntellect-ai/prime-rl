import subprocess
from subprocess import Popen
from threading import Event, Thread


def cleanup_threads(threads: list[Thread]):
    """Cleanup a list of threads"""
    for thread in threads:
        thread.join(timeout=5)


def cleanup_processes(processes: list[Popen]):
    """Cleanup a list of subprocesses"""
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()


def monitor_process(process: Popen, stop_event: Event, error_queue: list, process_name: str):
    """Monitor a subprocess and signal errors via shared queue."""
    process.wait()

    if process.returncode != 0:
        err_msg = f"{process_name.capitalize()} failed with exit code {process.returncode}"
        if process.stderr:
            err_msg += f"\n{process.stderr.read().decode('utf-8')}"
        error_queue.append(RuntimeError(err_msg))
    stop_event.set()
