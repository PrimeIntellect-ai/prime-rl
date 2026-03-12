import time


def prepare_initial_weight_broadcast(multi_run_manager, world, timeout_seconds: int) -> bool:
    """Mark active runs ready for initial step-0 weight broadcast."""
    deadline = time.perf_counter() + timeout_seconds
    while True:
        if world.is_master:
            multi_run_manager.discover_runs()
            for idx in multi_run_manager.used_idxs:
                multi_run_manager.ready_to_update[idx] = True

        multi_run_manager.synchronize_state()
        if multi_run_manager.ready_to_update_idxs:
            return True

        if time.perf_counter() >= deadline:
            return False
        time.sleep(1)
