import os
import sys

from sglang.launch_server import run_server
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

from prime_rl_sglang.patches import apply_patches, run_scheduler_process_with_prime_rl_patches

suppress_noisy_warnings()


if __name__ == "__main__":
    apply_patches()
    server_args = prepare_server_args(sys.argv[1:])

    try:
        if server_args.grpc_mode or server_args.encoder_only:
            run_server(server_args)
        else:
            launch_server(server_args, run_scheduler_process_func=run_scheduler_process_with_prime_rl_patches)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
