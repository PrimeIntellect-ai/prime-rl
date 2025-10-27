import shutil
import sys
from pathlib import Path

import uvloop

from prime_rl.inference.config import InferenceConfig
from prime_rl.inference.vllm.patch import apply_patches
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

# Ensure .pth file is installed in site-packages (for editable installs and child workers)
_pth_installed = False
for _site_path in sys.path:
    _site_path = Path(_site_path)
    if _site_path.name == "site-packages" and _site_path.exists():
        _pth_dest = _site_path / "prime_rl_vllm_patch.pth"
        if not _pth_dest.exists():
            # Path(__file__) is src/prime_rl/inference/vllm/server.py, go up to src/prime_rl
            _pth_source = Path(__file__).parent.parent.parent / "data" / "prime_rl_vllm_patch.pth"
            if _pth_source.exists():
                shutil.copy2(_pth_source, _pth_dest)
        _pth_installed = True
        break

# Apply patches for current process (source checkouts / first run)
# Child workers will get patches via .pth file above
apply_patches()


def server(config: InferenceConfig, vllm_args: list[str]):
    """
    Start vLLM API server with custom patches.

    Custom functionality (worker extension, custom endpoints) is injected via
    the prime_rl_vllm_patch.pth file installed in site-packages, which applies
    patches to every Python interpreter (parent and all spawned workers).
    """
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    validate_parsed_serve_args(args)

    # Raise error if logprobs_mode is not set to processed_logprobs
    if args.logprobs_mode != "processed_logprobs":
        raise ValueError("logprobs_mode must be 'processed_logprobs' to be compatible with the orchestrator.")

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            uvloop.run(run_server(args))
