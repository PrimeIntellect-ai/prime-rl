import shutil
import sys
import uvloop
from pathlib import Path

from prime_rl.inference.config import InferenceConfig
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

# Ensure .pth file is installed in site-packages (for editable installs)
_pth_installed = False
for _site_path in sys.path:
    _site_path = Path(_site_path)
    if _site_path.name == "site-packages" and _site_path.exists():
        _pth_dest = _site_path / "prime_rl_vllm_patch.pth"
        if not _pth_dest.exists():
            _pth_source = Path(__file__).parent.parent / "data" / "prime_rl_vllm_patch.pth"
            if _pth_source.exists():
                shutil.copy2(_pth_source, _pth_dest)
        _pth_installed = True
        break


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
