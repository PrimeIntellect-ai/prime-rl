import shutil
import site
from pathlib import Path

import uvloop

from prime_rl.inference.config import InferenceConfig
from prime_rl.inference.vllm.patch import apply_patches
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server, run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)

# Ensure .pth file is installed in site-packages (for editable installs and child workers)
# This handles both site-packages and dist-packages across different distributions
_pth_source = Path(__file__).parent / "prime_rl_vllm_patch.pth"
_pth_installed = False

if _pth_source.exists():
    # Get all site-packages directories (includes dist-packages on Debian/Ubuntu)
    for _site_dir in site.getsitepackages():
        _site_path = Path(_site_dir)
        if _site_path.exists():
            _pth_dest = _site_path / "prime_rl_vllm_patch.pth"
            if not _pth_dest.exists():
                try:
                    shutil.copy2(_pth_source, _pth_dest)
                    logger.info("Installed .pth file to %s for multi-API mode support", _pth_dest)
                    _pth_installed = True
                    break
                except (PermissionError, OSError) as e:
                    # Read-only file system or no permissions - continue trying other locations
                    logger.debug("Could not install .pth to %s: %s", _site_path, e)
                    continue
            else:
                _pth_installed = True
                break

if not _pth_installed:
    logger.warning(
        ".pth file not installed in site-packages. Multi-API mode may not work correctly. "
        "For editable installs, manually copy %s to your site-packages directory.",
        _pth_source
    )

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
