"""Run the full converter chain for one mini_moe arch.

Builds the fixture — a tiny random prime model exported as a source HF dir (the
byte-exactness reference) plus a DCP checkpoint in run-dir layout — then executes
each converter's CLI entrypoint in-process via runpy, so the whole chain pays the
heavy imports once. Building through the prime class keeps the source dir in the
canonical HF layout the converters emit, so comparisons need no canonicalization.

Usage: uv run python tests/converters/run_chain.py <arch> <run_dir>
Writes <run_dir>/source, <run_dir>/checkpoints/step_1/{trainer,weights,weights-FP8},
<run_dir>/weights-FP8-chained, <run_dir>/weights-dequant.
"""

import importlib.util
import json
import os
import runpy
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from transformers import AutoTokenizer

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import save_state_dict

REPO_ROOT = Path(__file__).parents[2]
CONVERTERS_DIR = REPO_ROOT / "tools" / "converters"

DIST_ENV_VARS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")


def load_mini_moe():
    path = REPO_ROOT / "scripts" / "mini_moe.py"
    spec = importlib.util.spec_from_file_location("mini_moe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source_dir(preset, source_dir: Path) -> None:
    """A tiny random prime model exported in canonical HF layout, with tokenizer."""
    config = preset["config_fn"]() if "config_fn" in preset else preset["config_class"](**preset["config_kwargs"])
    config._attn_implementation = "flash_attention_2"
    for subconfig_key in getattr(config, "sub_configs", {}):
        subconfig = getattr(config, subconfig_key, None)
        if subconfig is not None:
            subconfig._attn_implementation = "flash_attention_2"

    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = preset["prime_model_class"]._from_config(config)

    state_dict = model.convert_to_hf(dict(model.state_dict()))
    state_dict = {key: value.contiguous() for key, value in state_dict.items()}
    source_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(source_dir)
    save_state_dict(state_dict, source_dir)
    AutoTokenizer.from_pretrained(preset["tokenizer_source"], trust_remote_code=True).save_pretrained(source_dir)


def write_dcp_checkpoint(source_dir: Path, run_dir: Path) -> None:
    """Load the source dir the way the trainer does and save a DCP checkpoint."""
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
    os.environ.update(
        MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1"
    )
    setup_torch_distributed()

    model_config = ModelConfig(name=str(source_dir), compile=None, ac=None)
    resolve_ep(model_config)
    parallel_dims = get_parallel_dims(model_config)
    model = setup_model(model_config, parallel_dims)
    dcp_save({"app": AppState(model, [], None, None)}, checkpoint_id=run_dir / "checkpoints" / "step_1" / "trainer")
    dist.destroy_process_group()

    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    trainer_config = {"model": {"name": str(source_dir)}, "tokenizer": {}}
    (run_dir / "configs" / "trainer.json").write_text(json.dumps(trainer_config, indent=2) + "\n")


def run_converter(script: str, *args: Path) -> None:
    """Execute a converter's CLI entrypoint in-process."""
    sys.argv = [script, *[str(arg) for arg in args]]
    runpy.run_path(str(CONVERTERS_DIR / script), run_name="__main__")
    # Each converter must pick a fresh rendezvous port for its own process group.
    for key in DIST_ENV_VARS:
        os.environ.pop(key, None)
    torch.cuda.empty_cache()


def main() -> None:
    arch, run_dir = sys.argv[1], Path(sys.argv[2])
    setup_logger("info")
    sys.path.insert(0, str(CONVERTERS_DIR))  # dcp_to_fp8 imports its sibling converters

    preset = load_mini_moe().ARCH_PRESETS[arch]
    source_dir = run_dir / "source"
    write_source_dir(preset, source_dir)
    write_dcp_checkpoint(source_dir, run_dir)
    for key in DIST_ENV_VARS:
        os.environ.pop(key, None)

    step_dir = run_dir / "checkpoints" / "step_1"
    run_converter("dcp_to_bf16.py", step_dir)
    run_converter("dcp_to_fp8.py", step_dir)
    run_converter("bf16_to_fp8.py", step_dir / "weights", run_dir / "weights-FP8-chained")
    run_converter("fp8_to_bf16.py", step_dir / "weights-FP8", run_dir / "weights-dequant")
    print(f"chain complete: {run_dir}")


if __name__ == "__main__":
    main()
