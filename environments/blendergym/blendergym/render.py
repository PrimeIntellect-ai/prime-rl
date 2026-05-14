"""Blender subprocess wrapper.

This module is intentionally a thin, synchronous shell around ``subprocess.run``
so that it stays usable from both the verifiers env loop (Phase 5) and ad-hoc
debugging (``python -m blendergym.render ...``).

Design contract (see plan §"render.py 接口"):

* No ``.blend`` copy — Phase 0 ``chmod -R a-w data/blendergym/`` makes the
  dataset filesystem-level read-only, so all rollouts share the same
  ``--background`` path with zero I/O.
* ``CUDA_VISIBLE_DEVICES`` is injected per-process to pin Blender to the
  rollout's assigned GPU.
* ``BLENDER_USER_RESOURCES`` is redirected to ``output_dir/blender_user`` so
  parallel Blender instances don't fight over ``~/.config/blender/`` user
  prefs locks.
* Failure semantics are split:
    - ``CalledProcessError`` / ``TimeoutExpired`` → ``RenderResult(success=False, ...)``
      (these are model-controllable: bad code / runaway loop).
    - Anything else (missing binary, bad blend path, ENOMEM at fork, ...) is
      allowed to propagate — those are operator errors, not rollout outcomes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib.resources import files as _resource_files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact_manager import TurnPaths

DEFAULT_BLENDER_BIN = Path(
    "_reference_codes/VIGA/utils/third_party/infinigen/blender/blender"
)
DEFAULT_RENDER_SCRIPT_RESOURCE = ("blendergym.assets", "pipeline_render_script.py")
DEFAULT_TIMEOUT_S = 120
RENDER1_FILENAME = "render1.png"
CODE_FILENAME = "code.py"
LOG_FILENAME = "blender.log"


@dataclass
class RenderResult:
    """Outcome of a single Blender invocation."""

    success: bool
    image_paths: list[Path]
    stderr: str
    duration_s: float
    returncode: int | None = None
    code_path: Path | None = None
    log_path: Path | None = None
    timed_out: bool = False
    gpu_id: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


def default_render_script() -> Path:
    """Return the path to the bundled ``pipeline_render_script.py`` asset.

    Resolved via ``importlib.resources`` so editable installs and wheels both
    work; falls back to a sibling-directory lookup for source layouts where
    ``importlib.resources`` cannot resolve the asset (e.g. running directly
    from a checkout without an installed package).
    """
    package, resource = DEFAULT_RENDER_SCRIPT_RESOURCE
    try:
        return Path(str(_resource_files(package).joinpath(resource)))
    except (ModuleNotFoundError, FileNotFoundError):
        return Path(__file__).resolve().parent / "assets" / resource


def _build_subprocess_env(gpu_id: int, blender_user_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["BLENDER_USER_RESOURCES"] = str(blender_user_dir)
    # Reduce Blender's own logging noise; user-code stack traces still surface
    # via stderr because exec() lets exceptions propagate.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_blender(
    blend_file: str | Path,
    code: str,
    output_dir: str | Path,
    *,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    render_script: str | Path | None = None,
    gpu_id: int = 0,
    timeout: int = DEFAULT_TIMEOUT_S,
    paths: "TurnPaths | None" = None,
) -> RenderResult:
    """Run one Blender background render and capture all artifacts.

    Args:
        blend_file: read-only ``.blend`` from the BlenderGym dataset.
        code: the model-generated Blender Python program (string contents).
        output_dir: per-turn directory; ``code.py``, ``render1.png``,
            ``blender.log``, and ``blender_user/`` will live here.
        blender_bin: path to the Infinigen-bundled Blender binary.
        render_script: bundled or override ``pipeline_render_script.py``.
            Defaults to the packaged asset.
        gpu_id: GPU index passed via ``CUDA_VISIBLE_DEVICES``. Note this is
            independent of the env-worker main process's CUDA visibility.
        timeout: hard wall-clock cap (seconds) for the subprocess.

    Returns:
        :class:`RenderResult`. ``success=True`` iff the subprocess exited 0
        and ``render1.png`` is present in ``output_dir``.
    """
    blend_path = Path(blend_file).expanduser().resolve()
    if not blend_path.is_file():
        raise FileNotFoundError(f"blend_file not found: {blend_path}")

    blender_path = Path(blender_bin).expanduser().resolve()
    if not blender_path.is_file():
        raise FileNotFoundError(f"blender_bin not found: {blender_path}")

    script_path = Path(render_script).expanduser().resolve() if render_script else default_render_script()
    if not script_path.is_file():
        raise FileNotFoundError(f"render_script not found: {script_path}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if paths is not None:
        _code_path = paths.code
        _render_path = paths.render
        _log_path = paths.log
        _blender_user = paths.blender_user
    else:
        _code_path = out_dir / CODE_FILENAME
        _render_path = out_dir / RENDER1_FILENAME
        _log_path = out_dir / LOG_FILENAME
        _blender_user = out_dir / "blender_user"

    _blender_user.mkdir(parents=True, exist_ok=True)
    _code_path.write_text(code, encoding="utf-8")
    image_path = _render_path
    if image_path.exists():
        # Stale artifact from a prior invocation would cause a false success
        # signal if the new invocation crashes before re-rendering.
        image_path.unlink()

    cmd = [
        str(blender_path),
        "--background",
        str(blend_path),
        "--python",
        str(script_path),
        "--",
        str(_code_path),
        str(out_dir),
    ]
    env = _build_subprocess_env(gpu_id=gpu_id, blender_user_dir=_blender_user)

    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    stderr_text = ""
    stdout_text = ""
    success = False

    try:
        completed = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        returncode = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
        success = image_path.is_file()
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # ``exc.stdout`` / ``exc.stderr`` are bytes-or-None when text=True yet
        # the child gets killed; coerce defensively.
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        stderr_text += f"\n[blendergym] TIMEOUT after {timeout}s\n"
    except subprocess.CalledProcessError as exc:
        returncode = exc.returncode
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""

    duration = time.monotonic() - started

    if _log_path is not None:
        _log_path.write_text(
            "$ "
            + " ".join(shlex.quote(part) for part in cmd)
            + "\n\n"
            + "=== STDOUT ===\n"
            + stdout_text
            + "\n=== STDERR ===\n"
            + stderr_text,
            encoding="utf-8",
        )

    image_paths = [image_path] if image_path.is_file() else []

    return RenderResult(
        success=success,
        image_paths=image_paths,
        stderr=stderr_text,
        duration_s=duration,
        returncode=returncode,
        code_path=_code_path,
        log_path=_log_path,
        timed_out=timed_out,
    )


def _read_code(code_arg: str, code_file_arg: str | None) -> str:
    if code_file_arg:
        return Path(code_file_arg).read_text(encoding="utf-8")
    if code_arg == "-":
        return sys.stdin.read()
    return Path(code_arg).read_text(encoding="utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m blendergym.render",
        description="One-shot Blender render smoke for BlenderGym.",
    )
    parser.add_argument("--blend", required=True, help="Path to .blend file (read-only).")
    parser.add_argument(
        "--code",
        help="Path to Blender Python source file. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--code-file",
        dest="code_file",
        help="Alias for --code (kept for plan-spec compatibility).",
    )
    parser.add_argument("--output-dir", required=True, help="Per-turn work dir.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--blender-bin",
        default=str(DEFAULT_BLENDER_BIN),
        help="Override Blender binary path.",
    )
    parser.add_argument(
        "--render-script",
        default=None,
        help="Override bundled pipeline_render_script.py.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Hard wall-clock cap (seconds).",
    )
    # Cycles knobs are forwarded as env vars (see pipeline_render_script.py).
    # We intentionally do NOT add them to ``run_blender``'s signature: subprocess
    # already inherits dict(os.environ), so a CLI flag → env var → child process
    # path keeps the public Python API stable.
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Square render resolution; sets BLENDERGYM_RENDER_RESOLUTION.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Cycles samples; sets BLENDERGYM_CYCLES_SAMPLES.",
    )
    parser.add_argument(
        "--denoiser",
        default=None,
        help="Cycles denoiser; sets BLENDERGYM_CYCLES_DENOISER.",
    )
    parser.add_argument(
        "--compute-device",
        dest="compute_device",
        default=None,
        help="Cycles compute device; sets BLENDERGYM_CYCLES_COMPUTE_DEVICE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if not args.code and not args.code_file:
        parser.error("one of --code or --code-file must be provided")

    code_text = _read_code(args.code, args.code_file)

    if args.resolution is not None:
        os.environ["BLENDERGYM_RENDER_RESOLUTION"] = str(args.resolution)
    if args.samples is not None:
        os.environ["BLENDERGYM_CYCLES_SAMPLES"] = str(args.samples)
    if args.denoiser is not None:
        os.environ["BLENDERGYM_CYCLES_DENOISER"] = args.denoiser
    if args.compute_device is not None:
        os.environ["BLENDERGYM_CYCLES_COMPUTE_DEVICE"] = args.compute_device

    result = run_blender(
        blend_file=args.blend,
        code=code_text,
        output_dir=args.output_dir,
        blender_bin=args.blender_bin,
        render_script=args.render_script,
        gpu_id=args.gpu,
        timeout=args.timeout,
    )

    print(
        f"[blendergym] success={result.success} "
        f"returncode={result.returncode} "
        f"timed_out={result.timed_out} "
        f"duration={result.duration_s:.2f}s "
        f"images={[str(p) for p in result.image_paths]}"
    )
    print(f"[blendergym] log: {result.log_path}")
    if not result.success:
        sys.stderr.write(result.stderr[-2000:] + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
