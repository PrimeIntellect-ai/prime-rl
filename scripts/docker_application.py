import argparse
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path


def editable_paths(root: Path) -> list[Path]:
    lock = tomllib.loads((root / "uv.lock").read_text())
    paths = set()
    for package in lock["package"]:
        source = package["source"]
        if "directory" in source:
            raise ValueError(f"Non-editable local package: {package['name']}")
        if "editable" not in source:
            continue
        path = (root / source["editable"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Editable package outside the application: {path}")
        paths.add(path)
    return sorted(paths)


def set_fallback_version(pyproject: Path, version: str) -> None:
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+-]*", version):
        raise ValueError(f"Invalid package version: {version!r}")
    content, count = re.subn(
        r'(?m)^fallback-version = "[^"]*"$', f'fallback-version = "{version}"', pyproject.read_text()
    )
    if count != 1:
        raise ValueError(f"Expected one fallback-version in {pyproject}")
    pyproject.write_text(content)


def copy_installation_artifacts(venv: Path, baseline: set[Path], output: Path) -> None:
    # The runtime supplies the interpreter and third-party environment.
    for directory in (venv / "bin", venv / "lib/python3.12/site-packages"):
        for path in directory.iterdir():
            relative = path.relative_to(venv)
            if relative in baseline:
                continue
            target = output / ".venv" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(path, target)


def application_layer(root: Path, output: Path) -> None:
    if output.is_relative_to(root):
        raise ValueError("The output directory must be outside the application")
    if output.exists():
        raise FileExistsError(output)
    venv = root / ".venv"
    if venv.exists() or venv.is_symlink():
        raise FileExistsError(venv)
    versions = {
        "deps/verifiers": "VERIFIERS_PRETEND_VERSION",
        "deps/renderers": "RENDERERS_PRETEND_VERSION",
        "deps/pydantic-config": "PYDANTIC_CONFIG_PRETEND_VERSION",
    }
    for path, variable in versions.items():
        set_fallback_version(root / path / "pyproject.toml", os.environ[variable])

    subprocess.run(["uv", "venv", "--python", "/usr/bin/python3.12", str(venv)], check=True)
    baseline = {path.relative_to(venv) for path in venv.rglob("*")}
    command = ["uv", "pip", "install", "--python", str(venv / "bin/python"), "--no-deps"]
    for path in editable_paths(root):
        command.extend(["--editable", str(path)])
    subprocess.run(command, check=True, cwd=root)

    shutil.copytree(root, output, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc"))
    copy_installation_artifacts(venv, baseline, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source and editable-package Docker layer")
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    application_layer(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
