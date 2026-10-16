"""Prepare an explicitly requested finite docs smoke config."""

import argparse
import tomllib
from pathlib import Path


def prepare(source: Path, destination: Path, steps: int) -> None:
    if steps < 1:
        raise ValueError("A positive finite training-step cap is required")
    text = source.read_text()
    raw = tomllib.loads(text)
    lines = text.splitlines(keepends=True)
    if "max_steps" in raw:
        matches = [i for i, line in enumerate(lines) if line.startswith("max_steps =")]
        if len(matches) != 1:
            raise ValueError("Expected one top-level max_steps assignment")
        lines[matches[0]] = f"max_steps = {steps}\n"
        text = "".join(lines)
    else:
        text = f"max_steps = {steps}\n\n" + text
    # Hosted name is injected separately; RLConfig does not accept it.
    lines = text.splitlines(keepends=True)
    header = True
    output = []
    for line in lines:
        if line.startswith("["):
            header = False
        if header and line.startswith("name = "):
            continue
        output.append(line)
    destination.write_text("".join(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.steps)


if __name__ == "__main__":
    main()
