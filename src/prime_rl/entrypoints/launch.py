"""``uv run launch @ config.toml [@ more.toml ...] [overrides]``: run the entrypoint named by the config's ``mode``."""

import importlib
import sys

from prime_rl.utils.config import MODES, config_mode

USAGE = f"""\
usage: uv run launch @ config.toml [@ more.toml ...] [overrides]

Runs the entrypoint named by the top-level `mode` of the config files: {", ".join(MODES)}.
With a config, --help shows that entrypoint's help.
"""


def main():
    args = sys.argv[1:]
    mode = config_mode(args)
    if mode is None:
        if "-h" in args or "--help" in args:
            print(USAGE)
            return
        raise SystemExit(f"{USAGE}\nerror: set `mode` in the config files to one of: {', '.join(MODES)}")
    importlib.import_module(f"prime_rl.entrypoints.{mode}").main()


if __name__ == "__main__":
    main()
