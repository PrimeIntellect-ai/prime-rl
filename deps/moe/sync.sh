cp deps/moe/pyproject.toml .
cp deps/moe/uv.lock .
uv sync && uv sync --all-extras
