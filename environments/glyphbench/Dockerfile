# GlyphBench — verifiers + prime-rl container
#
# Build:   docker build -t glyphbench:latest .
#
# Run eval inside the container with your model cache mounted as needed.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# uv (Python & Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && cp /root/.local/bin/uv /usr/local/bin/uv \
  && cp /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /opt/glyphbench

# Cache dependency installation independently of project source changes.
COPY pyproject.toml uv.lock /opt/glyphbench/
RUN uv python install 3.12 \
  && uv sync --frozen --extra eval --no-install-project

# Install the package from the allowlisted build context, including its license.
COPY . /opt/glyphbench
RUN uv sync --frozen --extra eval

# Optional RL extra (prime-rl + flash-attn). Heavy; install at run time if needed.
# RUN uv sync --frozen --extra eval --extra rl

ENV PATH="/opt/glyphbench/.venv/bin:${PATH}"

# uv installs the managed Python under /root/.local/share/uv/...; the venv
# at /opt/glyphbench/.venv/bin/python symlinks to it. Loosen permissions so
# non-root container runtimes can execute the venv interpreter.
RUN chmod -R a+rX /root || true

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
