#inspired from https://github.com/astral-sh/uv-docker-example/blob/main/multistage.Dockerfile

# Build stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder
LABEL maintainer="prime intellect"
LABEL repository="prime-rl"

# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Set CUDA_HOME and update PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  build-essential \
  curl \
  sudo \
  git \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies (The gradual copies help with caching)
WORKDIR /app

COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./README.md ./README.md
COPY ./src/ ./src/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0


COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY README.md /app/README.md
COPY configs /app/configs

RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --locked --no-dev

RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --extra fa --locked --no-dev

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    --force-yes \
    build-essential \
    wget \
    clang \
    tmux \
    iperf \
    openssh-server \
    git-lfs \
    gpg \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid $GROUP_ID appuser && \
    useradd --uid $USER_ID --gid appuser --create-home --shell /bin/bash appuser

USER appuser
WORKDIR /app
# Copy the application from the builder
COPY --from=builder --chown=appuser:appuser /app /app

RUN rm /app/.venv/bin/python && ln -s /usr/local/bin/python /app/.venv/bin/python
RUN rm /app/.venv/bin/python3 && ln -s /usr/local/bin/python /app/.venv/bin/python3
RUN rm /app/.venv/bin/python3.11 && ln -s /usr/local/bin/python /app/.venv/bin/python3.11


# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "src/zeroband/infer.py"]