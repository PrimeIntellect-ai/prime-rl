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
  wget \
  git \
  vim \
  htop \
  nvtop \
  iperf \
  tmux \
  openssh-server \
  git-lfs \
  sudo \
  gpg \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install gsutil
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

# # Install Rust
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"
# RUN echo "export PATH=\"/opt/conda/bin:/root/.cargo/bin:\$PATH\"" >> /root/.bashrc

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies (The gradual copies help with caching)
WORKDIR /root/prime-rl

COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./README.md ./README.md
COPY ./src/ ./src/

# Create venv and install dependencies
RUN uv sync && uv sync --extra fa

# Runtime stage
FROM python:3.11-slim
WORKDIR /root/prime-rl

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes build-essential wget clang

# Copy virtual environment
COPY --from=builder /root/prime-rl/.venv /root/prime-rl/.venv
RUN rm /root/prime-rl/.venv/bin/python
RUN ln -s /usr/local/bin/python /root/prime-rl/.venv/bin/python
ENV PATH="/root/prime-rl/.venv/bin:$PATH"

# Note(Jack): Nothing should need to compile so we don't need these
# COPY --from=builder /usr/local/cuda-12.4 /usr/local/cuda
# ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# ENV CUDA_HOME=/usr/local/cuda-12.4
# ENV PATH="/usr/local/cuda-12.4/bin:$PATH"

# Copy application files
COPY --from=builder /root/prime-rl/src ./src
COPY ./configs/ ./configs/

ENTRYPOINT ["python", "src/zeroband/inference.py"]
CMD ["@", "configs/inference/Qwen1.5B/multistep.toml", "--output_path", "/share"]
