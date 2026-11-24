# metalog-jax GPU benchmark image
#
# The image .github/workflows/gpu-bench-image.yml builds and pushes to
# ghcr.io/tjefferies/metalog_jax/gpu-bench. Build it from the repository root:
#
#   docker build -f benchmarks/gpu/Dockerfile --platform linux/amd64 \
#     --build-arg GIT_SHA="$(git rev-parse HEAD)" \
#     --build-arg UV_LOCK_SHA256="$(shasum -a 256 uv.lock | cut -d' ' -f1)" \
#     --build-arg VERSION=3.0.0 -t gpu-bench .
#
# Benchmark runs pin the pushed image by digest. The CUDA libraries come from the
# jax[cuda12] wheels that uv.lock pins, so no CUDA base image is needed: the host
# provides only the driver.

# uv and Python, pinned by version and digest. Dependabot updates these lines.
FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv
FROM python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26

ARG GIT_SHA=unknown
ARG VERSION=unknown
ARG UV_LOCK_SHA256=unknown

LABEL org.opencontainers.image.source="https://github.com/tjefferies/metalog_jax" \
      org.opencontainers.image.description="metalog-jax GPU benchmark: batch fitting and sampling" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}" \
      io.github.tjefferies.metalog-jax.uv-lock-sha256="${UV_LOCK_SHA256}"

# JAX_PLATFORMS keeps the CPU too, for the correctness check. With preallocation off,
# measurements record their own peak, and XLA may use 95% of the GPU's memory.
# NVIDIA_DRIVER_CAPABILITIES' utility mounts nvidia-smi.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    JAX_PLATFORMS=cuda,cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    BENCH_RESULTS=/results

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/metalog_jax

# The locked dependencies first, so that a source change reuses this layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra gpu --no-install-project

COPY metalog_jax ./metalog_jax
RUN uv sync --frozen --no-dev --extra gpu --no-editable

COPY benchmarks/gpu ./benchmarks/gpu
RUN python -c "import json, os; json.dump({'git_sha': os.environ['GIT_SHA'], 'version': os.environ['VERSION'], 'uv_lock_sha256': os.environ['UV_LOCK_SHA256']}, open('/opt/metalog_jax/build-info.json', 'w'), indent=1)"

# Run as an unprivileged user. /results is writable by any user, so that
# docker run --user can write results to a mounted directory.
RUN useradd --create-home --uid 1000 bench && mkdir -m 1777 /results
USER 1000

ENTRYPOINT ["python", "-m", "benchmarks.gpu.run"]
