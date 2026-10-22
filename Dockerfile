# metalog-jax CI/CD base image
# Includes all dependencies for lint, test, security, docs, and development
#
# The Docker workflow builds one image for each Python version CI tests and passes the
# version as PYTHON_VERSION. A plain `docker build .` uses the default.
ARG PYTHON_VERSION=3.11

# uv, Syft and Grype come from their official images, pinned by version and digest.
# Dependabot updates these lines.
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv
FROM ghcr.io/anchore/syft:v1.54.1@sha256:3eb5379ba7b409c3f4069b686110527af0c47df993fa5c10d13e7cf34f49b1aa AS syft
FROM ghcr.io/anchore/grype:v0.120.1@sha256:e4a44ef45d285b829ce6efe2642980329661bd2d18eab5fc539138d4adaebbbe AS grype

FROM python:${PYTHON_VERSION}-slim-bookworm

LABEL maintainer="Travis Jefferies"
LABEL description="metalog-jax development and CI/CD image with all dependencies"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Install system dependencies (pandoc for nbsphinx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=uv /uv /uvx /usr/local/bin/

# Install Syft for SBOM generation and Grype for vulnerability scanning
COPY --from=syft /syft /usr/local/bin/
COPY --from=grype /grype /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install all dependencies (base + all groups). The project source is not copied into
# the image, so skip installing metalog-jax itself (pyproject sets tool.uv.package).
RUN uv sync --frozen --no-install-project --all-groups

# Default command
CMD ["/bin/bash"]
