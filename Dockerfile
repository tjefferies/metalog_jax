# metalog-jax CI/CD base image
# Includes all dependencies for lint, test, security, docs, and development
#
# The Docker workflow builds one image for each Python version CI tests and passes the
# version as PYTHON_VERSION. A plain `docker build .` uses the default.
ARG PYTHON_VERSION=3.11

# uv, Syft and Grype come from their official images, pinned by version and digest.
# Dependabot updates these lines.
FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv
FROM ghcr.io/anchore/syft:v1.55.0@sha256:4cc92b64ca74ffca4c94d4c8fcc01f217731375300be14d0b6eeb2e82eaa110a AS syft
FROM ghcr.io/anchore/grype:v0.121.0@sha256:621d56d1d68106b787022c81546f7e0a36202a25d1239f0c20195fab8c16f7ea AS grype

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
