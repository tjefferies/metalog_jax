# metalog-jax CI/CD base image
# Includes all dependencies for lint, test, security, docs, and development

FROM python:3.11-slim-bookworm

LABEL maintainer="Travis Jefferies"
LABEL description="metalog-jax development and CI/CD image with all dependencies"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Install system dependencies (curl needed for syft/grype install, pandoc for nbsphinx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install Syft for SBOM generation
RUN curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Install Grype for vulnerability scanning
RUN curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install all dependencies (base + all groups)
RUN uv sync --frozen --all-groups

# Default command
CMD ["/bin/bash"]
