# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        metalog-jax Development Makefile                       ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║ Run CI/CD quality gates locally before pushing. All commands use `uv run`    ║
# ║ to match the GitHub Actions pipeline exactly.                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: help install format format-check lint typecheck radon radon-mi radon-cc \
        test test-quick coverage-html license-check semgrep bandit sbom security \
        build docs docs-html docs-clean docs-serve docs-live docs-rebuild \
        quality-gate ci clean

# ══════════════════════════════════════════════════════════════════════════════
# Variables & Configuration
# ══════════════════════════════════════════════════════════════════════════════

PACKAGE := metalog_jax
TESTS := tests
DOCS_DIR := docs
DOCS_SOURCE := $(DOCS_DIR)/source
DOCS_BUILD := $(DOCS_DIR)/build
DOCS_PORT := 5337

# License allow list (from CI)
LICENSE_ALLOWED := MIT;BSD;Apache;PSF;Python Software Foundation;ISC;Unlicense;Public Domain;CC0;Zlib;Historical Permission Notice and Disclaimer;Mozilla Public License;Academic Free License;0BSD

# Packages to ignore in license check (dev/docs/security tools)
LICENSE_IGNORE := metalog-jax Unidecode text-unidecode css-html-js-minify semgrep docutils loro matplotlib-inline

# ══════════════════════════════════════════════════════════════════════════════
# Help Target
# ══════════════════════════════════════════════════════════════════════════════

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                        metalog-jax Development Makefile                       ║"
	@echo "╠══════════════════════════════════════════════════════════════════════════════╣"
	@echo "║ Run CI/CD quality gates locally before pushing. All commands use 'uv run'    ║"
	@echo "║ to match the GitHub Actions pipeline exactly.                                ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "SETUP"
	@echo "  make install          Install all dependencies (uv sync --all-groups)"
	@echo ""
	@echo "CODE QUALITY"
	@echo "  make format           Auto-format code with Ruff"
	@echo "  make format-check     Check formatting (CI: quality-gate.yml/ruff)"
	@echo "  make lint             Run Ruff linter (CI: quality-gate.yml/ruff)"
	@echo "  make typecheck        Run ty type checker (CI: quality-gate.yml/ty)"
	@echo ""
	@echo "CODE METRICS"
	@echo "  make radon            Run both maintainability and complexity checks"
	@echo "  make radon-mi         Maintainability Index (CI: quality-gate.yml/radon)"
	@echo "  make radon-cc         Cyclomatic Complexity (CI: quality-gate.yml/radon)"
	@echo ""
	@echo "TESTING"
	@echo "  make test             Run pytest with coverage (CI: quality-gate.yml/pytest)"
	@echo "  make test-quick       Run pytest without coverage (faster iteration)"
	@echo "  make coverage-html    Open coverage report in browser"
	@echo ""
	@echo "SECURITY"
	@echo "  make security         Run all security checks"
	@echo "  make semgrep          SAST scanning (CI: quality-gate.yml/semgrep)"
	@echo "  make bandit           Security linting (CI: quality-gate.yml/bandit)"
	@echo "  make sbom             Generate SBOM (requires syft/grype)"
	@echo ""
	@echo "LICENSE"
	@echo "  make license-check    Verify license compliance (CI: quality-gate.yml/license-check)"
	@echo ""
	@echo "BUILD"
	@echo "  make build            Build wheel (CI: build.yml)"
	@echo ""
	@echo "DOCUMENTATION"
	@echo "  make docs             Build HTML documentation (CI: docs.yml)"
	@echo "  make docs-html        Build HTML documentation (alias)"
	@echo "  make docs-serve       Serve docs at http://localhost:$(DOCS_PORT)"
	@echo "  make docs-live        Live-reload docs during development"
	@echo "  make docs-clean       Clean documentation build artifacts"
	@echo "  make docs-rebuild     Clean and rebuild documentation"
	@echo ""
	@echo "CI/CD COMPOSITES"
	@echo "  make quality-gate     Run all 8 quality checks (matches CI quality-gate.yml)"
	@echo "  make ci               Run full CI pipeline locally"
	@echo ""
	@echo "CLEANUP"
	@echo "  make clean            Remove all build artifacts"
	@echo ""
	@echo "TYPICAL WORKFLOWS"
	@echo "  Before committing:    make format lint typecheck test"
	@echo "  Before pushing:       make quality-gate"
	@echo "  Full CI simulation:   make ci"
	@echo "  Documentation work:   make docs-live"
	@echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Setup & Installation
# ══════════════════════════════════════════════════════════════════════════════

install:
	@echo "Installing all dependencies..."
	uv sync --all-groups

# ══════════════════════════════════════════════════════════════════════════════
# Code Quality (format, lint, type-check)
# ══════════════════════════════════════════════════════════════════════════════

format:
	@echo "Auto-formatting code with Ruff..."
	uv run ruff format $(PACKAGE)/ $(TESTS)/

format-check:
	@echo "Checking code formatting..."
	uv run ruff format --check $(PACKAGE)/ $(TESTS)/

lint:
	@echo "Running Ruff linter..."
	uv run ruff check $(PACKAGE)/ $(TESTS)/

typecheck:
	@echo "Running ty type checker..."
	uv run ty check

# ══════════════════════════════════════════════════════════════════════════════
# Code Metrics (radon)
# ══════════════════════════════════════════════════════════════════════════════

radon-mi:
	@echo "Checking Maintainability Index..."
	uv run radon mi $(PACKAGE)/ -s

radon-cc:
	@echo "Checking Cyclomatic Complexity..."
	uv run radon cc $(PACKAGE)/ -s --average

radon: radon-mi radon-cc

# ══════════════════════════════════════════════════════════════════════════════
# Testing & Coverage
# ══════════════════════════════════════════════════════════════════════════════

test:
	@echo "Running tests with coverage..."
	uv run pytest $(TESTS)/ \
		--cov=$(PACKAGE) \
		--cov-report=term \
		--cov-report=xml:coverage.xml \
		--cov-report=html:coverage_html \
		--cov-fail-under=75

test-quick:
	@echo "Running tests without coverage..."
	uv run pytest $(TESTS)/

coverage-html: test
	@echo "Opening coverage report in browser..."
	@if command -v open > /dev/null 2>&1; then \
		open coverage_html/index.html; \
	elif command -v xdg-open > /dev/null 2>&1; then \
		xdg-open coverage_html/index.html; \
	else \
		echo "Coverage report available at: coverage_html/index.html"; \
	fi

# ══════════════════════════════════════════════════════════════════════════════
# Security Scanning (semgrep, bandit, sbom)
# ══════════════════════════════════════════════════════════════════════════════

semgrep:
	@echo "Running Semgrep SAST scan..."
	uv run semgrep scan \
		--config=p/python \
		--config=p/security-audit \
		--config=p/owasp-top-ten \
		--config=p/cwe-top-25 \
		--config=p/secrets \
		--config=.semgrep.yml \
		--severity=ERROR --error \
		--no-git-ignore $(PACKAGE)/

bandit:
	@echo "Running Bandit security scan..."
	uv run bandit -r $(PACKAGE)/ \
		--severity-level high --confidence-level high

sbom:
	@echo "Generating SBOM..."
	@if command -v syft > /dev/null 2>&1; then \
		echo "Generating CycloneDX SBOM..."; \
		syft scan dir:.venv -o cyclonedx-json=sbom-cyclonedx.json; \
		echo "Generating SPDX SBOM..."; \
		syft scan dir:.venv -o spdx-json=sbom-spdx.json; \
	else \
		echo ""; \
		echo "WARNING: syft not installed. Install with:"; \
		echo "  curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"; \
		exit 1; \
	fi
	@if command -v grype > /dev/null 2>&1; then \
		echo "Scanning for vulnerabilities..."; \
		grype sbom:sbom-cyclonedx.json -o table || true; \
	else \
		echo ""; \
		echo "WARNING: grype not installed. Install with:"; \
		echo "  curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin"; \
	fi

security: semgrep bandit
	@echo "Security checks complete."

# ══════════════════════════════════════════════════════════════════════════════
# License Compliance
# ══════════════════════════════════════════════════════════════════════════════

license-check:
	@echo "Checking license compliance..."
	@echo "=== License Compliance Check ==="
	uv run pip-licenses --format=markdown --ignore-packages $(LICENSE_IGNORE)
	uv run pip-licenses --ignore-packages $(LICENSE_IGNORE) --partial-match --allow-only="$(LICENSE_ALLOWED)"
	@echo "All runtime dependencies have approved licenses!"

# ══════════════════════════════════════════════════════════════════════════════
# Build & Package
# ══════════════════════════════════════════════════════════════════════════════

build:
	@echo "Building wheel..."
	uv build --wheel --out-dir dist

# ══════════════════════════════════════════════════════════════════════════════
# Documentation
# ══════════════════════════════════════════════════════════════════════════════

docs: docs-html
	@echo "Documentation build complete. Open $(DOCS_BUILD)/html/index.html to view."

docs-html:
	@echo "Building HTML documentation..."
	cd $(DOCS_DIR) && uv run sphinx-build -b html source build/html

docs-clean:
	@echo "Cleaning documentation build..."
	rm -rf $(DOCS_BUILD)

docs-serve: docs-html
	@echo "Serving documentation at http://localhost:$(DOCS_PORT)"
	@echo "Press Ctrl+C to stop."
	uv run python -m http.server $(DOCS_PORT) --directory $(DOCS_BUILD)/html

docs-live:
	@echo "Starting live-reload documentation server..."
	cd $(DOCS_DIR) && uv run sphinx-autobuild source build/html --port $(DOCS_PORT)

docs-rebuild: docs-clean docs-html
	@echo "Clean rebuild complete."

# ══════════════════════════════════════════════════════════════════════════════
# CI/CD Composite Targets
# ══════════════════════════════════════════════════════════════════════════════

quality-gate: format-check lint typecheck radon test license-check semgrep bandit
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                       All quality checks passed!                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════╝"

ci: install quality-gate build docs
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                         Full CI pipeline complete!                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════╝"

# ══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════════

clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf $(DOCS_BUILD)/
	rm -rf coverage_html/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf sbom-cyclonedx.json
	rm -rf sbom-spdx.json
	rm -rf grype-vulnerabilities.json
	rm -rf semgrep-results.json
	rm -rf bandit-results.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
