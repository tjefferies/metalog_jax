# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Port for local development server
SERVE_PORT    = 5337

.PHONY: help clean html serve livehtml rebuild

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  html       to build HTML documentation"
	@echo "  clean      to clean build artifacts"
	@echo "  rebuild    to clean and rebuild documentation (use after CSS changes)"
	@echo "  serve      to serve documentation at http://localhost:$(SERVE_PORT)"
	@echo "  livehtml   to serve with auto-reload on changes"

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

serve: html
	@echo "Serving documentation at http://localhost:$(SERVE_PORT)"
	@echo "Press Ctrl+C to stop."
	uv run python -m http.server $(SERVE_PORT) --directory $(BUILDDIR)/html

livehtml:
	uv run sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html --port $(SERVE_PORT) $(SPHINXOPTS)

rebuild: clean html
	@echo "Clean rebuild complete."
