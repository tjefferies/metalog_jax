"""Metalog JAX library."""
# Copyright: Travis Jefferies 2026

import jax

# Every metalog_jax import path needs float64: the fits, the exact rational tables and
# the feasibility certificate are all below float32 resolution. Setting it here (the
# package root) covers direct imports such as ``import metalog_jax.feasibility``.
jax.config.update("jax_enable_x64", True)
