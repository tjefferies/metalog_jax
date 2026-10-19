"""Hypothesis profiles for the property-based tests.

Choose a profile with the HYPOTHESIS_PROFILE environment variable; CI uses the
default, ``ci``. The first call of each shape spends seconds compiling, so no
profile has a deadline, and strategies that fit metalogs are slow by design.
"""
# Copyright: Travis Jefferies 2026

import os

from hypothesis import HealthCheck, settings

_COMMON = {"deadline": None, "suppress_health_check": [HealthCheck.too_slow]}

# Reproducible: the same examples on every run (derandomize implies no database)
settings.register_profile("ci", max_examples=50, derandomize=True, **_COMMON)
settings.register_profile("dev", max_examples=20, **_COMMON)
settings.register_profile("nightly", max_examples=500, **_COMMON)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
