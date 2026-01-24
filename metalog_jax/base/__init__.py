# Copyright: Travis Jefferies 2026
"""Base module for the Metalog JAX library.

This module provides the core data structures and enumeration types for metalog
distributions. The module is organized into submodules:

Submodules:
    core: Contains MetalogBase, the abstract base class for metalog distributions.
    data: Contains MetalogBaseData and MetalogInputData for input data handling.
    enums: Contains enumeration types (MetalogBoundedness, MetalogFitMethod,
        MetalogPlotOptions, CustomIntEnum).
    parameters: Contains parameter configuration classes (MetalogParameters,
        MetalogParametersBase, MetalogRandomVariableParameters, SPTMetalogParameters).

All public classes are re-exported at the module level, allowing imports like:
    from metalog_jax.base import MetalogBase, MetalogParameters

See Also:
    metalog_jax.metalog: High-level fitting functions and Metalog/SPTMetalog classes.
    metalog_jax.regression: Regression models used for fitting metalog distributions.
"""

from metalog_jax.base.core import MetalogBase
from metalog_jax.base.data import MetalogBaseData, MetalogInputData
from metalog_jax.base.enums import (
    CustomIntEnum,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogPlotOptions,
)
from metalog_jax.base.parameters import (
    MetalogParameters,
    MetalogParametersBase,
    MetalogRandomVariableParameters,
    SPTMetalogParameters,
)

__all__ = [
    "CustomIntEnum",
    "MetalogBoundedness",
    "MetalogFitMethod",
    "MetalogPlotOptions",
    "MetalogBaseData",
    "MetalogInputData",
    "MetalogRandomVariableParameters",
    "MetalogParametersBase",
    "MetalogParameters",
    "SPTMetalogParameters",
    "MetalogBase",
]
