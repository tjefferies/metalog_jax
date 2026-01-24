# Copyright: Travis Jefferies 2026
"""Regression module for metalog_jax.

This module provides regression algorithms for fitting metalog distributions.
The module is organized into submodules for better maintainability:

Submodules:
    base: Base classes for regression models (RegressionModel, RegularizedParameters).
    ols: Ordinary Least Squares regression (no regularization).
        - OLSModel, fit_ordinary_least_squares, predict_ordinary_least_squares
    lasso: LASSO regression (L1 regularization).
        - LassoModel, LassoParameters, fit_lasso, soft_thresholding

All public classes and functions are re-exported at the module level, allowing
imports like:
    from metalog_jax.regression import fit_lasso, LassoParameters

The regression functions are called by metalog_jax.metalog.fit() based on the
MetalogFitMethod specified in MetalogParameters. The dispatch table maps:
    - MetalogFitMethod.OLS -> fit_ordinary_least_squares
    - MetalogFitMethod.Lasso -> fit_lasso

See Also:
    metalog_jax.metalog: High-level fitting functions that use these regression methods.
    metalog_jax.base.enums.MetalogFitMethod: Enumeration of available regression methods.
"""

from metalog_jax.regression.base import RegressionModel, RegularizedParameters
from metalog_jax.regression.lasso import (
    DEFAULT_LASSO_ITERATIONS,
    DEFAULT_LASSO_LAMBDA,
    DEFAULT_LASSO_LEARNING_RATE,
    DEFAULT_LASSO_MOMENTUM,
    DEFAULT_LASSO_PARAMETERS,
    DEFAULT_LASSO_TOLERANCE,
    LassoModel,
    LassoParameters,
    fit_lasso,
    soft_thresholding,
)
from metalog_jax.regression.ols import (
    OLSModel,
    fit_ordinary_least_squares,
    predict_ordinary_least_squares,
)

__all__ = [
    # Base
    "RegularizedParameters",
    "RegressionModel",
    # OLS
    "OLSModel",
    "fit_ordinary_least_squares",
    "predict_ordinary_least_squares",
    # LASSO
    "DEFAULT_LASSO_ITERATIONS",
    "DEFAULT_LASSO_LAMBDA",
    "DEFAULT_LASSO_LEARNING_RATE",
    "DEFAULT_LASSO_MOMENTUM",
    "DEFAULT_LASSO_PARAMETERS",
    "DEFAULT_LASSO_TOLERANCE",
    "LassoModel",
    "LassoParameters",
    "fit_lasso",
    "soft_thresholding",
]
