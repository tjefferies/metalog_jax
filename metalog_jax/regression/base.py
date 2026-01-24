# Copyright: Travis Jefferies 2026
"""Base classes for regression models and parameters.

This module provides the abstract base classes for regression models and their
parameters used in metalog distribution fitting.

Classes:
    RegularizedParameters: Base class for regularized regression hyperparameters.
        Subclasses: LassoParameters
    RegressionModel: Base class for trained regression model weights.
        Subclasses: OLSModel, LassoModel

See Also:
    metalog_jax.regression.ols: OLS regression implementation.
    metalog_jax.regression.lasso: LASSO regression implementation.
"""

import chex
from flax import struct


@struct.dataclass
class RegularizedParameters:
    """Base class for regularized regression hyperparameters.

    This abstract base class serves as the parent for all regularized regression
    parameter configurations in the metalog JAX library. It provides a common type
    hierarchy for regression methods that incorporate regularization penalties
    (L1, L2, or both) to prevent overfitting and improve generalization.

    Regularization adds penalty terms to the regression objective function to
    constrain model complexity. This base class enables polymorphic handling of
    different regularization strategies while maintaining type safety and
    consistency across the library.

    The class uses Flax's `struct.dataclass` decorator to ensure immutability
    and compatibility with JAX transformations (jit, vmap, grad, etc.).

    Subclasses:
        LassoParameters: L1 regularization parameters (LASSO regression).
            Defined in metalog_jax.regression.lasso.
            Penalizes the absolute magnitude of coefficients: λ||w||₁

    Design Pattern:
        This class follows the Template Method pattern, providing a common
        interface while allowing subclasses to define specific regularization
        configurations. It enables functions to accept any regularized regression
        parameters through polymorphic type hints.

    Example:
        Using the base class for polymorphic type hints:

            >>> from typing import Union
            >>> from metalog_jax.regression.base import RegularizedParameters
            >>> from metalog_jax.regression.lasso import LassoParameters
            >>>
            >>> def validate_regularization(params: RegularizedParameters):
            ...     '''Accept any regularized regression parameters.'''
            ...     if isinstance(params, LassoParameters):
            ...         print(f"Lasso with L1={params.lam}")
            >>>
            >>> lasso_params = LassoParameters(
            ...     lam=1.0,
            ...     learning_rate=0.01,
            ...     num_iters=500,
            ...     tol=1e-6,
            ...     momentum=0.9
            ... )
            >>> validate_regularization(lasso_params)  # Valid
            Lasso with L1=1.0

        Creating subclass instances:

            >>> # Lasso regression (L1 only)
            >>> from metalog_jax.regression.lasso import LassoParameters
            >>> lasso = LassoParameters(
            ...     lam=0.5,
            ...     learning_rate=0.01,
            ...     num_iters=500,
            ...     tol=1e-6,
            ...     momentum=0.9
            ... )
            >>> isinstance(lasso, RegularizedParameters)
            True

    Note:
        - This is an abstract base class with no attributes or methods
        - Direct instantiation is possible but not meaningful (use subclasses)
        - All instances are immutable due to `struct.dataclass` decorator
        - Subclasses must define their own specific regularization parameters

    See Also:
        metalog_jax.regression.lasso.LassoParameters: LASSO (L1) regression parameters.
        metalog_jax.regression.lasso.fit_lasso: Function that uses LassoParameters.

    References:
        Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.).
        Springer. Chapter 3: Linear Methods for Regression.
    """

    pass


@struct.dataclass
class RegressionModel:
    """Structure for trained Regression model weights.

    Attributes:
        weights: Coefficient vector of shape (n_features,).
    """

    weights: chex.Array
