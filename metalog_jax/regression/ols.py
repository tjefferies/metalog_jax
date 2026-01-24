# Copyright: Travis Jefferies 2026
"""Ordinary Least Squares (OLS) regression implementation.

This module provides OLS regression for fitting metalog distributions when
no regularization is needed.

Classes:
    OLSModel: Structure for trained OLS model weights.

Functions:
    fit_ordinary_least_squares: Fit an OLS regression model.
    predict_ordinary_least_squares: Make predictions using a fitted OLS model.

This module is used when MetalogFitMethod.OLS is specified in MetalogParameters.

See Also:
    metalog_jax.regression.lasso: LASSO (L1 regularization).
    metalog_jax.metalog.fit: High-level fitting function that dispatches to OLS.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from metalog_jax.regression.base import RegressionModel


@struct.dataclass
class OLSModel(RegressionModel):
    """Structure for trained Ordinary Least Squares model weights.

    Attributes:
        weights: Coefficient vector of shape (n_features,).
        bias: Intercept term (scalar).
    """

    pass


@jax.jit
def fit_ordinary_least_squares(
    X: chex.Array,
    y: chex.Array,
) -> OLSModel:
    """Fit an Ordinary Least Squares (OLS) regression model.

    Computes the closed-form solution to linear regression using the normal
    equations via least squares. This method finds the optimal weights and bias
    that minimize the sum of squared residuals with no regularization.

    The model solves: min ||y - (Xw + b)||^2

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).

    Returns:
        OLSModel containing the fitted weights and bias that minimize
        the squared error.

    Example:
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> y = jnp.array([1.0, 2.0, 3.0])
        >>> model = fit_ordinary_least_squares(X, y)
    """

    @jax.jit
    def least_squares(
        X: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Solve ordinary least squares using JAX's lstsq function.

        Computes the least-squares solution to the linear system Xw = y using
        singular value decomposition (SVD). This method is more numerically
        stable than the normal equation for ill-conditioned or rank-deficient
        matrices.

        Args:
            X: Design matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            Tuple containing:
            - solution: Weight vector of shape (n_features,) that minimizes ||y - Xw||^2
            - residuals: Sum of squared residuals
            - rank: Effective rank of matrix X
            - singular_values: Singular values of X
        """
        return jnp.linalg.lstsq(X, y)

    @jax.jit
    def normal_equation(
        X: chex.Array,
        y: chex.Array,
    ) -> chex.Array:
        """Solve ordinary least squares using the normal equation.

        Computes the OLS solution using the analytical formula:
        weights = (X^T X)^(-1) X^T y

        This is the closed-form solution to the linear regression problem
        that minimizes the sum of squared residuals.

        Args:
            X: Design matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            Weight vector of shape (n_features,) that minimizes ||y - Xw||^2.

        Note:
            This function requires X^T X to be invertible. For ill-conditioned
            or rank-deficient matrices.
        """
        return jnp.linalg.inv(X.T @ X) @ X.T @ y

    if len(X.shape) == 1:
        X = jnp.expand_dims(X, axis=1)
    lstsq_result = least_squares(X, y)
    weights = lstsq_result[0]  # Extract weights from tuple
    return OLSModel(weights=weights)


@jax.jit
def predict_ordinary_least_squares(
    X: chex.Array,
    model: OLSModel,
) -> chex.Array:
    """Make predictions using a fitted Regression model.

    Computes predictions for new data using the trained weights and bias:
    y_pred = X @ weights + bias

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        model: OLSModel containing fitted weights and bias.

    Returns:
        Predictions of shape (n_samples,).

    Example:
        >>> X_test = jnp.array([[2.0, 3.0], [4.0, 5.0]])
        >>> predictions = predict_ordinary_least_squares(X_test, model)
    """
    if len(X.shape) == 1:
        X = jnp.expand_dims(X, axis=1)
    predictions = jnp.dot(X, model.weights)
    return jnp.squeeze(predictions)
