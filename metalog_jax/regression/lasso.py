# Copyright: Travis Jefferies 2026
"""LASSO regression implementation with L1 regularization.

This module provides LASSO (Least Absolute Shrinkage and Selection Operator)
regression for fitting metalog distributions with L1 regularization using
proximal gradient descent with Nesterov acceleration.

Classes:
    LassoParameters: Hyperparameters for LASSO regression.
    LassoModel: Structure for trained LASSO model weights.

Functions:
    fit_lasso: Fit a LASSO regression model via proximal gradient descent.
    soft_thresholding: Apply the soft-thresholding operator (proximal for L1).

Constants:
    DEFAULT_LASSO_LAMBDA: Default L1 regularization strength (0).
    DEFAULT_LASSO_LEARNING_RATE: Default learning rate (0.01).
    DEFAULT_LASSO_ITERATIONS: Default maximum iterations (500).
    DEFAULT_LASSO_TOLERANCE: Default convergence tolerance (1e-6).
    DEFAULT_LASSO_MOMENTUM: Default Nesterov momentum factor (0.9).
    DEFAULT_LASSO_PARAMETERS: Default LassoParameters instance.

This module is used when MetalogFitMethod.Lasso is specified in MetalogParameters.

See Also:
    metalog_jax.regression.ols: OLS regression (no regularization).
    metalog_jax.metalog.fit: High-level fitting function that dispatches to LASSO.
"""

import chex
import jax
import jax.numpy as jnp
from flax import struct

from metalog_jax.regression.base import RegressionModel, RegularizedParameters

DEFAULT_LASSO_LAMBDA: float = 0.0
DEFAULT_LASSO_LEARNING_RATE: float = 0.01
DEFAULT_LASSO_ITERATIONS: int = 500
DEFAULT_LASSO_TOLERANCE: float = 1e-6
DEFAULT_LASSO_MOMENTUM: float = 0.9


@struct.dataclass
class LassoParameters(RegularizedParameters):
    """Structure for Lasso parameters.

    Attributes:
        lam: L1 regularization strength λ ≥ 0. Controls sparsity of the solution:
            - λ = 0: No regularization (equivalent to OLS, but solved iteratively)
            - λ > 0: Promotes sparsity; larger values → more coefficients set to zero
            - Typical values: 0.001 to 10.0, depending on data scale
        learning_rate: Learning rate (step size) for gradient descent. Default: 0.01.
            Controls the size of parameter updates. Should be tuned based on problem:
            - Too large: May cause divergence or oscillation
            - Too small: Slow convergence
            - Typical values: 0.001 to 0.1
        num_iters: Maximum number of iterations. Default: 500.
            Training will stop early if convergence is reached before this limit.
        tol: Convergence tolerance for weight changes. Default: 1e-6.
            Training stops when ||w_new - w|| < tol, indicating the solution has stabilized.
            Smaller values require more precise convergence but may take longer.
        momentum: Nesterov momentum factor, must satisfy 0 < momentum < 1. Default: 0.9.
            Controls acceleration of convergence:
            - Higher values (e.g., 0.9, 0.99): Faster convergence but less stable
            - Lower values (e.g., 0.5, 0.7): More stable but slower convergence
            - Standard choice: 0.9
    """

    lam: float
    learning_rate: float
    num_iters: int
    tol: float
    momentum: float


DEFAULT_LASSO_PARAMETERS = LassoParameters(
    lam=DEFAULT_LASSO_LAMBDA,
    learning_rate=DEFAULT_LASSO_LEARNING_RATE,
    num_iters=DEFAULT_LASSO_ITERATIONS,
    tol=DEFAULT_LASSO_TOLERANCE,
    momentum=DEFAULT_LASSO_MOMENTUM,
)


@struct.dataclass
class LassoModel(RegressionModel):
    """Structure for trained LASSO Regression model weights.

    Attributes:
        weights: Coefficient vector of shape (n_features,).
        bias: Intercept term (scalar).
    """

    pass


@jax.jit
def soft_thresholding(x: chex.Numeric, lam: float) -> chex.Numeric:
    """Apply the soft-thresholding operator (proximal operator for L1 norm).

    The soft-thresholding operator is the proximal operator for the L1 norm and is
    fundamental to LASSO regression and other sparse optimization methods. It shrinks
    values toward zero and sets values below the threshold to exactly zero, promoting
    sparsity in the solution.

    The operator is defined element-wise as:
        soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)

    Equivalently:
        - If x > λ: return x - λ
        - If x < -λ: return x + λ
        - If |x| ≤ λ: return 0

    This function is the proximal operator for the L1 penalty:
        prox_{λ||·||₁}(x) = argmin_z { (1/2)||z - x||² + λ||z||₁ }

    Args:
        x: Input value(s) to threshold. Can be a scalar or array of any shape.
        lam: Threshold parameter λ ≥ 0. Controls the amount of shrinkage:
            - λ = 0: No shrinkage (returns x unchanged)
            - λ > 0: Shrinks values toward zero, setting small values to 0
            - Larger λ produces sparser solutions with more zeros

    Returns:
        Thresholded value(s) with the same shape as x. Values are shrunk toward
        zero by amount λ, with values |x| ≤ λ set to exactly zero.

    Example:
        Scalar inputs:

            >>> import jax.numpy as jnp
            >>> soft_thresholding(5.0, 2.0)
            Array(3.0, dtype=float32)
            >>> soft_thresholding(-3.0, 1.0)
            Array(-2.0, dtype=float32)
            >>> soft_thresholding(1.5, 2.0)  # Below threshold -> zero
            Array(0.0, dtype=float32)

        Array inputs:

            >>> x = jnp.array([-5.0, -1.0, 0.5, 2.0, 4.0])
            >>> soft_thresholding(x, 1.5)
            Array([-3.5,  0. ,  0. ,  0.5,  2.5], dtype=float32)

    Note:
        - This function is JIT-compiled for efficient execution
        - The operation is element-wise and preserves the input shape
        - Setting lam=0 returns the input unchanged
        - This is also known as the "shrinkage operator" or "soft thresholding"

    See Also:
        fit_lasso: LASSO regression that uses this operator in proximal gradient descent.

    References:
        Parikh, N., & Boyd, S. (2014). Proximal Algorithms. Foundations and Trends
        in Optimization, 1(3), 127-239.

        Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
        Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences,
        2(1), 183-202.
    """
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lam, 0.0)


def fit_lasso(
    X: chex.Numeric,
    y: chex.Numeric,
    params: LassoParameters = DEFAULT_LASSO_PARAMETERS,
) -> LassoModel:
    """Fit a LASSO regression model using proximal gradient descent with Nesterov acceleration.

    Trains a linear regression model with L1 regularization (LASSO - Least Absolute Shrinkage
    and Selection Operator) using an accelerated proximal gradient method. LASSO promotes
    sparse solutions by shrinking coefficients toward zero and setting many coefficients to
    exactly zero, making it useful for feature selection and interpretable models.

    This implementation uses:
    - **Proximal gradient descent**: Combines gradient steps on the smooth loss (MSE) with
      the proximal operator (soft-thresholding) for the non-smooth L1 penalty
    - **Nesterov momentum**: Accelerates convergence using momentum-based lookahead steps
    - **Early stopping**: Terminates when weight changes fall below tolerance threshold
    - **JAX compatibility**: Fully vectorized and compatible with JAX transformations

    The LASSO objective function is:
        min_w { (1/2n)||y - Xw||² + λ||w||₁ }

    where λ is the L1 regularization strength that controls sparsity.

    Algorithm:
        1. Initialize weights w and velocity v to zero
        2. For each iteration (until convergence or max iterations):
           a. Compute lookahead weights: w_lookahead = w + momentum * v
           b. Compute gradient at lookahead position
           c. Update velocity with momentum and gradient: v = momentum * v - learning_rate * grad
           d. Apply proximal operator: w = soft_threshold(w + v, learning_rate * λ)
           e. Check convergence: stop if ||w_new - w|| < tol
        3. Return final weights

    Args:
        X: Feature matrix of shape (n_samples, n_features). The design matrix containing
            the independent variables for regression.
        y: Target vector of shape (n_samples,). The dependent variable values to predict.
        params: LassoParameters containing optimization hyperparameters.
            Defaults to DEFAULT_LASSO_PARAMETERS. The object contains:
            - lam: L1 regularization strength λ ≥ 0. Default: 0.
            - learning_rate (float): Step size for gradient descent. Default: 0.01.
            - num_iters (int): Maximum number of iterations. Default: 500.
            - tol (float): Convergence tolerance for weight changes. Default: 1e-6.
            - momentum (float): Nesterov momentum factor (0 < momentum < 1). Default: 0.9.

    Returns:
        LassoModel containing the fitted weights vector of shape (n_features,).
        Many coefficients may be exactly zero due to L1 regularization, achieving
        feature selection and model sparsity.

    Example:
        Basic LASSO regression with default parameters:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.regression.lasso import fit_lasso
            >>>
            >>> # Create sample data
            >>> X = jnp.array([[1.0, 2.0, 0.5],
            ...                [3.0, 4.0, 1.5],
            ...                [5.0, 6.0, 2.5]])
            >>> y = jnp.array([1.0, 2.0, 3.0])
            >>>
            >>> # Fit LASSO with defaults
            >>> model = fit_lasso(X, y)
            >>> print(model.weights)  # Some coefficients may be exactly zero

    Note:
        - Unlike Ridge regression, LASSO does not have a closed-form solution and requires
          iterative optimization
        - The algorithm may converge before num_iters if tol is reached
        - Feature scaling (standardization) is recommended before fitting LASSO

    See Also:
        soft_thresholding: The proximal operator used in each iteration.
        fit_ordinary_least_squares: No regularization (from regression.ols).
        LassoModel: Return type containing fitted weights.
        LassoParameters: Configuration dataclass for LASSO hyperparameters.

    References:
        Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
        Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.

        Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm
        for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    """
    lam = params.lam
    learning_rate: float = params.learning_rate
    num_iters: int = params.num_iters
    tol: float = params.tol
    momentum: float = params.momentum

    n_samples, n_features = X.shape
    w = jnp.zeros(n_features)
    v = jnp.zeros_like(w)  # momentum velocity

    def step(carry, _):
        """Perform one iteration of Nesterov accelerated proximal gradient descent."""
        w, v = carry

        # Nesterov lookahead: evaluate gradient at predicted future position
        w_lookahead = w + momentum * v

        # Gradient of MSE loss at lookahead position
        grad = (1.0 / n_samples) * X.T @ (X @ w_lookahead - y)

        # Update velocity with momentum decay and gradient
        v_new = momentum * v - learning_rate * grad

        # Apply proximal operator (soft-thresholding) for L1 penalty
        w_new = soft_thresholding(w + v_new, learning_rate * lam)

        # Check convergence
        delta = jnp.linalg.norm(w_new - w)
        done = delta < tol

        return (w_new, v_new), done

    def cond_fun(carry):
        """Check whether to continue optimization loop."""
        i, (w, v), done = carry
        return jnp.logical_and(i < num_iters, jnp.logical_not(done))

    def body_fun(carry):
        """Execute one iteration of the optimization loop."""
        i, (w, v), _ = carry
        (w_new, v_new), done = step((w, v), None)
        return (i + 1, (w_new, v_new), done)

    # Initialize optimization state
    carry = (0, (w, v), False)

    # Run optimization loop until convergence or maximum iterations
    carry = jax.lax.while_loop(cond_fun, body_fun, carry)

    # Extract final weights from optimization state
    w_final, _ = carry[1]

    return LassoModel(weights=w_final)
