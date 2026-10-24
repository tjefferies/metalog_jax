# Copyright: Travis Jefferies 2026
"""LASSO regression implementation with L1 regularization.

This module provides LASSO (Least Absolute Shrinkage and Selection Operator)
regression. fit_lasso solves it exactly by feature-sign search, an active-set
method (Lee, Battle, Raina and Ng 2007), and reports whether it converged.

Classes:
    LassoParameters: Hyperparameters for LASSO regression.
    LassoModel: A fitted LASSO model: its weights, and whether it converged.

Functions:
    fit_lasso: Fit a LASSO regression model by feature-sign search.

Constants:
    DEFAULT_LASSO_LAMBDA: Default L1 regularization strength (0).
    DEFAULT_LASSO_ITERATIONS: Default maximum number of active-set steps (500).
    DEFAULT_LASSO_TOLERANCE: Default optimality tolerance (1e-6).
    DEFAULT_LASSO_PARAMETERS: Default LassoParameters instance.

This module is used when MetalogFitMethod.Lasso is specified in MetalogParameters.

See Also:
    metalog_jax.regression.ols: OLS regression (no regularization).
    metalog_jax.metalog.fit: High-level fitting function that dispatches to LASSO.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct

from metalog_jax.regression.base import RegressionModel, RegularizedParameters

DEFAULT_LASSO_LAMBDA: float = 0.0
DEFAULT_LASSO_ITERATIONS: int = 500
DEFAULT_LASSO_TOLERANCE: float = 1e-6


@struct.dataclass
class LassoParameters(RegularizedParameters):
    """Structure for Lasso parameters.

    Attributes:
        lam: L1 regularization strength λ ≥ 0. Controls sparsity of the solution:
            - λ = 0: No regularization; the fit is the least-squares fit
            - λ > 0: Promotes sparsity; larger values → more coefficients set to zero
            - Typical values: 0.001 to 10.0, depending on data scale
        num_iters: Maximum number of active-set steps. Default: 500. Each step
            adds or removes coefficients from the active set, so a fit takes about
            as many steps as it has nonzero coefficients.
        tol: Optimality tolerance, relative to ``max |Xᵀy| / n``. Default: 1e-6. The
            fit has converged when every coefficient satisfies the Lasso's
            optimality (KKT) conditions to within it.
    """

    lam: float
    num_iters: int = DEFAULT_LASSO_ITERATIONS
    tol: float = DEFAULT_LASSO_TOLERANCE


DEFAULT_LASSO_PARAMETERS = LassoParameters(
    lam=DEFAULT_LASSO_LAMBDA,
    num_iters=DEFAULT_LASSO_ITERATIONS,
    tol=DEFAULT_LASSO_TOLERANCE,
)


@struct.dataclass
class LassoModel(RegressionModel):
    """Structure for a fitted LASSO regression model.

    Attributes:
        weights: Coefficient vector of shape (n_features,).
        converged: Whether the optimality conditions hold to within ``tol``. False
            when fit_lasso stops at ``num_iters`` steps, or when the data aren't
            finite. None for a model built by hand.
        iterations: The number of active-set steps taken. None for a model built
            by hand.
    """

    converged: chex.Array | None = None
    iterations: chex.Array | None = None


def _lasso_objective(
    R: chex.Array, c: chex.Array, n: int, lam: chex.Array, w: chex.Array
) -> chex.Array:
    """(1/2n) ||Rw - c||² + Σ lam_j |w_j|, for w or each row of a batch of w.

    With X = QR and c = Qᵀy, this differs from (1/2n) ||Xw - y||² + Σ lam_j |w_j|
    by a constant, the part of y outside X's column space.

    Args:
        R: The R factor of X's QR decomposition.
        c: Qᵀy.
        n: The number of samples.
        lam: The penalty on each coefficient.
        w: Coefficients, shape (k,) or (m, k).

    Returns:
        chex.Array: The objective, shape () or (m,).
    """
    residual = jnp.sum((w @ R.T - c) ** 2, axis=-1)
    return residual / (2 * n) + jnp.sum(lam * jnp.abs(w), axis=-1)


def _kkt_violations(
    R: chex.Array,
    c: chex.Array,
    n: int,
    lam: chex.Array,
    w: chex.Array,
    active: chex.Array,
    sign: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """How far w is from the Lasso's optimality (KKT) conditions.

    With g the gradient of the smooth part, an active coefficient needs
    g_j + lam_j sign_j = 0, and a zero coefficient needs |g_j| <= lam_j.

    Args:
        R: The R factor of X's QR decomposition.
        c: Qᵀy.
        n: The number of samples.
        lam: The penalty on each coefficient.
        w: Coefficients.
        active: Which coefficients are in the active set.
        sign: The signs of the active coefficients.

    Returns:
        tuple: ``(g, worst_active, excess)``: the gradient, the largest violation
        among active coefficients, and |g_j| - lam_j for each inactive one (-inf
        for active ones).
    """
    g = R.T @ (R @ w - c) / n
    worst_active = jnp.max(jnp.where(active, jnp.abs(g + lam * sign), 0.0))
    excess = jnp.where(active, -jnp.inf, jnp.abs(g) - lam)
    return g, worst_active, excess


def _signed_least_squares(
    R: chex.Array,
    c: chex.Array,
    n: int,
    lam: chex.Array,
    active: chex.Array,
    sign: chex.Array,
) -> chex.Array:
    """Minimize the objective over the active set, with the signs held fixed.

    The minimizer solves R_AᵀR_A w_A = R_Aᵀc - n lam_A sign_A. The
    pseudo-inverse P of R with its inactive columns zeroed gives
    w = Pc - n P Pᵀ (lam sign) without forming R_AᵀR_A, so the solve is as well
    conditioned as R itself, and inactive coefficients stay 0.

    Args:
        R: The R factor of X's QR decomposition.
        c: Qᵀy.
        n: The number of samples.
        lam: The penalty on each coefficient.
        active: Which coefficients are in the active set.
        sign: The signs of the active coefficients.

    Returns:
        chex.Array: The coefficients, 0 outside the active set.
    """
    P = jnp.linalg.pinv(R * active)
    w = P @ c - n * P @ (P.T @ (lam * sign))
    return jnp.where(active, w, 0.0)


def _line_search(
    R: chex.Array,
    c: chex.Array,
    n: int,
    lam: chex.Array,
    w: chex.Array,
    w_new: chex.Array,
) -> chex.Array:
    """The best point on the segment from w to w_new.

    The objective is quadratic between the points where a penalized coefficient
    crosses zero, so its minimum on the segment is w_new or one of those points.
    Each crossing point has its coefficient set exactly to 0.

    Args:
        R: The R factor of X's QR decomposition.
        c: Qᵀy.
        n: The number of samples.
        lam: The penalty on each coefficient.
        w: The current coefficients.
        w_new: The signed least-squares solution.

    Returns:
        chex.Array: The candidate with the smallest objective.
    """
    d = w_new - w
    crosses = (lam > 0) & (w != 0) & (jnp.sign(w_new) != jnp.sign(w))
    t = jnp.where(crosses, -w / jnp.where(crosses, d, 1.0), jnp.inf)
    t = jnp.where((t > 0) & (t < 1), t, jnp.inf)
    # Row j is the point where coefficient j reaches 0, or w_new if it doesn't
    points = w + jnp.minimum(t, 1.0)[:, None] * d
    at_zero = jnp.diag(jnp.isfinite(t))
    points = jnp.where(at_zero, 0.0, points)
    candidates = jnp.concatenate([points, w_new[None, :]])
    return candidates[jnp.argmin(_lasso_objective(R, c, n, lam, candidates))]


@partial(jax.jit, static_argnames=("intercept",))
def fit_lasso(
    X: chex.Numeric,
    y: chex.Numeric,
    params: LassoParameters = DEFAULT_LASSO_PARAMETERS,
    *,
    intercept: bool = False,
) -> LassoModel:
    """Fit a LASSO regression model by feature-sign search.

    Minimizes the LASSO objective exactly::

        (1/2n) ||y - Xw||² + λ Σ_j |w_j|

    The solver is the feature-sign search of Lee, Battle, Raina and Ng (2007), an
    active-set method. Each step:

    1. Adds the zero coefficient that most violates its optimality condition to
       the active set, once the active coefficients are optimal.
    2. Solves least squares on the active set with the signs of its coefficients
       fixed, a linear solve.
    3. Moves to the best point between the current coefficients and that
       solution: the solution, or a point where a coefficient reaches zero, which
       then leaves the active set.

    Each step lowers the objective, so the active set never repeats, and the
    search ends in a finite number of steps at the exact minimizer. Unlike
    gradient methods, it needs no step size, and badly conditioned designs such
    as the metalog basis don't slow it down. At λ = 0 it is the least-squares fit,
    in one step.

    The design is reduced by a QR decomposition first, so every step works with
    a k x k matrix, and solves are as well conditioned as X itself.

    Args:
        X: Design matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        params: LassoParameters: the penalty ``lam``, the step limit ``num_iters``
            and the optimality tolerance ``tol``.
        intercept: Whether X's first column is an intercept, whose coefficient
            isn't penalized, as for scikit-learn's ``fit_intercept``. Static.

    Returns:
        LassoModel: The weights, of shape (n_features,), with whether the
        optimality conditions hold to within ``tol`` (``converged``) and the
        number of steps taken (``iterations``).

    Example:
        At λ = 0 the fit is the least-squares fit:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.regression.lasso import LassoParameters, fit_lasso
            >>>
            >>> X = jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5], [1.0, 4.0]])
            >>> y = jnp.array([1.0, 2.0, 2.5, 4.0])
            >>> model = fit_lasso(X, y)
            >>> bool(model.converged)
            True
            >>> bool(jnp.allclose(model.weights, jnp.linalg.lstsq(X, y)[0]))
            True

        A penalty shrinks the slope, and with ``intercept=True`` leaves the first
        column's coefficient unpenalized:

            >>> model = fit_lasso(X, y, LassoParameters(lam=0.5), intercept=True)
            >>> [round(float(v), 4) for v in model.weights]
            [1.243, 0.5327]

    Note:
        - Feature scaling (standardization) is recommended before fitting LASSO,
          since the penalty treats every coefficient alike.
        - Works under ``jax.jit`` and ``jax.vmap``, for example over penalties in
          the grid search; the steps run in a ``while_loop``.

    See Also:
        fit_ordinary_least_squares: No regularization (from regression.ols).
        LassoModel: Return type containing the weights and convergence.
        LassoParameters: Configuration dataclass for LASSO hyperparameters.

    References:
        Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
        Journal of the Royal Statistical Society: Series B (Methodological), 58(1),
        267-288.

        Lee, H., Battle, A., Raina, R., & Ng, A. Y. (2007). Efficient sparse coding
        algorithms. Advances in Neural Information Processing Systems 19, 801-808.
    """
    n, k = X.shape
    Q, R = jnp.linalg.qr(X)
    c = Q.T @ y
    penalized = jnp.ones(k)
    if intercept:
        penalized = penalized.at[0].set(0.0)
    lam = params.lam * penalized
    kkt_tol = params.tol * jnp.max(jnp.abs(R.T @ c)) / n + jnp.finfo(c.dtype).tiny

    def converged(w, active, sign):
        _, worst_active, excess = _kkt_violations(R, c, n, lam, w, active, sign)
        return (worst_active <= kkt_tol) & (jnp.max(excess) <= kkt_tol)

    def step(state):
        w, active, sign, steps, _ = state
        g, worst_active, excess = _kkt_violations(R, c, n, lam, w, active, sign)
        # Once the active coefficients are optimal, add the worst zero coefficient
        j = jnp.argmax(excess)
        enter = (worst_active <= kkt_tol) & (excess[j] > kkt_tol)
        active = active.at[j].set(active[j] | enter)
        sign = sign.at[j].set(jnp.where(enter, -jnp.sign(g[j]), sign[j]))
        w_new = _signed_least_squares(R, c, n, lam, active, sign)
        w = _line_search(R, c, n, lam, w, w_new)
        # Coefficients that reached zero leave; unpenalized ones always stay
        active = (w != 0) | (lam == 0)
        sign = jnp.where(active, jnp.sign(w), 0.0)
        return w, active, sign, steps + 1, converged(w, active, sign)

    def keep_going(state):
        w, _, _, steps, done = state
        return ~done & (steps < params.num_iters) & jnp.all(jnp.isfinite(w))

    w0 = jnp.zeros(k, dtype=c.dtype)
    active0 = lam == 0
    sign0 = jnp.zeros(k, dtype=c.dtype)
    start = (w0, active0, sign0, 0, converged(w0, active0, sign0))
    w, _, _, steps, done = jax.lax.while_loop(keep_going, step, start)
    return LassoModel(weights=w, converged=done, iterations=steps)
