"""Metalog distribution implementations and fitting functions.

This module provides the main metalog distribution classes and fitting functions:

Classes:
    Metalog: Standard metalog distribution fitted via regression methods.
    SPTMetalog: Symmetric Percentile Triplet metalog with closed-form coefficients.
    GridResult: Results container for grid search and hyperparameter optimization.

Functions:
    fit: Fit a standard metalog distribution using configurable regression methods.
    fit_spt_metalog: Fit an SPT metalog distribution using closed-form formulas.

The fit function uses a dispatch table pattern to route to the appropriate regression
implementation based on the MetalogFitMethod specified in MetalogParameters:

- MetalogFitMethod.OLS -> metalog_jax.regression.ols.fit_ordinary_least_squares
- MetalogFitMethod.Lasso -> metalog_jax.regression.lasso.fit_lasso
- MetalogFitMethod.Feasible -> metalog_jax.feasibility.a_star.fit_feasible

See Also:
    metalog_jax.base: Base classes and parameter configurations.
    metalog_jax.regression: Regression implementations for fitting.
"""
# Copyright: Travis Jefferies 2026

from collections.abc import Callable
from functools import partial
from typing import NoReturn

import chex
import jax
import jax.numpy as jnp
from flax import struct

from metalog_jax.base import (
    MetalogBase,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    SPTMetalogParameters,
)
from metalog_jax.feasibility.a_star import fit_feasible
from metalog_jax.regression import (
    RegularizedParameters,
    fit_lasso,
    fit_ordinary_least_squares,
)
from metalog_jax.regression.base import assert_full_column_rank
from metalog_jax.utils import (
    assert_condition,
    assert_numeric_array,
)

# Largest feasible |a3| / a2 of a 3-term metalog: the unique positive root of
# theta * ln((2 + theta) / (2 - theta)) = 4 (Baucells et al. 2025, Proposition 6;
# Keelin 2016 rounds it to 1.66711).
THETA_3 = 1.6671131192019295

# Dispatch table mapping fit methods to their implementation functions
_FIT_METHOD_DISPATCH = {
    MetalogFitMethod.OLS: fit_ordinary_least_squares,
    MetalogFitMethod.Lasso: fit_lasso,
    MetalogFitMethod.Feasible: fit_feasible,
}

# Fit methods whose implementation accepts regression hyperparameters (``params=``)
_METHODS_WITH_HYPERPARAMS = frozenset({MetalogFitMethod.Lasso})


def _get_fit_function(
    method: MetalogFitMethod,
    hyperparams: RegularizedParameters = None,
) -> Callable:
    """Get the appropriate fit function for the given method.

    Uses a dispatch table pattern to map MetalogFitMethod enum values to their
    corresponding regression implementation functions from the metalog_jax.regression
    submodules.

    The dispatch table (_FIT_METHOD_DISPATCH) maps:
        - MetalogFitMethod.OLS -> fit_ordinary_least_squares (from regression.ols)
        - MetalogFitMethod.Lasso -> fit_lasso (from regression.lasso)
        - MetalogFitMethod.Feasible -> fit_feasible (from feasibility.a_star)

    Args:
        method: The regression method to use. Must be a valid MetalogFitMethod enum.
        hyperparams: Optional regularization hyperparameters. When provided, the
            hyperparameters are bound to the fit function using functools.partial.
            Should be an instance of metalog_jax.regression.lasso.LassoParameters
            for Lasso method. Must be None for OLS and Feasible, which take none.

    Returns:
        Callable fit function, optionally with hyperparams bound via partial.

    Raises:
        TypeError: If method is not a valid MetalogFitMethod, or if hyperparams are
            given for a method that takes none.
    """
    try:
        fit_func = _FIT_METHOD_DISPATCH[method]
    except KeyError:
        raise TypeError(
            f"method type {type(method)} not type MetalogFitMethod!"
        ) from None

    if hyperparams is None:
        return fit_func
    if method not in _METHODS_WITH_HYPERPARAMS:
        raise TypeError(
            f"regression_hyperparams is only supported for MetalogFitMethod.Lasso; "
            f"MetalogFitMethod.{method.name} takes no hyperparameters, so pass "
            "regression_hyperparams=None."
        )
    return partial(fit_func, params=hyperparams)


@struct.dataclass
class SPTMetalog(MetalogBase):
    """Symmetric Percentile Triplet (SPT) metalog distribution.

    SPTMetalog is a specialized metalog implementation that uses exactly three
    terms fitted from three quantiles: the alpha-quantile, median (0.5), and
    (1-alpha)-quantile, where alpha < 0.5. This method provides a computationally
    efficient way to fit metalog distributions with closed-form coefficient
    solutions, avoiding the need for optimization or linear regression.

    The SPT approach is particularly useful when you need:
    - A quick approximation using only three data points
    - Guaranteed feasibility through explicit feasibility checks
    - Analytical coefficient formulas for different boundedness types
    - A distribution that matches specific symmetric or asymmetric quantile triplets

    Unlike the standard `Metalog` class which uses OLS or LASSO regression
    to fit arbitrary numbers of terms, `SPTMetalog` always has exactly 3 terms
    and computes coefficients directly from the quantile triplet. This makes it
    faster for small datasets but less flexible than the full metalog approach.

    Attributes:
        metalog_params: Configuration parameters specific to SPT fitting,
            including boundedness type, boundary values, and the alpha parameter
            that determines which quantiles are used (alpha, 0.5, 1-alpha).
        a: Coefficient vector of shape (3,) representing the three fitted
            metalog distribution parameters. These are computed using closed-form
            formulas specific to the boundedness type. A list or tuple is
            converted to a float JAX array on construction.

    Examples:
        Fit an unbounded SPT metalog from data:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.base import MetalogBoundedness, SPTMetalogParameters
            >>> from metalog_jax.metalog import SPTMetalog, fit_spt_metalog
            >>>
            >>> # Generate sample data
            >>> data = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1])
            >>>
            >>> # Configure SPT parameters
            >>> params = SPTMetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     alpha=0.1,  # Use 10th, 50th, and 90th percentiles
            ...     lower_bound=0.0,
            ...     upper_bound=0.0,
            ... )
            >>>
            >>> # Fit the SPT metalog
            >>> spt = fit_spt_metalog(data, params)
            >>>
            >>> # The fit passes through the data's 10th, 50th and 90th percentiles
            >>> spt.ppf(jnp.array([0.1, 0.5, 0.9])).round(2).tolist()
            [2.08, 4.1, 8.06]
            >>>
            >>> # Summary statistics are properties
            >>> mean, std = spt.mean, spt.std
            >>>
            >>> # Serialize and load it back (save and load do the same with a file)
            >>> SPTMetalog.loads(spt.dumps()) == spt
            True

        Show the fitted quantile function. ``print`` writes one line (wrapped
        here), ``repr`` the same line at full precision, and the ``print`` method
        one term per line. With a lower bound of 0, Q(y) = 0 + exp(M(y)):

            >>> lower = fit_spt_metalog(
            ...     data, params.replace(boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND)
            ... )
            >>> print(lower)  # doctest: +NORMALIZE_WHITESPACE
            SPTMetalog(3 terms, alpha = 0.1, lower bound 0): Q(y) = 0 + exp(M(y)),
            M(y) = 1.41099 + 0.30824 ln(y/(1-y)) - 0.00153176 (y - 0.5) ln(y/(1-y))
            >>> lower.print()
            SPTMetalog: 3 terms, alpha = 0.1, lower bound 0
              Q(y) = 0 + exp(M(y))
              M(y) = 1.41099
                     + 0.30824 ln(y/(1-y))
                     - 0.00153176 (y - 0.5) ln(y/(1-y))
            >>> lower.print(m=False, verbose=True)  # doctest: +NORMALIZE_WHITESPACE
            SPTMetalog: 3 terms, alpha = 0.1, lower bound 0
              Q(y) = 0 + exp(1.41099 + 0.30824 ln(y/(1-y))
                     - 0.00153176 (y - 0.5) ln(y/(1-y)))

    Note:
        SPT metalog distributions may fail feasibility checks if the data does
        not satisfy specific constraints on the symmetry ratio r. The feasibility
        constraints depend on the boundedness type and alpha parameter. When
        feasibility fails, `fit_spt_metalog` will raise an AssertionError.

        The SPT method is described in Keelin (2016) as a special case of the
        metalog family that enables rapid approximation with minimal data.

    See Also:
        Metalog: Standard metalog implementation using regression fitting.
        MetalogBase: Base class defining the common interface.
        fit_spt_metalog: Function to fit SPT metalog distributions from data.

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4),
        243-277. https://doi.org/10.1287/deca.2016.0338
    """

    metalog_params: SPTMetalogParameters
    a: chex.Numeric

    # load and loads rebuild SPTMetalogParameters, which save alpha but no fit method
    _params_type = SPTMetalogParameters

    def __eq__(self, other: object) -> bool:
        """Check equality between two SPTMetalog distribution instances.

        Delegates to the parent MetalogBase.__eq__ method.

        Args:
            other: The object to compare against.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        return MetalogBase.__eq__(self, other)

    def __repr__(self) -> str:
        """Show the SPTMetalog and its quantile function on one line, at full precision.

        Delegates to MetalogBase.__repr__, which flax.struct.dataclass would
        otherwise replace with a list of the dataclass fields.

        Returns:
            str: The one-line equation, or a summary if the model is traced or
                batched.
        """
        return MetalogBase.__repr__(self)

    @property
    def num_terms(self) -> int:
        """Get the number of terms in the metalog expansion.

        Returns:
            3 - an SPT Metalog always has three terms.
        """
        return 3

    @property
    def method(self) -> NoReturn:
        """An SPT metalog has no fit method.

        fit_spt_metalog computes the coefficients in closed form (Keelin 2016,
        Propositions 1-4) rather than by regression, so SPTMetalogParameters has
        no ``method``.

        Raises:
            AttributeError: Always, so ``hasattr(spt, "method")`` is False.
        """
        raise AttributeError(
            "SPTMetalog has no fit method: fit_spt_metalog computes its "
            "coefficients in closed form, not by regression"
        )


@struct.dataclass
class Metalog(MetalogBase):
    """Standard metalog distribution fitted via regression.

    Metalog is the full-featured metalog implementation that extends `MetalogBase`
    and uses ordinary least squares (OLS) or LASSO regression to fit metalog
    distributions with arbitrary numbers of terms (typically 2-30). This approach
    provides maximum flexibility and accuracy for modeling complex distributions
    from larger datasets.

    Unlike `SPTMetalog` which uses exactly 3 terms with closed-form coefficient
    formulas, the standard `Metalog` class:
    - Supports any number of terms (configurable via `num_terms` parameter)
    - Uses regression-based fitting (OLS or LASSO methods)
    - Can model more complex distribution shapes with higher-order terms
    - Requires more data for reliable fitting (recommended minimum: 3 x num_terms)
    - May encounter numerical issues with very high term counts

    This class inherits all distribution methods from `MetalogBase` including
    ppf (quantile function), cdf, pdf, and statistical moments (mean, variance,
    std, skewness, kurtosis). It also provides serialization capabilities for
    saving and loading fitted distributions.

    Attributes:
        metalog_params: Configuration parameters for the standard metalog fit,
            including boundedness type (UNBOUNDED, STRICTLY_LOWER_BOUND,
            STRICTLY_UPPER_BOUND, or BOUNDED), boundary values, fitting method
            (OLS or LASSO), and number of terms in the expansion.
        a: Coefficient vector of shape (num_terms,) representing the fitted
            metalog distribution parameters. Each coefficient corresponds to
            one basis function in the metalog expansion. These are computed
            via regression from the input data. A list or tuple is converted
            to a float JAX array on construction.

    Examples:
        Fit an unbounded metalog with 9 terms:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
            >>> from metalog_jax.base import MetalogInputData, MetalogParameters
            >>> from metalog_jax.metalog import fit
            >>> from metalog_jax.utils import DEFAULT_Y
            >>>
            >>> # Sample data
            >>> samples = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1, 10.5])
            >>> data = MetalogInputData.from_values(
            ...     samples, DEFAULT_Y, precomputed_quantiles=False
            ... )
            >>>
            >>> # Configure metalog parameters
            >>> params = MetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     method=MetalogFitMethod.Feasible,
            ...     num_terms=9,
            ...     lower_bound=0.0,
            ...     upper_bound=0.0,
            ... )
            >>>
            >>> # Fit the metalog distribution
            >>> m = fit(data, params)
            >>>
            >>> # Evaluate the quantile function at probability 0.75
            >>> round(float(m.ppf(jnp.array([0.75]))[0]), 2)
            7.4
            >>>
            >>> # Summary statistics are properties
            >>> mean, variance, skewness = m.mean, m.var, m.skewness

        Show the fitted quantile function of a 5-term OLS fit bounded on (0, 15).
        ``print`` writes one line (wrapped here), ``repr`` the same line at full
        precision, and the ``print`` method one term per line, with options:

            >>> bounded = fit(
            ...     data,
            ...     MetalogParameters(
            ...         boundedness=MetalogBoundedness.BOUNDED,
            ...         method=MetalogFitMethod.OLS,
            ...         num_terms=5,
            ...         lower_bound=0.0,
            ...         upper_bound=15.0,
            ...     ),
            ... )
            >>> print(bounded)  # doctest: +NORMALIZE_WHITESPACE
            Metalog(5 terms, OLS, bounded on (0, 15)): Q(y) = (0 + 15 exp(M(y))) /
            (1 + exp(M(y))), M(y) = -0.754464 + 0.0719273 ln(y/(1-y)) - 0.110759
            (y - 0.5) ln(y/(1-y)) + 2.45794 (y - 0.5) + 1.1187 (y - 0.5)^2
            >>> bounded.print()
            Metalog: 5 terms, OLS, bounded on (0, 15)
              Q(y) = (0 + 15 exp(M(y))) / (1 + exp(M(y)))
              M(y) = -0.754464
                     + 0.0719273 ln(y/(1-y))
                     - 0.110759 (y - 0.5) ln(y/(1-y))
                     + 2.45794 (y - 0.5)
                     + 1.1187 (y - 0.5)^2
            >>> bounded.print(m=False, verbose=True)  # doctest: +NORMALIZE_WHITESPACE
            Metalog: 5 terms, OLS, bounded on (0, 15)
              Q(y) = (0 + 15 exp(-0.754464 + 0.0719273 ln(y/(1-y))
                     - 0.110759 (y - 0.5) ln(y/(1-y)) + 2.45794 (y - 0.5)
                     + 1.1187 (y - 0.5)^2))
                     / (1 + exp(-0.754464 + 0.0719273 ln(y/(1-y))
                     - 0.110759 (y - 0.5) ln(y/(1-y)) + 2.45794 (y - 0.5)
                     + 1.1187 (y - 0.5)^2))

    Note:
        The standard `Metalog` class is the recommended choice for most use cases
        involving sufficient data. Use `SPTMetalog` only when you need a quick
        three-term approximation or when you have exactly three representative
        quantiles. For datasets with at least 10-30 observations, the standard
        metalog typically provides superior fit quality.

        Higher term counts generally improve accuracy but require more data and
        may cause overfitting or numerical instability. As a rule of thumb, ensure
        your sample size is at least 3 times the number of terms.

    See Also:
        SPTMetalog: Specialized three-term metalog using closed-form coefficients.
        MetalogBase: Base class defining the common distribution interface.
        fit_metalog: Function to fit standard metalog distributions from data.

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4),
        243-277. https://doi.org/10.1287/deca.2016.0338
    """

    metalog_params: MetalogParameters
    a: chex.Numeric

    def __eq__(self, other: object) -> bool:
        """Check equality between two Metalog distribution instances.

        Delegates to the parent MetalogBase.__eq__ method.

        Args:
            other: The object to compare against.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        return MetalogBase.__eq__(self, other)

    def __repr__(self) -> str:
        """Show the Metalog and its quantile function on one line, at full precision.

        Delegates to MetalogBase.__repr__, which flax.struct.dataclass would
        otherwise replace with a list of the dataclass fields.

        Returns:
            str: The one-line equation, or a summary if the model is traced or
                batched.
        """
        return MetalogBase.__repr__(self)

    @property
    def num_terms(self) -> int:
        """Get the number of terms in the metalog expansion.

        Returns:
            Integer representing the number of basis functions used in the
            metalog approximation. Higher values generally provide better
            accuracy but require more data.
        """
        return self.metalog_params.num_terms

    @property
    def method(self) -> MetalogFitMethod:
        """Get the method used to fit the metalog distribution.

        Returns:
            MetalogFitMethod enum of regression methods for fitting metalog distribution.
        """
        return self.metalog_params.method


def fit(
    data: MetalogInputData,
    metalog_params: MetalogParameters,
    regression_hyperparams: RegularizedParameters = None,
) -> Metalog:
    """Fit a metalog distribution to data using the specified configuration.

    Estimates the metalog distribution coefficients by solving a linear regression
    problem that maps the basis functions (target matrix) to the transformed quantiles.
    The regression method specified in metalog_params determines the fitting approach,
    and optional regression_hyperparams allow fine-tuning of regularization settings.

    This function processes a `MetalogInputData` instance containing validated input
    data (quantiles and probability levels) and fits a metalog distribution according
    to the specified parameters. The data must be created using `MetalogInputData.from_values()`
    which performs comprehensive validation.

    Args:
        data: Validated input data created via `MetalogInputData.from_values()`.
            Contains:
            - x: Quantile values (precomputed or computed from raw samples)
            - y: Probability levels in (0, 1)
            - precomputed_quantiles: Flag indicating data type
            Direct instantiation of `MetalogInputData` is prevented by `__post_init__`
            validation to ensure data integrity.
        metalog_params: Configuration parameters specifying the distribution
            characteristics (boundedness, bounds, number of terms) and the regression
            method to use for fitting. The method field determines which regression
            approach is used:
            - MetalogFitMethod.OLS: Ordinary Least Squares (no regularization)
            - MetalogFitMethod.Lasso: L1 regularization
            - MetalogFitMethod.Feasible: Best least-squares fit among valid metalogs
        regression_hyperparams: Optional regularization hyperparameters for controlling
            the fitting process when using LASSO regression method.
            Must be None for OLS and Feasible. If None, default hyperparameters
            are used. Must be an instance of LassoParameters for MetalogFitMethod.Lasso.

    Returns:
        Metalog containing the fitted coefficient vector and the configuration
        parameters used to fit the metalog distribution. The returned object is
        validated via ``assert_fitted()`` to ensure the fit produces a feasible
        distribution with strictly positive PDF values.

    Raises:
        TypeError: If metalog_params.method is not a valid MetalogFitMethod instance,
            or if regression_hyperparams is given for OLS or Feasible.
        checkify.JaxRuntimeError: If the fitted distribution produces non-positive
            PDF values, indicating an infeasible fit. This validation is performed
            by ``assert_fitted()`` before returning.
        checkify.JaxRuntimeError: For OLS and Feasible, if the basis matrix is rank
            deficient so the fit is not unique (e.g. n = k symmetric probabilities
            with 7, 11, 15, ... terms; Baucells et al. 2025, Proposition 2), or if
            Feasible cannot certify its result.

    Example:
        Basic usage with OLS (no regularization):

            >>> import jax.numpy as jnp
            >>> from metalog_jax.base import MetalogInputData, MetalogParameters
            >>> from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
            >>> from metalog_jax.metalog import fit
            >>>
            >>> # Create validated input data using from_values()
            >>> raw_data = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6])
            >>> data = MetalogInputData.from_values(
            ...     x=raw_data,
            ...     y=jnp.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
            ...     precomputed_quantiles=False  # Raw samples, not precomputed quantiles
            ... )
            >>>
            >>> # Configure metalog parameters with OLS
            >>> metalog_params = MetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     method=MetalogFitMethod.OLS,
            ...     lower_bound=0.0,
            ...     upper_bound=0.0,
            ...     num_terms=5
            ... )
            >>>
            >>> # Fit the metalog distribution
            >>> m = fit(data, metalog_params)

    Note:
        - The regression_hyperparams parameter is only applicable for LASSO method.
          Passing it with OLS or Feasible raises a TypeError.
        - If regression_hyperparams is None, sensible defaults are used for each method.
        - For production use, consider tuning regularization hyperparameters via
          cross-validation to prevent overfitting while maintaining good fit quality.

    See Also:
        fit_spt_metalog: Alternative fitting method using Symmetric Percentile Triplet.
        MetalogInputData.from_values: Required method for creating validated input data.
        metalog_jax.regression.lasso.LassoParameters: Hyperparameters for LASSO regression.
        metalog_jax.feasibility.best_feasible_fit: Best feasible fit (Baucells et al.).

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4),
        243-277. https://doi.org/10.1287/deca.2016.0338

        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """  # noqa: DOC502
    quantiles = Metalog.get_quantiles(data=data, metalog_params=metalog_params)
    target = Metalog.get_target(data=data, metalog_params=metalog_params)

    fit_method = _get_fit_function(metalog_params.method, regression_hyperparams)
    if metalog_params.method != MetalogFitMethod.Lasso:  # Lasso is regularized
        assert_full_column_rank(target)
    model = fit_method(target, quantiles)
    metalog = Metalog(metalog_params=metalog_params, a=model.weights)
    metalog.assert_fitted()
    return metalog


@jax.jit
def _spt_k_alpha(alpha: chex.Scalar) -> chex.Scalar:
    """Return k_alpha, the SPT feasibility threshold for the median's position.

    An SPT metalog is feasible exactly when the median's relative position ``r`` on
    the metalog's unbounded scale satisfies ``k_alpha < r < 1 - k_alpha``: the
    3-term condition ``|a3| / a2 < THETA_3`` rewritten for the triplet (Keelin 2016,
    equation 18). ``k_alpha = 0.5 * (1 - THETA_3 * (0.5 - alpha))`` lies in
    (0.0832, 0.5) for alpha in (0, 0.5).

    Args:
        alpha: Lower percentile parameter in (0, 0.5).

    Returns:
        The feasibility threshold k_alpha.
    """
    return 0.5 * (1 - (THETA_3 * (0.5 - alpha)))


@jax.jit
def _spt_median_position(z: chex.Array) -> chex.Scalar:
    """Return the median's relative position between the outer SPT quantiles.

    ``r = (z[1] - z[0]) / (z[2] - z[0])``. A symmetric triplet has r = 0.5. When
    r < 0.5 the median is closer to the lower quantile and the distribution is
    right-skewed; when r > 0.5 it is left-skewed.

    Args:
        z: Array of length 3 containing the alpha, 0.5 and 1 - alpha quantiles, in
            ascending order.

    Returns:
        The relative position r, in (0, 1) for an ascending triplet.
    """
    return (z[1] - z[0]) / (z[2] - z[0])


@jax.jit
def _spt_coefficients(z: chex.Array, alpha: chex.Scalar) -> chex.Array:
    """Return the 3-term metalog coefficients that pass through an SPT triplet.

    Implements Keelin (2016), Proposition 1 (equations 16 and 17), with
    ``L = log((1 - alpha) / alpha)`` and median position ``r``:

    - a1 = z[1]
    - a2 = (z[2] - z[0]) / (2L)
    - a3 = (1 - 2r) * (z[2] - z[0]) / ((1 - 2 * alpha) * L)

    Propositions 3 and 4 apply the same formulas to the log-transformed and
    logit-transformed quantiles, and the upper-bounded metalog does the same with
    -log(upper_bound - q), so ``z`` is the triplet on the metalog's unbounded
    scale for every boundedness type. Defined at module level so that it compiles
    once, not on every fit.

    Args:
        z: Array of length 3 containing the alpha, 0.5 and 1 - alpha quantiles on
            the metalog's unbounded scale, in ascending order.
        alpha: Lower percentile parameter in (0, 0.5).

    Returns:
        Array of length 3 containing the coefficients [a1, a2, a3].

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4),
        243-277. https://doi.org/10.1287/deca.2016.0338
    """
    log_odds = jnp.log((1 - alpha) / alpha)
    # (1 - 2r) * (z[2] - z[0]) simplifies to z[2] + z[0] - 2 * z[1]
    return jnp.array(
        [
            z[1],
            (z[2] - z[0]) / (2 * log_odds),
            (z[2] + z[0] - 2 * z[1]) / ((1 - 2 * alpha) * log_odds),
        ]
    )


def fit_spt_metalog(
    array: chex.Numeric,
    spt_metalog_params: SPTMetalogParameters,
    *,
    precomputed_quantiles: bool = False,
) -> SPTMetalog:
    """Fit a Symmetric Percentile Triplet (SPT) metalog to data or assessed quantiles.

    This function fits a 3-term metalog distribution using the Symmetric Percentile
    Triplet (SPT) parameterization method described in Keelin (2016). Unlike the
    standard metalog fit which uses many quantiles, the SPT method uses exactly three
    symmetric quantiles: alpha, 0.5 (median), and (1 - alpha), where 0 < alpha < 0.5.

    The SPT approach provides a quick, analytical solution for fitting a metalog
    distribution when:
    - You have limited data or only three quantiles available
    - You want a simple, closed-form solution without regression
    - You need a fast approximation with 3 terms only

    The method computes the 3 metalog coefficients directly from the three quantile
    values without requiring least squares regression. The formulas vary based on
    the boundedness type (unbounded, semi-bounded, or bounded).

    Args:
        array: Input data array with at least 3 elements from which to compute
            empirical quantiles. The array will be used to compute the alpha-th,
            50th, and (1-alpha)-th percentiles. If ``precomputed_quantiles`` is
            True, it is instead the triplet [q_alpha, q_0.5, q_1-alpha] itself,
            for example an expert's assessed 10th, 50th and 90th percentiles.
        spt_metalog_params: Configuration parameters containing:

            - alpha: Lower percentile parameter in (0, 0.5). Common values are
              0.1 (10-50-90 percentiles) or 0.25 (25-50-75 percentiles/IQR).
            - boundedness: Domain constraint type (UNBOUNDED, STRICTLY_LOWER_BOUND,
              STRICTLY_UPPER_BOUND or BOUNDED).
            - lower_bound: Lower boundary value (used for STRICTLY_LOWER_BOUND
              and BOUNDED).
            - upper_bound: Upper boundary value (used for STRICTLY_UPPER_BOUND
              and BOUNDED).
        precomputed_quantiles: Whether ``array`` holds the three quantiles
            themselves rather than data. Defaults to False.

    Returns:
        SPTMetalog: Fitted SPT metalog distribution containing:

            - metalog_params: The input configuration parameters
            - a: Coefficient vector of length 3 containing the fitted metalog
              parameters [a1, a2, a3] that define the quantile function

    Raises:
        AssertionError: If array has fewer than 3 elements, is not rank 1,
            or contains non-numeric values.
        AssertionError: If precomputed_quantiles is True and array doesn't
            have shape (3,).
        AssertionError: If alpha is not positive or alpha >= 0.5.
        AssertionError: If the computed quantiles violate feasibility constraints
            (q_alpha < median < q_complement).
        AssertionError: If a quantile lies outside the support: at or below
            lower_bound (STRICTLY_LOWER_BOUND and BOUNDED), or at or above
            upper_bound (STRICTLY_UPPER_BOUND and BOUNDED).
        AssertionError: If the triplet is infeasible for the boundedness type
            (Keelin 2016, Propositions 2-4, and the mirror image of
            Proposition 3 for STRICTLY_UPPER_BOUND).
        AssertionError: If the coefficients are not finite, for example because
            the data or bounds contain NaN.
        checkify.JaxRuntimeError: Under ``jax.vmap``, for the value checks above
            (array shape and type checks still raise AssertionError).

    Example:
        >>> import jax.numpy as jnp
        >>> from metalog_jax.metalog import fit_spt_metalog, SPTMetalogParameters
        >>> from metalog_jax.metalog import MetalogBoundedness
        >>> # Generate sample data
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> # Configure SPT parameters using 10-50-90 percentiles
        >>> params = SPTMetalogParameters(
        ...     alpha=0.1,
        ...     boundedness=MetalogBoundedness.UNBOUNDED,
        ...     lower_bound=0.0,
        ...     upper_bound=1.0
        ... )
        >>> # Fit the SPT metalog
        >>> spt_metalog = fit_spt_metalog(data, params)
        >>> # The result contains 3 coefficients
        >>> assert len(spt_metalog.a) == 3
        >>>
        >>> # Fit several datasets at once
        >>> import jax
        >>> batch = jnp.stack([data, 2 * data])
        >>> jax.vmap(lambda d: fit_spt_metalog(d, params).a)(batch).shape
        (2, 3)
        >>>
        >>> # Fit an expert's 10th, 50th and 90th percentile assessments directly
        >>> assessed = jnp.array([2.08, 4.1, 8.06])
        >>> spt = fit_spt_metalog(assessed, params, precomputed_quantiles=True)
        >>> spt.ppf(jnp.array([0.1, 0.5, 0.9])).round(2).tolist()
        [2.08, 4.1, 8.06]

    Note:
        - The SPT method always produces exactly 3 metalog terms, regardless of
          data size. For more flexible fits with more terms, use the standard
          fit() function.
        - The method performs multiple feasibility checks to ensure the quantiles
          and parameters produce a valid probability distribution.
        - Pass ``precomputed_quantiles=True`` for quantiles you already have. By
          default a 3-element array is treated as data, and ``jnp.quantile``
          interpolates between its three values, so the fit would not pass
          through them.
        - For unbounded distributions, the median position ratio r must satisfy
          specific constraints related to k_alpha to ensure feasibility, where
          r = (median - q_alpha) / (q_complement - q_alpha).
        - Works under ``jax.vmap`` and ``jax.jit`` over the data, alpha and
          bounds, and the parameters can be passed as a jit argument (boundedness
          is a static field, so each boundedness type compiles once). Inside
          transforms the value checks use ``checkify``, so wrap a jitted call
          with ``checkify.checkify`` and call ``throw()`` on the returned error.

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4).
    """  # noqa: DOC502

    def _quantile_feasibility_check(q: chex.Numeric) -> None:
        """Check that quantiles are monotonically increasing for valid distribution.

        Validates that the three SPT quantiles (q_alpha, median, q_complement)
        satisfy the fundamental ordering constraint required for a valid
        probability distribution: q_alpha < median < q_complement.

        Args:
            q: Array of length 3 containing [q_alpha, median, q_complement] where:
                - q_alpha is the quantile at probability alpha
                - median is the quantile at probability 0.5
                - q_complement is the quantile at probability (1 - alpha)

        Raises:
            AssertionError: If q_alpha < median < q_complement does not hold
                (checkify.JaxRuntimeError under jax.vmap).

        Note:
            This is a fundamental requirement for any valid cumulative distribution
            function. If this check fails, the input data or alpha parameter
            produces quantiles that cannot represent a valid distribution.
        """  # noqa: DOC502
        assert_condition(
            (q[0] < q[1]) & (q[1] < q[2]),
            "SPT quantiles must be strictly ascending: q_alpha < median < q_complement",
        )

    def _unbounded_fit(
        q: chex.Numeric,
        y: chex.Numeric,
    ) -> chex.Numeric:
        """Fit SPT metalog coefficients for unbounded distributions.

        Computes the three metalog coefficients [a1, a2, a3] for an unbounded
        distribution using the analytical SPT formulas. An unbounded distribution
        has support on the entire real line (-∞, ∞).

        The coefficients are computed as (Keelin 2016, Proposition 1):

        - a1 = median (q[1])
        - a2 = 0.5 * (q_complement - q_alpha) / log((1-alpha)/alpha)
        - a3 = (1 - 2r) * (q_complement - q_alpha) /
          ((1 - 2*alpha) * log((1-alpha)/alpha)),
          where r = (median - q_alpha) / (q_complement - q_alpha)

        Args:
            q: Array of length 3 containing [q_alpha, median, q_complement],
                the empirical quantiles at probabilities [alpha, 0.5, 1-alpha].
            y: Array of length 3 containing [alpha, 0.5, 1-alpha], the
                probability levels corresponding to the quantiles.

        Returns:
            Array of length 3 containing the fitted metalog coefficients [a1, a2, a3].

        Raises:
            AssertionError: If the median position ratio r violates feasibility
                constraints (must satisfy k_alpha < r < 1 - k_alpha).

        Note:
            The unbounded fit requires that the median position ratio r satisfies
            specific constraints to ensure a valid (positive) PDF everywhere.
        """  # noqa: DOC502

        def _unbounded_fit_feasibility_check(q: chex.Numeric, alpha: float) -> None:
            """Check feasibility constraints for unbounded SPT metalog.

            Validates that the median position ratio r satisfies the constraints
            required for an unbounded SPT metalog to have a valid (positive) PDF.
            The ratio r must fall within bounds determined by k_alpha.

            The constraints are:
            - (1 - k_alpha) > r
            - r > k_alpha

            where r = (median - q_alpha) / (q_complement - q_alpha)

            These ensure that the resulting metalog coefficients produce positive
            probability density across the entire real line.

            Args:
                q: Array of length 3 containing [q_alpha, median, q_complement].
                alpha: Lower percentile parameter in (0, 0.5).

            Raises:
                AssertionError: If r is not in (k_alpha, 1 - k_alpha)
                    (checkify.JaxRuntimeError under jax.vmap).

            Note:
                This check is essential to prevent fitting infeasible distributions
                that would have negative or undefined PDF values. If this check fails,
                the median position relative to the quantiles is too extreme for the
                given alpha value.
            """  # noqa: DOC502
            r = _spt_median_position(q)
            k_alpha = _spt_k_alpha(alpha)
            assert_condition(
                (k_alpha < r) & (r < 1 - k_alpha),
                "Infeasible SPT triplet: the median's position must lie in "
                "(k_alpha, 1 - k_alpha) (Keelin 2016, Proposition 2)",
            )

        alpha = y[0]
        _unbounded_fit_feasibility_check(q, alpha)
        return _spt_coefficients(q, alpha)

    def _strictly_lower_bound_fit(
        q: chex.Numeric,
        y: chex.Numeric,
        lower_bound: float,
    ) -> chex.Numeric:
        """Fit SPT metalog coefficients for strictly lower-bounded distributions.

        Computes the three metalog coefficients [a1, a2, a3] for a distribution
        with a strict lower bound using the analytical SPT formulas. The distribution
        has support on (lower_bound, ∞).

        The method first transforms the quantiles by subtracting the lower bound
        to get gamma values, then computes coefficients in log-space:
        - a1 = log(gamma_median)
        - a2 = 0.5 * log(gamma_complement / gamma_alpha) / log((1-alpha)/alpha)
        - a3 = log((gamma_complement * gamma_alpha) / gamma_median^2) /
               ((1 - 2*alpha) * log((1-alpha)/alpha))

        Args:
            q: Array of length 3 containing [q_alpha, median, q_complement],
                the empirical quantiles at probabilities [alpha, 0.5, 1-alpha].
                All values must be > lower_bound.
            y: Array of length 3 containing [alpha, 0.5, 1-alpha], the
                probability levels corresponding to the quantiles.
            lower_bound: The lower boundary value. All quantiles must exceed
                this value.

        Returns:
            Array of length 3 containing the fitted metalog coefficients [a1, a2, a3]
            in the transformed (log) space.

        Raises:
            AssertionError: If q_alpha is not above lower_bound.
            AssertionError: If the transformed quantiles (gamma values) violate
                feasibility constraints for the given lower bound.

        Note:
            The coefficients are those of Proposition 1 applied to log(gamma)
            (Keelin 2016, Proposition 3), computed by _spt_coefficients.
        """  # noqa: DOC502

        def _strictly_lower_bound_fit_feasibility_check(
            q: chex.Numeric, gamma: chex.Numeric, alpha: float, lower_bound: float
        ) -> None:
            """Check feasibility constraints for lower-bounded SPT metalog.

            Validates that the transformed quantiles (gamma values) satisfy the
            constraints required for a lower-bounded SPT metalog to have a valid
            (positive) PDF. The constraints ensure the median falls within bounds
            determined by k_alpha and the gamma values.

            The constraints are:
            - median < lower_bound + gamma_alpha^k_alpha * gamma_complement^(1-k_alpha)
            - median > lower_bound + gamma_alpha^(1-k_alpha) * gamma_complement^k_alpha

            These ensure positive probability density for all x > lower_bound.

            Args:
                q: Array of length 3 containing [q_alpha, median, q_complement].
                gamma: Array of length 3 containing transformed quantiles
                    [gamma_alpha, gamma_median, gamma_complement] where
                    gamma_i = q_i - lower_bound.
                alpha: Lower percentile parameter in (0, 0.5).
                lower_bound: The lower boundary value of the distribution.

            Raises:
                AssertionError: If the median is not strictly between the two
                    limits (checkify.JaxRuntimeError under jax.vmap).

            Note:
                This check uses weighted geometric means of the gamma values with
                weights k_alpha and (1-k_alpha) to establish feasibility bounds.
            """  # noqa: DOC502
            median = q[1]
            gamma_alpha = gamma[0]
            gamma_compliment = gamma[2]

            k_alpha = _spt_k_alpha(alpha)
            lower_limit = lower_bound + jnp.power(gamma_alpha, 1 - k_alpha) * jnp.power(
                gamma_compliment, k_alpha
            )
            upper_limit = lower_bound + jnp.power(gamma_alpha, k_alpha) * jnp.power(
                gamma_compliment, 1 - k_alpha
            )
            assert_condition(
                (lower_limit < median) & (median < upper_limit),
                "Infeasible SPT triplet for a lower bound (Keelin 2016, Proposition 3)",
            )

        alpha = y[0]

        q_alpha = q[0]
        median = q[1]
        q_compliment = q[2]

        # The quantiles ascend, so they all lie in the support if q_alpha does
        assert_condition(
            q_alpha > lower_bound, "SPT quantiles must lie above lower_bound"
        )

        gamma_alpha = q_alpha - lower_bound
        gamma_median = median - lower_bound
        gamma_compliment = q_compliment - lower_bound

        gamma = jnp.array([gamma_alpha, gamma_median, gamma_compliment])
        _strictly_lower_bound_fit_feasibility_check(q, gamma, alpha, lower_bound)
        return _spt_coefficients(jnp.log(gamma), alpha)

    def _strictly_upper_bound_fit(
        q: chex.Numeric,
        y: chex.Numeric,
        upper_bound: float,
    ) -> chex.Numeric:
        """Fit SPT metalog coefficients for strictly upper-bounded distributions.

        Computes the three metalog coefficients [a1, a2, a3] for a distribution
        with a strict upper bound. The distribution has support on
        (-∞, upper_bound).

        Keelin (2016, Section 4.1) defines the upper-bounded (negative-log)
        metalog through the transform z = -log(upper_bound - x), but states SPT
        formulas only for the unbounded, log and logit metalogs. As in
        Propositions 3 and 4, Proposition 1 applies to the transformed
        quantiles. With delta_i = upper_bound - q_i:
        - a1 = -log(delta_median)
        - a2 = 0.5 * log(delta_alpha / delta_complement) / log((1-alpha)/alpha)
        - a3 = log(delta_median^2 / (delta_alpha * delta_complement)) /
               ((1 - 2*alpha) * log((1-alpha)/alpha))

        Args:
            q: Array of length 3 containing [q_alpha, median, q_complement],
                the empirical quantiles at probabilities [alpha, 0.5, 1-alpha].
                All values must be < upper_bound.
            y: Array of length 3 containing [alpha, 0.5, 1-alpha], the
                probability levels corresponding to the quantiles.
            upper_bound: The upper boundary value. All quantiles must be below
                this value.

        Returns:
            Array of length 3 containing the fitted metalog coefficients [a1, a2, a3]
            in the transformed (negative-log) space.

        Raises:
            AssertionError: If q_complement is not below upper_bound.
            AssertionError: If the triplet violates the feasibility constraints
                for the given upper bound.

        Note:
            The coefficients are those of Proposition 1 applied to -log(delta),
            computed by _spt_coefficients.
        """  # noqa: DOC502

        def _strictly_upper_bound_fit_feasibility_check(
            q: chex.Numeric, delta: chex.Numeric, alpha: float, upper_bound: float
        ) -> None:
            """Check feasibility constraints for upper-bounded SPT metalog.

            The mirror image of Proposition 3. With delta_i = upper_bound - q_i,
            the triplet is feasible if and only if:
            - median > upper_bound - delta_alpha^(1-k_alpha) * delta_complement^k_alpha
            - median < upper_bound - delta_alpha^k_alpha * delta_complement^(1-k_alpha)

            That is k_alpha < r < 1 - k_alpha for the median's position r on the
            -log(delta) scale, which ensures positive density for all
            x < upper_bound.

            Args:
                q: Array of length 3 containing [q_alpha, median, q_complement].
                delta: Array of length 3 containing upper_bound - q, which
                    descends because q ascends.
                alpha: Lower percentile parameter in (0, 0.5).
                upper_bound: The upper boundary value of the distribution.

            Raises:
                AssertionError: If the median is not strictly between the two
                    limits (checkify.JaxRuntimeError under jax.vmap).
            """  # noqa: DOC502
            median = q[1]
            delta_alpha = delta[0]
            delta_complement = delta[2]

            k_alpha = _spt_k_alpha(alpha)
            lower_limit = upper_bound - jnp.power(delta_alpha, 1 - k_alpha) * jnp.power(
                delta_complement, k_alpha
            )
            upper_limit = upper_bound - jnp.power(delta_alpha, k_alpha) * jnp.power(
                delta_complement, 1 - k_alpha
            )
            assert_condition(
                (lower_limit < median) & (median < upper_limit),
                "Infeasible SPT triplet for an upper bound "
                "(the mirror image of Keelin 2016, Proposition 3)",
            )

        alpha = y[0]

        # The quantiles ascend, so they all lie in the support if q_complement does
        assert_condition(q[2] < upper_bound, "SPT quantiles must lie below upper_bound")

        delta = upper_bound - q
        _strictly_upper_bound_fit_feasibility_check(q, delta, alpha, upper_bound)
        return _spt_coefficients(-jnp.log(delta), alpha)

    def _bounded_fit(
        q: chex.Numeric,
        y: chex.Numeric,
        lower_bound: float,
        upper_bound: float,
    ) -> chex.Numeric:
        """Fit SPT metalog coefficients for fully bounded distributions.

        Computes the three metalog coefficients [a1, a2, a3] for a distribution
        with both lower and upper bounds using the analytical SPT formulas. The
        distribution has support on (lower_bound, upper_bound).

        The method first transforms the quantiles to gamma values using the logit
        transformation: gamma_i = (q_i - lower_bound) / (upper_bound - q_i),
        then computes coefficients in log-space:
        - a1 = log(gamma_median)
        - a2 = 0.5 * log(gamma_complement / gamma_alpha) / log((1-alpha)/alpha)
        - a3 = log((gamma_complement * gamma_alpha) / gamma_median^2) /
               ((1 - 2*alpha) * log((1-alpha)/alpha))

        Args:
            q: Array of length 3 containing [q_alpha, median, q_complement],
                the empirical quantiles at probabilities [alpha, 0.5, 1-alpha].
                All values must be in (lower_bound, upper_bound).
            y: Array of length 3 containing [alpha, 0.5, 1-alpha], the
                probability levels corresponding to the quantiles.
            lower_bound: The lower boundary value. All quantiles must exceed
                this value.
            upper_bound: The upper boundary value. All quantiles must be less
                than this value.

        Returns:
            Array of length 3 containing the fitted metalog coefficients [a1, a2, a3]
            in the transformed (log) space.

        Raises:
            AssertionError: If q_alpha is not above lower_bound, or q_complement
                is not below upper_bound.
            AssertionError: If the transformed quantiles (gamma values) violate
                feasibility constraints for the given bounds.

        Note:
            The logit-like transformation maps (lower_bound, upper_bound) to
            (0, ∞). The coefficients are those of Proposition 1 applied to
            log(gamma) (Keelin 2016, Proposition 4), computed by _spt_coefficients.
        """  # noqa: DOC502

        def _bounded_fit_feasibility_check(
            q: chex.Numeric,
            gamma: chex.Numeric,
            alpha: float,
            lower_bound: float,
            upper_bound: float,
        ) -> None:
            """Check feasibility constraints for bounded SPT metalog.

            Validates that the transformed quantiles (gamma values) satisfy the
            constraints required for a bounded SPT metalog to have a valid
            (positive) PDF. The constraints ensure the median falls within bounds
            determined by k_alpha, the gamma values, and the boundary values.

            The constraints are:
            - median < (lower + upper * gamma_alpha^k_alpha * gamma_complement^(1-k_alpha)) /
                      (1 + gamma_alpha^k_alpha * gamma_complement^(1-k_alpha))
            - median > (lower + upper * gamma_alpha^(1-k_alpha) * gamma_complement^k_alpha) /
                      (1 + gamma_alpha^(1-k_alpha) * gamma_complement^k_alpha)

            These ensure positive probability density for all x in (lower_bound, upper_bound).

            Args:
                q: Array of length 3 containing [q_alpha, median, q_complement].
                gamma: Array of length 3 containing transformed quantiles
                    [gamma_alpha, gamma_median, gamma_complement] where
                    gamma_i = (q_i - lower_bound) / (upper_bound - q_i).
                alpha: Lower percentile parameter in (0, 0.5).
                lower_bound: The lower boundary value of the distribution.
                upper_bound: The upper boundary value of the distribution.

            Raises:
                AssertionError: If the median is not strictly between the two
                    limits (checkify.JaxRuntimeError under jax.vmap).

            Note:
                This check uses weighted geometric means of the gamma values with
                weights k_alpha and (1-k_alpha), transformed back to the bounded
                domain using the inverse logit-like mapping.
            """  # noqa: DOC502
            median = q[1]
            gamma_alpha = gamma[0]
            gamma_compliment = gamma[2]
            k_alpha = _spt_k_alpha(alpha)
            # Weighted geometric means of gamma, mapped back to (lower, upper)
            g_lower = jnp.power(gamma_alpha, 1 - k_alpha) * jnp.power(
                gamma_compliment, k_alpha
            )
            g_upper = jnp.power(gamma_alpha, k_alpha) * jnp.power(
                gamma_compliment, 1 - k_alpha
            )
            lower_limit = (lower_bound + upper_bound * g_lower) / (1 + g_lower)
            upper_limit = (lower_bound + upper_bound * g_upper) / (1 + g_upper)
            assert_condition(
                (lower_limit < median) & (median < upper_limit),
                "Infeasible SPT triplet for bounds (Keelin 2016, Proposition 4)",
            )

        alpha = y[0]

        q_alpha = q[0]
        median = q[1]
        q_compliment = q[2]

        # The quantiles ascend, so they all lie in the support if q_alpha and
        # q_complement do; this also requires upper_bound > lower_bound
        assert_condition(
            q_alpha > lower_bound, "SPT quantiles must lie above lower_bound"
        )
        assert_condition(
            q_compliment < upper_bound, "SPT quantiles must lie below upper_bound"
        )

        gamma_alpha = (q_alpha - lower_bound) / (upper_bound - q_alpha)
        gamma_median = (median - lower_bound) / (upper_bound - median)
        gamma_compliment = (q_compliment - lower_bound) / (upper_bound - q_compliment)

        gamma = jnp.array([gamma_alpha, gamma_median, gamma_compliment])
        _bounded_fit_feasibility_check(q, gamma, alpha, lower_bound, upper_bound)
        return _spt_coefficients(jnp.log(gamma), alpha)

    chex.assert_axis_dimension_gteq(array, 0, 3)
    chex.assert_rank(array, 1)
    assert_numeric_array(array)

    alpha = spt_metalog_params.alpha
    median_percentile = 0.5

    assert_condition(
        (alpha > 0) & (alpha < median_percentile), "alpha must be in (0, 0.5)"
    )

    compliment = 1.0 - alpha
    y = jnp.array([alpha, median_percentile, compliment])
    if precomputed_quantiles:
        chex.assert_shape(array, (3,))
        q = jnp.asarray(array, dtype=float)
    else:
        q = jnp.quantile(array, y)
    _quantile_feasibility_check(q)

    boundedness = spt_metalog_params.boundedness
    lower_bound = spt_metalog_params.lower_bound
    upper_bound = spt_metalog_params.upper_bound

    if boundedness == MetalogBoundedness.UNBOUNDED:
        a = _unbounded_fit(q, y)
    elif boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
        a = _strictly_lower_bound_fit(q, y, lower_bound)
    elif boundedness == MetalogBoundedness.BOUNDED:
        a = _bounded_fit(q, y, lower_bound, upper_bound)
    else:  # MetalogBoundedness.STRICTLY_UPPER_BOUND
        a = _strictly_upper_bound_fit(q, y, upper_bound)

    # Infinite data can pass the checks above and still give infinite coefficients
    assert_condition(
        jnp.all(jnp.isfinite(a)),
        "SPT coefficients must be finite; check the data and bounds",
    )
    return SPTMetalog(metalog_params=spt_metalog_params, a=a)


@struct.dataclass
class GridResult:
    """Results from metalog grid search or hyperparameter optimization.

    GridResult encapsulates the outcome of fitting a metalog distribution along
    with a goodness-of-fit metric. This dataclass is typically used when performing
    grid searches over hyperparameters (e.g., number of terms, regularization
    penalties) or when comparing multiple metalog configurations to select the
    best-fitting distribution.

    The primary use case is hyperparameter tuning via grid search with JAX's vmap
    to fit multiple metalog configurations in parallel, then selecting the best
    configuration based on the Kolmogorov-Smirnov distance.

    The class uses Flax's `struct.dataclass` decorator to ensure immutability and
    compatibility with JAX transformations (jit, vmap, grad), making it suitable
    for use in vectorized grid search operations.

    Attributes:
        metalog: The fitted metalog distribution. Can be either a standard Metalog
            (fitted via regression with configurable terms) or SPTMetalog (fitted
            using Symmetric Percentile Triplet with exactly 3 terms). Contains:
            - metalog_params: Configuration parameters (boundedness, method, etc.)
            - a: Coefficient vector representing the fitted distribution
        ks_dist: Scalar Kolmogorov-Smirnov distance measuring the goodness-of-fit
            between the fitted metalog distribution and the input random variable.
            This is the maximum absolute difference between the empirical CDF of
            the input data and the fitted metalog's CDF:
            - 0: Perfect fit (empirical CDFs are identical)
            - 1: Worst possible fit (completely non-overlapping distributions)
            - Lower values indicate better fit quality
            Used to select the best metalog configuration from a grid search.

    Note:
        - This class is immutable due to the `struct.dataclass` decorator
        - The ks_dist attribute provides a single goodness-of-fit metric for
          comparing different metalog configurations
        - When used with jax.vmap, the metalog and ks_dist fields will be
          batched arrays/PyTrees, allowing vectorized comparison
        - Lower KS distances indicate better fit quality, but extremely low
          values may indicate overfitting (especially with high term counts)
        - For production use, consider cross-validation instead of or in
          addition to KS distance for hyperparameter selection

    See Also:
        ks_distance: Function to compute the Kolmogorov-Smirnov distance.
        fit: Standard metalog fitting function.
        fit_spt_metalog: SPT metalog fitting function.
        Metalog: Standard metalog distribution class.
        SPTMetalog: Symmetric Percentile Triplet metalog class.

    References:
        Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge
        di distribuzione". Giornale dell'Istituto Italiano degli Attuari, 4: 83-91.

        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4),
        243-277. https://doi.org/10.1287/deca.2016.0338
    """

    metalog: Metalog | SPTMetalog
    ks_dist: chex.Scalar
