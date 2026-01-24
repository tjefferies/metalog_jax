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

See Also:
    metalog_jax.base: Base classes and parameter configurations.
    metalog_jax.regression: Regression implementations for fitting.
"""
# Copyright: Travis Jefferies 2026

from functools import partial
from typing import Callable, Union

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
from metalog_jax.regression import (
    RegularizedParameters,
    fit_lasso,
    fit_ordinary_least_squares,
)
from metalog_jax.utils import (
    assert_numeric_array,
)

# Dispatch table mapping fit methods to their implementation functions
_FIT_METHOD_DISPATCH = {
    MetalogFitMethod.OLS: fit_ordinary_least_squares,
    MetalogFitMethod.Lasso: fit_lasso,
}


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

    Args:
        method: The regression method to use. Must be a valid MetalogFitMethod enum.
        hyperparams: Optional regularization hyperparameters. When provided, the
            hyperparameters are bound to the fit function using functools.partial.
            Should be an instance of metalog_jax.regression.lasso.LassoParameters
            for Lasso method. Ignored for OLS method.

    Returns:
        Callable fit function, optionally with hyperparams bound via partial.

    Raises:
        TypeError: If method is not a valid MetalogFitMethod.
    """
    try:
        fit_func = _FIT_METHOD_DISPATCH[method]
    except KeyError:
        raise TypeError(
            f"method type {type(method)} not type MetalogFitMethod!"
        ) from None

    if hyperparams is not None:
        return partial(fit_func, params=hyperparams)
    return fit_func


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
            formulas specific to the boundedness type.

    Examples:
        Fit an unbounded SPT metalog from data:

            >>> import jax.numpy as jnp
            >>> from metalog_jax import fit_spt_metalog, SPTMetalogParameters
            >>> from metalog_jax import MetalogBoundedness
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
            >>> # Evaluate quantile function at probability 0.25
            >>> q_25 = spt.ppf(0.25)
            >>>
            >>> # Compute mean and standard deviation
            >>> mean = spt.mean()
            >>> std = spt.std()

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

    def __eq__(self, other: object) -> bool:
        """Check equality between two SPTMetalog distribution instances.

        Delegates to the parent MetalogBase.__eq__ method.

        Args:
            other: The object to compare against.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        return MetalogBase.__eq__(self, other)

    @property
    def num_terms(self) -> int:
        """Get the number of terms in the metalog expansion.

        Returns:
            3 - an SPT Metalog always has three terms.
        """
        return 3


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
            via regression from the input data.

    Examples:
        Fit an unbounded metalog with 9 terms using OLS:

            >>> import jax.numpy as jnp
            >>> from metalog_jax import fit_metalog, MetalogParameters
            >>> from metalog_jax import MetalogBoundedness, MetalogFitMethod
            >>>
            >>> # Generate sample data
            >>> data = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1, 10.5])
            >>>
            >>> # Configure metalog parameters
            >>> params = MetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     method=MetalogFitMethod.OLS,
            ...     num_terms=9,
            ...     lower_bound=0.0,
            ...     upper_bound=0.0,
            ... )
            >>>
            >>> # Fit the metalog distribution
            >>> m = fit_metalog(data, params)
            >>>
            >>> # Evaluate quantile function at probability 0.75
            >>> q_75 = m.ppf(0.75)
            >>>
            >>> # Compute distribution moments
            >>> mean = m.mean()
            >>> variance = m.variance()
            >>> skewness = m.skewness()

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
        regression_hyperparams: Optional regularization hyperparameters for controlling
            the fitting process when using LASSO regression method.
            This parameter is ignored when method=OLS. If None, default hyperparameters
            are used. Must be an instance of LassoParameters for MetalogFitMethod.Lasso.

    Returns:
        Metalog containing the fitted coefficient vector and the configuration
        parameters used to fit the metalog distribution. The returned object is
        validated via ``assert_fitted()`` to ensure the fit produces a feasible
        distribution with strictly positive PDF values.

    Raises:
        TypeError: If metalog_params.method is not a valid MetalogFitMethod instance.
        checkify.JaxRuntimeError: If the fitted distribution produces non-positive
            PDF values, indicating an infeasible fit. This validation is performed
            by ``assert_fitted()`` before returning.

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
          It is ignored when using OLS.
        - If regression_hyperparams is None, sensible defaults are used for each method.
        - For production use, consider tuning regularization hyperparameters via
          cross-validation to prevent overfitting while maintaining good fit quality.

    See Also:
        fit_spt_metalog: Alternative fitting method using Symmetric Percentile Triplet.
        MetalogInputData.from_values: Required method for creating validated input data.
        metalog_jax.regression.lasso.LassoParameters: Hyperparameters for LASSO regression.
    """
    quantiles = Metalog.get_quantiles(data=data, metalog_params=metalog_params)
    target = Metalog.get_target(data=data, metalog_params=metalog_params)

    fit_method = _get_fit_function(metalog_params.method, regression_hyperparams)
    model = fit_method(target, quantiles)
    metalog = Metalog(metalog_params=metalog_params, a=model.weights)
    metalog.assert_fitted()
    return metalog


def fit_spt_metalog(
    array: chex.Numeric,
    spt_metalog_params: SPTMetalogParameters,
) -> SPTMetalog:
    """Fit a Symmetric Percentile Triplet (SPT) metalog distribution to data.

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
            50th, and (1-alpha)-th percentiles.
        spt_metalog_params: Configuration parameters containing:
            - alpha: Lower percentile parameter in (0, 0.5). Common values are
              0.1 (10-50-90 percentiles) or 0.25 (25-50-75 percentiles/IQR).
            - boundedness: Domain constraint type (UNBOUNDED, STRICTLY_LOWER_BOUND,
              or BOUNDED). STRICTLY_UPPER_BOUND is not supported in SPT formulation.
            - lower_bound: Lower boundary value (used for semi-bounded/bounded).
            - upper_bound: Upper boundary value (used for bounded distributions).

    Returns:
        SPTMetalog: Fitted SPT metalog distribution containing:
            - metalog_params: The input configuration parameters
            - a: Coefficient vector of length 3 containing the fitted metalog
              parameters [a1, a2, a3] that define the quantile function

    Raises:
        AssertionError: If array has fewer than 3 elements, is not rank 1,
            or contains non-numeric values.
        AssertionError: If alpha is not positive or alpha >= 0.5.
        AssertionError: If the computed quantiles violate feasibility constraints
            (q_alpha < median < q_complement).
        AssertionError: If boundedness-specific feasibility checks fail
            (e.g., quantiles outside valid range for the given bounds).
        NotImplementedError: If boundedness is STRICTLY_UPPER_BOUND, which is
            undefined in the SPT formulation.

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

    Note:
        - The SPT method always produces exactly 3 metalog terms, regardless of
          data size. For more flexible fits with more terms, use the standard
          fit() function.
        - STRICTLY_UPPER_BOUND boundedness is not supported by the SPT formulation
          as defined in Keelin's paper.
        - The method performs multiple feasibility checks to ensure the quantiles
          and parameters produce a valid probability distribution.
        - For unbounded distributions, the median position ratio r must satisfy
          specific constraints related to k_alpha to ensure feasibility, where
          r = (median - q_alpha) / (q_complement - q_alpha).

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4).
    """

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
            AssertionError: If median <= q_alpha (violates q_alpha < median).
            AssertionError: If q_complement <= median (violates median < q_complement).

        Note:
            This is a fundamental requirement for any valid cumulative distribution
            function. If this check fails, the input data or alpha parameter
            produces quantiles that cannot represent a valid distribution.
        """
        q_alpha = q[0]
        median = q[1]
        q_compliment = q[2]
        chex.assert_scalar_positive(float(q_compliment - median))
        chex.assert_scalar_positive(float(median - q_alpha))

    @jax.jit
    def _k_alpha(alpha) -> float:
        """Compute the k_alpha correction factor for SPT feasibility constraints.

        Calculates a correction factor used in feasibility checks for bounded and
        semi-bounded SPT metalog distributions. This factor appears in the constraints
        that ensure the fitted distribution produces valid probability densities.

        The formula is: k_alpha = 0.5 * (1 - 1.66711) * (0.5 - alpha)

        The constant 1.66711 appears in Keelin's SPT formulation and represents
        a theoretical bound related to the metalog basis functions.

        Args:
            alpha: Lower percentile parameter in (0, 0.5), representing the
                symmetric percentile triplet's lower tail probability.

        Returns:
            Correction factor k_alpha used in feasibility constraints. The value
            is always negative for valid alpha in (0, 0.5).

        Note:
            This function is JIT-compiled for performance. The correction factor
            is used to check whether quantiles fall within acceptable ranges that
            guarantee a valid PDF (positive density everywhere).
        """
        return 0.5 * (1 - (1.66711 * (0.5 - alpha)))

    @jax.jit
    def _log_term(alpha: float) -> float:
        """Compute log((1 - alpha) / alpha) for SPT coefficient calculations.

        Calculates the natural logarithm of the odds ratio (1-alpha)/alpha,
        which appears in the analytical formulas for SPT metalog coefficients.
        This term captures the asymmetry in the symmetric percentile triplet.

        Args:
            alpha: Lower percentile parameter in (0, 0.5). Must be positive
                and less than 0.5.

        Returns:
            Natural logarithm of (1 - alpha) / alpha. The value is always
            positive for valid alpha in (0, 0.5), since (1-alpha)/alpha > 1.

        Note:
            This function is JIT-compiled for performance. It is used in
            computing the second and third metalog coefficients for all
            boundedness types.
        """
        return jnp.log((1 - alpha) / alpha)

    @jax.jit
    def _dist_symmetry(q: chex.Numeric) -> chex.Numeric:
        """Compute the symmetry ratio of the distribution from quantiles.

        Calculates r = (median - q_alpha) / (q_complement - q_alpha), which represents
        where the median falls within the range between the lower and upper quantiles.
        This ratio measures the symmetry of the distribution and is used in SPT metalog
        feasibility checks and coefficient calculations.

        When r = 0.5, the median is exactly centered between q_alpha and q_complement,
        indicating a symmetric distribution. Values of r < 0.5 indicate left skew
        (median closer to the lower quantile), while r > 0.5 indicates right skew
        (median closer to the upper quantile).

        Args:
            q: Array of length 3 containing [q_alpha, median, q_complement] where:
                - q[0] is the quantile at probability alpha
                - q[1] is the median (quantile at 0.5)
                - q[2] is the quantile at probability (1 - alpha)

        Returns:
            Symmetry ratio r = (median - q_alpha) / (q_complement - q_alpha), a value
            in [0, 1] representing the relative position of the median. For a valid
            SPT metalog distribution, this must be in the range (k_alpha, 1 - k_alpha)
            where k_alpha is the feasibility threshold.

        Note:
            This function is JIT-compiled for performance. The function name reflects
            that it measures distribution symmetry; r = 0.5 indicates perfect symmetry.
            The actual quantile spread (range) is computed as (q[2] - q[0]).
        """
        return (q[1] - q[0]) / (q[2] - q[0])

    def _unbounded_fit(
        q: chex.Numeric,
        y: chex.Numeric,
    ) -> chex.Numeric:
        """Fit SPT metalog coefficients for unbounded distributions.

        Computes the three metalog coefficients [a1, a2, a3] for an unbounded
        distribution using the analytical SPT formulas. An unbounded distribution
        has support on the entire real line (-∞, ∞).

        The coefficients are computed as:
        - a1 = median (q[1])
        - a2 = 0.5 * r / log((1-alpha)/alpha)
        - a3 = r * (1 - 2r) / ((1 - 2*alpha) * log((1-alpha)/alpha))
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
        """

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
                AssertionError: If (1 - k_alpha) - r <= 0.
                AssertionError: If r - k_alpha <= 0.

            Note:
                This check is essential to prevent fitting infeasible distributions
                that would have negative or undefined PDF values. If this check fails,
                the median position relative to the quantiles is too extreme for the
                given alpha value.
            """
            r = _dist_symmetry(q)
            k_alpha = _k_alpha(alpha)
            chex.assert_scalar_positive(float((1 - k_alpha) - r))
            chex.assert_scalar_positive(float(r - k_alpha))

        @jax.jit
        def _second_term(q: chex.Numeric, alpha: float) -> float:
            """Compute the second metalog coefficient (a2) for unbounded distributions.

            Calculates a2 using the SPT formula for unbounded distributions:
            a2 = 0.5 * r / log((1-alpha)/alpha)
            where r = (median - q_alpha) / (q_complement - q_alpha)

            This coefficient controls the scale/spread of the distribution. Larger
            r values (median positioned more toward the upper quantile) result in
            larger a2 values.

            Args:
                q: Array of length 3 containing [q_alpha, median, q_complement].
                alpha: Lower percentile parameter in (0, 0.5).

            Returns:
                Second metalog coefficient a2. The value is always positive for
                valid inputs since both r and log((1-alpha)/alpha) are positive.

            Note:
                This function is JIT-compiled and is only called for unbounded
                distributions within _unbounded_fit.
            """
            return 0.5 * jnp.power(_log_term(alpha), -1) * _dist_symmetry(q)

        @jax.jit
        def _third_term(q: chex.Numeric, alpha: float) -> float:
            """Compute the third metalog coefficient (a3) for unbounded distributions.

            Calculates a3 using the SPT formula for unbounded distributions:
            a3 = r * (1 - 2r) / ((1 - 2*alpha) * log((1-alpha)/alpha))
            where r = (median - q_alpha) / (q_complement - q_alpha)

            This coefficient controls the asymmetry/skewness of the distribution.
            When r = 0.5 (median is centered), a3 = 0 and the distribution is symmetric.
            Non-zero a3 values introduce skewness.

            Args:
                q: Array of length 3 containing [q_alpha, median, q_complement].
                alpha: Lower percentile parameter in (0, 0.5).

            Returns:
                Third metalog coefficient a3. The value can be positive (right skew),
                negative (left skew), or zero (symmetric).

            Note:
                This function is JIT-compiled and is only called for unbounded
                distributions within _unbounded_fit.
            """
            r = (q[1] - q[0]) / (q[2] - q[0])
            return (
                jnp.power((1 - (2 * alpha)) * _log_term(alpha), -1)
                * (1 - (2 * r))
                * _dist_symmetry(q)
            )

        alpha = y[0]
        _unbounded_fit_feasibility_check(q, alpha)
        return jnp.array([q[1], _second_term(q, alpha), _third_term(q, alpha)])

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
            AssertionError: If the transformed quantiles (gamma values) violate
                feasibility constraints for the given lower bound.

        Note:
            This function is JIT-compiled for performance. The coefficients are
            computed in log-space after subtracting the lower bound from all quantiles.
        """

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
                AssertionError: If lower_bound + geometric_mean_upper - median <= 0.
                AssertionError: If median - lower_bound - geometric_mean_lower <= 0.

            Note:
                This check uses weighted geometric means of the gamma values with
                weights k_alpha and (1-k_alpha) to establish feasibility bounds.
            """
            median = q[1]
            gamma_alpha = gamma[0]
            gamma_compliment = gamma[2]

            k_alpha = _k_alpha(alpha)
            chex.assert_scalar_positive(
                float(
                    lower_bound
                    + (
                        jnp.power(gamma_alpha, k_alpha)
                        * jnp.power(gamma_compliment, 1 - k_alpha)
                    )
                    - median
                )
            )
            chex.assert_scalar_positive(
                float(
                    median
                    - lower_bound
                    + (
                        jnp.power(gamma_alpha, 1 - k_alpha)
                        * jnp.power(gamma_compliment, k_alpha)
                    )
                )
            )

        @jax.jit
        def _second_term(gamma: chex.Numeric, alpha: float) -> float:
            """Compute the second coefficient (a2) for lower-bounded distributions.

            Calculates a2 using the SPT formula for lower-bounded distributions:
            a2 = 0.5 * log(gamma_complement / gamma_alpha) / log((1-alpha)/alpha)

            where gamma values are the quantiles transformed by subtracting the
            lower bound. This coefficient controls the scale/spread in log-space.

            Args:
                gamma: Array of length 3 containing transformed quantiles
                    [gamma_alpha, gamma_median, gamma_complement] where
                    gamma_i = q_i - lower_bound.
                alpha: Lower percentile parameter in (0, 0.5).

            Returns:
                Second metalog coefficient a2 in log-space. Can be positive or
                negative depending on the relative magnitudes of gamma values.

            Note:
                This function is JIT-compiled and is only called for lower-bounded
                distributions within _strictly_lower_bound_fit.
            """
            gamma_alpha = gamma[0]
            gamma_compliment = gamma[2]
            return (
                0.5
                * jnp.power(_log_term(alpha), -1)
                * jnp.log(gamma_compliment / gamma_alpha)
            )

        @jax.jit
        def _third_term(gamma: chex.Numeric, alpha: float) -> float:
            """Compute the third coefficient (a3) for lower-bounded distributions.

            Calculates a3 using the SPT formula for lower-bounded distributions:
            a3 = log((gamma_complement * gamma_alpha) / gamma_median^2) /
                 ((1 - 2*alpha) * log((1-alpha)/alpha))

            where gamma values are the quantiles transformed by subtracting the
            lower bound. This coefficient controls asymmetry/skewness in log-space.

            Args:
                gamma: Array of length 3 containing transformed quantiles
                    [gamma_alpha, gamma_median, gamma_complement] where
                    gamma_i = q_i - lower_bound.
                alpha: Lower percentile parameter in (0, 0.5).

            Returns:
                Third metalog coefficient a3 in log-space. Can be positive (right skew),
                negative (left skew), or zero (log-symmetric).

            Note:
                This function is JIT-compiled and is only called for lower-bounded
                distributions within _strictly_lower_bound_fit.
            """
            gamma_alpha = gamma[0]
            gamma_median = gamma[1]
            gamma_compliment = gamma[2]
            return jnp.power((1 - (2 * alpha)) * _log_term(alpha), -1) * jnp.log(
                (gamma_compliment * gamma_alpha) / jnp.power(gamma_median, 2)
            )

        alpha = y[0]

        q_alpha = q[0]
        median = q[1]
        q_compliment = q[2]

        gamma_alpha = q_alpha - lower_bound
        gamma_median = median - lower_bound
        gamma_compliment = q_compliment - lower_bound

        gamma = jnp.array([gamma_alpha, gamma_median, gamma_compliment])
        _strictly_lower_bound_fit_feasibility_check(q, gamma, alpha, lower_bound)
        return jnp.array(
            [
                jnp.log(gamma_median),
                _second_term(gamma, alpha),
                _third_term(gamma, alpha),
            ]
        )

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
            AssertionError: If the transformed quantiles (gamma values) violate
                feasibility constraints for the given bounds.

        Note:
            This function is JIT-compiled for performance. The coefficients are
            computed in log-space after applying the logit-like transformation
            to map (lower_bound, upper_bound) to (0, ∞).
        """

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
                AssertionError: If upper_bound_ratio - median <= 0.
                AssertionError: If median - lower_bound_ratio <= 0.

            Note:
                This check uses weighted geometric means of the gamma values with
                weights k_alpha and (1-k_alpha), transformed back to the bounded
                domain using the inverse logit-like mapping.
            """
            median = q[1]
            gamma_alpha = gamma[0]
            gamma_compliment = gamma[2]
            k_alpha = _k_alpha(alpha)

            chex.assert_scalar_positive(
                float(
                    (
                        (
                            lower_bound
                            + (
                                upper_bound
                                * jnp.power(gamma_alpha, k_alpha)
                                * jnp.power(gamma_compliment, 1 - k_alpha)
                            )
                        )
                        / (
                            1
                            + (
                                jnp.power(gamma_alpha, k_alpha)
                                * jnp.power(gamma_compliment, 1 - k_alpha)
                            )
                        )
                    )
                    - median
                )
            )
            chex.assert_scalar_positive(
                float(
                    median
                    - (
                        (
                            lower_bound
                            + (
                                upper_bound
                                * jnp.power(gamma_alpha, 1 - k_alpha)
                                * jnp.power(gamma_compliment, k_alpha)
                            )
                        )
                        / (
                            1
                            + (
                                jnp.power(gamma_alpha, 1 - k_alpha)
                                * jnp.power(gamma_compliment, k_alpha)
                            )
                        )
                    )
                )
            )

        @jax.jit
        def _second_term(gamma: chex.Numeric, alpha: float) -> float:
            """Compute the second coefficient (a2) for bounded distributions.

            Calculates a2 using the SPT formula for bounded distributions:
            a2 = 0.5 * log(gamma_complement / gamma_alpha) / log((1-alpha)/alpha)

            where gamma values are the quantiles transformed using the logit-like
            mapping: gamma_i = (q_i - lower) / (upper - q_i). This coefficient
            controls the scale/spread in log-space.

            Args:
                gamma: Array of length 3 containing transformed quantiles
                    [gamma_alpha, gamma_median, gamma_complement] where
                    gamma_i = (q_i - lower_bound) / (upper_bound - q_i).
                alpha: Lower percentile parameter in (0, 0.5).

            Returns:
                Second metalog coefficient a2 in log-space. Can be positive or
                negative depending on the relative magnitudes of gamma values.

            Note:
                This function is JIT-compiled and is only called for bounded
                distributions within _bounded_fit. The formula is identical to
                that used for lower-bounded distributions, but operates on
                differently transformed gamma values.
            """
            gamma_alpha = gamma[0]
            gamma_compliment = gamma[2]
            return (
                0.5
                * jnp.power(_log_term(alpha), -1)
                * jnp.log(gamma_compliment / gamma_alpha)
            )

        @jax.jit
        def _third_term(gamma: chex.Numeric, alpha: float) -> float:
            """Compute the third coefficient (a3) for bounded distributions.

            Calculates a3 using the SPT formula for bounded distributions:
            a3 = log((gamma_complement * gamma_alpha) / gamma_median^2) /
                 ((1 - 2*alpha) * log((1-alpha)/alpha))

            where gamma values are the quantiles transformed using the logit-like
            mapping: gamma_i = (q_i - lower) / (upper - q_i). This coefficient
            controls asymmetry/skewness in log-space.

            Args:
                gamma: Array of length 3 containing transformed quantiles
                    [gamma_alpha, gamma_median, gamma_complement] where
                    gamma_i = (q_i - lower_bound) / (upper_bound - q_i).
                alpha: Lower percentile parameter in (0, 0.5).

            Returns:
                Third metalog coefficient a3 in log-space. Can be positive (right skew),
                negative (left skew), or zero (log-symmetric).

            Note:
                This function is JIT-compiled and is only called for bounded
                distributions within _bounded_fit. The formula is identical to
                that used for lower-bounded distributions, but operates on
                differently transformed gamma values.
            """
            gamma_alpha = gamma[0]
            gamma_median = gamma[1]
            gamma_compliment = gamma[2]
            return jnp.power((1 - (2 * alpha)) * _log_term(alpha), -1) * jnp.log(
                (gamma_compliment * gamma_alpha) / jnp.power(gamma_median, 2)
            )

        alpha = y[0]

        q_alpha = q[0]
        median = q[1]
        q_compliment = q[2]

        gamma_alpha = (q_alpha - lower_bound) / (upper_bound - q_alpha)
        gamma_median = (median - lower_bound) / (upper_bound - median)
        gamma_compliment = (q_compliment - lower_bound) / (upper_bound - q_compliment)

        gamma = jnp.array([gamma_alpha, gamma_median, gamma_compliment])
        _bounded_fit_feasibility_check(q, gamma, alpha, lower_bound, upper_bound)
        return jnp.array(
            [
                jnp.log(gamma_median),
                _second_term(gamma, alpha),
                _third_term(gamma, alpha),
            ]
        )

    chex.assert_axis_dimension_gteq(array, 0, 3)
    chex.assert_rank(array, 1)
    assert_numeric_array(array)

    alpha = spt_metalog_params.alpha
    median_percentile = 0.5

    chex.assert_scalar_positive(alpha)
    chex.assert_scalar_positive(float(median_percentile - alpha))

    compliment = 1.0 - alpha
    y = jnp.array([alpha, median_percentile, compliment])
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
    elif boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
        raise NotImplementedError("This formulation is undefined in Keelin's paper!")
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

    metalog: Union[Metalog, SPTMetalog]
    ks_dist: chex.Scalar
