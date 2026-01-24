# Copyright: Travis Jefferies 2026
"""Core MetalogBase class for metalog distributions.

This module contains the MetalogBase class, which is the abstract base class for all
metalog distribution implementations. It provides the common interface and core
functionality shared by Metalog and SPTMetalog classes defined in metalog_jax.metalog.

The MetalogBase class includes:
    - Quantile function (PPF) computation
    - Probability density function (PDF) computation
    - Cumulative distribution function (CDF) computation
    - Statistical moments (mean, median, variance, std, mode)
    - Random variable sampling (rvs)
    - Serialization/deserialization (save, load, dumps, loads)
    - Visualization (plot)

See Also:
    metalog_jax.base.data: Input data container classes.
    metalog_jax.base.enums: Enumeration types for boundedness and fit methods.
    metalog_jax.base.parameters: Parameter configuration classes.
    metalog_jax.metalog: Metalog and SPTMetalog distribution classes.
"""

import json
from functools import partial
from pathlib import Path
from typing import Type, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from flax import struct
from jax.experimental import checkify

from metalog_jax.base.data import MetalogInputData
from metalog_jax.base.enums import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogPlotOptions,
)
from metalog_jax.base.parameters import (
    MetalogParameters,
    MetalogRandomVariableParameters,
    SPTMetalogParameters,
)
from metalog_jax.utils import (
    DEFAULT_Y,
    DEFAULT_Y_FULL,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
    NotFittedError,
    assert_float_array,
    assert_probability_range,
    find_nearest_index,
    hdrprng,
)

T_MetalogBase = TypeVar("T_MetalogBase", bound="MetalogBase")


@struct.dataclass
class MetalogBase:
    """Base class for metalog distribution implementations.

    This abstract base class provides the common interface and core functionality
    shared by all metalog distribution types (Metalog and SPTMetalog). It implements
    the fundamental metalog distribution operations including quantile function (PPF),
    probability density function (PDF), cumulative distribution function (CDF),
    and statistical moments.

    Subclasses:
        Metalog: Standard metalog fitted via regression (OLS, Ridge, ElasticNet, Lasso).
            Defined in metalog_jax.metalog.
        SPTMetalog: Symmetric Percentile Triplet metalog with closed-form coefficients.
            Defined in metalog_jax.metalog.

    Attributes:
        metalog_params: Configuration parameters defining the distribution type.
            Can be MetalogParameters (for Metalog) or SPTMetalogParameters
            (for SPTMetalog). Defined in metalog_jax.base.parameters.
        a: Coefficient vector defining the metalog quantile function. Shape is
            (num_terms,) where num_terms depends on the configuration.

    Example:
        MetalogBase is not instantiated directly. Use Metalog or SPTMetalog:

            >>> from metalog_jax.metalog import fit, Metalog
            >>> from metalog_jax.base import MetalogInputData, MetalogParameters
            >>> from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
            >>>
            >>> data = MetalogInputData.from_values(x=samples, y=probs, precomputed_quantiles=False)
            >>> params = MetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     method=MetalogFitMethod.OLS,
            ...     num_terms=5, lower_bound=0.0, upper_bound=0.0
            ... )
            >>> m = fit(data, params)  # Returns Metalog instance
            >>> isinstance(m, MetalogBase)  # True

    See Also:
        metalog_jax.metalog.Metalog: Standard metalog distribution class.
        metalog_jax.metalog.SPTMetalog: SPT metalog distribution class.
        metalog_jax.base.parameters.MetalogParameters: Configuration for Metalog.
        metalog_jax.base.parameters.SPTMetalogParameters: Configuration for SPTMetalog.
    """

    metalog_params: Union[SPTMetalogParameters, MetalogParameters]
    a: chex.Numeric

    def __eq__(self, other: object) -> bool:
        """Check equality between two metalog distribution instances.

        Compares the metalog parameters and coefficient arrays to determine
        equality. For JAX arrays, uses `jnp.allclose` for numerical comparison.

        Args:
            other: The object to compare against.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if not isinstance(other, MetalogBase):
            return False

        # Compare metalog_params attributes
        if self.metalog_params != other.metalog_params:
            return False

        # Compare coefficient arrays using allclose for numerical comparison
        # Check shape first, then values
        if self.a.shape != other.a.shape:
            return False

        return bool(jnp.allclose(self.a, other.a))

    @property
    def boundedness(self) -> MetalogBoundedness:
        """Get the boundedness type of the metalog distribution.

        Returns:
            MetalogBoundedness: The boundedness type (UNBOUNDED, STRICTLY_LOWER_BOUND,
                STRICTLY_UPPER_BOUND, or BOUNDED).
        """
        return self.metalog_params.boundedness

    @property
    def lower_bound(self) -> chex.Scalar:
        """Get the lower bound of the metalog distribution.

        Returns:
            chex.Scalar: The lower bound value. Only meaningful when boundedness
                is STRICTLY_LOWER_BOUND or BOUNDED.
        """
        return self.metalog_params.lower_bound

    @property
    def upper_bound(self) -> chex.Scalar:
        """Get the upper bound of the metalog distribution.

        Returns:
            chex.Scalar: The upper bound value. Only meaningful when boundedness
                is STRICTLY_UPPER_BOUND or BOUNDED.
        """
        return self.metalog_params.upper_bound

    @property
    def num_terms(self) -> int:
        """Get the number of terms in the metalog expansion.

        Returns:
            int: The number of terms (coefficients) in the metalog quantile function.
        """
        return self.metalog_params.num_terms

    @property
    def method(self) -> MetalogFitMethod:
        """Get the fit method used for the metalog distribution.

        Returns:
            MetalogFitMethod: The regression method (OLS or Lasso).
        """
        return self.metalog_params.method

    def _assert_has_coefficients(self) -> None:
        """Check that the distribution has been fitted with coefficients.

        This is a lightweight check used internally by methods like ``pdf`` to
        avoid recursion when ``assert_fitted`` calls ``pdf`` for validation.

        Raises:
            NotFittedError: If the distribution has not been fitted (no coefficients).
        """
        if len(self.a) < 1:
            raise NotFittedError(
                "Metalog has not been fit yet! Distribution must be fit "
                "before you can call methods on the distribution."
            )

    def assert_fitted(self) -> None:
        """Validate that the metalog distribution has been fitted and is feasible.

        Performs two validation checks:
        1. Verifies that coefficients exist (distribution has been fit).
        2. Computes the PDF at standard quantile points to verify feasibility,
           which internally calls ``assert_fit_valid`` to ensure all PDF values
           are strictly positive.

        This method is vmap-compatible through the use of ``jax.experimental.checkify``
        in the underlying PDF validation.

        Raises:
            NotFittedError: If the distribution has not been fitted (no coefficients).
            checkify.JaxRuntimeError: If the fitted distribution produces non-positive
                PDF values, indicating an infeasible fit.
        """
        self._assert_has_coefficients()
        self.pdf(DEFAULT_Y)

    @staticmethod
    def assert_fit_valid(m: chex.Numeric) -> None:
        """Validate that PDF values are strictly positive.

        Checks that all probability density values are positive, which is a
        necessary condition for a valid probability distribution. A metalog
        fit that produces non-positive PDF values is theoretically infeasible.

        This method is vmap-compatible by using ``jax.experimental.checkify``
        for runtime error checking inside JAX transforms.

        Args:
            m: Array of PDF values to validate. All elements must be > 0.

        Raises:
            checkify.JaxRuntimeError: If any PDF value is less than or equal
                to zero, indicating an infeasible metalog fit.
        """
        checkify.check(
            jnp.all(m > 0),
            "Metalog PDF returned a non-positive value! "
            "This is a theoretically invalid PDF.",
        )

    @staticmethod
    @jax.jit
    def strictly_lower_bound_quantile_transform(
        x: chex.Numeric, lower_bound: chex.Scalar
    ) -> chex.Numeric:
        """Apply logarithmic transformation for strictly lower-bounded quantiles.

        Transforms quantiles from (lower_bound, inf) to (-inf, inf) using log(x - L).

        Args:
            x: Quantile values to transform. Must be > lower_bound.
            lower_bound: The strict lower bound of the distribution.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        return jnp.log(x - lower_bound)

    @staticmethod
    @jax.jit
    def strictly_upper_bound_quantile_transform(
        x: chex.Numeric, upper_bound: chex.Scalar
    ) -> chex.Numeric:
        """Apply negative logarithmic transformation for strictly upper-bounded quantiles.

        Transforms quantiles from (-inf, upper_bound) to (-inf, inf) using -log(U - x).

        Args:
            x: Quantile values to transform. Must be < upper_bound.
            upper_bound: The strict upper bound of the distribution.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        return -jnp.log(upper_bound - x)

    @staticmethod
    @jax.jit
    def bounded_quantile_transform(
        x: chex.Numeric,
        lower_bound: chex.Scalar,
        upper_bound: chex.Scalar,
    ) -> chex.Numeric:
        """Apply logit-like transformation for fully bounded quantiles.

        Transforms quantiles from (lower_bound, upper_bound) to (-inf, inf)
        using log((x - L) / (U - x)).

        Args:
            x: Quantile values to transform. Must be in (lower_bound, upper_bound).
            lower_bound: The strict lower bound of the distribution.
            upper_bound: The strict upper bound of the distribution.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        return jnp.log((x - lower_bound) / (upper_bound - x))

    @staticmethod
    @jax.jit
    def delta_term(x: chex.Numeric) -> chex.Numeric:
        """Compute the delta term (x - 0.5) used in metalog basis functions.

        Args:
            x: Probability values in (0, 1).

        Returns:
            chex.Numeric: The delta term values (x - 0.5).
        """
        return x - 0.5

    @staticmethod
    @jax.jit
    def log_term(x: chex.Numeric) -> chex.Numeric:
        """Compute the logit transformation log(x / (1 - x)).

        Args:
            x: Probability values in (0, 1).

        Returns:
            chex.Numeric: The logit-transformed values.
        """
        return jnp.log(x / (1 - x))

    @staticmethod
    @jax.jit
    def product_term(x: chex.Numeric) -> chex.Numeric:
        """Compute the product term x * (1 - x).

        This term appears in the derivative of the logit function and is used
        in PDF computation.

        Args:
            x: Probability values in (0, 1).

        Returns:
            chex.Numeric: The product term values x * (1 - x).
        """
        return x * (1 - x)

    @staticmethod
    def get_quantiles(
        data: MetalogInputData,
        metalog_params: MetalogParameters,
    ) -> chex.Numeric:
        """Extract and transform quantiles from validated input data.

        Applies the appropriate boundedness transformation to convert quantiles
        from their natural domain to the unbounded (-inf, inf) scale used
        internally by the metalog.

        Args:
            data: Validated input data containing quantile values (x) and
                probability levels (y).
            metalog_params: Configuration parameters specifying boundedness
                type and bound values.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.

        Raises:
            chex.AssertionError: If the number of data points is less than num_terms.
        """
        x = data.x
        chex.assert_axis_dimension_gteq(x, 0, metalog_params.num_terms)
        boundedness = metalog_params.boundedness

        if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            lower_bound = metalog_params.lower_bound
            x = MetalogBase.strictly_lower_bound_quantile_transform(x, lower_bound)
        elif boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            upper_bound = metalog_params.upper_bound
            x = MetalogBase.strictly_upper_bound_quantile_transform(x, upper_bound)
        elif boundedness == MetalogBoundedness.BOUNDED:
            lower_bound = metalog_params.lower_bound
            upper_bound = metalog_params.upper_bound
            x = MetalogBase.bounded_quantile_transform(x, lower_bound, upper_bound)
        return x

    @staticmethod
    @partial(jax.jit, static_argnames=["num_terms"])
    def _get_target(y: chex.Numeric, num_terms: int) -> chex.Numeric:
        """Construct the design matrix of metalog basis functions from input data.

        Builds the matrix Y where each column corresponds to a metalog basis function
        evaluated at the probability levels y. The basis functions are:
        - Term 1: 1 (intercept)
        - Term 2: log(y / (1-y))
        - Term 3: (y - 0.5) * log(y / (1-y))
        - Term 4: (y - 0.5)
        - Term 5+: Alternating powers of (y - 0.5) and products with log term.

        Args:
            y: Probability levels in (0, 1), shape (m,).
            num_terms: Number of metalog terms (basis functions) to include.

        Returns:
            chex.Numeric: Design matrix of shape (m, num_terms).
        """

        @jax.jit
        def _delta_term(y: chex.Numeric) -> chex.Numeric:
            return y - 0.5

        @jax.jit
        def _log_term(y: chex.Numeric) -> chex.Numeric:
            return jnp.log(y / (1 - y))

        @partial(jax.jit, static_argnames=["term", "m"])
        def odd_term_transformation(y: chex.Numeric, term: int, m: int) -> chex.Numeric:
            return (jnp.power(_delta_term(y), term // 2)).reshape((m, 1))

        @partial(jax.jit, static_argnames=["m"])
        def even_term_transformation(
            y: chex.Numeric, previous_term: chex.Numeric, m: int
        ) -> chex.Numeric:
            return (_log_term(y) * jnp.squeeze(previous_term)).reshape((m, 1))

        m = y.shape[0]
        intercepts = jnp.ones(m).reshape((m, 1))
        y_terms = [
            intercepts,
            _log_term(y).reshape((m, 1)),
            (_delta_term(y) * _log_term(y)).reshape((m, 1)),
        ]
        if num_terms > 3:
            y_terms.append(_delta_term(y).reshape((m, 1)))
            if num_terms > 4:
                for term in range(5, num_terms + 1):
                    if term % 2 == 1:
                        odd_term = odd_term_transformation(y, term, m)
                        y_terms.append(odd_term)
                    else:
                        y_terms.append(even_term_transformation(y, odd_term, m))
        return jnp.hstack(y_terms)

    @staticmethod
    def get_target(
        data: MetalogInputData, metalog_params: MetalogParameters
    ) -> chex.Numeric:
        """Construct the design matrix from input data and parameters.

        Convenience wrapper around _get_target that extracts the probability
        levels and num_terms from the input objects.

        Args:
            data: Validated input data containing probability levels (y).
            metalog_params: Configuration parameters specifying num_terms.

        Returns:
            chex.Numeric: Design matrix of shape (m, num_terms).
        """
        y = data.y
        num_terms = metalog_params.num_terms
        return MetalogBase._get_target(y=y, num_terms=num_terms)

    @partial(jax.jit, static_argnames=["num_terms"])
    def _ppf(self, x: chex.Numeric, num_terms: int) -> chex.Numeric:
        """Compute the unbounded metalog quantile function (internal PPF computation).

        Evaluates the metalog quantile function on the unbounded scale before
        applying any boundedness transformations. This is the core computation
        M(y) = sum(a_k * basis_k(y)).

        Args:
            x: Probability values in (0, 1) at which to evaluate the quantile function.
            num_terms: Number of terms in the metalog expansion.

        Returns:
            chex.Numeric: Quantile values on the unbounded (-inf, inf) scale.
        """
        delta_term = self.delta_term(x)
        log_term = self.log_term(x)
        ppf = self.a[0] + (self.a[1] * log_term)
        if num_terms > 2:
            ppf += self.a[2] * delta_term * log_term
        if num_terms > 3:
            ppf += self.a[3] * delta_term
        if num_terms > 4:
            odd_term = 2
            even_term = 2
            for term in range(5, num_terms + 1):
                if term % 2 == 1:  # odd term
                    ppf += self.a[term - 1] * jnp.power(delta_term, odd_term)
                    odd_term += 1
                if term % 2 == 0:  # even term
                    ppf += (
                        self.a[term - 1] * jnp.power(delta_term, even_term) * log_term
                    )
                    even_term += 1
        return ppf

    def ppf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the percent point function (quantile function) of the distribution.

        The PPF is the inverse of the CDF: for a probability p, ppf(p) returns
        the value x such that P(X <= x) = p.

        Args:
            x: Probability values in (0, 1) at which to evaluate the quantile function.
                Can be a scalar or array.

        Returns:
            chex.Numeric: Quantile values corresponding to the input probabilities.
                Same shape as input.

        Raises:
            ValueError: If x contains values outside (0, 1).
            NotFittedError: If the distribution has not been fitted.
        """
        assert_float_array(x)
        assert_probability_range(x)
        self.assert_fitted()

        num_terms = self.num_terms
        ppf = self._ppf(x, num_terms)
        if self.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            ppf = self.lower_bound + jnp.exp(ppf)
        elif self.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            ppf = self.upper_bound - jnp.exp(-1 * ppf)
        elif self.boundedness == MetalogBoundedness.BOUNDED:
            ppf = (self.lower_bound + self.upper_bound * jnp.exp(ppf)) / (
                1 + jnp.exp(ppf)
            )
        return ppf

    def pdf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the probability density function of the metalog distribution.

        The PDF is computed as the reciprocal of the derivative of the quantile
        function (PPF), with appropriate transformations for bounded distributions.

        Args:
            x: Probability values in (0, 1) at which to evaluate the PDF.
                Can be a scalar or array.

        Returns:
            chex.Numeric: Probability density values. Same shape as input.

        Raises:
            ValueError: If x contains values outside (0, 1).
            NotFittedError: If the distribution has not been fitted.
            checkify.JaxRuntimeError: If the PDF contains non-positive values,
                indicating an infeasible metalog fit.
        """

        @partial(jax.jit, static_argnames=["num_terms"])
        def _pdf(x: chex.Numeric, num_terms: int) -> chex.Array:
            product_term = self.product_term(x)
            delta_term = self.delta_term(x)
            log_term = self.log_term(x)

            if num_terms == 2:
                return product_term / self.a[1]

            m_inv = self.a[1] / product_term
            if num_terms >= 3:
                m_inv += self.a[2] * ((delta_term / product_term) + log_term)
            if num_terms >= 4:
                m_inv += self.a[3]
            if num_terms >= 5:
                odd_term = 1
                even_term = 1
                for term in range(5, num_terms + 1):
                    if term % 2 == 1:  # Odd term
                        m_inv += (
                            (odd_term + 1)
                            * self.a[term - 1]
                            * jnp.power(delta_term, odd_term)
                        )
                        odd_term += 1
                    else:  # Even term
                        m_inv += self.a[term - 1] * (
                            (jnp.power(delta_term, even_term + 1) / product_term)
                            + (even_term + 1)
                            * jnp.power(delta_term, even_term)
                            * log_term
                        )
                        even_term += 1
            m = 1.0 / m_inv
            return m

        assert_float_array(x)
        assert_probability_range(x)
        self._assert_has_coefficients()
        num_terms = self.num_terms
        m = _pdf(x, num_terms)
        if self.boundedness != MetalogBoundedness.UNBOUNDED:
            M = self._ppf(x, num_terms)
            if self.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
                m = m * jnp.exp(-M)
            elif self.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
                m = m * jnp.exp(M)
            elif self.boundedness == MetalogBoundedness.BOUNDED:
                m = (m * (1 + jnp.exp(M)) ** 2) / (
                    (self.upper_bound - self.lower_bound) * jnp.exp(M)
                )
        self.assert_fit_valid(m=m)
        return m

    def cdf(self, q: chex.Numeric) -> chex.Numeric:
        """Compute the cumulative distribution function.

        Returns P(X <= q) for the fitted metalog distribution. Uses numerical
        inversion of the quantile function via nearest-neighbor lookup.

        Args:
            q: Quantile values at which to evaluate the CDF. Can be a scalar
                or 1D array.

        Returns:
            chex.Numeric: Probability values in (0, 1). Same shape as input.
        """
        ppf = self.ppf(DEFAULT_Y_FULL)
        if not isinstance(q, jax.Array):
            chex.assert_rank(q, 0)  # assert scalar
            index = find_nearest_index(ppf, q)
            return DEFAULT_Y_FULL[index]
        else:
            chex.assert_rank(q, 1)  # assert vector
            find_nearest_batch = jax.vmap(find_nearest_index, in_axes=(None, 0))
            indices = find_nearest_batch(ppf, q)
            return DEFAULT_Y_FULL[indices]

    def rvs(self, rv_params: MetalogRandomVariableParameters) -> chex.Numeric:
        """Generate random variates from the fitted metalog distribution.

        Samples are generated by drawing uniform random values and applying
        the quantile function (inverse transform sampling).

        Args:
            rv_params: Parameters controlling random variate generation, including
                PRNG configuration (HDR or JAX-based) and sample size.

        Returns:
            chex.Numeric: Random samples from the distribution, shape (size,).

        Raises:
            chex.AssertionError: If size is not positive or exceeds max_draws.
            checkify.JaxRuntimeError: If the PDF is infeasible (non-positive values).
        """

        @jax.jit
        def _hdr_uniform(trial: int, maxval: int) -> chex.Array:
            params = HDRPRNGParameters(
                trial=trial,
                variable=variable,
                entity=entity,
                time=time,
                agent=agent,
            )
            rv = hdrprng(params)
            return (rv * (maxval - 1)).astype(int)

        @partial(jax.jit, static_argnames=["size", "maxval"])
        def _jax_uniform(key: chex.PRNGKey, size: int, maxval: int) -> chex.Array:
            return jax.random.randint(
                key, shape=(size,), minval=0, maxval=maxval, dtype="int32"
            )

        prng_params = rv_params.prng_params
        size = rv_params.size
        chex.assert_scalar_positive(size)
        chex.assert_scalar_non_negative(rv_params.max_draws - size)
        maxval = len(DEFAULT_Y_FULL)
        if isinstance(prng_params, HDRPRNGParameters):
            variable = prng_params.variable
            entity = prng_params.entity
            time = prng_params.time
            agent = prng_params.agent
            trials = jnp.arange(1, size + 1, dtype=int)

            vectorized_hdr = jax.vmap(_hdr_uniform, in_axes=(0, None))
            mask = vectorized_hdr(trials, maxval)
        elif isinstance(prng_params, JaxUniformDistributionParameters):
            seed = prng_params.seed
            key = jax.random.PRNGKey(seed)
            mask = _jax_uniform(key, size, maxval)

        self.pdf(DEFAULT_Y_FULL)  # asserts feasibility
        ppf = self.ppf(DEFAULT_Y_FULL)
        return ppf[mask]

    def logpdf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the natural logarithm of the probability density function.

        Args:
            x: Probability values in (0, 1) at which to evaluate log(PDF).

        Returns:
            chex.Numeric: Log-density values. Same shape as input.
        """
        return jnp.log(self.pdf(x))

    def logppf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the natural logarithm of the percent point function.

        Args:
            x: Probability values in (0, 1) at which to evaluate log(PPF).

        Returns:
            chex.Numeric: Log-quantile values. Same shape as input.
        """
        return jnp.log(self.ppf(x))

    def sf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the survival function (complement of CDF).

        The survival function is S(x) = 1 - F(x) = P(X > x).

        Args:
            x: Quantile values at which to evaluate the survival function.

        Returns:
            chex.Numeric: Survival probabilities in (0, 1). Same shape as input.
        """
        return 1.0 - self.cdf(x)

    def logsf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the natural logarithm of the survival function.

        Args:
            x: Quantile values at which to evaluate log(SF).

        Returns:
            chex.Numeric: Log-survival probabilities. Same shape as input.
        """
        return jnp.log(self.sf(x))

    def isf(self, q: chex.Numeric) -> chex.Numeric:
        """Compute the inverse survival function (inverse complementary CDF).

        Returns the value x such that P(X > x) = q, equivalent to ppf(1 - q).

        Args:
            q: Probability values in (0, 1).

        Returns:
            chex.Numeric: Quantile values corresponding to survival probabilities.
        """
        return self.ppf(1 - q)

    @property
    def mean(self) -> chex.Scalar:
        """Compute the mean (expected value) of the metalog distribution.

        Estimated via Monte Carlo sampling with 20,000 draws.

        Returns:
            chex.Scalar: The estimated mean of the distribution.
        """
        rv = self.rvs(
            MetalogRandomVariableParameters(
                prng_params=JaxUniformDistributionParameters(seed=0), size=20_000
            )
        )
        return jnp.mean(rv)

    @property
    def median(self) -> chex.Scalar:
        """Compute the median of the metalog distribution.

        The median is the 50th percentile, computed exactly via ppf(0.5).

        Returns:
            chex.Scalar: The median of the distribution.
        """
        return self.ppf(0.5)

    @property
    def var(self) -> chex.Scalar:
        """Compute the variance of the metalog distribution.

        Estimated via Monte Carlo sampling with 20,000 draws.

        Returns:
            chex.Scalar: The estimated variance of the distribution.
        """
        rv = self.rvs(
            MetalogRandomVariableParameters(
                prng_params=JaxUniformDistributionParameters(seed=0), size=20_000
            )
        )
        return jnp.var(rv)

    @property
    def std(self) -> chex.Scalar:
        """Compute the standard deviation of the metalog distribution.

        Estimated via Monte Carlo sampling with 20,000 draws.

        Returns:
            chex.Scalar: The estimated standard deviation of the distribution.
        """
        rv = self.rvs(
            MetalogRandomVariableParameters(
                prng_params=JaxUniformDistributionParameters(seed=0), size=20_000
            )
        )
        return jnp.std(rv)

    @property
    def mode(self) -> chex.Scalar:
        """Compute the mode (most likely value) of the fitted metalog distribution.

        The mode is found by locating the maximum of the PDF over a fine grid
        of probability values.

        Returns:
            chex.Scalar: The mode of the distribution.
        """
        ppf = self.ppf(DEFAULT_Y_FULL)
        pdf = self.pdf(DEFAULT_Y_FULL)
        return ppf[pdf.argmax()]

    def plot(self, plot_option: MetalogPlotOptions) -> go.Figure:
        """Visualize the fitted metalog distribution using interactive Plotly plots.

        Creates an interactive Plotly figure showing the selected distribution
        function (PDF, CDF, or survival function).

        Args:
            plot_option: Which function to plot. One of MetalogPlotOptions.PDF,
                MetalogPlotOptions.CDF, or MetalogPlotOptions.SF.

        Returns:
            go.Figure: The Plotly figure object.

        Raises:
            NotFittedError: If the distribution has not been fitted.
        """
        self.assert_fitted()
        x = DEFAULT_Y_FULL
        ppf = self.ppf(x)
        fig = go.Figure()
        if plot_option == MetalogPlotOptions.PDF:
            pdf = self.pdf(x)
            fig.add_trace(go.Scatter(x=ppf, y=pdf, mode="markers", name="Metalog Fit"))
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Density",
                title="Probability Density Function",
            )
        elif plot_option == MetalogPlotOptions.CDF:
            fig.add_trace(go.Scatter(x=ppf, y=x, mode="lines", name="Metalog Fit"))
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Probability",
                title="Cumulative Density Function",
            )
        elif plot_option == MetalogPlotOptions.SF:
            fig.add_trace(go.Scatter(x=ppf, y=1 - x, mode="lines", name="Metalog Fit"))
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Probability",
                title="Survival Function",
            )
        fig.show()
        return fig

    def save(self, path: Path) -> None:
        """Save the fitted metalog distribution to a JSON file.

        Serializes the distribution coefficients and parameters to JSON format.

        Args:
            path: File path where the JSON file will be written.
        """
        data = {
            "a": self.a.tolist(),
            "metalog_params": {
                "boundedness": int(self.boundedness),
                "method": int(self.method),
                "lower_bound": float(self.lower_bound),
                "upper_bound": float(self.upper_bound),
                "num_terms": int(self.num_terms),
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def dumps(self) -> str:
        """Serialize the fitted metalog distribution to a JSON string.

        Returns:
            str: JSON string representation of the distribution.
        """
        data = {
            "a": self.a.tolist(),
            "metalog_params": {
                "boundedness": int(self.boundedness),
                "method": int(self.method),
                "lower_bound": float(self.lower_bound),
                "upper_bound": float(self.upper_bound),
                "num_terms": int(self.num_terms),
            },
        }
        return json.dumps(data)

    @staticmethod
    def _deserialize_boundedness(boundedness: int) -> MetalogBoundedness:
        """Convert integer value to MetalogBoundedness enum member.

        Args:
            boundedness: Integer representation of the boundedness type.

        Returns:
            MetalogBoundedness: The corresponding enum member.
        """
        return MetalogBoundedness.from_value(boundedness)

    @staticmethod
    def _deserialize_method(method: int) -> MetalogFitMethod:
        """Convert integer value to MetalogFitMethod enum member.

        Args:
            method: Integer representation of the fit method.

        Returns:
            MetalogFitMethod: The corresponding enum member.
        """
        return MetalogFitMethod.from_value(method)

    @classmethod
    def load(cls: Type[T_MetalogBase], path: Path) -> T_MetalogBase:
        """Load a fitted metalog distribution from a JSON file.

        Args:
            path: File path to the JSON file containing the serialized distribution.

        Returns:
            T_MetalogBase: A new instance of the metalog class with loaded
                coefficients and parameters.
        """
        with open(path) as f:
            input_data = json.load(f)

        a = jnp.array(input_data["a"])
        params_dict = input_data["metalog_params"]
        metalog_params = MetalogParameters(
            boundedness=MetalogBase._deserialize_boundedness(
                params_dict["boundedness"]
            ),
            method=MetalogBase._deserialize_method(params_dict["method"]),
            lower_bound=params_dict["lower_bound"],
            upper_bound=params_dict["upper_bound"],
            num_terms=params_dict["num_terms"],
        )

        return cls(metalog_params=metalog_params, a=a)

    @classmethod
    def loads(cls: Type[T_MetalogBase], json_string: str) -> T_MetalogBase:
        """Load a fitted metalog distribution from a JSON string.

        Args:
            json_string: JSON string containing the serialized distribution.

        Returns:
            T_MetalogBase: A new instance of the metalog class with loaded
                coefficients and parameters.
        """
        input_data = json.loads(json_string)

        a = jnp.array(input_data["a"])
        params_dict = input_data["metalog_params"]
        metalog_params = MetalogParameters(
            boundedness=MetalogBase._deserialize_boundedness(
                params_dict["boundedness"]
            ),
            method=MetalogBase._deserialize_method(params_dict["method"]),
            lower_bound=params_dict["lower_bound"],
            upper_bound=params_dict["upper_bound"],
            num_terms=params_dict["num_terms"],
        )

        return cls(metalog_params=metalog_params, a=a)

    def q(self, x: chex.Numeric) -> chex.Numeric:
        """Quantile function alias for R metalog package compatibility.

        This method provides compatibility with the R metalog package's `qmetalog`
        function naming convention.

        Args:
            x: Probability values in (0, 1) at which to evaluate the quantile function.

        Returns:
            chex.Numeric: Quantile values corresponding to the input probabilities.

        Raises:
            ValueError: If x contains values outside (0, 1).
            NotFittedError: If the distribution has not been fitted.
        """
        assert_float_array(x)
        assert_probability_range(x)
        self.assert_fitted()

        boundedness = self.boundedness
        target = MetalogBase._get_target(x, self.num_terms)
        ppf = target @ self.a

        if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            lower_bound = self.lower_bound
            ppf = MetalogBase.strictly_lower_bound_quantile_transform(ppf, lower_bound)
        elif boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            upper_bound = self.upper_bound
            ppf = MetalogBase.strictly_upper_bound_quantile_transform(ppf, upper_bound)
        elif boundedness == MetalogBoundedness.BOUNDED:
            lower_bound = self.lower_bound
            upper_bound = self.upper_bound
            ppf = MetalogBase.bounded_quantile_transform(ppf, lower_bound, upper_bound)
        return ppf

    def p(self, q: chex.Numeric) -> chex.Numeric:
        """Probability function alias for R metalog package compatibility.

        This method provides compatibility with the R metalog package's `pmetalog`
        function naming convention. Equivalent to cdf().

        Args:
            q: Quantile values at which to evaluate the CDF.

        Returns:
            chex.Numeric: Probability values in (0, 1).
        """
        return self.cdf(q)

    def d(self, q: chex.Numeric) -> chex.Numeric:
        """Density function alias for R metalog package compatibility.

        This method provides compatibility with the R metalog package's `dmetalog`
        function naming convention. Computes pdf(cdf(q)).

        Args:
            q: Quantile values at which to evaluate the density.

        Returns:
            chex.Numeric: Probability density values.
        """
        return self.pdf(self.cdf(q))
