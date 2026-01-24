"""Utility functions for the Metalog JAX library."""
# Copyright: Travis Jefferies 2026

import chex
import jax
import jax.numpy as jnp
from flax import struct

jax.config.update("jax_enable_x64", True)

DEFAULT_Y: chex.Numeric = jnp.concatenate(
    [
        jnp.array([0.001, 0.003, 0.006], dtype=jnp.float32),
        jnp.arange(0.01, 1.0, 0.01, dtype=jnp.float32),
        jnp.array([0.994, 0.997, 0.999], dtype=jnp.float32),
    ]
)
DEFAULT_Y_FULL: chex.Numeric = jnp.arange(0.001, 1.0, 0.0001, dtype="float32")
DEFAULT_HDR_TRIAL: int = 1
DEFAULT_HDR_VARIABLE: int = 0
DEFAULT_HDR_ENTITY: int = 0
DEFAULT_HDR_TIME: int = 0
DEFAULT_HDR_AGENT: int = 0
DEFAULT_MAX_NUMBER_RV_DRAWS: int = 100_000


class NotFittedError(Exception):
    """A custom exception that raises when metalog has not been fit yet."""

    pass


class InvalidPDFError(Exception):
    """A custom exception that raises when metalog returns an invalid PDF."""

    pass


@struct.dataclass
class JaxUniformDistributionParameters:
    """Configuration parameters for a JAX uniform distribution.

    Attributes:
        seed: Seed for the psuedorandom number generator.
    """

    seed: int


@struct.dataclass
class HDRPRNGParameters:
    """Structure for Hubbard Decision Research Parameterized PRNG.

    See paper below:

    Douglas W. Hubbard.
    2019.
    A multi-dimensional,
    counter-based psuedo random number generator
    as a standard for Monte Carlo simulations.
    In Proceedings of the Winter Simulation Conference.
    IEEE Press, 3064 - 3073.

    Attributes:
        trial: Seed representing the simulation trial or iteration. Defaults to 1.
        variable: Parameter representing the random variable being sampled. Defaults to 0.
        entity: Parameter representing the entity or object being simulated. Defaults to 0.
        time: Parameter representing the time step in the simulation. Defaults to 0.
        agent: Parameter representing the agent or actor in the simulation. Defaults to 0.
    """

    trial: int = DEFAULT_HDR_TRIAL
    variable: int = DEFAULT_HDR_VARIABLE
    entity: int = DEFAULT_HDR_ENTITY
    time: int = DEFAULT_HDR_TIME
    agent: int = DEFAULT_HDR_AGENT


def assert_numeric_array(array: chex.Array) -> None:
    """Assert that an array contains all float or int values.

    Args:
        array: Input array to validate.

    Raises:
        AssertionError: If the array does not contain numeric (float or int) values.
    """
    assert_float_array(array.astype(float))


def assert_float_array(array: chex.Array) -> None:
    """Assert that an array contains all float values.

    Args:
        array: Input array to validate.

    Raises:
        AssertionError: If the array does not contain numeric float values.
    """
    chex.assert_type(array, float)


def assert_probability_range(array: chex.Array) -> None:
    """Assert that all array values are in the open interval (0, 1).

    Validates that the minimum value in the array is strictly greater than 0
    and the maximum value is strictly less than 1, ensuring all values represent
    valid probabilities in the open unit interval.

    Args:
        array: Input array to validate.

    Raises:
        AssertionError: If any value is <= 0 or >= 1.

    Example:
        >>> valid_probs = jnp.array([0.1, 0.5, 0.9])
        >>> assert_probability_range(valid_probs)  # No error
        >>> invalid_probs = jnp.array([0.0, 0.5, 1.0])
        >>> assert_probability_range(invalid_probs)  # Raises AssertionError
    """
    min_val = jnp.min(array)
    max_val = jnp.max(array)

    if min_val <= 0:
        raise ValueError(f"Array minimum {min_val} must be strictly greater than 0")
    if max_val >= 1:
        raise ValueError(f"Array maximum {max_val} must be strictly less than 1")


def assert_strictly_ascending(x: chex.Numeric, tol: float = 1e-12) -> None:
    """Assert that array values are strictly ascending (monotonically increasing).

    Validates that each element in the array is strictly greater than the previous
    element, with differences exceeding a specified tolerance threshold. This ensures
    the array represents a strictly increasing sequence, which is required for valid
    probability levels and quantile values in metalog distributions.

    The function computes consecutive differences between array elements and verifies
    that all differences are strictly positive (greater than the tolerance). This is
    more robust than simple inequality checks as it accounts for numerical precision
    issues in floating-point comparisons.

    Args:
        x: Input array to validate. Should be a 1D numeric array representing values
            that must be in strictly ascending order (e.g., probability levels,
            quantile values, or time steps).
        tol: Tolerance threshold for considering differences as strictly positive.
            Must be a small positive value. Default is 1e-12, which is appropriate
            for double-precision floating-point arithmetic. Values separated by less
            than this tolerance are considered equal (not strictly ascending).

    Raises:
        AssertionError: If any consecutive difference is less than or equal to the
            tolerance threshold, indicating the array is not strictly ascending.
            This can occur if:
            - The array contains duplicate values (difference = 0)
            - The array is not sorted in ascending order (negative differences)
            - Consecutive values are too close together (difference < tol)

    Note:
        This function is used to validate inputs for metalog fitting, particularly
        probability levels (y values) which must be strictly increasing to ensure
        a valid probability distribution. The tolerance parameter helps handle
        numerical precision issues while still enforcing strict monotonicity.

        Unlike simple sorting checks, this function ensures strict inequality
        (no duplicate values allowed), which is essential for metalog distributions
        where each probability level must map to a unique quantile value.

    Example:
        Valid strictly ascending array:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.utils import assert_strictly_ascending
            >>>
            >>> x = jnp.array([0.1, 0.2, 0.5, 0.9])
            >>> assert_strictly_ascending(x)  # No error
            >>> # Differences: [0.1, 0.3, 0.4] all > 1e-12

        Invalid array with duplicate values:

            >>> x_dup = jnp.array([0.1, 0.2, 0.2, 0.9])
            >>> assert_strictly_ascending(x_dup)
            >>> # Raises AssertionError: difference of 0.0 at index 2

        Invalid array not in ascending order:

            >>> x_unsorted = jnp.array([0.1, 0.5, 0.2, 0.9])
            >>> assert_strictly_ascending(x_unsorted)
            >>> # Raises AssertionError: negative difference at index 2

        Using custom tolerance:

            >>> # Values very close together (within default tolerance)
            >>> x_close = jnp.array([0.1, 0.1 + 1e-13, 0.2])
            >>> assert_strictly_ascending(x_close)  # May fail with default tol
            >>> assert_strictly_ascending(x_close, tol=1e-14)  # Passes

    See Also:
        assert_probability_range: Validates array values are in (0, 1).
        jnp.diff: Computes consecutive differences between array elements.

    References:
        Used extensively in metalog fitting to validate probability level arrays
        (y values) before computing quantile functions or fitting coefficients.
    """
    diffs = jnp.diff(x)
    chex.assert_scalar_positive(bool(jnp.all(diffs > tol)))


def hdrprng(
    hdr_parameters: HDRPRNGParameters,
) -> float:
    """Generate a pseudo-random number using the Hubbard Decision Research PRNG algorithm.

    Implements a multi-dimensional, counter-based pseudo-random number generator
    designed for reproducible Monte Carlo simulations. This method validates the
    input parameters and delegates to the JIT-compiled internal implementation.

    This PRNG is particularly useful for large-scale simulations where:
    - Reproducibility is critical
    - Multiple independent random streams are needed
    - Parallel execution requires uncorrelated random sequences

    Args:
        hdr_parameters: HDRPRNGParameters instance containing the five integer
            parameters (trial, variable, entity, time, agent) that determine
            the random output.

    Returns:
        A pseudo-random float in the open interval (0, 1).

    Raises:
        AssertionError: If hdr_parameters.trial is not a positive scalar
            (validated by HDRPRNGParameters.__post_init__).

    Example:
        >>> from metalog_jax.utils import HDRPRNGParameters, hdrprng
        >>> params = HDRPRNGParameters(trial=1, variable=0, entity=0, time=0, agent=0)
        >>> random_value = hdrprng(params)
        >>> # Same parameters always produce the same output
        >>> random_value_2 = hdrprng(params)
        >>> assert random_value == random_value_2

    References:
        Douglas W. Hubbard. 2019.
        A multi-dimensional, counter-based pseudo random number generator
        as a standard for Monte Carlo simulations.
        In Proceedings of the Winter Simulation Conference.
        IEEE Press, 3064-3073.

    Note:
        This is the public API method. The actual computation is performed by
        the internal _hdrprng() function, which is JIT-compiled for performance.
        The algorithm uses a combination of large prime numbers and modular
        arithmetic to generate uniformly distributed pseudo-random values.
    """

    @jax.jit
    def _hdrprng(
        trial: int,
        variable: int,
        entity: int,
        time: int,
        agent: int,
    ) -> float:
        """Compute the pseudo-random value using the HDR algorithm (internal computation).

        This internal JIT-compiled function implements the core Hubbard Decision
        Research PRNG algorithm. It uses a counter-based approach with multiple
        prime numbers and modular arithmetic operations to generate deterministic
        pseudo-random values.

        Args:
            trial: Trial or iteration number (must be >= 1, validated by caller).
            variable: Random variable identifier.
            entity: Entity or object identifier.
            time: Time step identifier.
            agent: Agent or actor identifier.

        Returns:
            A pseudo-random float in the open interval (0, 1).

        Note:
            This is an internal method that is JIT-compiled for performance. Use
            hdrprng() for the public API, which includes parameter validation.

            The algorithm generates random values through two parallel computation
            paths, each using different sets of prime numbers, which are then
            combined to produce the final output. The result is guaranteed to be
            in the range (0, 1) exclusive and is deterministic for any given set
            of input parameters.
        """
        trial_primes = jnp.array([2499997, 2246527], dtype="float64")
        variable_primes = jnp.array([1800451, 2399993], dtype="float64")
        entity_primes = jnp.array([2000371, 2100869], dtype="float64")
        time_primes = jnp.array([1796777, 1918303], dtype="float64")
        agent_primes = jnp.array([2299603, 1624729], dtype="float64")

        return (
            jnp.mod(
                (
                    jnp.mod(
                        jnp.mod(
                            999999999999989,
                            jnp.mod(
                                trial * trial_primes[0]
                                + variable * variable_primes[0]
                                + entity * entity_primes[0]
                                + time * time_primes[0]
                                + agent * agent_primes[0],
                                7450589,
                            )
                            * 4658
                            + 7450581,
                        )
                        * 383,
                        99991,
                    )
                    * 7440893
                    + jnp.mod(
                        jnp.mod(
                            999999999999989,
                            jnp.mod(
                                trial * trial_primes[1]
                                + variable * variable_primes[1]
                                + entity * entity_primes[1]
                                + time * time_primes[1]
                                + agent * agent_primes[1],
                                7450987,
                            )
                            * 7580
                            + 7560584,
                        )
                        * 17669,
                        7440893,
                    )
                )
                * 1343,
                4294967296,
            )
            + 0.5
        ) / 4294967296

    trial = hdr_parameters.trial
    variable = hdr_parameters.variable
    entity = hdr_parameters.entity
    time = hdr_parameters.time
    agent = hdr_parameters.agent

    return _hdrprng(
        trial=trial,
        variable=variable,
        entity=entity,
        time=time,
        agent=agent,
    )


@jax.jit
def find_nearest_index(array: chex.Numeric, value: chex.Numeric) -> chex.Numeric:
    """Find the index of the nearest value in an array.

    Args:
        array: JAX array to search
        value: Target value to find nearest match for

    Returns:
        Index of the nearest value
    """
    differences = jnp.abs(array - value)
    nearest_idx = jnp.argmin(differences)
    return nearest_idx


@jax.jit
def ks_distance(x: chex.Numeric, y: chex.Numeric) -> chex.Scalar:
    """Compute the two-sample Kolmogorov-Smirnov (KS) distance between two samples.

    Calculates the maximum absolute difference between the empirical cumulative
    distribution functions (ECDFs) of two samples. This is the Kolmogorov-Smirnov
    statistic for the two-sample test, which measures the degree of dissimilarity
    between two probability distributions.

    The KS distance is defined as:
        D = sup |F_x(t) - F_y(t)|
    where F_x and F_y are the empirical CDFs of samples x and y, and the supremum
    is taken over all values t.

    This implementation is fully vectorized and JIT-compiled for efficient execution
    with JAX, making it suitable for use in gradient-based optimization and parallel
    computation workflows. The function uses `jnp.searchsorted` for efficient CDF
    computation, avoiding explicit loops.

    Args:
        x: First sample array of shape (n,). Contains n independent observations
            from the first distribution. Can be unsorted (will be sorted internally).
        y: Second sample array of shape (m,). Contains m independent observations
            from the second distribution. Can be unsorted (will be sorted internally).
            Does not need to have the same length as x.

    Returns:
        Scalar KS distance in the range [0, 1]. The KS statistic represents the
        maximum vertical distance between the two empirical CDFs:
        - 0: The two samples have identical empirical distributions
        - 1: The two samples are completely non-overlapping
        - Values closer to 0 indicate more similar distributions

    Note:
        - Both input samples are sorted internally, so there's no need to pre-sort
        - The function is JIT-compiled, so the first call may be slower due to
          compilation overhead, but subsequent calls will be very fast
        - This computes only the KS distance (D statistic), not the p-value for
          the hypothesis test
        - The function is differentiable and can be used in gradient-based
          optimization (e.g., for distribution matching)
        - Memory complexity is O(n + m) due to the combined sorted values array

    Example:
        Basic usage with identical distributions:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.utils import ks_distance
            >>>
            >>> # Two samples from the same distribution
            >>> x = jnp.array([0.1, 0.5, 0.3, 0.7, 0.9])
            >>> y = jnp.array([0.2, 0.4, 0.6, 0.8])
            >>> distance = ks_distance(x, y)
            >>> # Small distance indicates similar distributions

        Comparing different distributions:

            >>> # Sample from uniform(0, 1)
            >>> uniform_sample = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
            >>> # Sample from distribution concentrated near 0
            >>> skewed_sample = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25])
            >>> distance = ks_distance(uniform_sample, skewed_sample)
            >>> # Larger distance indicates different distributions

        Using for metalog goodness-of-fit:

            >>> from metalog_jax.metalog import fit
            >>> from metalog_jax.base import MetalogInputData, MetalogParameters
            >>>
            >>> # Fit metalog to data
            >>> data = MetalogInputData.from_values(x, DEFAULT_Y, False)
            >>> params = MetalogParameters(...)
            >>> metalog = fit(data, params)
            >>>
            >>> # Generate samples from fitted distribution
            >>> fitted_samples = metalog.rvs(...)
            >>>
            >>> # Compare original vs fitted distribution
            >>> ks_dist = ks_distance(x, fitted_samples)
            >>> # Low KS distance indicates good fit

    See Also:
        jnp.searchsorted: Efficient binary search used for CDF computation.
        scipy.stats.ks_2samp: SciPy equivalent (includes p-value calculation).

    References:
        Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge
        di distribuzione". Giornale dell'Istituto Italiano degli Attuari, 4: 83-91.

        Smirnov, N. V. (1948). "Table for estimating the goodness of fit of
        empirical distributions". Annals of Mathematical Statistics, 19(2): 279-281.

    Implementation Details:
        The algorithm works as follows:
        1. Sort both input samples (O(n log n + m log m))
        2. Combine all unique values from both samples and sort (O((n+m) log(n+m)))
        3. Use binary search to compute empirical CDFs at all combined points (O((n+m) log n))
        4. Find maximum absolute difference between the CDFs (O(n+m))
        Overall complexity: O((n+m) log(n+m))
    """
    # Sort both samples
    x = jnp.sort(x)
    y = jnp.sort(y)
    n = x.size
    m = y.size

    # Combine and sort all unique sample points
    all_values = jnp.sort(jnp.concatenate([x, y]))

    # Compute empirical CDFs efficiently using searchsorted
    cdf_x = jnp.searchsorted(x, all_values, side="right") / n
    cdf_y = jnp.searchsorted(y, all_values, side="right") / m

    # Compute KS statistic (supremum of absolute difference)
    d_stat = jnp.max(jnp.abs(cdf_x - cdf_y))
    return d_stat
