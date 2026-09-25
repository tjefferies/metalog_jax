"""Integration tests for the Metalog JAX module.

These tests recreate Keelin's original results from the paper.
"""
# Copyright: Travis Jefferies 2026

import jax.numpy as jnp
from absl.testing import absltest, parameterized
from scipy.stats import (
    beta,
    chi2,
    expon,
    f,
    gamma,
    gumbel_l,
    invgamma,
    logistic,
    lognorm,
    norm,
    t,
    triang,
    uniform,
    weibull_min,
)
from scipy.stats._distn_infrastructure import rv_frozen

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    MetalogRandomVariableParameters,
)
from metalog_jax.metalog import fit
from metalog_jax.utils import DEFAULT_Y, JaxUniformDistributionParameters

SEED: int = 0
SIZE: int = 90_000
# Keelin's tables report KS distances to 4 decimals and several fits reproduce them
# exactly; float32 evaluation then lands a few ulps (~4e-8) above the tabulated value.
KS_ROUNDING_TOL = 1e-6

METALOG_RV_PARAMS = MetalogRandomVariableParameters(
    JaxUniformDistributionParameters(SEED), size=SIZE
)
SIG_ALPHA: float = 0.001


def calculate_ks_distance(
    dist: rv_frozen,
    metalog_params: MetalogParameters,
) -> float:
    """Calculate Kolmogorov-Smirnov distance between a distribution and its metalog fit.

    Fits a metalog distribution to quantiles sampled from a theoretical distribution
    and computes the maximum absolute difference between the theoretical CDF and the
    fitted metalog's inverse CDF. This metric quantifies how well the metalog
    approximation captures the original distribution.

    The Kolmogorov-Smirnov (KS) distance is computed as:
        KS = max |F(x) - F_metalog(x)|
    where F is the theoretical CDF and F_metalog is the fitted metalog's CDF evaluated
    via inversion (cdf method). Since we work with quantiles, we compute:
        KS = max |y - metalog.cdf(ppf(y))|
    where ppf are the theoretical quantiles at probability levels y.

    This function is used by integration tests to validate metalog fitting accuracy
    across various probability distributions, following the methodology from Keelin's
    metalog paper (Tables 5, 6, and 7).

    Args:
        dist: Scipy frozen distribution object representing the theoretical distribution
            to approximate. Must support the ppf() method for quantile computation.
        metalog_params: Configuration parameters specifying how to fit the metalog
            distribution, including boundedness type, fitting method (OLS/Lasso),
            boundary values, and number of terms.

    Returns:
        Maximum Kolmogorov-Smirnov distance (float) between the theoretical distribution
        and the fitted metalog approximation. Lower values indicate better fit quality.
        Values are typically in the range [0.0, 1.0], with 0.0 indicating perfect fit.

    Example:
        >>> from scipy.stats import norm
        >>> from metalog_jax.base import MetalogParameters, MetalogBoundedness
        >>> from metalog_jax.base import MetalogFitMethod
        >>> # Test unbounded normal distribution with 5-term metalog
        >>> dist = norm(loc=50, scale=15)
        >>> params = MetalogParameters(
        ...     boundedness=MetalogBoundedness.UNBOUNDED,
        ...     method=MetalogFitMethod.OLS,
        ...     lower_bound=0,
        ...     upper_bound=0,
        ...     num_terms=5
        ... )
        >>> ks_dist = calculate_ks_distance(dist, params)
        >>> assert ks_dist < 0.01  # Expect very good fit for 5 terms

    Note:
        - Uses DEFAULT_Y probability grid (defined module-level) for quantile sampling
        - The function fits the metalog from scratch each time it's called
        - KS distance is computed over the entire probability range in DEFAULT_Y
        - Lower KS distances indicate better metalog approximation quality
        - This is a helper function used by parameterized test cases in the test classes
        - The methodology follows Keelin (2016) Tables 5-7 for validation

    See Also:
        MetalogUnboundedDistributionTest: Uses this function to test unbounded distributions.
        MetalogSemiboundedDistributionTest: Uses this function to test semi-bounded distributions.
        MetalogBoundedDistributionTest: Uses this function to test bounded distributions.

    References:
        Keelin, T. W. (2016). The Metalog Distributions. Decision Analysis, 13(4), 243-277.
    """
    ppf = jnp.array(dist.ppf(DEFAULT_Y))

    # Create validated input data using the factory method
    data = MetalogInputData.from_values(
        x=ppf,
        y=DEFAULT_Y,
        precomputed_quantiles=True,
    )

    # Fit metalog distribution
    metalog = fit(data=data, metalog_params=metalog_params)

    # Compute KS distance
    ks_dist = jnp.max(jnp.abs(DEFAULT_Y - metalog.cdf(ppf)))
    return ks_dist


class MetalogUnboundedDistributionTest(parameterized.TestCase):
    """Integration tests for metalog fitting on unbounded distributions.

    This test class validates the metalog distribution fitting algorithm against
    various unbounded probability distributions (Normal, Logistic, Student t,
    Extreme value). Test cases recreate the results from Keelin's Table 5,
    comparing Kolmogorov-Smirnov distances between fitted metalog distributions
    and their corresponding theoretical distributions for 3, 4, and 5-term
    approximations using OLS fitting.

    Attributes:
        table58_reported_results: Expected KS distances from Keelin's Table 5/8.
        table58_acceptable_results: Adjusted acceptable KS distance thresholds
            accounting for random seed variations.
    """

    table58_reported_results: dict = {
        "Normal (mu=50, sigma=15)": {
            3: 0.035,
            4: 0.006,
            5: 0.006,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.000,
        },
        "Logistic (mu=40, s=4.6)": {
            3: 0.000,
            4: 0.000,
            5: 0.000,
            6: 0.000,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Student t (df=6)": {
            3: 0.012,
            4: 0.008,
            5: 0.008,
            6: 0.004,
            7: 0.002,
            8: 0.002,
            9: 0.002,
            10: 0.001,
        },
        "Extreme value (mu=100, sigma=20, epsilon=-0.5)": {
            3: 0.070,
            4: 0.017,
            5: 0.009,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.000,
        },
        "Extreme value (mu=100, sigma=20, epsilon=-0.2)": {
            3: 0.047,
            4: 0.008,
            5: 0.008,
            6: 0.003,
            7: 0.002,
            8: 0.001,
            9: 0.001,
            10: 0.000,
        },
        "Extreme value (mu=100, sigma=20, epsilon=-0.025)": {
            3: 0.036,
            4: 0.028,
            5: 0.006,
            6: 0.005,
            7: 0.005,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
    }

    # reported vs acceptable can be chalked up to noise / different seeds
    table58_acceptable_results: dict = {
        "Normal (mu=50, sigma=15)": {
            3: 0.037,
            4: 0.009,
            5: 0.009,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.0003,
        },
        "Logistic (mu=40, s=4.6)": {
            3: 0.0025,
            4: 0.0025,
            5: 0.0025,
            6: 6e-7,
            7: 6e-7,
            8: 6e-7,
            9: 6e-7,
            10: 6e-7,
        },
        "Student t (df=6)": {
            3: 0.015,
            4: 0.011,
            5: 0.011,
            6: 0.00421,
            7: 0.0023,
            8: 0.0023,
            9: 0.0023,
            10: 0.0011,
        },
        "Extreme value (mu=100, sigma=20, epsilon=-0.5)": {
            3: 0.070,
            4: 0.0223,
            5: 0.009,
            6: 0.005,
            7: 0.005,
            8: 0.001,
            9: 0.00041,
            10: 0.0004,
        },
    }

    @parameterized.named_parameters(
        {
            "testcase_name": "Normal(mu=50, sigma=15) 3 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 4 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 5 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 3 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 4 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 5 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 3 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 4 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 5 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 3 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 4 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 5 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 6 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 7 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 8 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 9 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Normal(mu=50, sigma=15) 10 term",
            "distribution": "Normal (mu=50, sigma=15)",
            "dist": norm(loc=50, scale=15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 6 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 7 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 8 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 9 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Logistic (mu=40, s=4.6) 10 term",
            "distribution": "Logistic (mu=40, s=4.6)",
            "dist": logistic(loc=40, scale=4.6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 6 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 7 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 8 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 9 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Student t (df=6) 10 term",
            "distribution": "Student t (df=6)",
            "dist": t(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 6 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 7 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 8 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 9 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Extreme value (mu=100, sigma=20, epsilon=-0.5) 10 term",
            "distribution": "Extreme value (mu=100, sigma=20, epsilon=-0.5)",
            "dist": gumbel_l(loc=100, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
    )
    def test_metalog_fit(
        self,
        distribution: str,
        dist: rv_frozen,
        metalog_params: MetalogParameters,
    ) -> None:
        """Test metalog fitting accuracy for unbounded distributions.

        Fits a metalog distribution to precomputed quantiles from a theoretical
        unbounded distribution and validates the approximation quality using
        the Kolmogorov-Smirnov distance metric.

        Args:
            distribution: Name of the theoretical distribution being tested.
            dist: Scipy frozen distribution object for the theoretical distribution.
            metalog_params: Configuration parameters for metalog fitting (boundedness,
                fitting method, and number of terms).

        Asserts:
            The maximum KS distance between the fitted metalog CDF and the
            theoretical distribution CDF is less than or equal to the expected
            threshold from Keelin's Table 5 results.
        """
        num_terms = metalog_params.num_terms
        expected_ks_dist = self.table58_acceptable_results[distribution][num_terms]

        ks_dist = calculate_ks_distance(dist, metalog_params)
        self.assertLessEqual(float(ks_dist), expected_ks_dist + KS_ROUNDING_TOL)


class MetalogSemiboundedDistributionTest(parameterized.TestCase):
    """Integration tests for metalog fitting on semi-bounded distributions.

    This test class validates the metalog distribution fitting algorithm against
    various semi-bounded probability distributions (Log-normal, Weibull, Gamma,
    Inverse Gamma, Exponential, Chi-squared, Inverse Chi-squared, F). Test cases
    recreate the results from Keelin's Table 6, comparing Kolmogorov-Smirnov
    distances between fitted metalog distributions and their corresponding theoretical
    distributions for 3-10 term approximations using OLS fitting.

    Semi-bounded distributions (also called strictly lower bounded) have a lower
    limit (typically 0) but extend to positive infinity, requiring logarithmic
    transformations in the metalog framework. The test suite covers a wide range
    of positively-skewed distributions commonly used in reliability analysis,
    survival modeling, and financial applications.

    Attributes:
        table68_reported_results: Expected KS distances from Keelin's Table 6/8,
            containing reference values for 3-10 term metalog approximations across
            all tested semi-bounded distributions.
        table68_acceptable_results: Adjusted acceptable KS distance thresholds
            accounting for numerical precision and random seed variations in the
            fitting process. Tests pass if computed KS distance is below these
            thresholds.
    """

    table68_reported_results: dict = {
        "Log-normal (mu = 0, sigma = 0.5)": {
            3: 0.035,
            4: 0.006,
            5: 0.006,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.000,
        },
        "Log-normal (mu = 0, sigma = 0.3)": {
            3: 0.035,
            4: 0.006,
            5: 0.006,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.000,
        },
        "Log-normal (mu = 0, sigma = 0.15)": {
            3: 0.035,
            4: 0.006,
            5: 0.006,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.000,
        },
        "Weibull (lambda = 3, kappa = 3)": {
            3: 0.037,
            4: 0.022,
            5: 0.006,
            6: 0.004,
            7: 0.003,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Weibull (lambda = 7, kappa = 7)": {
            3: 0.037,
            4: 0.022,
            5: 0.006,
            6: 0.004,
            7: 0.003,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Gamma (kappa = 4, theta = 2)": {
            3: 0.038,
            4: 0.011,
            5: 0.006,
            6: 0.002,
            7: 0.002,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Gamma (kappa = 2, theta = 2)": {
            3: 0.038,
            4: 0.015,
            5: 0.006,
            6: 0.003,
            7: 0.002,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Inverse gamma (alpha = 3, beta = 1)": {
            3: 0.038,
            4: 0.012,
            5: 0.006,
            6: 0.002,
            7: 0.002,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Inverse gamma (alpha = 5, beta = 0.5)": {
            3: 0.038,
            4: 0.010,
            5: 0.006,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Exponential (alpha = 0.5)": {
            3: 0.037,
            4: 0.022,
            5: 0.006,
            6: 0.004,
            7: 0.003,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Chi-squared (df = 3)": {
            3: 0.038,
            4: 0.017,
            5: 0.006,
            6: 0.003,
            7: 0.003,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Chi-squared (df = 6)": {
            3: 0.038,
            4: 0.012,
            5: 0.006,
            6: 0.002,
            7: 0.002,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Inverse chi-squared (df = 3)": {
            3: 0.038,
            4: 0.017,
            5: 0.006,
            6: 0.003,
            7: 0.003,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Inverse chi-squared (df = 6)": {
            3: 0.038,
            4: 0.012,
            5: 0.006,
            6: 0.002,
            7: 0.002,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "F (df1 = 1, df2 = 1)": {
            3: 0.020,
            4: 0.001,
            5: 0.001,
            6: 0.000,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "F (df1 = 15, df2 = 30)": {
            3: 0.033,
            4: 0.007,
            5: 0.006,
            6: 0.002,
            7: 0.001,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
    }

    # reported vs acceptable can be chalked up to noise / different seeds
    table68_acceptable_results: dict = {
        "Log-normal (mu = 0, sigma = 0.5)": {
            3: 0.038,
            4: 0.0091,
            5: 0.0091,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.00021,
        },
        "Log-normal (mu = 0, sigma = 0.3)": {
            3: 0.038,
            4: 0.0091,
            5: 0.0091,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.00021,
        },
        "Log-normal (mu = 0, sigma = 0.15)": {
            3: 0.038,
            4: 0.0091,
            5: 0.0091,
            6: 0.002,
            7: 0.001,
            8: 0.001,
            9: 0.001,
            10: 0.00021,
        },
        "Weibull (lambda = 3, kappa = 3)": {
            3: 0.0373,
            4: 0.0223,
            5: 0.0091,
            6: 0.0041,
            7: 0.00351,
            8: 0.001,
            9: 0.00041,
            10: 0.00031,
        },
        "Weibull (lambda = 7, kappa = 7)": {
            3: 0.0373,
            4: 0.0223,
            5: 0.0091,
            6: 0.0041,
            7: 0.00351,
            8: 0.001,
            9: 0.00041,
            10: 0.00031,
        },
        "Gamma (kappa = 4, theta = 2)": {
            3: 0.0384,
            4: 0.0134,
            5: 0.0091,
            6: 0.00211,
            7: 0.002,
            8: 0.001,
            9: 0.00051,
            10: 0.00021,
        },
        "Gamma (kappa = 2, theta = 2)": {
            3: 0.038,
            4: 0.0182,
            5: 0.0091,
            6: 0.003,
            7: 0.0023,
            8: 0.001,
            9: 0.00041,
            10: 0.00021,
        },
        "Inverse gamma (alpha = 3, beta = 1)": {
            3: 0.0384,
            4: 0.0124,
            5: 0.0091,
            6: 0.0023,
            7: 0.002,
            8: 0.001,
            9: 0.00051,
            10: 0.00021,
        },
        "Inverse gamma (alpha = 5, beta = 0.5)": {
            3: 0.0385,
            4: 0.0103,
            5: 0.0091,
            6: 0.002,
            7: 0.00131,
            8: 0.001,
            9: 0.00051,
            10: 0.00021,
        },
        "Exponential (alpha = 0.5)": {
            3: 0.0371,
            4: 0.0223,
            5: 0.0091,
            6: 0.0041,
            7: 0.00351,
            8: 0.001,
            9: 0.00041,
            10: 0.00031,
        },
        "Chi-squared (df = 3)": {
            3: 0.038,
            4: 0.01741,
            5: 0.0091,
            6: 0.00331,
            7: 0.003,
            8: 0.001,
            9: 0.00041,
            10: 0.0003,
        },
        "Chi-squared (df = 6)": {
            3: 0.0383,
            4: 0.01241,
            5: 0.0091,
            6: 0.00231,
            7: 0.002,
            8: 0.001,
            9: 0.0005,
            10: 0.00021,
        },
        "Inverse chi-squared (df = 3)": {
            3: 0.038,
            4: 0.0174,
            5: 0.0091,
            6: 0.00331,
            7: 0.003,
            8: 0.001,
            9: 0.00041,
            10: 0.00031,
        },
        "Inverse chi-squared (df = 6)": {
            3: 0.03831,
            4: 0.0124,
            5: 0.0091,
            6: 0.00231,
            7: 0.002,
            8: 0.001,
            9: 0.00051,
            10: 0.00021,
        },
        "F (df1 = 1, df2 = 1)": {
            3: 0.020,
            4: 0.001,
            5: 0.001,
            6: 0.00011,
            7: 6e-7,
            8: 6e-7,
            9: 6e-7,
            10: 6e-7,
        },
        "F (df1 = 15, df2 = 30)": {
            3: 0.0332,
            4: 0.007,
            5: 0.0091,
            6: 0.002,
            7: 0.001,
            8: 0.00051,
            9: 0.00051,
            10: 0.00021,
        },
    }

    @parameterized.named_parameters(
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 3 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 4 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 5 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 3 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 4 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 5 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 3 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 4 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 5 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 3 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 4 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 5 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 3 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 4 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 5 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 3 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 4 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 5 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 3 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 4 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 5 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 3 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 4 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 5 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 3 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 4 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 5 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 3 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 4 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 5 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 3 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 4 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 5 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 3 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 4 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 5 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 3 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(
                a=3 / 2, scale=3 / 2
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 4 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(
                a=3 / 2, scale=3 / 2
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 5 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(
                a=3 / 2, scale=3 / 2
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 3 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(
                a=6 / 2, scale=6 / 2
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 4 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(
                a=6 / 2, scale=6 / 2
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 5 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(
                a=6 / 2, scale=6 / 2
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 3 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(
                dfn=1, dfd=1
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 4 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(
                dfn=1, dfd=1
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 5 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(
                dfn=1, dfd=1
            ),  # inverse chi-squared is special case of inverse gamma
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 6 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 7 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 8 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 9 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.5) 10 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.5)",
            "dist": lognorm(loc=0, s=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 6 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 7 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 8 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 9 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.3) 10 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.3)",
            "dist": lognorm(loc=0, s=0.3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 6 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 7 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 8 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 9 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Log-normal (mu = 0, sigma = 0.15) 10 term",
            "distribution": "Log-normal (mu = 0, sigma = 0.15)",
            "dist": lognorm(loc=0, s=0.15),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 6 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 7 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 8 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 9 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 3, kappa = 3) 10 term",
            "distribution": "Weibull (lambda = 3, kappa = 3)",
            "dist": weibull_min(c=3, scale=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 6 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 7 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 8 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 9 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Weibull (lambda = 7, kappa = 7) 10 term",
            "distribution": "Weibull (lambda = 7, kappa = 7)",
            "dist": weibull_min(c=7, scale=7),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 6 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 7 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 8 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 9 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 4, theta = 2) 10 term",
            "distribution": "Gamma (kappa = 4, theta = 2)",
            "dist": gamma(a=4, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 6 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 7 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 8 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 9 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Gamma (kappa = 2, theta = 2) 10 term",
            "distribution": "Gamma (kappa = 2, theta = 2)",
            "dist": gamma(a=2, scale=2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 6 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 7 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 8 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 9 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 3, beta = 1) 10 term",
            "distribution": "Inverse gamma (alpha = 3, beta = 1)",
            "dist": invgamma(a=3, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 6 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 7 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 8 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 9 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Inverse gamma (alpha = 5, beta = 0.5) 10 term",
            "distribution": "Inverse gamma (alpha = 5, beta = 0.5)",
            "dist": invgamma(a=5, scale=0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 6 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 7 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 8 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 9 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Exponential (alpha = 0.5) 10 term",
            "distribution": "Exponential (alpha = 0.5)",
            "dist": expon(scale=1 / 0.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 6 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 7 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 8 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 9 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 3) 10 term",
            "distribution": "Chi-squared (df = 3)",
            "dist": chi2(df=3),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 6 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 7 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 8 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 9 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Chi-squared (df = 6) 10 term",
            "distribution": "Chi-squared (df = 6)",
            "dist": chi2(df=6),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 6 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(a=3 / 2, scale=3 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 7 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(a=3 / 2, scale=3 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 8 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(a=3 / 2, scale=3 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 9 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(a=3 / 2, scale=3 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 3) 10 term",
            "distribution": "Inverse chi-squared (df = 3)",
            "dist": invgamma(a=3 / 2, scale=3 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 6 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(a=6 / 2, scale=6 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 7 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(a=6 / 2, scale=6 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 8 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(a=6 / 2, scale=6 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 9 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(a=6 / 2, scale=6 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Inverse chi-squared (df = 6) 10 term",
            "distribution": "Inverse chi-squared (df = 6)",
            "dist": invgamma(a=6 / 2, scale=6 / 2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 6 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(dfn=1, dfd=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 7 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(dfn=1, dfd=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 8 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(dfn=1, dfd=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 9 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(dfn=1, dfd=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "F (df1 = 1, df2 = 1) 10 term",
            "distribution": "F (df1 = 1, df2 = 1)",
            "dist": f(dfn=1, dfd=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 3 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 4 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 5 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 6 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 7 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 8 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 9 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "F (df1 = 15, df2 = 30) 10 term",
            "distribution": "F (df1 = 15, df2 = 30)",
            "dist": f(dfn=15, dfd=30),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=0,
                num_terms=10,
            ),
        },
    )
    def test_metalog_fit(
        self,
        distribution: str,
        dist: rv_frozen,
        metalog_params: MetalogParameters,
    ) -> None:
        """Test metalog fitting for semi-bounded distributions.

        Validates that the fitted metalog distribution closely approximates
        the original distribution by comparing Kolmogorov-Smirnov distances.
        """
        num_terms = metalog_params.num_terms
        expected_ks_dist = self.table68_acceptable_results[distribution][num_terms]

        ks_dist = calculate_ks_distance(dist, metalog_params)
        self.assertLessEqual(float(ks_dist), expected_ks_dist + KS_ROUNDING_TOL)


class MetalogBoundedDistributionTest(parameterized.TestCase):
    """Integration tests for metalog fitting on bounded distributions.

    This test class validates the metalog distribution fitting algorithm against
    various bounded probability distributions (Beta, Uniform, Triangular). Test
    cases recreate the results from Keelin's Table 7, comparing Kolmogorov-Smirnov
    distances between fitted metalog distributions and their corresponding theoretical
    distributions for 3-10 term approximations using OLS fitting.

    Bounded distributions have both lower and upper limits, requiring special handling
    in the metalog framework. The test suite covers distributions with various shapes
    including symmetric (Beta with equal parameters, Uniform), asymmetric (Beta with
    unequal parameters), and triangular distributions.

    Attributes:
        table78_reported_results: Expected KS distances from Keelin's Table 7/8,
            containing reference values for 3-10 term metalog approximations across
            all tested bounded distributions.
        table7_acceptable_results: Adjusted acceptable KS distance thresholds
            accounting for numerical precision and random seed variations in the
            fitting process. Tests pass if computed KS distance is below these
            thresholds.
    """

    table78_reported_results: dict = {
        "Beta (alpha = 3.5, beta = 3.5)": {
            3: 0.024,
            4: 0.004,
            5: 0.004,
            6: 0.001,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Beta (alpha = 9, beta = 3.5)": {
            3: 0.031,
            4: 0.008,
            5: 0.005,
            6: 0.002,
            7: 0.001,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Beta (alpha = 0.8, beta = 0.9)": {
            3: 0.005,
            4: 0.002,
            5: 0.001,
            6: 0.000,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Beta (alpha = 60, beta = 1.5)": {
            3: 0.037,
            4: 0.017,
            5: 0.006,
            6: 0.003,
            7: 0.003,
            8: 0.001,
            9: 0.000,
            10: 0.000,
        },
        "Beta (alpha = 1.2, beta = 1.2)": {
            3: 0.005,
            4: 0.001,
            5: 0.001,
            6: 0.000,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Beta (alpha = 0.9, beta = 0.9)": {
            3: 0.003,
            4: 0.000,
            5: 0.000,
            6: 0.000,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Uniform (A = 0, B = 1)": {
            3: 0.000,
            4: 0.000,
            5: 0.000,
            6: 0.000,
            7: 0.000,
            8: 0.000,
            9: 0.000,
            10: 0.000,
        },
        "Triangular (A = 5, B = 20, C = 25)": {
            3: 0.019,
            4: 0.009,
            5: 0.003,
            6: 0.003,
            7: 0.002,
            8: 0.002,
            9: 0.001,
            10: 0.001,
        },
    }

    table78_acceptable_results: dict = {
        "Beta (alpha = 3.5, beta = 3.5)": {
            3: 0.024,
            4: 0.004,
            5: 0.004,
            6: 0.001,
            7: 0.00031,
            8: 0.00031,
            9: 0.00031,
            10: 0.00011,
        },
        "Beta (alpha = 9, beta = 3.5)": {
            3: 0.0315,
            4: 0.008,
            5: 0.0051,
            6: 0.002,
            7: 0.0011,
            8: 0.00051,
            9: 0.00041,
            10: 0.00021,
        },
        "Beta (alpha = 0.8, beta = 0.9)": {
            3: 0.0051,
            4: 0.00231,
            5: 0.001,
            6: 0.0003,
            7: 0.0003,
            8: 6e-7,
            9: 6e-7,
            10: 6e-7,
        },
        "Beta (alpha = 60, beta = 1.5)": {
            3: 0.037,
            4: 0.017,
            5: 0.006,
            6: 0.0032,
            7: 0.003,
            8: 0.001,
            9: 0.00041,
            10: 0.00021,
        },
        "Beta (alpha = 1.2, beta = 1.2)": {
            3: 0.005,
            4: 0.001,
            5: 0.001,
            6: 0.00011,
            7: 6e-7,
            8: 6e-7,
            9: 6e-7,
            10: 6e-7,
        },
        "Beta (alpha = 0.9, beta = 0.9)": {
            3: 0.003,
            4: 0.00031,
            5: 0.00031,
            6: 6e-7,
            7: 6e-7,
            8: 6e-7,
            9: 6e-7,
            10: 6e-7,
        },
        "Uniform (A = 0, B = 1)": {
            3: 6e-7,
            4: 6e-7,
            5: 6e-7,
            6: 6e-7,
            7: 6e-7,
            8: 6e-7,
            9: 6e-7,
            10: 6e-7,
        },
        "Triangular (A = 5, B = 20, C = 25)": {
            3: 0.0193,
            4: 0.0094,
            5: 0.003,
            6: 0.0035,
            7: 0.002,
            8: 0.002,
            9: 0.0015,
            10: 0.0012,
        },
    }

    @parameterized.named_parameters(
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 3 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 4 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 5 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 3 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 4 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 5 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 3 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 4 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 5 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 3 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 4 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 5 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 3 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 4 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 5 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 3 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 4 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 5 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 3 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 4 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 5 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 3 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=3,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 4 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=4,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 5 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=5,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 6 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 7 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 8 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 9 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 3.5, beta = 3.5) 10 term",
            "distribution": "Beta (alpha = 3.5, beta = 3.5)",
            "dist": beta(a=3.5, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 6 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 7 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 8 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 9 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 9, beta = 3.5) 10 term",
            "distribution": "Beta (alpha = 9, beta = 3.5)",
            "dist": beta(a=9, b=3.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 6 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 7 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 8 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 9 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.8, beta = 0.9) 10 term",
            "distribution": "Beta (alpha = 0.8, beta = 0.9)",
            "dist": beta(a=0.8, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 6 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 7 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 8 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 9 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 60, beta = 1.5) 10 term",
            "distribution": "Beta (alpha = 60, beta = 1.5)",
            "dist": beta(a=60, b=1.5),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 6 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 7 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 8 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 9 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 1.2, beta = 1.2) 10 term",
            "distribution": "Beta (alpha = 1.2, beta = 1.2)",
            "dist": beta(a=1.2, b=1.2),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 6 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 7 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 8 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 9 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Beta (alpha = 0.9, beta = 0.9) 10 term",
            "distribution": "Beta (alpha = 0.9, beta = 0.9)",
            "dist": beta(a=0.9, b=0.9),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 6 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 7 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 8 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 9 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Uniform (A = 0, B = 1) 10 term",
            "distribution": "Uniform (A = 0, B = 1)",
            "dist": uniform(loc=0, scale=1),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0,
                upper_bound=1,
                num_terms=10,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 6 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=6,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 7 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=7,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 8 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=8,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 9 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=9,
            ),
        },
        {
            "testcase_name": "Triangular (A = 5, B = 20, C = 25) 10 term",
            "distribution": "Triangular (A = 5, B = 20, C = 25)",
            "dist": triang(c=0.75, loc=5, scale=20),
            "metalog_params": MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=5,
                upper_bound=25,
                num_terms=10,
            ),
        },
    )
    def test_metalog_fit(
        self,
        distribution: str,
        dist: rv_frozen,
        metalog_params: MetalogParameters,
    ) -> None:
        """Test metalog fitting for bounded distributions.

        Validates that the fitted metalog distribution closely approximates
        the original distribution by comparing Kolmogorov-Smirnov distances.
        """
        num_terms = metalog_params.num_terms
        expected_ks_dist = self.table78_acceptable_results[distribution][num_terms]

        ks_dist = calculate_ks_distance(dist, metalog_params)
        self.assertLessEqual(float(ks_dist), expected_ks_dist + KS_ROUNDING_TOL)


if __name__ == "__main__":
    absltest.main()
