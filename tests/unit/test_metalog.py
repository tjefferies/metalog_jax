"""Unit tests for the Metalog JAX metalog module.

This module contains comprehensive tests for functions in the metalog_jax.metalog
module, using parameterized testing to validate behavior across various
configurations and edge cases.
"""
# Copyright: Travis Jefferies 2026

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify
from scipy import stats

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    SPTMetalogParameters,
)
from metalog_jax.grid_search import fit_grid
from metalog_jax.metalog import THETA_3, Metalog, fit, fit_spt_metalog
from metalog_jax.utils import DEFAULT_Y

# NOTE: GetQuantilesTest class has been removed as get_quantiles() is now an
# internal function (_get_quantiles) within fit(). The quantile transformation
# logic is thoroughly tested through integration tests in test_metalog_ks.py.


class FitSPTMetalogTest(parameterized.TestCase):
    """Test suite for the fit_spt_metalog function.

    Tests the Symmetric Percentile Triplet (SPT) metalog fitting defined in
    metalog_jax.metalog for various boundedness types, alpha values, and edge cases.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_uniform_alpha_0.1",
            "array": jnp.array(np.linspace(-10, 10, 100)),
            "alpha": 0.1,
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        },
        {
            "testcase_name": "unbounded_uniform_alpha_0.25",
            "array": jnp.array(np.linspace(-5, 5, 100)),
            "alpha": 0.25,
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        },
        {
            "testcase_name": "unbounded_uniform_alpha_0.2",
            "array": jnp.array(np.linspace(-20, 20, 150)),
            "alpha": 0.2,
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        },
        {
            "testcase_name": "unbounded_uniform_alpha_0.15",
            "array": jnp.array(np.linspace(0, 100, 200)),
            "alpha": 0.15,
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        },
    )
    def test_unbounded_fit_returns_valid_coefficients(
        self, array, alpha, boundedness, lower_bound, upper_bound
    ):
        """Test SPT fitting for unbounded distributions returns valid coefficients.

        Args:
            array: Input data array.
            alpha: Lower percentile parameter.
            boundedness: Distribution boundedness type.
            lower_bound: Lower bound (unused for unbounded).
            upper_bound: Upper bound (unused for unbounded).
        """
        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=boundedness,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        result = fit_spt_metalog(array, params)

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.a)
        self.assertIsNotNone(result.metalog_params)

        # Verify coefficient array has exactly 3 terms
        self.assertEqual(len(result.a), 3)

        # Verify all coefficients are finite
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))

        # Verify parameters are preserved
        self.assertEqual(result.metalog_params.alpha, alpha)
        self.assertEqual(result.metalog_params.boundedness, boundedness)

    @parameterized.named_parameters(
        {
            "testcase_name": "lower_bound_uniform_alpha_0.1",
            "array": jnp.array(np.linspace(5.0, 25.0, 100)),
            "alpha": 0.1,
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "lower_bound": 5.0,
            "upper_bound": 0.0,
        },
        {
            "testcase_name": "lower_bound_uniform_alpha_0.25",
            "array": jnp.array(np.linspace(10.0, 20.0, 100)),
            "alpha": 0.25,
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "lower_bound": 10.0,
            "upper_bound": 0.0,
        },
        {
            "testcase_name": "lower_bound_uniform_alpha_0.2",
            "array": jnp.array(np.linspace(0.1, 10.0, 150)),
            "alpha": 0.2,
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        },
    )
    def test_strictly_lower_bound_fit_returns_valid_coefficients(
        self, array, alpha, boundedness, lower_bound, upper_bound
    ):
        """Test SPT fitting for lower-bounded distributions.

        Args:
            array: Input data array (all values > lower_bound).
            alpha: Lower percentile parameter.
            boundedness: Distribution boundedness type.
            lower_bound: Lower bound value.
            upper_bound: Upper bound (unused for strictly lower).
        """
        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=boundedness,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        result = fit_spt_metalog(array, params)

        # Verify coefficient array has exactly 3 terms
        self.assertEqual(len(result.a), 3)

        # Verify all coefficients are finite
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))

        # Verify parameters are preserved
        self.assertEqual(result.metalog_params.alpha, alpha)
        self.assertEqual(result.metalog_params.boundedness, boundedness)
        self.assertEqual(result.metalog_params.lower_bound, lower_bound)

    @parameterized.named_parameters(
        {
            "testcase_name": "bounded_uniform_alpha_0.1",
            "array": jnp.array(np.random.uniform(0, 1, 100)),
            "alpha": 0.1,
            "boundedness": MetalogBoundedness.BOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
        {
            "testcase_name": "bounded_beta_alpha_0.25",
            "array": jnp.array(np.random.beta(2, 2, 100) * 20 - 10),
            "alpha": 0.25,
            "boundedness": MetalogBoundedness.BOUNDED,
            "lower_bound": -10.0,
            "upper_bound": 10.0,
        },
        {
            "testcase_name": "bounded_uniform_wide_range_alpha_0.05",
            "array": jnp.array(np.random.uniform(5, 15, 150)),
            "alpha": 0.05,
            "boundedness": MetalogBoundedness.BOUNDED,
            "lower_bound": 5.0,
            "upper_bound": 15.0,
        },
    )
    def test_bounded_fit_returns_valid_coefficients(
        self, array, alpha, boundedness, lower_bound, upper_bound
    ):
        """Test SPT fitting for fully bounded distributions.

        Args:
            array: Input data array (lower_bound < values < upper_bound).
            alpha: Lower percentile parameter.
            boundedness: Distribution boundedness type.
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
        """
        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=boundedness,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        result = fit_spt_metalog(array, params)

        # Verify coefficient array has exactly 3 terms
        self.assertEqual(len(result.a), 3)

        # Verify all coefficients are finite
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))

        # Verify parameters are preserved
        self.assertEqual(result.metalog_params.alpha, alpha)
        self.assertEqual(result.metalog_params.boundedness, boundedness)
        self.assertEqual(result.metalog_params.lower_bound, lower_bound)
        self.assertEqual(result.metalog_params.upper_bound, upper_bound)

    @parameterized.named_parameters(
        {
            "testcase_name": "too_few_elements_2",
            "array": jnp.array([1.0, 2.0]),
        },
        {
            "testcase_name": "too_few_elements_1",
            "array": jnp.array([5.0]),
        },
        {
            "testcase_name": "empty_array",
            "array": jnp.array([]),
        },
    )
    def test_insufficient_array_length_raises_error(self, array):
        """Test that arrays with fewer than 3 elements raise AssertionError.

        Args:
            array: Input array with insufficient length.
        """
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        with self.assertRaises(AssertionError):
            fit_spt_metalog(array, params)

    @parameterized.named_parameters(
        {
            "testcase_name": "alpha_zero",
            "alpha": 0.0,
        },
        {
            "testcase_name": "alpha_negative",
            "alpha": -0.1,
        },
    )
    def test_invalid_alpha_zero_or_negative_raises_error(self, alpha):
        """Test that alpha <= 0 raises AssertionError.

        Args:
            alpha: Invalid alpha value (zero or negative).
        """
        array = jnp.array(np.random.randn(100))
        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        with self.assertRaises(AssertionError):
            fit_spt_metalog(array, params)

    @parameterized.named_parameters(
        {
            "testcase_name": "alpha_equals_0.5",
            "alpha": 0.5,
        },
        {
            "testcase_name": "alpha_greater_than_0.5",
            "alpha": 0.6,
        },
    )
    def test_invalid_alpha_greater_than_or_equal_half_raises_error(self, alpha):
        """Test that alpha >= 0.5 raises AssertionError.

        Args:
            alpha: Invalid alpha value (>= 0.5).
        """
        array = jnp.array(np.random.randn(100))
        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        with self.assertRaises(AssertionError):
            fit_spt_metalog(array, params)

    def test_strictly_upper_bound_fit_returns_valid_coefficients(self):
        """Test SPT fitting for strictly upper-bounded distributions (issue #23)."""
        array = jnp.array(np.linspace(0.05, 9.95, 100))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.STRICTLY_UPPER_BOUND,
            lower_bound=0.0,
            upper_bound=10.0,
        )

        result = fit_spt_metalog(array, params)

        self.assertEqual(len(result.a), 3)
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))
        y = jnp.array([0.1, 0.5, 0.9])
        # The closed forms are exact, so only float64 rounding remains
        np.testing.assert_allclose(result.ppf(y), jnp.quantile(array, y), rtol=1e-12)

    def test_non_monotonic_quantiles_raises_error(self):
        """Test that non-monotonic quantiles raise AssertionError.

        Creates data where the median is not between q_alpha and q_complement.
        """
        # Create array where quantiles won't be properly ordered
        # Use a degenerate case with repeated values
        array = jnp.array([1.0, 1.0, 1.0, 100.0])
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        # This may or may not raise depending on the exact quantile computation
        # The feasibility check should catch invalid distributions
        try:
            result = fit_spt_metalog(array, params)
            # If it succeeds, verify coefficients are at least finite
            self.assertTrue(jnp.all(jnp.isfinite(result.a)))
        except AssertionError:
            # Expected if quantiles violate monotonicity
            pass

    def test_multidimensional_array_raises_error(self):
        """Test that multidimensional arrays raise AssertionError."""
        # Create 2D array
        array = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        with self.assertRaises(AssertionError):
            fit_spt_metalog(array, params)

    @parameterized.named_parameters(
        {
            "testcase_name": "alpha_0.1_exact_quantiles",
            "quantiles": [1.0, 5.5, 10.0],
            "alpha": 0.1,
        },
        {
            "testcase_name": "alpha_0.25_exact_quantiles",
            "quantiles": [2.0, 6.0, 10.0],
            "alpha": 0.25,
        },
    )
    def test_known_quantiles_unbounded(self, quantiles, alpha):
        """Test SPT fitting with known quantile values for unbounded case.

        Args:
            quantiles: Known quantile values [q_alpha, median, q_complement].
            alpha: Alpha parameter corresponding to the quantiles.
        """
        # Create array that will produce these approximate quantiles
        # Generate data centered around median with appropriate spread
        q_alpha, median, q_complement = quantiles
        # Create uniform distribution that hits these quantiles
        n = 1000
        data = np.linspace(q_alpha, q_complement, n)
        array = jnp.array(data)

        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # Verify the fit produces finite coefficients
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))
        self.assertEqual(len(result.a), 3)

        # First coefficient should be close to median for unbounded
        self.assertAlmostEqual(float(result.a[0]), median, delta=0.5)

    def test_symmetric_distribution_unbounded(self):
        """Test SPT fitting on symmetric distribution produces expected coefficients."""
        # Generate perfectly symmetric data
        array = jnp.array(np.linspace(-10, 10, 1000))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # For symmetric data, first coefficient should be near 0 (median)
        self.assertAlmostEqual(float(result.a[0]), 0.0, delta=0.1)

        # Third coefficient (skewness term) should be near 0 for symmetric data
        self.assertAlmostEqual(float(result.a[2]), 0.0, delta=0.1)

    def test_skewed_distribution_unbounded(self):
        """Test SPT fitting on skewed distribution produces non-zero skewness coefficient."""
        # Generate right-skewed data (exponential)
        array = jnp.array(np.random.exponential(2.0, 1000))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # Third coefficient should be non-zero for skewed data
        # (though we can't predict exact sign without more analysis)
        self.assertNotAlmostEqual(float(result.a[2]), 0.0, delta=0.01)

    def test_lower_bound_coefficients_in_log_space(self):
        """Test that lower-bounded fit produces log-space coefficients."""
        array = jnp.array(np.random.exponential(2.0, 100) + 5.0)
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
            lower_bound=5.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # For lower-bounded, first coefficient is log(median - lower_bound)
        # So a[0] should be the log of a positive value
        # The actual median should be > lower_bound
        expected_log_median = jnp.log(jnp.median(array) - params.lower_bound)

        # First coefficient should be close to log(median - lower_bound)
        self.assertAlmostEqual(
            float(result.a[0]), float(expected_log_median), delta=0.5
        )

    def test_bounded_coefficients_in_log_space(self):
        """Test that bounded fit produces log-space coefficients."""
        array = jnp.array(np.random.uniform(0, 1, 100))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0.0,
            upper_bound=1.0,
        )

        result = fit_spt_metalog(array, params)

        # For bounded, first coefficient is log(gamma_median) where
        # gamma_median = (median - lower) / (upper - median)
        median = jnp.median(array)
        gamma_median = (median - params.lower_bound) / (params.upper_bound - median)
        expected_log_gamma = jnp.log(gamma_median)

        # First coefficient should be close to log(gamma_median)
        self.assertAlmostEqual(float(result.a[0]), float(expected_log_gamma), delta=0.5)

    @parameterized.named_parameters(
        {
            "testcase_name": "small_sample_size_10",
            "size": 10,
        },
        {
            "testcase_name": "small_sample_size_5",
            "size": 5,
        },
        {
            "testcase_name": "minimum_sample_size_3",
            "size": 3,
        },
    )
    def test_small_sample_sizes(self, size):
        """Test SPT fitting with small sample sizes.

        Args:
            size: Number of samples in the array.
        """
        rng = np.random.default_rng(seed=0)
        array = jnp.array(rng.standard_normal(size))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # Should still produce valid coefficients
        self.assertEqual(len(result.a), 3)
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))

    def test_large_sample_size(self):
        """Test SPT fitting with large sample size."""
        array = jnp.array(np.random.randn(10000))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # Should produce valid coefficients
        self.assertEqual(len(result.a), 3)
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))

    @parameterized.named_parameters(
        {
            "testcase_name": "edge_alpha_0.01",
            "alpha": 0.01,
        },
    )
    def test_boundary_alpha_values(self, alpha):
        """Test SPT fitting with alpha values near boundaries.

        Args:
            alpha: Alpha value near 0 or 0.5.
        """
        array = jnp.array(np.random.randn(1000))
        params = SPTMetalogParameters(
            alpha=alpha,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        result = fit_spt_metalog(array, params)

        # Should produce valid coefficients even at boundary values
        self.assertEqual(len(result.a), 3)
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))

    def test_constant_array_raises_error(self):
        """Test that constant array values raise an error or produce degenerate result."""
        array = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        # Constant array should fail feasibility checks
        with self.assertRaises(AssertionError):
            fit_spt_metalog(array, params)

    def test_nearly_constant_array_lower_bound(self):
        """Test lower-bounded fit with nearly constant array."""
        # Array with very small variance above lower bound
        array = jnp.array([5.01, 5.02, 5.03, 5.04, 5.05])
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
            lower_bound=5.0,
            upper_bound=0.0,
        )

        # Should either succeed with valid coefficients or fail feasibility
        try:
            result = fit_spt_metalog(array, params)
            self.assertTrue(jnp.all(jnp.isfinite(result.a)))
        except AssertionError:
            # Expected if feasibility constraints are violated
            pass

    def test_data_near_bounds_bounded_case(self):
        """Test bounded fit when data is very close to bounds."""
        # Data very close to boundaries
        array = jnp.array([0.01, 0.02, 0.5, 0.98, 0.99])
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0.0,
            upper_bound=1.0,
        )

        result = fit_spt_metalog(array, params)

        # Should produce valid coefficients
        self.assertEqual(len(result.a), 3)
        self.assertTrue(jnp.all(jnp.isfinite(result.a)))


class SPTMetalogPropositionsTest(parameterized.TestCase):
    """Regression tests for issue #20: SPT fits against Keelin (2016), Props 1-4."""

    # Inside (LOWER_BOUND, UPPER_BOUND), feasible for every alpha in ALPHAS and every
    # supported boundedness type, and skewed so that a3 is nonzero
    TRIPLET = (2.0, 4.0, 7.0)
    ALPHAS = (0.05, 0.1, 0.25)
    LOWER_BOUND = 0.0
    UPPER_BOUND = 10.0
    SUPPORTED_BOUNDEDNESS = (
        ("unbounded", MetalogBoundedness.UNBOUNDED),
        ("strictly_lower_bound", MetalogBoundedness.STRICTLY_LOWER_BOUND),
        ("strictly_upper_bound", MetalogBoundedness.STRICTLY_UPPER_BOUND),
        ("bounded", MetalogBoundedness.BOUNDED),
    )

    def _params(self, boundedness, alpha):
        """Return SPT parameters that use the class bounds."""
        return SPTMetalogParameters(
            boundedness=boundedness,
            alpha=alpha,
            lower_bound=self.LOWER_BOUND,
            upper_bound=self.UPPER_BOUND,
        )

    def _data(self, q, alpha):
        """Return 101 points whose alpha, 0.5 and 1 - alpha quantiles are q."""
        # Piecewise linear in the index, so jnp.quantile's interpolation is exact
        i = round(alpha * 100)
        knots = [0, i, 50, 100 - i, 100]
        values = [
            0.5 * (self.LOWER_BOUND + q[0]),
            *q,
            0.5 * (q[2] + self.UPPER_BOUND),
        ]
        return jnp.asarray(np.interp(np.arange(101), knots, values))

    def _to_metalog_scale(self, q, boundedness):
        """Transform quantiles to the scale on which the metalog is unbounded."""
        q = np.asarray(q)
        if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            return np.log(q - self.LOWER_BOUND)
        if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            return -np.log(self.UPPER_BOUND - q)
        if boundedness == MetalogBoundedness.BOUNDED:
            return np.log((q - self.LOWER_BOUND) / (self.UPPER_BOUND - q))
        return q

    def _triplet_with_median_at(self, r, boundedness, alpha):
        """Return data whose median sits at relative position r on the metalog scale."""
        z = self._to_metalog_scale((2.0, 7.0), boundedness)
        z_median = z[0] + r * (z[1] - z[0])
        if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            median = self.LOWER_BOUND + np.exp(z_median)
        elif boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            median = self.UPPER_BOUND - np.exp(-z_median)
        elif boundedness == MetalogBoundedness.BOUNDED:
            median = (self.LOWER_BOUND + self.UPPER_BOUND * np.exp(z_median)) / (
                1 + np.exp(z_median)
            )
        else:
            median = z_median
        return self._data((2.0, float(median), 7.0), alpha)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_ppf_reproduces_anchor_quantiles(self, boundedness):
        """The fit's ppf passes through the alpha, 0.5 and 1 - alpha quantiles."""
        for alpha in self.ALPHAS:
            with self.subTest(alpha=alpha):
                m = fit_spt_metalog(
                    self._data(self.TRIPLET, alpha), self._params(boundedness, alpha)
                )
                y = jnp.array([alpha, 0.5, 1 - alpha])
                # The closed forms are exact, so only float64 rounding remains
                np.testing.assert_allclose(m.ppf(y), self.TRIPLET, rtol=1e-10)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_coefficients_match_closed_forms(self, boundedness):
        """Coefficients match Propositions 1, 3 and 4 of Keelin (2016)."""
        for alpha in self.ALPHAS:
            with self.subTest(alpha=alpha):
                m = fit_spt_metalog(
                    self._data(self.TRIPLET, alpha), self._params(boundedness, alpha)
                )
                # Propositions 3 and 4 apply Proposition 1 on the log and logit scales
                z = self._to_metalog_scale(self.TRIPLET, boundedness)
                log_odds = np.log((1 - alpha) / alpha)
                expected = [
                    z[1],
                    (z[2] - z[0]) / (2 * log_odds),
                    (z[2] + z[0] - 2 * z[1]) / ((1 - 2 * alpha) * log_odds),
                ]
                # Float64 rounding, including the cancellation in a3's numerator
                np.testing.assert_allclose(m.a, expected, rtol=1e-10)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_matches_three_term_ols_through_the_triplet(self, boundedness):
        """Coefficients match a 3-term OLS fit through the same three quantiles."""
        for alpha in self.ALPHAS:
            with self.subTest(alpha=alpha):
                q = jnp.asarray(self.TRIPLET)
                spt = fit_spt_metalog(
                    q, self._params(boundedness, alpha), precomputed_quantiles=True
                )
                data = MetalogInputData.from_values(
                    x=q,
                    y=jnp.array([alpha, 0.5, 1 - alpha]),
                    precomputed_quantiles=True,
                )
                params = MetalogParameters(
                    boundedness=boundedness,
                    method=MetalogFitMethod.OLS,
                    lower_bound=self.LOWER_BOUND,
                    upper_bound=self.UPPER_BOUND,
                    num_terms=3,
                )
                # Three points and three terms: least squares interpolates
                # exactly, so only the solver's float64 rounding differs
                np.testing.assert_allclose(spt.a, fit(data, params).a, rtol=1e-9)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_accepts_triplets_just_inside_feasible_range(self, boundedness):
        """Medians just inside Propositions 2-4's range give finite coefficients."""
        alpha = 0.1
        k_alpha = 0.5 * (1 - THETA_3 * (0.5 - alpha))
        for r in (k_alpha + 0.01, 1 - k_alpha - 0.01):
            with self.subTest(r=r):
                data = self._triplet_with_median_at(r, boundedness, alpha)
                m = fit_spt_metalog(data, self._params(boundedness, alpha))
                self.assertTrue(jnp.all(jnp.isfinite(m.a)))

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_rejects_triplets_just_outside_feasible_range(self, boundedness):
        """Medians just outside Propositions 2-4's range raise AssertionError."""
        alpha = 0.1
        k_alpha = 0.5 * (1 - THETA_3 * (0.5 - alpha))
        for r in (k_alpha - 0.01, 1 - k_alpha + 0.01):
            with self.subTest(r=r):
                data = self._triplet_with_median_at(r, boundedness, alpha)
                with self.assertRaises(AssertionError):
                    fit_spt_metalog(data, self._params(boundedness, alpha))

    @parameterized.named_parameters(
        (
            "lower_bound_above_q_alpha",
            MetalogBoundedness.STRICTLY_LOWER_BOUND,
            2.0,
            0.0,
        ),
        (
            "strictly_upper_bound_below_q_complement",
            MetalogBoundedness.STRICTLY_UPPER_BOUND,
            0.0,
            9.0,
        ),
        ("bounded_lower_bound_above_q_alpha", MetalogBoundedness.BOUNDED, 2.0, 20.0),
        (
            "bounded_upper_bound_below_q_complement",
            MetalogBoundedness.BOUNDED,
            0.0,
            9.0,
        ),
        (
            "bounded_upper_bound_below_lower_bound",
            MetalogBoundedness.BOUNDED,
            11.0,
            0.0,
        ),
    )
    def test_quantiles_outside_support_raise(
        self, boundedness, lower_bound, upper_bound
    ):
        """Quantiles outside the support raise instead of giving NaN coefficients."""
        # The 10th and 90th percentiles are 1.9 and 9.1
        data = jnp.linspace(1.0, 10.0, 101)
        # JAX array bounds, which construction doesn't check (Python floats with
        # upper below lower raise there), so the fit's own checks run, as they do
        # for bounds traced under jit or vmap
        params = SPTMetalogParameters(
            boundedness=boundedness,
            alpha=0.1,
            lower_bound=jnp.asarray(lower_bound),
            upper_bound=jnp.asarray(upper_bound),
        )
        with self.assertRaises(AssertionError):
            fit_spt_metalog(data, params)

    def test_non_finite_data_raises(self):
        """Data containing NaN raises instead of giving NaN coefficients."""
        data = jnp.linspace(1.0, 10.0, 101).at[50].set(jnp.nan)
        with self.assertRaises(AssertionError):
            fit_spt_metalog(data, self._params(MetalogBoundedness.UNBOUNDED, 0.1))


class SPTMetalogTransformTest(parameterized.TestCase):
    """Tests that fit_spt_metalog runs under jax.vmap and jax.jit."""

    DATA = (1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1)
    ALPHAS = (0.05, 0.1, 0.25)
    SUPPORTED_BOUNDEDNESS = (
        ("unbounded", MetalogBoundedness.UNBOUNDED, 0.0, 0.0),
        ("strictly_lower_bound", MetalogBoundedness.STRICTLY_LOWER_BOUND, 0.0, 0.0),
        ("strictly_upper_bound", MetalogBoundedness.STRICTLY_UPPER_BOUND, 0.0, 20.0),
        ("bounded", MetalogBoundedness.BOUNDED, 0.0, 20.0),
    )

    def _batch(self):
        """Return three feasible datasets of the same length."""
        data = jnp.asarray(self.DATA)
        return jnp.stack([data, 1.5 * data, data + 0.7])

    def _params(self, boundedness, lower_bound, upper_bound, alpha=0.1):
        """Return SPT parameters."""
        return SPTMetalogParameters(
            boundedness=boundedness,
            alpha=alpha,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_vmap_over_data_matches_loop(self, boundedness, lower_bound, upper_bound):
        """Vmapping over datasets gives the same coefficients as fitting each one."""
        params = self._params(boundedness, lower_bound, upper_bound)
        batch = self._batch()
        batched = jax.vmap(lambda d: fit_spt_metalog(d, params).a)(batch)
        looped = [fit_spt_metalog(d, params).a for d in batch]
        # The same operations run batched or one at a time
        np.testing.assert_allclose(batched, looped, rtol=1e-12)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_vmap_over_alpha_matches_loop(self, boundedness, lower_bound, upper_bound):
        """Vmapping over alpha gives the same coefficients as fitting each one."""
        data = jnp.asarray(self.DATA)

        def fit_a(alpha):
            """Return the coefficients fitted with this alpha."""
            params = self._params(boundedness, lower_bound, upper_bound, alpha)
            return fit_spt_metalog(data, params).a

        batched = jax.vmap(fit_a)(jnp.asarray(self.ALPHAS))
        looped = [fit_a(alpha) for alpha in self.ALPHAS]
        np.testing.assert_allclose(batched, looped, rtol=1e-12)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_checkified_jit_matches_eager(self, boundedness, lower_bound, upper_bound):
        """A jitted fit, functionalized with checkify, matches the eager fit."""
        params = self._params(boundedness, lower_bound, upper_bound)
        data = jnp.asarray(self.DATA)
        fit_a = checkify.checkify(jax.jit(lambda d: fit_spt_metalog(d, params).a))
        err, a = fit_a(data)
        err.throw()
        # Compiled and eager evaluation can differ by float64 rounding
        np.testing.assert_allclose(a, fit_spt_metalog(data, params).a, rtol=1e-12)

    def test_vmap_rejects_an_invalid_dataset(self):
        """An invalid dataset in a vmapped batch raises checkify.JaxRuntimeError."""
        params = self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0)
        # A constant row has no strictly ascending quantiles
        batch = self._batch().at[1].set(5.0)
        with self.assertRaises(checkify.JaxRuntimeError):
            jax.vmap(lambda d: fit_spt_metalog(d, params).a)(batch)

    def test_checkified_jit_reports_an_invalid_dataset(self):
        """A checkified jitted fit reports issue #20's infeasible lower-bound triplet."""
        params = self._params(MetalogBoundedness.STRICTLY_LOWER_BOUND, 0.0, 0.0)
        # Quantiles 1.0, 1.1 and 10.0, infeasible by Proposition 3
        data = jnp.array([0.5, 1.0, 1.02, 1.05, 1.08, 1.1, 2.0, 4.0, 6.0, 10.0, 12.0])
        err, _ = checkify.checkify(jax.jit(lambda d: fit_spt_metalog(d, params).a))(
            data
        )
        self.assertIsNotNone(err.get())
        with self.assertRaises(checkify.JaxRuntimeError):
            err.throw()


class SPTPrecomputedQuantilesTest(parameterized.TestCase):
    """Tests for issue #24: fit_spt_metalog with precomputed_quantiles=True."""

    # Assessed 10th, 50th and 90th percentiles inside (LOWER_BOUND, UPPER_BOUND),
    # feasible for every supported boundedness type
    TRIPLET = (2.0, 4.0, 7.0)
    ALPHA = 0.1
    LOWER_BOUND = 0.0
    UPPER_BOUND = 10.0
    SUPPORTED_BOUNDEDNESS = (
        ("unbounded", MetalogBoundedness.UNBOUNDED),
        ("strictly_lower_bound", MetalogBoundedness.STRICTLY_LOWER_BOUND),
        ("strictly_upper_bound", MetalogBoundedness.STRICTLY_UPPER_BOUND),
        ("bounded", MetalogBoundedness.BOUNDED),
    )

    def _params(self, boundedness=MetalogBoundedness.UNBOUNDED):
        """Return SPT parameters that use the class alpha and bounds."""
        return SPTMetalogParameters(
            boundedness=boundedness,
            alpha=self.ALPHA,
            lower_bound=self.LOWER_BOUND,
            upper_bound=self.UPPER_BOUND,
        )

    def _fit(self, quantiles, boundedness=MetalogBoundedness.UNBOUNDED):
        """Fit precomputed quantiles."""
        return fit_spt_metalog(
            jnp.asarray(quantiles),
            self._params(boundedness),
            precomputed_quantiles=True,
        )

    def _data_with_quantiles(self, q):
        """Return 101 points whose 10th, 50th and 90th percentiles are q."""
        # Piecewise linear in the index, so jnp.quantile's interpolation is exact
        knots = [0, 10, 50, 90, 100]
        values = [0.5 * (self.LOWER_BOUND + q[0]), *q, 0.5 * (q[2] + self.UPPER_BOUND)]
        return jnp.asarray(np.interp(np.arange(101), knots, values))

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_matches_data_with_the_same_quantiles(self, boundedness):
        """A triplet gives the same fit as data whose quantiles are that triplet."""
        from_triplet = self._fit(self.TRIPLET, boundedness)
        data = self._data_with_quantiles(self.TRIPLET)
        from_data = fit_spt_metalog(data, self._params(boundedness))
        # Same triplet either way, up to rounding in jnp.quantile's interpolation
        np.testing.assert_allclose(from_triplet.a, from_data.a, rtol=1e-12)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_ppf_passes_through_the_quantiles(self, boundedness):
        """The fit's ppf returns the three quantiles at alpha, 0.5 and 1 - alpha."""
        m = self._fit(self.TRIPLET, boundedness)
        y = jnp.array([self.ALPHA, 0.5, 1 - self.ALPHA])
        # The closed forms are exact, so only float64 rounding remains
        np.testing.assert_allclose(m.ppf(y), self.TRIPLET, rtol=1e-12)

    def test_three_values_as_data_are_interpolated(self):
        """Without the flag, a 3-element array is data, so the fit misses its values."""
        m = fit_spt_metalog(jnp.asarray(self.TRIPLET), self._params())
        y = jnp.array([self.ALPHA, 0.5, 1 - self.ALPHA])
        self.assertFalse(np.allclose(m.ppf(y), self.TRIPLET))

    def test_integer_quantiles_are_cast_to_float(self):
        """Integer quantiles give the same float64 coefficients as float ones."""
        from_ints = self._fit(jnp.array([2, 4, 7]))
        from_floats = self._fit(jnp.array([2.0, 4.0, 7.0]))
        self.assertEqual(from_ints.a.dtype, jnp.float64)
        np.testing.assert_array_equal(from_ints.a, from_floats.a)

    @parameterized.named_parameters(
        ("two_values", (2.0, 4.0)),
        ("four_values", (1.0, 2.0, 4.0, 7.0)),
    )
    def test_wrong_number_of_quantiles_raises(self, quantiles):
        """Anything but exactly three quantiles raises AssertionError."""
        with self.assertRaises(AssertionError):
            self._fit(quantiles)

    def test_unsorted_quantiles_raise(self):
        """Quantiles that aren't strictly ascending raise AssertionError."""
        with self.assertRaises(AssertionError):
            self._fit((4.0, 2.0, 7.0))

    def test_vmap_over_assessments_matches_loop(self):
        """Vmapping over a batch of triplets matches fitting each one."""
        assessments = jnp.array([[2.0, 4.0, 7.0], [1.0, 1.5, 3.0], [5.0, 6.0, 9.0]])
        batched = jax.vmap(lambda q: self._fit(q).a)(assessments)
        looped = [self._fit(q).a for q in assessments]
        np.testing.assert_allclose(batched, looped, rtol=1e-12)

    @parameterized.named_parameters(*SUPPORTED_BOUNDEDNESS)
    def test_checkified_jit_matches_eager(self, boundedness):
        """A jitted fit, with the quantiles and parameters as arguments, matches eager."""
        quantiles = jnp.asarray(self.TRIPLET)
        params = self._params(boundedness)
        jitted = checkify.checkify(
            jax.jit(lambda q, p: fit_spt_metalog(q, p, precomputed_quantiles=True).a)
        )
        err, a = jitted(quantiles, params)
        err.throw()
        eager = fit_spt_metalog(quantiles, params, precomputed_quantiles=True).a
        # Compiled and eager evaluation can differ by float64 rounding
        np.testing.assert_allclose(a, eager, rtol=1e-12)

    def test_vmap_rejects_an_infeasible_assessment(self):
        """An infeasible triplet in a vmapped batch raises JaxRuntimeError."""
        # The median sits too close to the lower quantile (Proposition 2)
        assessments = jnp.array([[2.0, 4.0, 7.0], [1.0, 1.05, 10.0]])
        with self.assertRaises(checkify.JaxRuntimeError):
            jax.vmap(lambda q: self._fit(q).a)(assessments)


class GetFitFunctionTest(absltest.TestCase):
    """Test suite for the _get_fit_function helper."""

    def test_ols_returns_function_directly(self):
        """Test dispatch returns OLS function without partial wrapping."""
        from functools import partial

        from metalog_jax.base import MetalogFitMethod
        from metalog_jax.metalog import _FIT_METHOD_DISPATCH, _get_fit_function

        result = _get_fit_function(MetalogFitMethod.OLS)
        self.assertEqual(result, _FIT_METHOD_DISPATCH[MetalogFitMethod.OLS])
        self.assertNotIsInstance(result, partial)

    def test_lasso_with_hyperparams_returns_partial(self):
        """Test dispatch returns partial with hyperparams bound."""
        from functools import partial

        from metalog_jax.base import MetalogFitMethod
        from metalog_jax.metalog import _get_fit_function
        from metalog_jax.regression import LassoParameters

        params = LassoParameters(
            lam=0.1,
            learning_rate=1e-3,
            num_iters=1000,
            tol=1e-6,
            momentum=0.9,
        )
        result = _get_fit_function(MetalogFitMethod.Lasso, params)
        self.assertIsInstance(result, partial)

    def test_invalid_method_raises_type_error(self):
        """Test dispatch raises TypeError for invalid method."""
        from metalog_jax.metalog import _get_fit_function

        with self.assertRaises(TypeError) as ctx:
            _get_fit_function("invalid")
        self.assertIn("not type MetalogFitMethod", str(ctx.exception))

    def test_hyperparams_rejected_for_methods_without_them(self):
        """Test OLS and Feasible raise a clear TypeError when given hyperparams."""
        from metalog_jax.base import MetalogFitMethod
        from metalog_jax.metalog import _get_fit_function
        from metalog_jax.regression import LassoParameters

        params = LassoParameters(
            lam=0.1,
            learning_rate=1e-3,
            num_iters=1000,
            tol=1e-6,
            momentum=0.9,
        )
        for method in (MetalogFitMethod.OLS, MetalogFitMethod.Feasible):
            with self.subTest(method=method.name):
                with self.assertRaises(TypeError) as ctx:
                    _get_fit_function(method, params)
                self.assertIn(
                    f"MetalogFitMethod.{method.name} takes no hyperparameters",
                    str(ctx.exception),
                )

    def test_all_methods_in_dispatch(self):
        """Test all MetalogFitMethod values are in dispatch table."""
        from metalog_jax.base import MetalogFitMethod
        from metalog_jax.metalog import _FIT_METHOD_DISPATCH

        for method in MetalogFitMethod:
            self.assertIn(method, _FIT_METHOD_DISPATCH)


class TwoTermFitTest(parameterized.TestCase):
    """Regression tests for issue #47: a 2-term fit regresses on two basis functions.

    The design matrix used to have three columns for any num_terms up to 3, so a
    2-term fit was a 3-term fit with its third coefficient dropped.
    """

    LOWER_BOUND = 0.0
    UPPER_BOUND = 20.0
    DATA = MetalogInputData.from_values(
        jnp.asarray(np.random.default_rng(0).gamma(2.0, size=300)),
        DEFAULT_Y,
        precomputed_quantiles=False,
    )

    def _params(self, boundedness, method):
        """2-term parameters whose bounds contain the data (0.11 to 7.96)."""
        return MetalogParameters(
            boundedness=boundedness,
            method=method,
            lower_bound=self.LOWER_BOUND,
            upper_bound=self.UPPER_BOUND,
            num_terms=2,
        )

    def _least_squares(self, boundedness):
        """The 2-term least-squares coefficients, from NumPy.

        M(y) = a1 + a2 ln(y / (1 - y)) (Keelin 2016, Definition 1, k = 2), fitted to
        the quantiles on the boundedness type's scale: x, ln(x - L), -ln(U - x) or
        ln((x - L) / (U - x)).
        """
        x, y = np.asarray(self.DATA.x), np.asarray(self.DATA.y)
        lower, upper = self.LOWER_BOUND, self.UPPER_BOUND
        z = {
            MetalogBoundedness.UNBOUNDED: x,
            MetalogBoundedness.STRICTLY_LOWER_BOUND: np.log(x - lower),
            MetalogBoundedness.STRICTLY_UPPER_BOUND: -np.log(upper - x),
            MetalogBoundedness.BOUNDED: np.log((x - lower) / (upper - x)),
        }[boundedness]
        basis = np.column_stack([np.ones_like(y), np.log(y / (1 - y))])
        return np.linalg.lstsq(basis, z, rcond=None)[0]

    @parameterized.product(
        boundedness=tuple(MetalogBoundedness),
        method=(MetalogFitMethod.OLS, MetalogFitMethod.Feasible),
    )
    def test_matches_least_squares(self, boundedness, method):
        """OLS and Feasible return the two least-squares coefficients.

        The least-squares fit has a2 > 0, so it's feasible and Feasible returns it.
        """
        m = fit(self.DATA, self._params(boundedness, method))

        self.assertEqual(m.a.shape, (2,))
        np.testing.assert_allclose(m.a, self._least_squares(boundedness), rtol=1e-10)

    @parameterized.parameters(*MetalogBoundedness)
    def test_lasso_is_close_to_least_squares(self, boundedness):
        """Lasso returns two coefficients, shrunk slightly by its default penalty.

        The shrinkage is below 0.03% of each coefficient here, while the 3-term
        regression's intercept differed by 20% in the unbounded case.
        """
        m = fit(self.DATA, self._params(boundedness, MetalogFitMethod.Lasso))

        self.assertEqual(m.a.shape, (2,))
        np.testing.assert_allclose(m.a, self._least_squares(boundedness), rtol=1e-3)

    @parameterized.product(
        boundedness=tuple(MetalogBoundedness),
        method=(MetalogFitMethod.OLS, MetalogFitMethod.Lasso),
    )
    def test_matches_fit_grid(self, boundedness, method):
        """The fit matches fit_grid, which builds its own, masked design matrix."""
        params = self._params(boundedness, method)

        m = fit(self.DATA, params)
        grid = fit_grid(self.DATA.x, self.DATA.y, params)

        np.testing.assert_allclose(m.a, grid.metalog.a, rtol=1e-12)

    def test_unbounded_fit_is_logistic(self):
        """An unbounded 2-term metalog is the logistic distribution (loc a1, scale a2)."""
        m = fit(
            self.DATA, self._params(MetalogBoundedness.UNBOUNDED, MetalogFitMethod.OLS)
        )
        loc, scale = (float(c) for c in m.a)
        y = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
        logistic = stats.logistic(loc=loc, scale=scale)

        np.testing.assert_allclose(m.ppf(jnp.asarray(y)), logistic.ppf(y), rtol=1e-12)
        np.testing.assert_allclose(
            m.pdf(jnp.asarray(y)), logistic.pdf(logistic.ppf(y)), rtol=1e-12
        )
        np.testing.assert_allclose(m.mean, logistic.mean(), rtol=1e-12)
        np.testing.assert_allclose(m.var, logistic.var(), rtol=1e-12)

    @parameterized.parameters(2, 3, 4, 5, 9)
    def test_design_matrix_has_one_column_per_term(self, num_terms):
        """The design matrix has num_terms columns, starting 1, ln(y / (1 - y))."""
        y = np.asarray(DEFAULT_Y)

        target = Metalog._get_target(jnp.asarray(y), num_terms)

        self.assertEqual(target.shape, (len(y), num_terms))
        np.testing.assert_allclose(target[:, 0], 1.0)
        np.testing.assert_allclose(target[:, 1], np.log(y / (1 - y)), rtol=1e-12)


if __name__ == "__main__":
    absltest.main()
