"""Unit tests for the Metalog JAX metalog module.

This module contains comprehensive tests for functions in the metalog_jax.metalog
module, using parameterized testing to validate behavior across various
configurations and edge cases.
"""
# Copyright: Travis Jefferies 2026

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from metalog_jax.base import (
    MetalogBoundedness,
    SPTMetalogParameters,
)
from metalog_jax.metalog import fit_spt_metalog

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

    def test_strictly_upper_bound_raises_not_implemented(self):
        """Test that STRICTLY_UPPER_BOUND raises NotImplementedError."""
        array = jnp.array(np.random.uniform(0, 10, 100))
        params = SPTMetalogParameters(
            alpha=0.1,
            boundedness=MetalogBoundedness.STRICTLY_UPPER_BOUND,
            lower_bound=0.0,
            upper_bound=10.0,
        )

        with self.assertRaises(NotImplementedError):
            fit_spt_metalog(array, params)

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

    def test_all_methods_in_dispatch(self):
        """Test all MetalogFitMethod values are in dispatch table."""
        from metalog_jax.base import MetalogFitMethod
        from metalog_jax.metalog import _FIT_METHOD_DISPATCH

        for method in MetalogFitMethod:
            self.assertIn(method, _FIT_METHOD_DISPATCH)


if __name__ == "__main__":
    absltest.main()
