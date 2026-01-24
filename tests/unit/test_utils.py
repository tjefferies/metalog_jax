"""Unit tests for the Metalog JAX utility functions.

This module contains comprehensive tests for all utility functions in the
metalog_jax.utils module, using parameterized testing to validate behavior
across a wide range of inputs.
"""
# Copyright: Travis Jefferies 2026

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from metalog_jax.utils import (
    HDRPRNGParameters,
    assert_numeric_array,
    assert_probability_range,
    hdrprng,
    ks_distance,
)

# Enable 64-bit precision for JAX (required for hdrprng large integers)
jax.config.update("jax_enable_x64", True)


class AssertNumericArrayTest(parameterized.TestCase):
    """Test suite for the assert_numeric_array function.

    Tests validation of arrays to ensure they contain only numeric (int or float)
    data types, using parameterized inputs to cover various scenarios.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "int32_array",
            "array": jnp.array([1, 2, 3], dtype=jnp.int32),
        },
        {
            "testcase_name": "int64_array",
            "array": jnp.array([10, 20, 30], dtype=jnp.int64),
        },
        {
            "testcase_name": "float32_array",
            "array": jnp.array([1.0, 2.5, 3.7], dtype=jnp.float32),
        },
        {
            "testcase_name": "float64_array",
            "array": jnp.array([1.0, 2.5, 3.7], dtype=jnp.float64),
        },
        {
            "testcase_name": "numpy_int_array",
            "array": np.array([1, 2, 3], dtype=np.int32),
        },
        {
            "testcase_name": "numpy_float_array",
            "array": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        },
        {
            "testcase_name": "single_element_int",
            "array": jnp.array([42], dtype=jnp.int32),
        },
        {
            "testcase_name": "single_element_float",
            "array": jnp.array([42.0], dtype=jnp.float32),
        },
        {
            "testcase_name": "empty_int_array",
            "array": jnp.array([], dtype=jnp.int32),
        },
        {
            "testcase_name": "empty_float_array",
            "array": jnp.array([], dtype=jnp.float32),
        },
        {
            "testcase_name": "multidimensional_int_array",
            "array": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        },
        {
            "testcase_name": "multidimensional_float_array",
            "array": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        },
        {
            "testcase_name": "zero_array",
            "array": jnp.zeros(10, dtype=jnp.float32),
        },
        {
            "testcase_name": "ones_array",
            "array": jnp.ones(5, dtype=jnp.int32),
        },
    )
    def test_valid_numeric_arrays(self, array):
        """Test that valid numeric arrays pass assertion without error.

        Args:
            array: A valid numeric array (int or float type).
        """
        # Should not raise any exception
        assert_numeric_array(array)

    @parameterized.named_parameters(
        {
            "testcase_name": "str_array",
            "array": np.array(["a", "b", "c"]),
        },
    )
    def test_invalid_non_numeric_arrays(self, array):
        """Test that non-numeric arrays raise AssertionError.

        Args:
            array: An invalid non-numeric array (bool, complex, str, etc.).
        """
        with self.assertRaises(ValueError):
            assert_numeric_array(array)


class AssertProbabilityRangeTest(parameterized.TestCase):
    """Test suite for the assert_probability_range function.

    Tests validation of arrays to ensure all values are in the open interval (0, 1),
    representing valid probability values that exclude the boundary points.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "valid_probabilities_mid_range",
            "array": jnp.array([0.1, 0.5, 0.9]),
        },
        {
            "testcase_name": "valid_probabilities_near_lower",
            "array": jnp.array([0.01, 0.02, 0.03]),
        },
        {
            "testcase_name": "valid_probabilities_near_upper",
            "array": jnp.array([0.97, 0.98, 0.99]),
        },
        {
            "testcase_name": "valid_single_probability",
            "array": jnp.array([0.5]),
        },
        {
            "testcase_name": "valid_uniform_spacing",
            "array": jnp.arange(0.01, 0.99, 0.01),
        },
        {
            "testcase_name": "valid_quartiles",
            "array": jnp.array([0.25, 0.5, 0.75]),
        },
        {
            "testcase_name": "valid_deciles",
            "array": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        },
        {
            "testcase_name": "valid_very_close_to_zero",
            "array": jnp.array([1e-10, 0.5, 0.9]),
        },
        {
            "testcase_name": "valid_very_close_to_one",
            "array": jnp.array([0.1, 0.5, 1.0 - 1e-7]),
        },
        {
            "testcase_name": "valid_multidimensional_array",
            "array": jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        },
        {
            "testcase_name": "valid_float32",
            "array": jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32),
        },
        {
            "testcase_name": "valid_float64",
            "array": jnp.array([0.25, 0.5, 0.75], dtype=jnp.float64),
        },
    )
    def test_valid_probability_arrays(self, array):
        """Test that valid probability arrays pass assertion without error.

        Args:
            array: Array with all values in the open interval (0, 1).
        """
        # Should not raise any exception
        assert_probability_range(array)

    @parameterized.named_parameters(
        {
            "testcase_name": "invalid_contains_zero",
            "array": jnp.array([0.0, 0.5, 0.9]),
            "error_pattern": "minimum.*greater than 0",
        },
        {
            "testcase_name": "invalid_contains_one",
            "array": jnp.array([0.1, 0.5, 1.0]),
            "error_pattern": "maximum.*less than 1",
        },
        {
            "testcase_name": "invalid_all_zeros",
            "array": jnp.zeros(5),
            "error_pattern": "minimum.*greater than 0",
        },
        {
            "testcase_name": "invalid_all_ones",
            "array": jnp.ones(5),
            "error_pattern": "maximum.*less than 1",
        },
        {
            "testcase_name": "invalid_negative_value",
            "array": jnp.array([-0.1, 0.5, 0.9]),
            "error_pattern": "minimum.*greater than 0",
        },
        {
            "testcase_name": "invalid_greater_than_one",
            "array": jnp.array([0.1, 0.5, 1.1]),
            "error_pattern": "maximum.*less than 1",
        },
        {
            "testcase_name": "invalid_both_boundaries",
            "array": jnp.array([0.0, 0.5, 1.0]),
            "error_pattern": "minimum.*greater than 0",
        },
        {
            "testcase_name": "invalid_out_of_range_negative",
            "array": jnp.array([-1.0, -0.5, 0.5]),
            "error_pattern": "minimum.*greater than 0",
        },
        {
            "testcase_name": "invalid_out_of_range_positive",
            "array": jnp.array([0.5, 1.5, 2.0]),
            "error_pattern": "maximum.*less than 1",
        },
        {
            "testcase_name": "invalid_multidimensional_with_zero",
            "array": jnp.array([[0.0, 0.2], [0.3, 0.4]]),
            "error_pattern": "minimum.*greater than 0",
        },
        {
            "testcase_name": "invalid_multidimensional_with_one",
            "array": jnp.array([[0.5, 0.6], [0.7, 1.0]]),
            "error_pattern": "maximum.*less than 1",
        },
    )
    def test_invalid_probability_arrays(self, array, error_pattern):
        """Test that invalid probability arrays raise ValueError.

        Args:
            array: Array containing values outside the open interval (0, 1).
            error_pattern: Regular expression pattern to match in error message.
        """
        with self.assertRaisesRegex(ValueError, error_pattern):
            assert_probability_range(array)

    def test_boundary_case_exactly_zero(self):
        """Test that array with minimum value exactly 0 raises ValueError."""
        array = jnp.array([0.0, 0.3, 0.7])
        with self.assertRaisesRegex(ValueError, "minimum.*greater than 0"):
            assert_probability_range(array)

    def test_boundary_case_exactly_one(self):
        """Test that array with maximum value exactly 1 raises ValueError."""
        array = jnp.array([0.3, 0.7, 1.0])
        with self.assertRaisesRegex(ValueError, "maximum.*less than 1"):
            assert_probability_range(array)

    def test_mixed_valid_invalid_lower_bound(self):
        """Test array with valid values but one at lower boundary."""
        array = jnp.array([0.0, 0.25, 0.5, 0.75, 0.99])
        with self.assertRaisesRegex(ValueError, "minimum.*greater than 0"):
            assert_probability_range(array)

    def test_mixed_valid_invalid_upper_bound(self):
        """Test array with valid values but one at upper boundary."""
        array = jnp.array([0.01, 0.25, 0.5, 0.75, 1.0])
        with self.assertRaisesRegex(ValueError, "maximum.*less than 1"):
            assert_probability_range(array)

    def test_numpy_array_valid(self):
        """Test that valid numpy arrays work correctly."""
        array = np.array([0.2, 0.4, 0.6, 0.8])
        # Should not raise any exception
        assert_probability_range(array)

    def test_numpy_array_invalid(self):
        """Test that invalid numpy arrays raise ValueError."""
        array = np.array([0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "minimum.*greater than 0"):
            assert_probability_range(array)


class HDRPRNGParametersTest(parameterized.TestCase):
    """Test suite for the HDRPRNGParameters dataclass.

    Tests validation of HDRPRNGParameters to ensure proper initialization
    and constraint enforcement, particularly the trial >= 1 requirement.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "default_parameters",
            "trial": 1,
            "variable": 0,
            "entity": 0,
            "time": 0,
            "agent": 0,
        },
        {
            "testcase_name": "all_positive_values",
            "trial": 100,
            "variable": 50,
            "entity": 25,
            "time": 10,
            "agent": 5,
        },
        {
            "testcase_name": "large_values",
            "trial": 1000000,
            "variable": 999999,
            "entity": 888888,
            "time": 777777,
            "agent": 666666,
        },
        {
            "testcase_name": "trial_one_others_nonzero",
            "trial": 1,
            "variable": 100,
            "entity": 200,
            "time": 300,
            "agent": 400,
        },
        {
            "testcase_name": "all_same_positive",
            "trial": 42,
            "variable": 42,
            "entity": 42,
            "time": 42,
            "agent": 42,
        },
    )
    def test_valid_parameters(self, trial, variable, entity, time, agent):
        """Test that valid HDRPRNGParameters can be created without error.

        Args:
            trial: Trial parameter (must be >= 1).
            variable: Variable parameter.
            entity: Entity parameter.
            time: Time parameter.
            agent: Agent parameter.
        """
        params = HDRPRNGParameters(
            trial=trial,
            variable=variable,
            entity=entity,
            time=time,
            agent=agent,
        )
        self.assertEqual(params.trial, trial)
        self.assertEqual(params.variable, variable)
        self.assertEqual(params.entity, entity)
        self.assertEqual(params.time, time)
        self.assertEqual(params.agent, agent)

    def test_default_initialization(self):
        """Test that HDRPRNGParameters uses correct default values."""
        params = HDRPRNGParameters()
        self.assertEqual(params.trial, 1)
        self.assertEqual(params.variable, 0)
        self.assertEqual(params.entity, 0)
        self.assertEqual(params.time, 0)
        self.assertEqual(params.agent, 0)


class HDRPRNGTest(parameterized.TestCase):
    """Test suite for the hdrprng function.

    Tests the Hubbard Decision Research PRNG algorithm to ensure it generates
    deterministic, properly distributed pseudo-random values in the range (0, 1)
    based on the input parameters.
    """

    def test_deterministic_output(self):
        """Test that identical parameters always produce identical outputs."""
        params = HDRPRNGParameters(trial=1, variable=0, entity=0, time=0, agent=0)
        result1 = hdrprng(params)
        result2 = hdrprng(params)
        result3 = hdrprng(params)

        # All calls with same parameters should produce same result
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)

    def test_output_range(self):
        """Test that all outputs are in the open interval (0, 1)."""
        test_cases = [
            HDRPRNGParameters(trial=1, variable=0, entity=0, time=0, agent=0),
            HDRPRNGParameters(trial=100, variable=50, entity=25, time=10, agent=5),
            HDRPRNGParameters(trial=999, variable=999, entity=999, time=999, agent=999),
            HDRPRNGParameters(trial=1, variable=1000000, entity=0, time=0, agent=0),
            HDRPRNGParameters(trial=50, variable=-100, entity=-50, time=-25, agent=-10),
        ]

        for params in test_cases:
            result = hdrprng(params)
            self.assertGreater(result, 0.0, f"Output {result} not greater than 0")
            self.assertLess(result, 1.0, f"Output {result} not less than 1")

    @parameterized.named_parameters(
        {
            "testcase_name": "change_trial",
            "params1": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=0
            ),
            "params2": HDRPRNGParameters(
                trial=2, variable=0, entity=0, time=0, agent=0
            ),
        },
        {
            "testcase_name": "change_variable",
            "params1": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=0
            ),
            "params2": HDRPRNGParameters(
                trial=1, variable=1, entity=0, time=0, agent=0
            ),
        },
        {
            "testcase_name": "change_entity",
            "params1": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=0
            ),
            "params2": HDRPRNGParameters(
                trial=1, variable=0, entity=1, time=0, agent=0
            ),
        },
        {
            "testcase_name": "change_time",
            "params1": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=0
            ),
            "params2": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=1, agent=0
            ),
        },
        {
            "testcase_name": "change_agent",
            "params1": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=0
            ),
            "params2": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=1
            ),
        },
        {
            "testcase_name": "change_multiple",
            "params1": HDRPRNGParameters(
                trial=1, variable=0, entity=0, time=0, agent=0
            ),
            "params2": HDRPRNGParameters(
                trial=5, variable=10, entity=15, time=20, agent=25
            ),
        },
    )
    def test_parameter_independence(self, params1, params2):
        """Test that different parameters produce different outputs.

        Args:
            params1: First set of parameters.
            params2: Second set of parameters (different from params1).
        """
        result1 = hdrprng(params1)
        result2 = hdrprng(params2)
        self.assertNotEqual(
            result1, result2, "Different parameters should produce different outputs"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "default_params",
            "params": HDRPRNGParameters(),
        },
        {
            "testcase_name": "trial_42",
            "params": HDRPRNGParameters(trial=42),
        },
        {
            "testcase_name": "all_zeros_except_trial",
            "params": HDRPRNGParameters(trial=1, variable=0, entity=0, time=0, agent=0),
        },
        {
            "testcase_name": "all_equal_positive",
            "params": HDRPRNGParameters(
                trial=100, variable=100, entity=100, time=100, agent=100
            ),
        },
        {
            "testcase_name": "mixed_positive_negative",
            "params": HDRPRNGParameters(
                trial=50, variable=-25, entity=75, time=-100, agent=200
            ),
        },
        {
            "testcase_name": "large_trial",
            "params": HDRPRNGParameters(trial=1000000),
        },
        {
            "testcase_name": "large_variable",
            "params": HDRPRNGParameters(trial=1, variable=1000000),
        },
        {
            "testcase_name": "sequential_trial_1",
            "params": HDRPRNGParameters(trial=1, variable=1, entity=1, time=1, agent=1),
        },
        {
            "testcase_name": "sequential_trial_2",
            "params": HDRPRNGParameters(trial=2, variable=1, entity=1, time=1, agent=1),
        },
        {
            "testcase_name": "sequential_trial_3",
            "params": HDRPRNGParameters(trial=3, variable=1, entity=1, time=1, agent=1),
        },
    )
    def test_specific_parameter_combinations(self, params):
        """Test specific parameter combinations produce valid outputs.

        Args:
            params: HDRPRNGParameters to test.
        """
        result = hdrprng(params)
        self.assertIsInstance(result, (float, jnp.ndarray))
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_simulation_trial_sequence(self):
        """Test a realistic simulation scenario with sequential trials."""
        # Simulate 100 trials for the same variable/entity/time/agent combination
        results = []
        for trial in range(1, 11):
            params = HDRPRNGParameters(
                trial=trial, variable=0, entity=0, time=0, agent=0
            )
            results.append(float(hdrprng(params)))

        # All results should be unique (different trials)
        unique_results = set(results)
        self.assertEqual(
            len(unique_results), 10, "Each trial should produce a unique value"
        )

        # All results should be in valid range
        for result in results:
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_multiple_variables_same_trial(self):
        """Test multiple independent random variables in the same trial."""
        trial = 42
        num_variables = 50

        results = []
        for variable in range(num_variables):
            params = HDRPRNGParameters(
                trial=trial, variable=variable, entity=0, time=0, agent=0
            )
            results.append(float(hdrprng(params)))

        # All results should be unique (different variables)
        unique_results = set(results)
        self.assertEqual(
            len(unique_results),
            num_variables,
            "Each variable should produce a unique value",
        )

        # All results should be in valid range
        for result in results:
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_negative_parameters_valid(self):
        """Test that negative values for non-trial parameters work correctly."""
        # The algorithm should handle negative values for all parameters except trial
        params = HDRPRNGParameters(
            trial=1, variable=-100, entity=-200, time=-300, agent=-400
        )
        result = hdrprng(params)

        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_uniform_distribution_approximation(self):
        """Test that outputs are approximately uniformly distributed."""

        def hdrgen(variable: int):
            params = HDRPRNGParameters(
                trial=1,
                variable=variable,
                entity=0,
                time=0,
                agent=0,
            )
            return hdrprng(params)

        # Generate many random values
        num_samples = 10000
        vectorized_hdrgen = jax.vmap(hdrgen)
        arr = jnp.arange(num_samples, dtype=int)
        results_array = vectorized_hdrgen(arr)

        # Test that mean is approximately 0.5 (uniform distribution on (0,1))
        mean = np.mean(results_array)
        self.assertAlmostEqual(mean, 0.5, delta=0.02)

        # Test that samples span the full range
        min_val = np.min(results_array)
        max_val = np.max(results_array)
        self.assertLess(min_val, 0.1, "Minimum should be close to 0")
        self.assertGreater(max_val, 0.9, "Maximum should be close to 1")

        # Test distribution across bins
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(results_array, bins=bins)

        # Each bin should have roughly 10% of samples (1000 samples per bin)
        expected_per_bin = num_samples / 10
        for count in hist:
            # Allow 20% deviation from expected
            self.assertGreater(
                count,
                expected_per_bin * 0.8,
                f"Bin has too few samples: {count}",
            )
            self.assertLess(
                count,
                expected_per_bin * 1.2,
                f"Bin has too many samples: {count}",
            )


class KSDistanceTest(parameterized.TestCase):
    """Test suite for the ks_distance function.

    Tests the two-sample Kolmogorov-Smirnov distance calculation to ensure
    accurate measurement of distributional differences between two samples.
    """

    def test_identical_samples(self):
        """Test that identical samples have zero KS distance."""
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        distance = ks_distance(x, y)
        self.assertAlmostEqual(float(distance), 0.0, places=7)

    def test_identical_samples_unsorted(self):
        """Test that identical but unsorted samples have zero KS distance."""
        x = jnp.array([0.5, 0.1, 0.3, 0.2, 0.4])
        y = jnp.array([0.3, 0.5, 0.1, 0.4, 0.2])
        distance = ks_distance(x, y)
        self.assertAlmostEqual(float(distance), 0.0, places=7)

    def test_completely_separate_samples(self):
        """Test that completely non-overlapping samples have maximum distance."""
        x = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
        y = jnp.array([0.6, 0.7, 0.8, 0.9, 1.0])
        distance = ks_distance(x, y)
        # KS distance should be 1.0 for completely separate samples
        self.assertAlmostEqual(float(distance), 1.0, places=10)

    def test_different_sample_sizes(self):
        """Test KS distance calculation with different sample sizes."""
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = jnp.array([0.15, 0.25, 0.35])
        distance = ks_distance(x, y)
        # Distance should be finite and in valid range
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    @parameterized.named_parameters(
        {
            "testcase_name": "small_shift",
            "x": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "y": jnp.array([0.15, 0.25, 0.35, 0.45, 0.55]),
        },
        {
            "testcase_name": "large_shift",
            "x": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "y": jnp.array([0.4, 0.5, 0.6, 0.7, 0.8]),
        },
        {
            "testcase_name": "partial_overlap",
            "x": jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "y": jnp.array([0.3, 0.4, 0.5, 0.6, 0.7]),
        },
    )
    def test_shifted_distributions(self, x, y):
        """Test KS distance for shifted but similar distributions.

        Args:
            x: First sample array.
            y: Second sample array (shifted from x).
        """
        distance = ks_distance(x, y)
        # Shifted distributions should have positive distance
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_symmetry(self):
        """Test that KS distance is symmetric: D(x, y) = D(y, x)."""
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = jnp.array([0.15, 0.25, 0.35, 0.45, 0.55])
        distance_xy = ks_distance(x, y)
        distance_yx = ks_distance(y, x)
        self.assertAlmostEqual(float(distance_xy), float(distance_yx), places=7)

    def test_single_element_samples(self):
        """Test KS distance with single-element samples."""
        x = jnp.array([0.3])
        y = jnp.array([0.7])
        distance = ks_distance(x, y)
        # Single element samples at different points should have distance 1.0
        self.assertAlmostEqual(float(distance), 1.0, places=10)

    def test_single_element_identical(self):
        """Test KS distance with identical single-element samples."""
        x = jnp.array([0.5])
        y = jnp.array([0.5])
        distance = ks_distance(x, y)
        self.assertAlmostEqual(float(distance), 0.0, places=10)

    def test_two_element_samples(self):
        """Test KS distance with small two-element samples."""
        x = jnp.array([0.2, 0.4])
        y = jnp.array([0.6, 0.8])
        distance = ks_distance(x, y)
        # Non-overlapping samples should have distance 1.0
        self.assertAlmostEqual(float(distance), 1.0, places=10)

    @parameterized.named_parameters(
        {
            "testcase_name": "uniform_vs_skewed_left",
            "x": jnp.linspace(0.01, 0.99, 100),
            "y": jnp.concatenate(
                [jnp.linspace(0.01, 0.3, 80), jnp.linspace(0.3, 0.99, 20)]
            ),
        },
        {
            "testcase_name": "uniform_vs_skewed_right",
            "x": jnp.linspace(0.01, 0.99, 100),
            "y": jnp.concatenate(
                [jnp.linspace(0.01, 0.7, 20), jnp.linspace(0.7, 0.99, 80)]
            ),
        },
    )
    def test_different_distribution_shapes(self, x, y):
        """Test KS distance for distributions with different shapes.

        Args:
            x: Sample from first distribution.
            y: Sample from second distribution (different shape).
        """
        distance = ks_distance(x, y)
        # Different distribution shapes should have positive distance
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_large_samples(self):
        """Test KS distance calculation with large sample sizes."""
        np.random.seed(42)
        # Generate large samples from similar distributions
        x = jnp.array(np.random.uniform(0.0, 1.0, 1000))
        y = jnp.array(np.random.uniform(0.1, 0.9, 1000))
        distance = ks_distance(x, y)
        # Should compute without error and return valid distance
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_jit_compilation(self):
        """Test that ks_distance is JIT-compiled and works correctly."""
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = jnp.array([0.15, 0.25, 0.35, 0.45, 0.55])

        # First call (compilation)
        distance1 = ks_distance(x, y)

        # Second call (should use compiled version)
        distance2 = ks_distance(x, y)

        # Results should be identical
        self.assertEqual(float(distance1), float(distance2))

    def test_output_range(self):
        """Test that KS distance is always in the range [0, 1]."""
        test_cases = [
            (jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2, 0.3])),  # Identical
            (jnp.array([0.1, 0.2, 0.3]), jnp.array([0.7, 0.8, 0.9])),  # Separate
            (jnp.array([0.1, 0.5, 0.9]), jnp.array([0.2, 0.5, 0.8])),  # Overlapping
            (jnp.linspace(0.0, 1.0, 50), jnp.linspace(0.1, 0.9, 50)),  # Shifted
        ]

        for x, y in test_cases:
            distance = ks_distance(x, y)
            self.assertGreaterEqual(
                float(distance), 0.0, f"Distance {distance} is less than 0"
            )
            self.assertLessEqual(
                float(distance), 1.0, f"Distance {distance} is greater than 1"
            )

    def test_deterministic_output(self):
        """Test that KS distance produces deterministic outputs."""
        x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y = jnp.array([0.2, 0.4, 0.6, 0.8])

        # Call multiple times
        distances = [ks_distance(x, y) for _ in range(5)]

        # All results should be identical
        for d in distances[1:]:
            self.assertEqual(float(distances[0]), float(d))

    def test_uniform_samples(self):
        """Test KS distance between samples from uniform distributions."""
        np.random.seed(123)
        # Two samples from the same uniform(0, 1) distribution
        x = jnp.array(np.random.uniform(0.0, 1.0, 200))
        y = jnp.array(np.random.uniform(0.0, 1.0, 200))
        distance = ks_distance(x, y)

        # Should have small distance (same distribution)
        # Typically < 0.15 for n=200 samples from same distribution
        self.assertLess(float(distance), 0.2)

    def test_normal_approximation_samples(self):
        """Test KS distance for samples approximating normal distributions."""
        np.random.seed(456)
        # Sample from normal(0, 1) truncated to (0, 1)
        x_raw = np.random.randn(500)
        x = jnp.array(x_raw[(x_raw > 0) & (x_raw < 1)][:200])

        # Sample from normal(0.5, 0.3) truncated to (0, 1)
        y_raw = np.random.randn(500) * 0.3 + 0.5
        y = jnp.array(y_raw[(y_raw > 0) & (y_raw < 1)][:200])

        distance = ks_distance(x, y)
        # Different normal distributions should have measurable distance
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_duplicate_values(self):
        """Test KS distance with duplicate values in samples."""
        x = jnp.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
        y = jnp.array([0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
        distance = ks_distance(x, y)

        # Should compute valid distance despite duplicates
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_all_same_value(self):
        """Test KS distance when all values in a sample are identical."""
        x = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        y = jnp.array([0.7, 0.7, 0.7, 0.7, 0.7])
        distance = ks_distance(x, y)

        # Two different point masses should have distance 1.0
        self.assertAlmostEqual(float(distance), 1.0, places=10)

    def test_mixed_precision_float32(self):
        """Test KS distance with float32 arrays."""
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=jnp.float32)
        y = jnp.array([0.15, 0.25, 0.35, 0.45, 0.55], dtype=jnp.float32)
        distance = ks_distance(x, y)

        # Should work with float32 precision
        self.assertIsInstance(distance, (float, jnp.ndarray))
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_mixed_precision_float64(self):
        """Test KS distance with float64 arrays."""
        x = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=jnp.float64)
        y = jnp.array([0.15, 0.25, 0.35, 0.45, 0.55], dtype=jnp.float64)
        distance = ks_distance(x, y)

        # Should work with float64 precision
        self.assertIsInstance(distance, (float, jnp.ndarray))
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_very_small_distance(self):
        """Test KS distance for nearly identical distributions."""
        # Create two very similar samples
        x = jnp.linspace(0.01, 0.99, 1000)
        y = x + 1e-6  # Tiny perturbation
        distance = ks_distance(x, y)

        # Distance should be very small but positive
        self.assertGreater(float(distance), 0.0)
        self.assertLess(float(distance), 0.01)

    @parameterized.named_parameters(
        {
            "testcase_name": "size_10_vs_100",
            "n1": 10,
            "n2": 100,
        },
        {
            "testcase_name": "size_50_vs_500",
            "n1": 50,
            "n2": 500,
        },
        {
            "testcase_name": "size_1_vs_1000",
            "n1": 1,
            "n2": 1000,
        },
    )
    def test_asymmetric_sample_sizes(self, n1, n2):
        """Test KS distance with very different sample sizes.

        Args:
            n1: Size of first sample.
            n2: Size of second sample.
        """
        np.random.seed(789)
        x = jnp.array(np.random.uniform(0.0, 1.0, n1))
        y = jnp.array(np.random.uniform(0.0, 1.0, n2))
        distance = ks_distance(x, y)

        # Should handle asymmetric sizes correctly
        self.assertGreaterEqual(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_comparison_with_scipy_reference(self):
        """Test that our KS distance matches scipy.stats.ks_2samp statistic."""
        from scipy.stats import ks_2samp

        np.random.seed(42)
        x = np.random.uniform(0.0, 1.0, 100)
        y = np.random.uniform(0.1, 0.9, 100)

        # Our implementation
        jax_distance = float(ks_distance(jnp.array(x), jnp.array(y)))

        # SciPy reference
        scipy_result = ks_2samp(x, y)
        scipy_distance = scipy_result.statistic

        # Should match within numerical precision
        self.assertAlmostEqual(
            jnp.round(jax_distance, 2),
            scipy_distance,
            places=11,
            msg=f"JAX: {jax_distance}, SciPy: {scipy_distance}",
        )

    def test_edge_case_minimal_separation(self):
        """Test KS distance with minimally separated samples."""
        x = jnp.array([0.499, 0.4999, 0.49999])
        y = jnp.array([0.5, 0.50001, 0.5001])
        distance = ks_distance(x, y)

        # Even minimal separation should be detected
        self.assertGreater(float(distance), 0.0)
        self.assertLessEqual(float(distance), 1.0)

    def test_monotonicity_with_increasing_shift(self):
        """Test that KS distance increases as distribution shift increases."""
        base = jnp.linspace(0.2, 0.8, 50)

        # Compute distances with increasing shifts
        distances = []
        for shift in [0.0, 0.05, 0.1, 0.15, 0.2]:
            shifted = base + shift
            # Clip to valid range
            shifted = jnp.clip(shifted, 0.0, 1.0)
            d = float(ks_distance(base, shifted))
            distances.append(d)

        # Distances should generally increase with shift (with shift > 0)
        for i in range(1, len(distances)):
            if distances[i - 1] < 1.0:  # Not already at maximum
                self.assertGreaterEqual(
                    distances[i],
                    distances[i - 1] - 1e-10,
                    f"Distance decreased: {distances}",
                )


if __name__ == "__main__":
    absltest.main()
