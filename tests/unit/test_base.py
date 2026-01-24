"""Unit tests for the Metalog JAX base module.

This module contains comprehensive tests for base classes, enumerations, and
dataclasses in the metalog_jax.base module, using parameterized testing to validate
behavior across various configurations and edge cases.
"""
# Copyright: Travis Jefferies 2026

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
)
from metalog_jax.metalog import Metalog
from metalog_jax.utils import DEFAULT_Y


class MetalogBoundednessTest(parameterized.TestCase):
    """Test suite for the MetalogBoundedness enumeration.

    Tests the enumeration values and properties defined in metalog_jax.base
    to ensure correct representation of metalog distribution boundedness types.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "expected_value": 1,
        },
        {
            "testcase_name": "strictly_lower_bound",
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "expected_value": 2,
        },
        {
            "testcase_name": "strictly_upper_bound",
            "boundedness": MetalogBoundedness.STRICTLY_UPPER_BOUND,
            "expected_value": 3,
        },
        {
            "testcase_name": "bounded",
            "boundedness": MetalogBoundedness.BOUNDED,
            "expected_value": 4,
        },
    )
    def test_enum_values(self, boundedness, expected_value):
        """Test that enum members have correct integer values.

        Args:
            boundedness: MetalogBoundedness enum member to test.
            expected_value: Expected integer value of the enum member.
        """
        self.assertEqual(boundedness.value, expected_value)

    def test_enum_member_count(self):
        """Test that the enum has exactly four members."""
        self.assertEqual(len(MetalogBoundedness), 4)

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_from_value",
            "value": 1,
            "expected": MetalogBoundedness.UNBOUNDED,
        },
        {
            "testcase_name": "strictly_lower_from_value",
            "value": 2,
            "expected": MetalogBoundedness.STRICTLY_LOWER_BOUND,
        },
        {
            "testcase_name": "strictly_upper_from_value",
            "value": 3,
            "expected": MetalogBoundedness.STRICTLY_UPPER_BOUND,
        },
        {
            "testcase_name": "bounded_from_value",
            "value": 4,
            "expected": MetalogBoundedness.BOUNDED,
        },
    )
    def test_enum_from_value(self, value, expected):
        """Test that enum members can be constructed from their integer values.

        Args:
            value: Integer value to construct enum from.
            expected: Expected enum member.
        """
        self.assertEqual(MetalogBoundedness(value), expected)

    def test_int_enum_type(self):
        """Test that enum members are instances of int."""
        self.assertIsInstance(MetalogBoundedness.UNBOUNDED, int)
        self.assertIsInstance(MetalogBoundedness.STRICTLY_LOWER_BOUND, int)
        self.assertIsInstance(MetalogBoundedness.STRICTLY_UPPER_BOUND, int)
        self.assertIsInstance(MetalogBoundedness.BOUNDED, int)


class MetalogParametersTest(parameterized.TestCase):
    """Test suite for the MetalogParameters dataclass.

    Tests the creation and validation of metalog distribution configuration
    parameters defined in metalog_jax.base.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_params",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 3,
        },
        {
            "testcase_name": "strictly_lower_bound_params",
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 5,
        },
        {
            "testcase_name": "strictly_upper_bound_params",
            "boundedness": MetalogBoundedness.STRICTLY_UPPER_BOUND,
            "lower_bound": 0.0,
            "upper_bound": 10.0,
            "num_terms": 7,
        },
        {
            "testcase_name": "bounded_params",
            "boundedness": MetalogBoundedness.BOUNDED,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
            "num_terms": 9,
        },
        {
            "testcase_name": "large_num_terms",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 20,
        },
    )
    def test_metalog_params_creation(
        self, boundedness, lower_bound, upper_bound, num_terms
    ):
        """Test that MetalogParameters can be created with valid values.

        Args:
            boundedness: Type of boundedness for the distribution.
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
            num_terms: Number of metalog terms.
        """
        params = MetalogParameters(
            boundedness=boundedness,
            method=MetalogFitMethod.OLS,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_terms=num_terms,
        )
        self.assertEqual(params.boundedness, boundedness)
        self.assertEqual(params.method, MetalogFitMethod.OLS)
        self.assertEqual(params.lower_bound, lower_bound)
        self.assertEqual(params.upper_bound, upper_bound)
        self.assertEqual(params.num_terms, num_terms)

    def test_metalog_params_immutability(self):
        """Test that MetalogParameters instances are immutable (frozen dataclass)."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=3,
        )
        with self.assertRaises((AttributeError, TypeError)):
            params.num_terms = 5


class MetalogBaseSaveTest(parameterized.TestCase):
    """Test suite for the MetalogBase.save method.

    Tests the save functionality defined in metalog_jax.base.MetalogBase to ensure
    metalog distributions can be properly serialized to JSON files.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_3_terms",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 3,
            "coefficients": [1.0, 2.0, 3.0],
        },
        {
            "testcase_name": "strictly_lower_5_terms",
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 5.0,
            "upper_bound": 0.0,
            "num_terms": 5,
            "coefficients": [0.5, 1.5, 2.5, 3.5, 4.5],
        },
        {
            "testcase_name": "strictly_upper_4_terms",
            "boundedness": MetalogBoundedness.STRICTLY_UPPER_BOUND,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "num_terms": 4,
            "coefficients": [10.0, 20.0, 30.0, 40.0],
        },
        {
            "testcase_name": "bounded_7_terms",
            "boundedness": MetalogBoundedness.BOUNDED,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": -10.0,
            "upper_bound": 10.0,
            "num_terms": 7,
            "coefficients": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        },
    )
    def test_save_creates_file(
        self, boundedness, method, lower_bound, upper_bound, num_terms, coefficients
    ):
        """Test that MetalogBase.save creates a JSON file with correct content.

        Args:
            boundedness: Type of boundedness for the distribution.
            method: Fitting method used.
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
            num_terms: Number of metalog terms.
            coefficients: Coefficient vector for the distribution.
        """
        import json
        import tempfile
        from pathlib import Path

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create MetalogParameters
            params = MetalogParameters(
                boundedness=boundedness,
                method=method,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                num_terms=num_terms,
            )

            # Create Metalog with test coefficients
            results = Metalog(
                metalog_params=params,
                a=jnp.array(coefficients),
            )

            # Save to temporary file
            save_path = Path(tmpdir) / "test_metalog.json"
            results.save(save_path)

            # Verify file exists
            self.assertTrue(save_path.exists())

            # Load and verify content
            with open(save_path, "r") as f:
                loaded_data = json.load(f)

            # Check that JSON has expected structure
            self.assertIn("a", loaded_data)
            self.assertIn("metalog_params", loaded_data)

            # Verify parameters are saved correctly
            self.assertEqual(loaded_data["metalog_params"]["boundedness"], boundedness)
            self.assertEqual(loaded_data["metalog_params"]["method"], method)
            self.assertEqual(loaded_data["metalog_params"]["lower_bound"], lower_bound)
            self.assertEqual(loaded_data["metalog_params"]["upper_bound"], upper_bound)
            self.assertEqual(loaded_data["metalog_params"]["num_terms"], num_terms)

            # Verify coefficients are saved correctly
            np.testing.assert_array_almost_equal(loaded_data["a"], coefficients)

    def test_save_overwrites_existing_file(self):
        """Test that MetalogBase.save overwrites an existing file."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            # Create first Metalog and save
            params1 = MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=3,
            )
            results1 = Metalog(
                metalog_params=params1,
                a=jnp.array([1.0, 2.0, 3.0]),
            )
            results1.save(save_path)

            # Create second Metalog with different values and save to same path
            params2 = MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.Lasso,
                lower_bound=-5.0,
                upper_bound=5.0,
                num_terms=5,
            )
            results2 = Metalog(
                metalog_params=params2,
                a=jnp.array([5.0, 6.0, 7.0, 8.0, 9.0]),
            )
            results2.save(save_path)

            # Load and verify the file contains the second set of values
            with open(save_path, "r") as f:
                loaded_data = json.load(f)

            self.assertEqual(
                loaded_data["metalog_params"]["boundedness"], MetalogBoundedness.BOUNDED
            )
            self.assertEqual(
                loaded_data["metalog_params"]["method"], MetalogFitMethod.Lasso
            )
            np.testing.assert_array_almost_equal(
                loaded_data["a"], [5.0, 6.0, 7.0, 8.0, 9.0]
            )

    def test_save_with_negative_coefficients(self):
        """Test that MetalogBase.save handles negative coefficient values correctly."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            params = MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=4,
            )
            results = Metalog(
                metalog_params=params,
                a=jnp.array([-10.5, -5.3, 0.0, 7.8]),
            )
            results.save(save_path)

            with open(save_path, "r") as f:
                loaded_data = json.load(f)

            np.testing.assert_array_almost_equal(
                loaded_data["a"], [-10.5, -5.3, 0.0, 7.8]
            )

    def test_save_with_large_coefficient_array(self):
        """Test that MetalogBase.save handles larger coefficient arrays correctly."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            # Create a distribution with many terms
            num_terms = 20
            coefficients = np.random.randn(num_terms)

            params = MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=num_terms,
            )
            results = Metalog(
                metalog_params=params,
                a=jnp.array(coefficients),
            )
            results.save(save_path)

            with open(save_path, "r") as f:
                loaded_data = json.load(f)

            self.assertEqual(len(loaded_data["a"]), num_terms)
            np.testing.assert_array_almost_equal(loaded_data["a"], coefficients)


class MetalogBaseLoadTest(parameterized.TestCase):
    """Test suite for the MetalogBase.load method.

    Tests the load functionality defined in metalog_jax.base.MetalogBase to ensure
    metalog distributions can be properly deserialized from JSON files created by
    the save method.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_3_terms",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 3,
            "coefficients": [1.0, 2.0, 3.0],
        },
        {
            "testcase_name": "strictly_lower_5_terms",
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 5.0,
            "upper_bound": 0.0,
            "num_terms": 5,
            "coefficients": [0.5, 1.5, 2.5, 3.5, 4.5],
        },
        {
            "testcase_name": "strictly_upper_4_terms",
            "boundedness": MetalogBoundedness.STRICTLY_UPPER_BOUND,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "num_terms": 4,
            "coefficients": [10.0, 20.0, 30.0, 40.0],
        },
        {
            "testcase_name": "bounded_7_terms",
            "boundedness": MetalogBoundedness.BOUNDED,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": -10.0,
            "upper_bound": 10.0,
            "num_terms": 7,
            "coefficients": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        },
    )
    def test_load_reconstructs_metalog_distribution(
        self, boundedness, method, lower_bound, upper_bound, num_terms, coefficients
    ):
        """Test that MetalogBase.load correctly reconstructs a Metalog instance from JSON.

        Args:
            boundedness: Type of boundedness for the distribution.
            method: Fitting method used.
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
            num_terms: Number of metalog terms.
            coefficients: Coefficient vector for the distribution.
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save original Metalog
            original_params = MetalogParameters(
                boundedness=boundedness,
                method=method,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                num_terms=num_terms,
            )
            original_results = Metalog(
                metalog_params=original_params,
                a=jnp.array(coefficients),
            )
            save_path = Path(tmpdir) / "test_metalog.json"
            original_results.save(save_path)

            # Load the Metalog
            loaded_results = Metalog.load(save_path)

            # Verify all attributes match
            self.assertEqual(loaded_results.boundedness, boundedness)
            self.assertEqual(loaded_results.metalog_params.method, method)
            self.assertEqual(loaded_results.lower_bound, lower_bound)
            self.assertEqual(loaded_results.upper_bound, upper_bound)
            self.assertEqual(loaded_results.num_terms, num_terms)
            np.testing.assert_array_almost_equal(loaded_results.a, coefficients)

    def test_load_enum_deserialization(self):
        """Test that MetalogBase.load correctly deserializes IntEnum values."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            # Create and save with specific enum values
            params = MetalogParameters(
                boundedness=MetalogBoundedness.BOUNDED,
                method=MetalogFitMethod.Lasso,
                lower_bound=-5.0,
                upper_bound=5.0,
                num_terms=3,
            )
            results = Metalog(
                metalog_params=params,
                a=jnp.array([1.0, 2.0, 3.0]),
            )
            results.save(save_path)

            # Load and verify enum types
            loaded_results = Metalog.load(save_path)

            # Check that loaded values are proper IntEnum instances
            self.assertIsInstance(loaded_results.boundedness, MetalogBoundedness)
            self.assertIsInstance(
                loaded_results.metalog_params.method, MetalogFitMethod
            )
            self.assertEqual(loaded_results.boundedness, MetalogBoundedness.BOUNDED)
            self.assertEqual(
                loaded_results.metalog_params.method, MetalogFitMethod.Lasso
            )

    def test_load_jax_array_reconstruction(self):
        """Test that MetalogBase.load correctly converts list back to JAX array."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            # Create and save with specific coefficients
            coefficients = jnp.array([1.5, 2.5, 3.5, 4.5])
            params = MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=4,
            )
            results = Metalog(metalog_params=params, a=coefficients)
            results.save(save_path)

            # Load and verify array type and values
            loaded_results = Metalog.load(save_path)

            # Verify it's a JAX array
            self.assertIsInstance(loaded_results.a, jnp.ndarray)
            # Verify values match
            np.testing.assert_array_almost_equal(loaded_results.a, coefficients)
            # Verify shape matches
            self.assertEqual(loaded_results.a.shape, coefficients.shape)

    def test_load_with_negative_coefficients(self):
        """Test that MetalogBase.load handles negative coefficient values correctly."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            coefficients = jnp.array([-10.5, -5.3, 0.0, 7.8])
            params = MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=4,
            )
            results = Metalog(metalog_params=params, a=coefficients)
            results.save(save_path)

            loaded_results = Metalog.load(save_path)

            np.testing.assert_array_almost_equal(loaded_results.a, coefficients)

    def test_load_with_large_coefficient_array(self):
        """Test that MetalogBase.load handles larger coefficient arrays correctly."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_metalog.json"

            num_terms = 20
            coefficients = np.random.randn(num_terms)

            params = MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=num_terms,
            )
            results = Metalog(metalog_params=params, a=jnp.array(coefficients))
            results.save(save_path)

            loaded_results = Metalog.load(save_path)

            self.assertEqual(len(loaded_results.a), num_terms)
            np.testing.assert_array_almost_equal(loaded_results.a, coefficients)

    def test_load_nonexistent_file(self):
        """Test that MetalogBase.load raises FileNotFoundError for nonexistent files."""
        from pathlib import Path

        nonexistent_path = Path("/tmp/nonexistent_metalog_file_12345.json")

        with self.assertRaises(FileNotFoundError):
            Metalog.load(nonexistent_path)

    def test_load_invalid_json(self):
        """Test that MetalogBase.load raises JSONDecodeError for invalid JSON files."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "invalid.json"

            # Write invalid JSON to file
            with open(save_path, "w") as f:
                f.write("{invalid json content")

            with self.assertRaises(json.JSONDecodeError):
                Metalog.load(save_path)

    def test_load_missing_required_fields(self):
        """Test that MetalogBase.load raises KeyError when JSON is missing required fields."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "incomplete.json"

            # Write JSON missing required fields
            incomplete_data = {
                "a": [1.0, 2.0, 3.0],
                # Missing "metalog_params" field
            }
            with open(save_path, "w") as f:
                json.dump(incomplete_data, f)

            with self.assertRaises(KeyError):
                Metalog.load(save_path)

    def test_load_invalid_enum_value(self):
        """Test that MetalogBase.load raises ValueError for invalid enum values."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "invalid_enum.json"

            # Write JSON with invalid enum value
            invalid_data = {
                "a": [1.0, 2.0, 3.0],
                "metalog_params": {
                    "boundedness": 999,  # Invalid enum value
                    "method": 1,
                    "lower_bound": 0.0,
                    "upper_bound": 0.0,
                    "num_terms": 3,
                },
            }
            with open(save_path, "w") as f:
                json.dump(invalid_data, f)

            with self.assertRaises(ValueError):
                Metalog.load(save_path)

    def test_save_load_roundtrip(self):
        """Test that save followed by load preserves all data (roundtrip test)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "roundtrip.json"

            # Create original Metalog with varied data
            original_params = MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.Lasso,
                lower_bound=10.0,
                upper_bound=0.0,
                num_terms=6,
            )
            original_coefficients = jnp.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            original_results = Metalog(
                metalog_params=original_params,
                a=original_coefficients,
            )

            # Save and load
            original_results.save(save_path)
            loaded_results = Metalog.load(save_path)

            # Verify complete equality using __eq__ method
            self.assertEqual(loaded_results, original_results)

            # Verify types are correct
            self.assertIsInstance(loaded_results.boundedness, MetalogBoundedness)
            self.assertIsInstance(
                loaded_results.metalog_params.method, MetalogFitMethod
            )
            self.assertIsInstance(loaded_results.a, jnp.ndarray)


class MetalogBaseDumpsTest(parameterized.TestCase):
    """Test suite for the MetalogBase.dumps method.

    Tests the dumps functionality defined in metalog_jax.base.MetalogBase to ensure
    metalog distributions can be properly serialized to JSON strings.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_3_terms",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 3,
            "coefficients": [1.0, 2.0, 3.0],
        },
        {
            "testcase_name": "strictly_lower_5_terms",
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 5.0,
            "upper_bound": 0.0,
            "num_terms": 5,
            "coefficients": [0.5, 1.5, 2.5, 3.5, 4.5],
        },
        {
            "testcase_name": "strictly_upper_4_terms",
            "boundedness": MetalogBoundedness.STRICTLY_UPPER_BOUND,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "num_terms": 4,
            "coefficients": [10.0, 20.0, 30.0, 40.0],
        },
        {
            "testcase_name": "bounded_7_terms",
            "boundedness": MetalogBoundedness.BOUNDED,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": -10.0,
            "upper_bound": 10.0,
            "num_terms": 7,
            "coefficients": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        },
    )
    def test_dumps_returns_valid_json_string(
        self, boundedness, method, lower_bound, upper_bound, num_terms, coefficients
    ):
        """Test that MetalogBase.dumps returns a valid JSON string with correct content.

        Args:
            boundedness: Type of boundedness for the distribution.
            method: Fitting method used.
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
            num_terms: Number of metalog terms.
            coefficients: Coefficient vector for the distribution.
        """
        import json

        # Create MetalogParameters
        params = MetalogParameters(
            boundedness=boundedness,
            method=method,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_terms=num_terms,
        )

        # Create Metalog with test coefficients
        results = Metalog(
            metalog_params=params,
            a=jnp.array(coefficients),
        )

        # Get JSON string
        json_str = results.dumps()

        # Verify it's a valid JSON string
        self.assertIsInstance(json_str, str)
        loaded_data = json.loads(json_str)

        # Check that JSON has expected structure
        self.assertIn("a", loaded_data)
        self.assertIn("metalog_params", loaded_data)

        # Verify parameters are serialized correctly
        self.assertEqual(loaded_data["metalog_params"]["boundedness"], boundedness)
        self.assertEqual(loaded_data["metalog_params"]["method"], method)
        self.assertEqual(loaded_data["metalog_params"]["lower_bound"], lower_bound)
        self.assertEqual(loaded_data["metalog_params"]["upper_bound"], upper_bound)
        self.assertEqual(loaded_data["metalog_params"]["num_terms"], num_terms)

        # Verify coefficients are serialized correctly
        np.testing.assert_array_almost_equal(loaded_data["a"], coefficients)

    def test_dumps_with_negative_coefficients(self):
        """Test that MetalogBase.dumps handles negative coefficient values correctly."""
        import json

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=4,
        )
        results = Metalog(
            metalog_params=params,
            a=jnp.array([-10.5, -5.3, 0.0, 7.8]),
        )

        json_str = results.dumps()
        loaded_data = json.loads(json_str)

        np.testing.assert_array_almost_equal(loaded_data["a"], [-10.5, -5.3, 0.0, 7.8])

    def test_dumps_with_large_coefficient_array(self):
        """Test that MetalogBase.dumps handles larger coefficient arrays correctly."""
        import json

        num_terms = 20
        coefficients = np.random.randn(num_terms)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=num_terms,
        )
        results = Metalog(
            metalog_params=params,
            a=jnp.array(coefficients),
        )

        json_str = results.dumps()
        loaded_data = json.loads(json_str)

        self.assertEqual(len(loaded_data["a"]), num_terms)
        np.testing.assert_array_almost_equal(loaded_data["a"], coefficients)


class MetalogBaseLoadsTest(parameterized.TestCase):
    """Test suite for the MetalogBase.loads method.

    Tests the loads functionality defined in metalog_jax.base.MetalogBase to ensure
    metalog distributions can be properly deserialized from JSON strings created by
    the dumps method.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "unbounded_3_terms",
            "boundedness": MetalogBoundedness.UNBOUNDED,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "num_terms": 3,
            "coefficients": [1.0, 2.0, 3.0],
        },
        {
            "testcase_name": "strictly_lower_5_terms",
            "boundedness": MetalogBoundedness.STRICTLY_LOWER_BOUND,
            "method": MetalogFitMethod.OLS,
            "lower_bound": 5.0,
            "upper_bound": 0.0,
            "num_terms": 5,
            "coefficients": [0.5, 1.5, 2.5, 3.5, 4.5],
        },
        {
            "testcase_name": "strictly_upper_4_terms",
            "boundedness": MetalogBoundedness.STRICTLY_UPPER_BOUND,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "num_terms": 4,
            "coefficients": [10.0, 20.0, 30.0, 40.0],
        },
        {
            "testcase_name": "bounded_7_terms",
            "boundedness": MetalogBoundedness.BOUNDED,
            "method": MetalogFitMethod.Lasso,
            "lower_bound": -10.0,
            "upper_bound": 10.0,
            "num_terms": 7,
            "coefficients": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        },
    )
    def test_loads_reconstructs_metalog_distribution(
        self, boundedness, method, lower_bound, upper_bound, num_terms, coefficients
    ):
        """Test that MetalogBase.loads correctly reconstructs a Metalog instance from JSON string.

        Args:
            boundedness: Type of boundedness for the distribution.
            method: Fitting method used.
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
            num_terms: Number of metalog terms.
            coefficients: Coefficient vector for the distribution.
        """
        # Create and serialize original Metalog
        original_params = MetalogParameters(
            boundedness=boundedness,
            method=method,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_terms=num_terms,
        )
        original_results = Metalog(
            metalog_params=original_params,
            a=jnp.array(coefficients),
        )
        json_str = original_results.dumps()

        # Load the Metalog from JSON string
        loaded_results = Metalog.loads(json_str)

        # Verify all attributes match
        self.assertEqual(loaded_results.boundedness, boundedness)
        self.assertEqual(loaded_results.metalog_params.method, method)
        self.assertEqual(loaded_results.lower_bound, lower_bound)
        self.assertEqual(loaded_results.upper_bound, upper_bound)
        self.assertEqual(loaded_results.num_terms, num_terms)
        np.testing.assert_array_almost_equal(loaded_results.a, coefficients)

    def test_loads_enum_deserialization(self):
        """Test that MetalogBase.loads correctly deserializes IntEnum values."""
        # Create and serialize with specific enum values
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            method=MetalogFitMethod.Lasso,
            lower_bound=-5.0,
            upper_bound=5.0,
            num_terms=3,
        )
        results = Metalog(
            metalog_params=params,
            a=jnp.array([1.0, 2.0, 3.0]),
        )
        json_str = results.dumps()

        # Load and verify enum types
        loaded_results = Metalog.loads(json_str)

        # Check that loaded values are proper IntEnum instances
        self.assertIsInstance(loaded_results.boundedness, MetalogBoundedness)
        self.assertIsInstance(loaded_results.metalog_params.method, MetalogFitMethod)
        self.assertEqual(loaded_results.boundedness, MetalogBoundedness.BOUNDED)
        self.assertEqual(loaded_results.metalog_params.method, MetalogFitMethod.Lasso)

    def test_loads_jax_array_reconstruction(self):
        """Test that MetalogBase.loads correctly converts list back to JAX array."""
        # Create and serialize with specific coefficients
        coefficients = jnp.array([1.5, 2.5, 3.5, 4.5])
        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=4,
        )
        results = Metalog(metalog_params=params, a=coefficients)
        json_str = results.dumps()

        # Load and verify array type and values
        loaded_results = Metalog.loads(json_str)

        # Verify it's a JAX array
        self.assertIsInstance(loaded_results.a, jnp.ndarray)
        # Verify values match
        np.testing.assert_array_almost_equal(loaded_results.a, coefficients)
        # Verify shape matches
        self.assertEqual(loaded_results.a.shape, coefficients.shape)

    def test_loads_with_negative_coefficients(self):
        """Test that MetalogBase.loads handles negative coefficient values correctly."""
        coefficients = jnp.array([-10.5, -5.3, 0.0, 7.8])
        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=4,
        )
        results = Metalog(metalog_params=params, a=coefficients)
        json_str = results.dumps()

        loaded_results = Metalog.loads(json_str)

        np.testing.assert_array_almost_equal(loaded_results.a, coefficients)

    def test_loads_with_large_coefficient_array(self):
        """Test that MetalogBase.loads handles larger coefficient arrays correctly."""
        num_terms = 20
        coefficients = np.random.randn(num_terms)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=num_terms,
        )
        results = Metalog(metalog_params=params, a=jnp.array(coefficients))
        json_str = results.dumps()

        loaded_results = Metalog.loads(json_str)

        self.assertEqual(len(loaded_results.a), num_terms)
        np.testing.assert_array_almost_equal(loaded_results.a, coefficients)

    def test_loads_invalid_json(self):
        """Test that MetalogBase.loads raises JSONDecodeError for invalid JSON strings."""
        import json

        invalid_json = "{invalid json content"

        with self.assertRaises(json.JSONDecodeError):
            Metalog.loads(invalid_json)

    def test_loads_missing_required_fields(self):
        """Test that MetalogBase.loads raises KeyError when JSON is missing required fields."""
        import json

        # JSON missing required fields
        incomplete_data = {
            "a": [1.0, 2.0, 3.0],
            # Missing "metalog_params" field
        }
        json_str = json.dumps(incomplete_data)

        with self.assertRaises(KeyError):
            Metalog.loads(json_str)

    def test_loads_invalid_enum_value(self):
        """Test that MetalogBase.loads raises ValueError for invalid enum values."""
        import json

        # JSON with invalid enum value
        invalid_data = {
            "a": [1.0, 2.0, 3.0],
            "metalog_params": {
                "boundedness": 999,  # Invalid enum value
                "method": 1,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "num_terms": 3,
            },
        }
        json_str = json.dumps(invalid_data)

        with self.assertRaises(ValueError):
            Metalog.loads(json_str)

    def test_dumps_loads_roundtrip(self):
        """Test that dumps followed by loads preserves all data (roundtrip test)."""
        # Create original Metalog with varied data
        original_params = MetalogParameters(
            boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
            method=MetalogFitMethod.Lasso,
            lower_bound=10.0,
            upper_bound=0.0,
            num_terms=6,
        )
        original_coefficients = jnp.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        original_results = Metalog(
            metalog_params=original_params,
            a=original_coefficients,
        )

        # Dumps and loads roundtrip
        json_str = original_results.dumps()
        loaded_results = Metalog.loads(json_str)

        # Verify complete equality using __eq__ method
        self.assertEqual(loaded_results, original_results)

        # Verify types are correct
        self.assertIsInstance(loaded_results.boundedness, MetalogBoundedness)
        self.assertIsInstance(loaded_results.metalog_params.method, MetalogFitMethod)
        self.assertIsInstance(loaded_results.a, jnp.ndarray)


class MetalogInputDataTest(parameterized.TestCase):
    """Test suite for the MetalogInputData dataclass.

    Tests the creation, validation, and behavior of MetalogInputData instances,
    including comprehensive validation of well-formed and invalid inputs.
    """

    # Test valid instance creation with precomputed quantiles
    @parameterized.named_parameters(
        {
            "testcase_name": "minimal_valid_precomputed",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "many_terms_precomputed",
            "x": jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]),
            "y": jnp.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "negative_quantiles_precomputed",
            "x": jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "small_range_precomputed",
            "x": jnp.array([0.001, 0.002, 0.003]),
            "y": jnp.array([0.25, 0.5, 0.75]),
            "precomputed_quantiles": True,
        },
    )
    def test_from_values_valid_precomputed_quantiles(self, x, y, precomputed_quantiles):
        """Test that from_values creates valid instances with precomputed quantiles.

        Args:
            x: Array of quantile values.
            y: Array of probability levels.
            precomputed_quantiles: Flag indicating precomputed quantiles.
        """
        data = MetalogInputData.from_values(
            x=x, y=y, precomputed_quantiles=precomputed_quantiles
        )

        # Verify all fields are set correctly
        np.testing.assert_array_almost_equal(data.x, x)
        np.testing.assert_array_almost_equal(data.y, y)
        self.assertEqual(data.precomputed_quantiles, precomputed_quantiles)

    # Test valid instance creation with raw sample data
    @parameterized.named_parameters(
        {
            "testcase_name": "minimal_valid_samples",
            "x": jnp.array([3.0, 7.0, 1.0]),  # Unordered raw samples
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": False,
        },
        {
            "testcase_name": "many_samples",
            "x": jnp.array([5.2, 3.1, 8.7, 2.4, 9.9, 1.5, 7.3, 4.8, 6.1]),
            "y": jnp.array([0.1, 0.25, 0.5, 0.75, 0.9]),
            "precomputed_quantiles": False,
        },
        {
            "testcase_name": "negative_samples",
            "x": jnp.array([-5.0, 10.0, -2.0, 3.0, -8.0, 1.0]),
            "y": jnp.array([0.2, 0.5, 0.8]),
            "precomputed_quantiles": False,
        },
    )
    def test_from_values_valid_raw_samples(self, x, y, precomputed_quantiles):
        """Test that from_values creates valid instances with raw samples.

        When precomputed_quantiles=False, the raw sample data should be
        automatically converted to quantiles.

        Args:
            x: Array of raw sample values (not necessarily sorted).
            y: Array of probability levels.
            precomputed_quantiles: Flag indicating raw samples (False).
        """
        data = MetalogInputData.from_values(
            x=x, y=y, precomputed_quantiles=precomputed_quantiles
        )

        # Verify probability levels are preserved
        np.testing.assert_array_almost_equal(data.y, y)
        self.assertEqual(data.precomputed_quantiles, precomputed_quantiles)

        # Verify x has been transformed to quantiles (should be sorted)
        # The transformed x should have the same length as y
        self.assertEqual(len(data.x), len(y))

        # Verify the transformed quantiles are strictly ascending
        diffs = jnp.diff(data.x)
        self.assertTrue(jnp.all(diffs > 0))

    # Test validation: insufficient array size
    @parameterized.named_parameters(
        {
            "testcase_name": "x_too_small",
            "x": jnp.array([1.0, 2.0]),  # Only 2 elements
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_too_small",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.5, 0.9]),  # Only 2 elements
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "both_too_small",
            "x": jnp.array([1.0]),  # Only 1 element
            "y": jnp.array([0.5]),  # Only 1 element
            "precomputed_quantiles": True,
        },
    )
    def test_from_values_insufficient_array_size(self, x, y, precomputed_quantiles):
        """Test that from_values raises error for arrays with fewer than 3 elements.

        Args:
            x: Array with insufficient elements.
            y: Array with insufficient elements.
            precomputed_quantiles: Flag for quantile type.
        """
        with self.assertRaises(Exception):  # chex.assert_axis_dimension_gteq raises
            MetalogInputData.from_values(
                x=x, y=y, precomputed_quantiles=precomputed_quantiles
            )

    # Test validation: probability range violations
    @parameterized.named_parameters(
        {
            "testcase_name": "y_contains_zero",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.0, 0.5, 0.9]),  # Contains 0
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_contains_one",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.5, 1.0]),  # Contains 1
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_contains_negative",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([-0.1, 0.5, 0.9]),  # Negative value
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_contains_value_greater_than_one",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.5, 1.5]),  # > 1
            "precomputed_quantiles": True,
        },
    )
    def test_from_values_invalid_probability_range(self, x, y, precomputed_quantiles):
        """Test that from_values raises error for probabilities outside (0, 1).

        Args:
            x: Array of quantile values.
            y: Array with invalid probability values.
            precomputed_quantiles: Flag for quantile type.
        """
        with self.assertRaises(
            ValueError
        ):  # assert_probability_range raises ValueError
            MetalogInputData.from_values(
                x=x, y=y, precomputed_quantiles=precomputed_quantiles
            )

    # Test validation: non-ascending probability levels
    @parameterized.named_parameters(
        {
            "testcase_name": "y_duplicate_values",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.5, 0.5]),  # Duplicate 0.5
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_descending",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.9, 0.5, 0.1]),  # Descending order
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_unsorted",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.9, 0.5]),  # Unsorted
            "precomputed_quantiles": True,
        },
    )
    def test_from_values_non_ascending_probabilities(self, x, y, precomputed_quantiles):
        """Test that from_values raises error for non-ascending probability levels.

        Args:
            x: Array of quantile values.
            y: Array with non-ascending probabilities.
            precomputed_quantiles: Flag for quantile type.
        """
        with self.assertRaises(Exception):  # assert_strictly_ascending raises
            MetalogInputData.from_values(
                x=x, y=y, precomputed_quantiles=precomputed_quantiles
            )

    # Test validation: non-ascending quantiles (precomputed only)
    @parameterized.named_parameters(
        {
            "testcase_name": "x_duplicate_values_precomputed",
            "x": jnp.array([1.0, 5.0, 5.0]),  # Duplicate 5.0
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "x_descending_precomputed",
            "x": jnp.array([10.0, 5.0, 1.0]),  # Descending
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "x_unsorted_precomputed",
            "x": jnp.array([5.0, 1.0, 10.0]),  # Unsorted
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": True,
        },
    )
    def test_from_values_non_ascending_precomputed_quantiles(
        self, x, y, precomputed_quantiles
    ):
        """Test that from_values raises error for non-ascending precomputed quantiles.

        Args:
            x: Array with non-ascending quantile values.
            y: Array of probability levels.
            precomputed_quantiles: Must be True for this test.
        """
        with self.assertRaises(Exception):  # assert_strictly_ascending raises
            MetalogInputData.from_values(
                x=x, y=y, precomputed_quantiles=precomputed_quantiles
            )

    # Test validation: invalid data types
    @parameterized.named_parameters(
        {
            "testcase_name": "x_not_numeric",
            "x": ["a", "b", "c"],  # Strings instead of numbers
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "y_not_numeric",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": ["low", "medium", "high"],  # Strings
            "precomputed_quantiles": True,
        },
        {
            "testcase_name": "precomputed_not_bool",
            "x": jnp.array([1.0, 5.0, 10.0]),
            "y": jnp.array([0.1, 0.5, 0.9]),
            "precomputed_quantiles": 1,  # Integer instead of boolean
        },
    )
    def test_from_values_invalid_data_types(self, x, y, precomputed_quantiles):
        """Test that from_values raises error for invalid data types.

        Args:
            x: Input with potentially invalid type.
            y: Input with potentially invalid type.
            precomputed_quantiles: Input with potentially invalid type.
        """
        with self.assertRaises(Exception):  # Various assertion errors
            MetalogInputData.from_values(
                x=x, y=y, precomputed_quantiles=precomputed_quantiles
            )

    # Test immutability
    def test_instance_immutability(self):
        """Test that MetalogInputData instances are immutable."""
        data = MetalogInputData.from_values(
            x=jnp.array([1.0, 5.0, 10.0]),
            y=jnp.array([0.1, 0.5, 0.9]),
            precomputed_quantiles=True,
        )

        # Attempt to modify fields should raise error
        with self.assertRaises((AttributeError, TypeError)):
            data.x = jnp.array([2.0, 6.0, 11.0])

        with self.assertRaises((AttributeError, TypeError)):
            data.y = jnp.array([0.2, 0.6, 0.95])

        with self.assertRaises((AttributeError, TypeError)):
            data.precomputed_quantiles = False

    # Test direct instantiation is disabled
    def test_direct_instantiation_disabled(self):
        """Test that direct instantiation via __init__() raises TypeError.

        While __init__() technically accepts parameters, the __post_init__
        validation method raises TypeError when instances are not created
        through the from_values() factory method.
        """
        with self.assertRaises(TypeError):
            # This should fail due to __post_init__ validation
            MetalogInputData(
                x=jnp.array([1.0, 5.0, 10.0]),
                y=jnp.array([0.1, 0.5, 0.9]),
                precomputed_quantiles=True,
            )

    # Test quantile transformation for raw samples
    def test_quantile_transformation_for_raw_samples(self):
        """Test that raw samples are correctly transformed to quantiles.

        When precomputed_quantiles=False, the input x (raw samples) should be
        transformed to quantiles using jnp.quantile(x, y).
        """
        # Create raw sample data (unordered)
        raw_samples = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0])
        y = jnp.array([0.1, 0.5, 0.9])

        # Create MetalogInputData with raw samples
        data = MetalogInputData.from_values(
            x=raw_samples, y=y, precomputed_quantiles=False
        )

        # Manually compute expected quantiles
        expected_quantiles = jnp.quantile(raw_samples, y)

        # Verify the transformation occurred correctly
        np.testing.assert_array_almost_equal(data.x, expected_quantiles)

        # Verify the transformed quantiles are strictly ascending
        diffs = jnp.diff(data.x)
        self.assertTrue(jnp.all(diffs > 0))

    # Test with DEFAULT_Y
    def test_with_default_y(self):
        """Test that MetalogInputData works with the DEFAULT_Y probability grid."""
        x_precomputed = jnp.quantile(
            jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), DEFAULT_Y
        )

        data = MetalogInputData.from_values(
            x=x_precomputed, y=DEFAULT_Y, precomputed_quantiles=True
        )

        # Verify instance was created successfully
        self.assertEqual(len(data.x), len(DEFAULT_Y))
        self.assertEqual(len(data.y), len(DEFAULT_Y))
        np.testing.assert_array_almost_equal(data.y, DEFAULT_Y)

    # Test with very close probability values (edge case)
    def test_very_close_probability_values(self):
        """Test that very close (but still strictly ascending) probabilities work."""
        x = jnp.array([1.0, 1.001, 1.002])
        y = jnp.array([0.1, 0.1 + 1e-6, 0.1 + 2e-6])  # Very close but ascending

        # This should succeed with default tolerance
        data = MetalogInputData.from_values(x=x, y=y, precomputed_quantiles=True)

        np.testing.assert_array_almost_equal(data.x, x)
        np.testing.assert_array_almost_equal(data.y, y)

    # Test edge case: probabilities very close to boundaries
    def test_probabilities_near_boundaries(self):
        """Test probabilities very close to 0 and 1 (but still in valid range)."""
        x = jnp.array([0.001, 5.0, 999.999])
        y = jnp.array([0.001, 0.5, 0.999])  # Close to 0 and 1 but valid

        data = MetalogInputData.from_values(x=x, y=y, precomputed_quantiles=True)

        np.testing.assert_array_almost_equal(data.x, x)
        np.testing.assert_array_almost_equal(data.y, y)

    # Test with large arrays
    def test_large_arrays(self):
        """Test MetalogInputData with larger arrays."""
        n = 100
        x = jnp.sort(jnp.array(np.random.randn(n)))  # Sorted random values
        y = jnp.linspace(0.01, 0.99, n)  # 100 evenly spaced probabilities

        data = MetalogInputData.from_values(x=x, y=y, precomputed_quantiles=True)

        self.assertEqual(len(data.x), n)
        self.assertEqual(len(data.y), n)
        np.testing.assert_array_almost_equal(data.x, x)
        np.testing.assert_array_almost_equal(data.y, y)


if __name__ == "__main__":
    absltest.main()
