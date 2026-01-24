# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.regression.base module."""

import unittest

import jax.numpy as jnp

from metalog_jax.regression.base import RegressionModel, RegularizedParameters


class RegularizedParametersTest(unittest.TestCase):
    """Tests for RegularizedParameters base class."""

    def test_regularized_parameters_is_base_class(self):
        """Test that RegularizedParameters can be instantiated as abstract base."""
        params = RegularizedParameters()
        self.assertIsInstance(params, RegularizedParameters)

    def test_regularized_parameters_immutability(self):
        """Test that RegularizedParameters instances are immutable."""
        params = RegularizedParameters()
        with self.assertRaises(AttributeError):
            params.some_attribute = 1.0


class RegressionModelTest(unittest.TestCase):
    """Tests for RegressionModel base class."""

    def test_regression_model_creation_single_feature(self):
        """Test creating RegressionModel with single feature weight."""
        weights = jnp.array([1.0])
        model = RegressionModel(weights=weights)
        self.assertEqual(model.weights.shape, (1,))
        self.assertTrue(jnp.allclose(model.weights, jnp.array([1.0])))

    def test_regression_model_creation_multiple_features(self):
        """Test creating RegressionModel with multiple feature weights."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = RegressionModel(weights=weights)
        self.assertEqual(model.weights.shape, (5,))

    def test_regression_model_immutability(self):
        """Test that RegressionModel instances are immutable."""
        weights = jnp.array([1.0, 2.0])
        model = RegressionModel(weights=weights)
        with self.assertRaises(AttributeError):
            model.weights = jnp.array([3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
