# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.regression.lasso module."""

import unittest

import jax.numpy as jnp

from metalog_jax.regression.base import RegressionModel, RegularizedParameters
from metalog_jax.regression.lasso import (
    DEFAULT_LASSO_ITERATIONS,
    DEFAULT_LASSO_LAMBDA,
    DEFAULT_LASSO_PARAMETERS,
    DEFAULT_LASSO_TOLERANCE,
    LassoModel,
    LassoParameters,
    fit_lasso,
)


class LassoParametersTest(unittest.TestCase):
    """Tests for LassoParameters class."""

    def test_lasso_params_creation(self):
        """Test creating LassoParameters with custom values."""
        params = LassoParameters(
            lam=0.1,
            num_iters=500,
            tol=1e-6,
        )
        self.assertEqual(params.lam, 0.1)
        self.assertEqual(params.num_iters, 500)
        self.assertEqual(params.tol, 1e-6)

    def test_lasso_params_defaults(self):
        """Only lam is required; the step limit and tolerance have defaults."""
        params = LassoParameters(lam=0.1)
        self.assertEqual(params.num_iters, DEFAULT_LASSO_ITERATIONS)
        self.assertEqual(params.tol, DEFAULT_LASSO_TOLERANCE)

    def test_lasso_params_inherits_from_regularized_parameters(self):
        """Test that LassoParameters inherits from RegularizedParameters."""
        params = LassoParameters(
            lam=0.1,
            num_iters=100,
            tol=1e-6,
        )
        self.assertIsInstance(params, RegularizedParameters)

    def test_lasso_params_immutability(self):
        """Test that LassoParameters instances are immutable."""
        params = LassoParameters(
            lam=0.1,
            num_iters=100,
            tol=1e-6,
        )
        with self.assertRaises(AttributeError):
            params.lam = 0.5

    def test_default_lasso_parameters(self):
        """Test DEFAULT_LASSO_PARAMETERS constant."""
        self.assertEqual(DEFAULT_LASSO_PARAMETERS.lam, DEFAULT_LASSO_LAMBDA)
        self.assertEqual(DEFAULT_LASSO_PARAMETERS.num_iters, DEFAULT_LASSO_ITERATIONS)
        self.assertEqual(DEFAULT_LASSO_PARAMETERS.tol, DEFAULT_LASSO_TOLERANCE)


class LassoModelTest(unittest.TestCase):
    """Tests for LassoModel class."""

    def test_convergence_fields_default_to_none(self):
        """A model built by hand has no convergence record."""
        model = LassoModel(weights=jnp.array([1.0, 2.0]))
        self.assertIsNone(model.converged)
        self.assertIsNone(model.iterations)

    def test_fit_records_convergence(self):
        """fit_lasso records that it converged and how many steps it took."""
        X = jnp.array([[1.0, 2.0], [3.0, 5.0], [5.0, 6.0]])
        model = fit_lasso(X, jnp.array([1.0, 2.0, 3.0]))
        self.assertTrue(bool(model.converged))
        self.assertEqual(int(model.iterations), 1)

    def test_lasso_model_creation(self):
        """Test creating LassoModel with weights."""
        weights = jnp.array([1.0, 2.0, 3.0])
        model = LassoModel(weights=weights)
        self.assertEqual(model.weights.shape, (3,))

    def test_lasso_model_inherits_from_regression_model(self):
        """Test that LassoModel inherits from RegressionModel."""
        weights = jnp.array([1.0, 2.0])
        model = LassoModel(weights=weights)
        self.assertIsInstance(model, RegressionModel)

    def test_lasso_model_immutability(self):
        """Test that LassoModel instances are immutable."""
        weights = jnp.array([1.0, 2.0])
        model = LassoModel(weights=weights)
        with self.assertRaises(AttributeError):
            model.weights = jnp.array([3.0, 4.0])


class FitLassoTest(unittest.TestCase):
    """Tests for fit_lasso function."""

    def test_fit_returns_lasso_model(self):
        """Test that fit returns a LassoModel instance."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        model = fit_lasso(X, y)
        self.assertIsInstance(model, LassoModel)

    def test_fit_weights_shape_matches_features(self):
        """Test that fitted weights shape matches number of features."""
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        model = fit_lasso(X, y)
        self.assertEqual(model.weights.shape, (3,))

    def test_fit_with_custom_params(self):
        """Test fitting with custom parameters."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        params = LassoParameters(
            lam=0.1,
            num_iters=100,
            tol=1e-6,
        )
        model = fit_lasso(X, y, params)
        self.assertIsInstance(model, LassoModel)

    def test_large_regularization_produces_sparse_weights(self):
        """Test that large lambda produces sparse (near-zero) weights."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        params = LassoParameters(
            lam=100.0,
            num_iters=500,
            tol=1e-6,
        )
        model = fit_lasso(X, y, params)
        # Large lambda should shrink weights toward zero
        weight_norm = jnp.linalg.norm(model.weights)
        self.assertLess(float(weight_norm), 0.1)


if __name__ == "__main__":
    unittest.main()
