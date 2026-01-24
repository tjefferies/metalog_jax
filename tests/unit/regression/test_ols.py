# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.regression.ols module."""

import unittest

import jax.numpy as jnp

from metalog_jax.regression.base import RegressionModel
from metalog_jax.regression.ols import (
    OLSModel,
    fit_ordinary_least_squares,
    predict_ordinary_least_squares,
)


class OLSModelTest(unittest.TestCase):
    """Tests for OLSModel class."""

    def test_ols_model_creation_single_feature(self):
        """Test creating OLSModel with single feature weight."""
        weights = jnp.array([1.0])
        model = OLSModel(weights=weights)
        self.assertEqual(model.weights.shape, (1,))

    def test_ols_model_creation_multiple_features(self):
        """Test creating OLSModel with multiple feature weights."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = OLSModel(weights=weights)
        self.assertEqual(model.weights.shape, (5,))

    def test_ols_model_inherits_from_regression_model(self):
        """Test that OLSModel inherits from RegressionModel."""
        weights = jnp.array([1.0, 2.0])
        model = OLSModel(weights=weights)
        self.assertIsInstance(model, RegressionModel)

    def test_ols_model_immutability(self):
        """Test that OLSModel instances are immutable."""
        weights = jnp.array([1.0, 2.0])
        model = OLSModel(weights=weights)
        with self.assertRaises(AttributeError):
            model.weights = jnp.array([3.0, 4.0])


class FitOrdinaryLeastSquaresTest(unittest.TestCase):
    """Tests for fit_ordinary_least_squares function."""

    def test_fit_returns_ols_model(self):
        """Test that fit returns an OLSModel instance."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        model = fit_ordinary_least_squares(X, y)
        self.assertIsInstance(model, OLSModel)

    def test_fit_weights_shape_matches_features(self):
        """Test that fitted weights shape matches number of features."""
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = jnp.array([1.0, 2.0, 3.0])
        model = fit_ordinary_least_squares(X, y)
        self.assertEqual(model.weights.shape, (3,))

    def test_fit_perfect_reconstruction(self):
        """Test perfect reconstruction with deterministic relationship."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2*x
        model = fit_ordinary_least_squares(X, y)
        predictions = X @ model.weights
        self.assertTrue(jnp.allclose(predictions, y, atol=1e-5))


class PredictOrdinaryLeastSquaresTest(unittest.TestCase):
    """Tests for predict_ordinary_least_squares function."""

    def test_predict_output_shape(self):
        """Test that predictions have correct shape."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model = OLSModel(weights=jnp.array([0.5, 0.5]))
        predictions = predict_ordinary_least_squares(X, model)
        self.assertEqual(predictions.shape, (3,))

    def test_predict_correct_values(self):
        """Test that predictions are computed correctly."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        model = OLSModel(weights=jnp.array([1.0, 1.0]))
        predictions = predict_ordinary_least_squares(X, model)
        expected = jnp.array([3.0, 7.0])  # 1+2=3, 3+4=7
        self.assertTrue(jnp.allclose(predictions, expected))

    def test_predict_with_fitted_model(self):
        """Test predictions using a fitted model."""
        X_train = jnp.array([[1.0], [2.0], [3.0]])
        y_train = jnp.array([2.0, 4.0, 6.0])
        model = fit_ordinary_least_squares(X_train, y_train)

        X_test = jnp.array([[4.0], [5.0]])
        predictions = predict_ordinary_least_squares(X_test, model)
        expected = jnp.array([8.0, 10.0])
        self.assertTrue(jnp.allclose(predictions, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
