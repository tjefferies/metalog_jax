"""Unit tests for the Metalog JAX regression models.

This module contains comprehensive tests for all regression models, dataclasses,
and functions in the metalog_jax.regression.models module, using parameterized
testing to validate behavior across various configurations and scenarios.
"""
# Copyright: Travis Jefferies 2026

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SklearnLasso

from metalog_jax.regression import (
    LassoModel,
    LassoParameters,
    OLSModel,
    RegressionModel,
    fit_lasso,
    fit_ordinary_least_squares,
    predict_ordinary_least_squares,
    soft_thresholding,
)


class RegressionModelTest(parameterized.TestCase):
    """Test suite for the RegressionModel base dataclass.

    Tests the creation and properties of the base regression model structure.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "single_feature",
            "weights": jnp.array([1.5]),
        },
        {
            "testcase_name": "two_features",
            "weights": jnp.array([1.0, 2.0]),
        },
        {
            "testcase_name": "multiple_features",
            "weights": jnp.array([0.5, 1.0, 1.5, 2.0]),
        },
    )
    def test_regression_model_creation(self, weights):
        """Test that RegressionModel can be created with valid values.

        Args:
            weights: Weight vector.
        """
        model = RegressionModel(weights=weights)
        np.testing.assert_array_equal(model.weights, weights)

    def test_regression_model_immutability(self):
        """Test that RegressionModel instances are immutable."""
        model = RegressionModel(weights=jnp.array([1.0, 2.0]))
        with self.assertRaises((AttributeError, TypeError)):
            model.weights = jnp.array([3.0, 4.0])


class OLSModelTest(parameterized.TestCase):
    """Test suite for the OLSModel dataclass.

    Tests the creation and properties of OLS model structures.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "single_feature",
            "weights": jnp.array([1.5]),
        },
        {
            "testcase_name": "two_features",
            "weights": jnp.array([1.0, 2.0]),
        },
        {
            "testcase_name": "multiple_features",
            "weights": jnp.array([0.5, 1.0, 1.5, 2.0]),
        },
        {
            "testcase_name": "zero_weights",
            "weights": jnp.zeros(3),
        },
        {
            "testcase_name": "negative_weights",
            "weights": jnp.array([-1.0, -2.0, -3.0]),
        },
    )
    def test_ols_model_creation(self, weights):
        """Test that OLSModel can be created with valid values.

        Args:
            weights: Weight vector.
        """
        model = OLSModel(weights=weights)
        np.testing.assert_array_equal(model.weights, weights)

    def test_ols_model_immutability(self):
        """Test that OLSModel instances are immutable."""
        model = OLSModel(weights=jnp.array([1.0, 2.0]))
        with self.assertRaises((AttributeError, TypeError)):
            model.weights = jnp.array([3.0, 4.0])

    def test_ols_model_inherits_from_regression_model(self):
        """Test that OLSModel is a subclass of RegressionModel."""
        model = OLSModel(weights=jnp.array([1.0, 2.0]))
        self.assertIsInstance(model, RegressionModel)


class FitOrdinaryLeastSquaresTest(parameterized.TestCase):
    """Test suite for the fit_ordinary_least_squares function.

    Tests OLS model fitting with various data configurations and validates
    the closed-form solution accuracy.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "simple_linear_data",
            "X": jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
            "y": jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        },
        {
            "testcase_name": "multivariate_linear",
            "X": jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
            "y": jnp.array([5.0, 8.0, 11.0, 14.0]),
        },
        {
            "testcase_name": "three_features",
            "X": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            "y": jnp.array([6.0, 15.0, 24.0]),
        },
    )
    def test_fit_returns_valid_model(self, X, y):
        """Test that fit_ordinary_least_squares returns a valid model structure.

        Args:
            X: Feature matrix.
            y: Target vector.
        """
        model = fit_ordinary_least_squares(X, y)

        # Check that model is a RegressionModel (or OLSModel)
        self.assertIsInstance(model, RegressionModel)

        # Check that weights have correct shape
        self.assertEqual(model.weights.shape, (X.shape[1],))

        # Check that weights are finite
        self.assertTrue(jnp.all(jnp.isfinite(model.weights)))

    @parameterized.named_parameters(
        {
            "testcase_name": "perfect_linear_relationship",
            "X": jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
            "y": jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            "expected_weight": 2.0,
            "tolerance": 1e-5,
        },
    )
    def test_fit_learns_exact_relationship(self, X, y, expected_weight, tolerance):
        """Test that OLS finds exact solution for perfectly linear data.

        Args:
            X: Feature matrix.
            y: Target vector.
            expected_weight: Expected weight value.
            tolerance: Tolerance for equality check.
        """
        model = fit_ordinary_least_squares(X, y)

        # Check that learned parameters match expected values
        np.testing.assert_allclose(
            float(model.weights[0]), expected_weight, atol=tolerance
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "large_dataset",
            "n_samples": 1000,
            "n_features": 10,
        },
        {
            "testcase_name": "wide_data",
            "n_samples": 50,
            "n_features": 20,
        },
        {
            "testcase_name": "single_feature_many_samples",
            "n_samples": 500,
            "n_features": 1,
        },
    )
    def test_fit_with_random_data(self, n_samples, n_features):
        """Test fitting with randomly generated data.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
        """
        np.random.seed(42)
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        model = fit_ordinary_least_squares(X, y)

        # Check output shapes and validity
        self.assertEqual(model.weights.shape, (n_features,))
        self.assertTrue(jnp.all(jnp.isfinite(model.weights)))

    def test_fit_perfect_reconstruction(self):
        """Test that OLS perfectly reconstructs training data for well-posed problems."""
        # Create data with more samples than features
        np.random.seed(123)
        n_samples, n_features = 100, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        true_weights = jnp.array(np.random.randn(n_features))
        y = jnp.dot(X, true_weights)

        # Fit model
        model = fit_ordinary_least_squares(X, y)

        # Predictions should match training data very closely
        predictions = jnp.dot(X, model.weights)
        mse = jnp.mean((y - predictions) ** 2)

        # MSE should be extremely small (near machine precision)
        self.assertLess(float(mse), 1e-10)


class PredictOrdinaryLeastSquaresTest(parameterized.TestCase):
    """Test suite for the predict_ordinary_least_squares function.

    Tests prediction functionality for OLS models with various configurations.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "single_sample_single_feature",
            "X": jnp.array([[2.0]]),
            "weights": jnp.array([3.0]),
            "expected": jnp.array([6.0]),
        },
        {
            "testcase_name": "multiple_samples_single_feature",
            "X": jnp.array([[1.0], [2.0], [3.0]]),
            "weights": jnp.array([2.0]),
            "expected": jnp.array([2.0, 4.0, 6.0]),
        },
        {
            "testcase_name": "single_sample_multiple_features",
            "X": jnp.array([[1.0, 2.0, 3.0]]),
            "weights": jnp.array([1.0, 2.0, 3.0]),
            "expected": jnp.array([14.0]),
        },
        {
            "testcase_name": "multiple_samples_multiple_features",
            "X": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "weights": jnp.array([1.0, 2.0]),
            "expected": jnp.array([5.0, 11.0]),
        },
        {
            "testcase_name": "zero_weights",
            "X": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "weights": jnp.array([0.0, 0.0]),
            "expected": jnp.array([0.0, 0.0]),
        },
        {
            "testcase_name": "negative_weights",
            "X": jnp.array([[1.0, 2.0]]),
            "weights": jnp.array([-1.0, -2.0]),
            "expected": jnp.array([-5.0]),
        },
    )
    def test_predict_correct_values(self, X, weights, expected):
        """Test that predictions match expected values.

        Args:
            X: Feature matrix.
            weights: Model weights.
            expected: Expected predictions.
        """
        model = OLSModel(weights=weights)
        predictions = predict_ordinary_least_squares(X, model)

        np.testing.assert_array_almost_equal(predictions, expected, decimal=5)

    @parameterized.named_parameters(
        {
            "testcase_name": "large_batch",
            "n_samples": 1000,
            "n_features": 10,
        },
        {
            "testcase_name": "wide_features",
            "n_samples": 10,
            "n_features": 100,
        },
        {
            "testcase_name": "single_feature",
            "n_samples": 50,
            "n_features": 1,
        },
    )
    def test_predict_output_shape(self, n_samples, n_features):
        """Test that predictions have correct output shape.

        Args:
            n_samples: Number of test samples.
            n_features: Number of features.
        """
        X = jnp.array(np.random.randn(n_samples, n_features))
        weights = jnp.array(np.random.randn(n_features))

        model = OLSModel(weights=weights)
        predictions = predict_ordinary_least_squares(X, model)

        # Check output shape
        self.assertEqual(predictions.shape, (n_samples,))

    def test_predict_with_fitted_model(self):
        """Test prediction using a model fitted with fit_ordinary_least_squares."""
        # Create training data
        X_train = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_train = jnp.array([5.0, 8.0, 11.0, 14.0])

        # Fit model
        model = fit_ordinary_least_squares(X_train, y_train)

        # Make predictions on new data
        X_test = jnp.array([[5.0, 6.0], [6.0, 7.0]])
        predictions = predict_ordinary_least_squares(X_test, model)

        # Check that predictions are valid
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(predictions)))

    def test_predict_training_data_exact_fit(self):
        """Test that OLS predictions on training data are exact."""
        # Create simple linear data
        X = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])

        # Fit model
        model = fit_ordinary_least_squares(X, y)

        # Predict on training data
        predictions = predict_ordinary_least_squares(X, model)

        # Predictions should match training data exactly (or very close)
        np.testing.assert_array_almost_equal(predictions, y, decimal=5)

    def test_predict_with_regression_model_base_class(self):
        """Test that predict works with RegressionModel base class."""
        # Create a RegressionModel (base class) instance
        weights = jnp.array([2.0, 3.0])
        model = RegressionModel(weights=weights)

        X = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        predictions = predict_ordinary_least_squares(X, model)

        expected = jnp.array([5.0, 10.0])  # (1*2 + 1*3), (2*2 + 2*3)
        np.testing.assert_array_almost_equal(predictions, expected, decimal=5)


class LassoModelTest(parameterized.TestCase):
    """Test suite for the LassoModel dataclass.

    Tests the creation and properties of trained Lasso model structures.
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "single_feature",
            "weights": jnp.array([1.5]),
        },
        {
            "testcase_name": "two_features",
            "weights": jnp.array([1.0, 2.0]),
        },
        {
            "testcase_name": "multiple_features",
            "weights": jnp.array([0.5, 1.0, 1.5, 2.0]),
        },
        {
            "testcase_name": "zero_weights",
            "weights": jnp.zeros(3),
        },
        {
            "testcase_name": "negative_weights",
            "weights": jnp.array([-1.0, -2.0, -3.0]),
        },
    )
    def test_lasso_model_creation(self, weights):
        """Test that LassoModel can be created with valid values.

        Args:
            weights: Weight vector.
        """
        model = LassoModel(weights=weights)
        np.testing.assert_array_equal(model.weights, weights)

    def test_lasso_model_immutability(self):
        """Test that LassoModel instances are immutable."""
        model = LassoModel(weights=jnp.array([1.0, 2.0]))
        with self.assertRaises((AttributeError, TypeError)):
            model.weights = jnp.array([3.0, 4.0])

    def test_lasso_model_inherits_from_regression_model(self):
        """Test that LassoModel is a subclass of RegressionModel."""
        model = LassoModel(weights=jnp.array([1.0, 2.0]))
        self.assertIsInstance(model, RegressionModel)


class SoftThresholdingTest(absltest.TestCase):
    """Test suite for the soft-thresholding operator."""

    def test_positive_scalar(self):
        """Test soft thresholding with positive scalar values."""
        # Case where x > lambda
        result = soft_thresholding(jnp.array(3.0), 1.0)
        self.assertAlmostEqual(float(result), 2.0, places=7)

        # Case where x = lambda
        result = soft_thresholding(jnp.array(1.0), 1.0)
        self.assertAlmostEqual(float(result), 0.0, places=7)

        # Case where 0 < x < lambda
        result = soft_thresholding(jnp.array(0.5), 1.0)
        self.assertAlmostEqual(float(result), 0.0, places=7)

    def test_negative_scalar(self):
        """Test soft thresholding with negative scalar values."""
        # Case where x < -lambda
        result = soft_thresholding(jnp.array(-3.0), 1.0)
        self.assertAlmostEqual(float(result), -2.0, places=7)

        # Case where x = -lambda
        result = soft_thresholding(jnp.array(-1.0), 1.0)
        self.assertAlmostEqual(float(result), 0.0, places=7)

        # Case where -lambda < x < 0
        result = soft_thresholding(jnp.array(-0.5), 1.0)
        self.assertAlmostEqual(float(result), 0.0, places=7)

    def test_zero(self):
        """Test soft thresholding with zero."""
        result = soft_thresholding(jnp.array(0.0), 1.0)
        self.assertAlmostEqual(float(result), 0.0, places=7)

    def test_array_input(self):
        """Test soft thresholding with array input."""
        x = jnp.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
        lam = 1.0
        expected = jnp.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        result = soft_thresholding(x, lam)
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_zero_threshold(self):
        """Test soft thresholding with zero threshold (identity)."""
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = soft_thresholding(x, 0.0)
        np.testing.assert_allclose(result, x, rtol=1e-7)

    def test_large_threshold(self):
        """Test soft thresholding with very large threshold."""
        x = jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = soft_thresholding(x, 100.0)
        np.testing.assert_allclose(result, jnp.zeros_like(x), rtol=1e-7)

    def test_jit_compilation(self):
        """Test that soft thresholding works with JIT compilation."""
        x = jnp.array([1.0, 2.0, 3.0])
        jitted_fn = jax.jit(soft_thresholding)
        result1 = soft_thresholding(x, 1.5)
        result2 = jitted_fn(x, 1.5)
        np.testing.assert_allclose(result1, result2, rtol=1e-7)


class LassoNesterovTest(absltest.TestCase):
    """Test suite for the LASSO regression with Nesterov acceleration."""

    def setUp(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    def test_zero_regularization(self):
        """Test that zero regularization gives OLS solution."""
        n_samples, n_features = 100, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        w_true = jnp.array(np.random.randn(n_features))
        y = X @ w_true + 0.01 * jnp.array(np.random.randn(n_samples))

        # LASSO with lambda=0 should give OLS solution
        params = LassoParameters(
            lam=0.0, learning_rate=0.1, num_iters=1000, tol=1e-6, momentum=0.9
        )
        w_lasso = fit_lasso(X, y, params=params).weights
        w_ols = jnp.linalg.lstsq(X, y)[0]

        np.testing.assert_allclose(w_lasso, w_ols, atol=1e-3)

    def test_large_regularization_zeros(self):
        """Test that very large regularization gives zero weights."""
        n_samples, n_features = 50, 10
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        # Very large lambda should zero out all weights
        params = LassoParameters(
            lam=1000.0, learning_rate=0.01, num_iters=100, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        np.testing.assert_allclose(w, jnp.zeros(n_features), atol=1e-6)

    def test_sparsity_inducing(self):
        """Test that LASSO induces sparsity."""
        n_samples, n_features = 100, 20

        # Create sparse ground truth
        w_true = np.zeros(n_features)
        w_true[:5] = np.array([3.0, -2.0, 1.5, -1.0, 2.5])
        w_true = jnp.array(w_true)

        X = jnp.array(np.random.randn(n_samples, n_features))
        y = X @ w_true + 0.1 * jnp.array(np.random.randn(n_samples))

        # Run LASSO with moderate regularization
        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=2000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights

        # Check that solution is sparse
        sparsity = jnp.sum(jnp.abs(w) < 1e-4) / n_features
        self.assertGreater(
            float(sparsity), 0.5
        )  # At least 50% of weights should be near zero

    def test_convergence(self):
        """Test that the algorithm converges."""
        n_samples, n_features = 50, 10
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        # Run with tight tolerance
        params1 = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=1000, tol=1e-8, momentum=0.9
        )
        w1 = fit_lasso(X, y, params=params1).weights

        # Run more iterations - should give same result if converged
        params2 = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=2000, tol=1e-8, momentum=0.9
        )
        w2 = fit_lasso(X, y, params=params2).weights

        np.testing.assert_allclose(w1, w2, atol=1e-7)

    def test_objective_decrease(self):
        """Test that objective function decreases during optimization."""
        n_samples, n_features = 50, 10
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))
        lam = 0.1

        def lasso_objective(w):
            return (1 / (2 * n_samples)) * jnp.mean((y - X @ w) ** 2) + lam * jnp.sum(
                jnp.abs(w)
            )

        # Initial objective (at one weights)
        obj_init = lasso_objective(jnp.ones(n_features))

        # Final objective
        params = LassoParameters(
            lam=lam, learning_rate=0.01, num_iters=1000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        obj_final = lasso_objective(w)

        self.assertLess(float(obj_final), float(obj_init))

    def test_compare_with_sklearn(self):
        """Compare results with sklearn's Lasso implementation."""
        n_samples, n_features = 100, 10
        X_np = np.random.randn(n_samples, n_features)
        y_np = np.random.randn(n_samples)

        # Normalize data for fair comparison
        X_np = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)
        y_np = (y_np - y_np.mean()) / y_np.std()

        X = jnp.array(X_np)
        y = jnp.array(y_np)

        lam = 0.01
        alpha = lam

        # Our implementation
        params = LassoParameters(
            lam=lam, learning_rate=0.1, num_iters=2000, tol=1e-6, momentum=0.9
        )
        w_ours = fit_lasso(X, y, params=params).weights

        # Sklearn implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sklearn_lasso = SklearnLasso(
                alpha=alpha, fit_intercept=False, max_iter=2000, tol=1e-6
            )
            sklearn_lasso.fit(X_np, y_np)
            w_sklearn = sklearn_lasso.coef_

        # Should be close (allowing for differences in optimization)
        np.testing.assert_allclose(w_ours, w_sklearn, atol=0.1)

    def test_single_feature(self):
        """Test with single feature (1D regression)."""
        n_samples = 50
        X = jnp.array(np.random.randn(n_samples, 1))
        y = 2.0 * X[:, 0] + 0.1 * jnp.array(np.random.randn(n_samples))

        params = LassoParameters(
            lam=0.01, learning_rate=0.1, num_iters=500, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertEqual(w.shape, (1,))
        self.assertLess(
            abs(float(w[0]) - 2.0), 0.2
        )  # Should be close to true coefficient

    def test_single_sample(self):
        """Test with single sample."""
        X = jnp.array([[1.0, 2.0, 3.0]])
        y = jnp.array([1.0])

        # With single sample, solution is not unique but should run
        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=100, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertEqual(w.shape, (3,))
        self.assertFalse(jnp.any(jnp.isnan(w)))

    def test_perfect_fit(self):
        """Test with perfect linear relationship."""
        n_samples, n_features = 50, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        w_true = jnp.array([1.0, -1.0, 2.0, -2.0, 0.5])
        y = X @ w_true  # Perfect fit, no noise

        # With small regularization, should recover true weights
        params = LassoParameters(
            lam=1e-6, learning_rate=0.1, num_iters=1000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        np.testing.assert_allclose(w, w_true, atol=1e-3)

    def test_overdetermined_system(self):
        """Test with overdetermined system (n_samples >> n_features)."""
        n_samples, n_features = 1000, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        params = LassoParameters(
            lam=0.01, learning_rate=0.01, num_iters=1000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertEqual(w.shape, (n_features,))
        self.assertFalse(jnp.any(jnp.isnan(w)))

    def test_underdetermined_system(self):
        """Test with underdetermined system (n_samples < n_features)."""
        n_samples, n_features = 20, 50
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        # LASSO should still work and produce sparse solution
        params = LassoParameters(
            lam=0.1, learning_rate=0.001, num_iters=2000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertEqual(w.shape, (n_features,))
        self.assertFalse(jnp.any(jnp.isnan(w)))

        # Should be sparse
        sparsity = jnp.sum(jnp.abs(w) < 1e-4) / n_features
        self.assertGreater(float(sparsity), 0.2)

    def test_reproducibility(self):
        """Test that results are reproducible."""
        n_samples, n_features = 50, 10
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=500, tol=1e-6, momentum=0.9
        )
        w1 = fit_lasso(X, y, params=params).weights
        w2 = fit_lasso(X, y, params=params).weights

        np.testing.assert_allclose(w1, w2)

    def test_jit_compatibility(self):
        """Test that the function works with JIT compilation."""
        n_samples, n_features = 30, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        # JIT compile the function
        lasso_jitted = jax.jit(fit_lasso, static_argnums=(2,))

        # Compare results
        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=500, tol=1e-6, momentum=0.9
        )
        w1 = fit_lasso(X, y, params=params).weights
        w2 = lasso_jitted(X, y, params=params).weights

        np.testing.assert_allclose(w1, w2)


class LearningRateTest(parameterized.TestCase):
    """Parameterized tests for different learning rates."""

    @parameterized.parameters(0.001, 0.01, 0.1)
    def test_different_learning_rates(self, lr):
        """Test stability with different learning rates."""
        np.random.seed(42)
        n_samples, n_features = 50, 10
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        params = LassoParameters(
            lam=0.1, learning_rate=lr, num_iters=2000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertFalse(jnp.any(jnp.isnan(w)))

        # Check that solution is reasonable (not exploding)
        self.assertLess(float(jnp.max(jnp.abs(w))), 100.0)


class MomentumTest(parameterized.TestCase):
    """Parameterized tests for momentum values."""

    @parameterized.parameters(0.0, 0.5, 0.9, 0.99)
    def test_momentum_values(self, momentum):
        """Test different momentum values."""
        np.random.seed(42)
        n_samples, n_features = 100, 20
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=500, tol=1e-6, momentum=momentum
        )
        w = fit_lasso(X, y, params=params).weights

        # Should produce valid solutions for all momentum values
        self.assertFalse(jnp.any(jnp.isnan(w)))
        self.assertFalse(jnp.any(jnp.isinf(w)))


class EdgeCasesTest(absltest.TestCase):
    """Test edge cases and numerical stability."""

    def test_collinear_features(self):
        """Test with perfectly collinear features."""
        np.random.seed(42)
        n_samples = 50
        X = np.random.randn(n_samples, 3)
        X[:, 2] = X[:, 0] + X[:, 1]  # Third feature is sum of first two
        X = jnp.array(X)
        y = jnp.array(np.random.randn(n_samples))

        # Should handle collinearity gracefully
        params = LassoParameters(
            lam=0.1, learning_rate=0.001, num_iters=1000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertFalse(jnp.any(jnp.isnan(w)))
        self.assertFalse(jnp.any(jnp.isinf(w)))

    def test_constant_features(self):
        """Test with constant features."""
        np.random.seed(42)
        n_samples, n_features = 50, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        X = X.at[:, 2].set(1.0)  # Make third feature constant
        y = jnp.array(np.random.randn(n_samples))

        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=500, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertFalse(jnp.any(jnp.isnan(w)))

    def test_zero_features(self):
        """Test with zero features."""
        n_samples, n_features = 50, 5
        X = jnp.zeros((n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        # With zero features, weights should be zero
        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=100, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        np.testing.assert_allclose(w, jnp.zeros(n_features))

    def test_zero_targets(self):
        """Test with zero targets."""
        np.random.seed(42)
        n_samples, n_features = 50, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.zeros(n_samples)

        # With zero targets, optimal weights are zero
        params = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=500, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        np.testing.assert_allclose(w, jnp.zeros(n_features), atol=1e-6)

    def test_large_values(self):
        """Test numerical stability with large values."""
        np.random.seed(42)
        n_samples, n_features = 50, 5
        X = jnp.array(np.random.randn(n_samples, n_features)) * 1000
        y = jnp.array(np.random.randn(n_samples)) * 1000

        # Should handle large values without overflow
        params = LassoParameters(
            lam=10.0, learning_rate=1e-7, num_iters=500, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertFalse(jnp.any(jnp.isnan(w)))
        self.assertFalse(jnp.any(jnp.isinf(w)))

    def test_small_values(self):
        """Test numerical stability with small values."""
        np.random.seed(42)
        n_samples, n_features = 50, 5
        X = jnp.array(np.random.randn(n_samples, n_features)) * 1e-6
        y = jnp.array(np.random.randn(n_samples)) * 1e-6

        # Should handle small values without underflow
        params = LassoParameters(
            lam=1e-8, learning_rate=0.1, num_iters=500, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights
        self.assertFalse(jnp.any(jnp.isnan(w)))


class PerformanceTest(absltest.TestCase):
    """Test performance characteristics."""

    def test_early_stopping(self):
        """Test that early stopping works correctly."""
        np.random.seed(42)
        n_samples, n_features = 50, 5
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        # Track if converges before max iterations
        max_iters = 10000
        tight_tol = 1e-10

        params1 = LassoParameters(
            lam=0.1,
            learning_rate=0.01,
            num_iters=max_iters,
            tol=tight_tol,
            momentum=0.9,
        )
        w = fit_lasso(X, y, params=params1).weights

        # Run again with fewer iterations but loose tolerance
        # Should give similar result if early stopping worked
        params2 = LassoParameters(
            lam=0.1, learning_rate=0.01, num_iters=100, tol=1e-4, momentum=0.9
        )
        w2 = fit_lasso(X, y, params=params2).weights

        np.testing.assert_allclose(w, w2, atol=12e-4)

    def test_scaling_with_features(self):
        """Test that algorithm scales reasonably with number of features."""
        np.random.seed(42)
        n_samples = 100

        for n_features in [10, 50, 100]:
            X = jnp.array(np.random.randn(n_samples, n_features))
            y = jnp.array(np.random.randn(n_samples))

            params = LassoParameters(
                lam=0.1, learning_rate=0.01, num_iters=500, tol=1e-6, momentum=0.9
            )
            w = fit_lasso(X, y, params=params).weights
            self.assertEqual(w.shape, (n_features,))
            self.assertFalse(jnp.any(jnp.isnan(w)))


class IntegrationTest(absltest.TestCase):
    """Integration tests for the full LASSO pipeline."""

    def test_full_pipeline(self):
        """Integration test of the full LASSO pipeline."""
        # Generate a realistic regression problem
        np.random.seed(42)
        X, y = make_regression(
            n_samples=200, n_features=100, n_informative=10, noise=0.1, random_state=42
        )

        X = jnp.array(X)
        y = jnp.array(y)

        # Standardize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / y.std()

        # Run LASSO regression
        params = LassoParameters(
            lam=0.05, learning_rate=0.01, num_iters=2000, tol=1e-6, momentum=0.9
        )
        w = fit_lasso(X, y, params=params).weights

        # Check basic properties
        self.assertEqual(w.shape, (100,))
        self.assertFalse(jnp.any(jnp.isnan(w)))
        self.assertFalse(jnp.any(jnp.isinf(w)))

        # Check sparsity (should have selected subset of features)
        n_nonzero = jnp.sum(jnp.abs(w) > 1e-4)
        self.assertBetween(int(n_nonzero), 5, 50)  # Reasonable sparsity

        # Check prediction quality
        y_pred = X @ w
        r2 = 1 - jnp.sum((y - y_pred) ** 2) / jnp.sum((y - y.mean()) ** 2)
        self.assertGreater(float(r2), 0.5)  # Should explain at least 50% of variance


class RegularizationPathTest(parameterized.TestCase):
    """Test regularization path properties."""

    @parameterized.parameters((0.001, 0.01, 0.1, 1.0))
    def test_regularization_path(self, *lambdas):
        """Test that sparsity increases with lambda."""
        np.random.seed(42)
        n_samples, n_features = 100, 20
        X = jnp.array(np.random.randn(n_samples, n_features))
        y = jnp.array(np.random.randn(n_samples))

        sparsities = []
        for lam in lambdas:
            params = LassoParameters(
                lam=lam, learning_rate=0.01, num_iters=1000, tol=1e-6, momentum=0.9
            )
            w = fit_lasso(X, y, params=params).weights
            sparsity = jnp.sum(jnp.abs(w) < 1e-6) / n_features
            sparsities.append(float(sparsity))

        # Sparsity should be non-decreasing with lambda
        for i in range(len(sparsities) - 1):
            self.assertLessEqual(
                sparsities[i], sparsities[i + 1] + 0.1
            )  # Allow small tolerance


if __name__ == "__main__":
    absltest.main()
