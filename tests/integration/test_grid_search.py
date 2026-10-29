"""Integration tests for the grid search module.

Tests the generalized vmap-based grid search functions for fitting metalog
distributions across multiple datasets, hyperparameters, and term counts.
"""
# Copyright: Travis Jefferies 2026

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from scipy.stats import beta, lognorm

from metalog_jax.base import (
    MetalogBaseData,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
)
from metalog_jax.feasibility import (
    TermOrder,
    best_feasible_fit,
    check_feasibility,
    get_engine,
)
from metalog_jax.grid_search import (
    extract_best_from_grid,
    extract_metalog,
    find_best_config,
    fit_grid,
    fit_grid_datasets,
    fit_grid_datasets_hyperparams,
    fit_grid_datasets_num_terms,
    fit_grid_full,
    fit_grid_hyperparams,
    fit_grid_num_terms,
    make_batch,
    unvmap,
)
from metalog_jax.metalog import GridResult, Metalog, fit
from metalog_jax.regression import DEFAULT_LASSO_PARAMETERS, LassoParameters
from metalog_jax.utils import DEFAULT_Y, ks_distance


class TestMakeBatch(absltest.TestCase):
    """Tests for make_batch function."""

    def test_batches_multiple_datasets(self):
        """Test that make_batch correctly stacks multiple datasets."""
        # Create individual datasets
        data1 = MetalogBaseData(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([0.25, 0.5, 0.75]),
            precomputed_quantiles=True,
        )
        data2 = MetalogBaseData(
            x=jnp.array([4.0, 5.0, 6.0]),
            y=jnp.array([0.25, 0.5, 0.75]),
            precomputed_quantiles=True,
        )

        batched = make_batch([data1, data2])

        self.assertEqual(batched.x.shape, (2, 3))
        self.assertEqual(batched.y.shape, (2, 3))
        self.assertTrue(jnp.allclose(batched.x[0], data1.x))
        self.assertTrue(jnp.allclose(batched.x[1], data2.x))


class TestUnvmap(absltest.TestCase):
    """Tests for unvmap function."""

    def test_converts_batched_to_list(self):
        """Test that unvmap correctly unbatches output."""
        # Create a simple batched GridResult
        batched = GridResult(
            metalog=None,  # Will use a simpler structure for test
            ks_dist=jnp.array([0.1, 0.2, 0.3]),
        )

        result = unvmap(batched)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(float(result[0].ks_dist), 0.1, places=5)
        self.assertAlmostEqual(float(result[1].ks_dist), 0.2, places=5)
        self.assertAlmostEqual(float(result[2].ks_dist), 0.3, places=5)


class TestFitGridHyperparams(absltest.TestCase):
    """Tests for fit_grid_hyperparams function - vmap over hyperparameters."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        self.data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.base_data = MetalogBaseData(
            x=self.data.x,
            y=self.data.y,
            precomputed_quantiles=self.data.precomputed_quantiles,
        )

    def test_fit_multiple_l1_penalties(self):
        """Test fitting with multiple L1 penalties."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        l1_penalties = jnp.array([0.0, 0.1, 1.0, 10.0])

        results = fit_grid_hyperparams(
            data=self.base_data,
            params=params,
            l1_penalties=l1_penalties,
        )

        # Check output shape
        self.assertEqual(results.metalog.a.shape, (4, 7))
        self.assertEqual(results.ks_dist.shape, (4,))

        # Check that all KS distances are reasonable
        self.assertTrue(jnp.all(results.ks_dist >= 0))
        self.assertTrue(jnp.all(results.ks_dist < 1))

    def test_regularization_shrinks_coefficients(self):
        """Test that higher L1 penalty shrinks coefficients."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        l1_penalties = jnp.array([0.0, 10.0, 100.0])

        results = fit_grid_hyperparams(
            data=self.base_data,
            params=params,
            l1_penalties=l1_penalties,
        )

        # Coefficient norms should generally decrease with higher penalty
        norms = jnp.linalg.norm(results.metalog.a, axis=1)
        # Allow some tolerance due to fitting dynamics
        self.assertGreater(float(norms[0]), float(norms[2]) * 0.5)


class TestFitGridDatasets(absltest.TestCase):
    """Tests for fit_grid_datasets function - vmap over datasets."""

    def setUp(self):
        """Set up test data with multiple datasets."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)
        dist3 = beta(a=3.5, b=3.5)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)
        rv3 = dist3.rvs(size=100, random_state=44)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)
        data3 = MetalogInputData.from_values(rv3, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x, data3.x])
        self.batched_y = jnp.stack([data1.y, data2.y, data3.y])

    def test_fit_multiple_datasets_ols(self):
        """Test fitting multiple datasets with OLS."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=7,
            # One beta dataset's 7-term fit is feasible only in the 2016 order
            term_order=TermOrder.KEELIN_2016,
        )

        results = fit_grid_datasets(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
        )

        # Check output shape
        self.assertEqual(results.metalog.a.shape, (3, 7))
        self.assertEqual(results.ks_dist.shape, (3,))

        # Check that all KS distances are reasonable
        self.assertTrue(jnp.all(results.ks_dist >= 0))
        self.assertTrue(jnp.all(results.ks_dist < 1))

    def test_fit_multiple_datasets_lasso(self):
        """Test fitting multiple datasets with Lasso regularization."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        lasso_params = LassoParameters(
            lam=1.0,
            num_iters=500,
            tol=1e-6,
        )

        results = fit_grid_datasets(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            fit_params=lasso_params,
        )

        # Check output shape
        self.assertEqual(results.metalog.a.shape, (3, 7))
        self.assertEqual(results.ks_dist.shape, (3,))

    def test_datasets_produce_different_results(self):
        """Test that different datasets produce different coefficients."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=7,
            # One beta dataset's 7-term fit is feasible only in the 2016 order
            term_order=TermOrder.KEELIN_2016,
        )

        results = fit_grid_datasets(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
        )

        # Coefficients should differ across datasets
        self.assertFalse(jnp.allclose(results.metalog.a[0], results.metalog.a[1]))
        self.assertFalse(jnp.allclose(results.metalog.a[1], results.metalog.a[2]))


class TestFitGridDatasetsHyperparams(absltest.TestCase):
    """Tests for fit_grid_datasets_hyperparams - 2D grid over datasets x hyperparams."""

    def setUp(self):
        """Set up test data with multiple datasets."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)
        dist3 = beta(a=3.5, b=3.5)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)
        rv3 = dist3.rvs(size=100, random_state=44)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)
        data3 = MetalogInputData.from_values(rv3, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x, data3.x])
        self.batched_y = jnp.stack([data1.y, data2.y, data3.y])

    def test_2d_grid_shape(self):
        """Test that 2D grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
            # One beta dataset's 7-term fit is feasible only in the 2016 order
            term_order=TermOrder.KEELIN_2016,
        )
        l1_penalties = jnp.array([0.0, 1.0, 10.0])

        results = fit_grid_datasets_hyperparams(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            l1_penalties=l1_penalties,
        )

        # Shape: (n_datasets, n_penalties, num_terms)
        self.assertEqual(results.metalog.a.shape, (3, 3, 7))
        # Shape: (n_datasets, n_penalties)
        self.assertEqual(results.ks_dist.shape, (3, 3))

    def test_2d_grid_ks_distances_reasonable(self):
        """Test that all KS distances are reasonable values."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
            # One beta dataset's 7-term fit is feasible only in the 2016 order
            term_order=TermOrder.KEELIN_2016,
        )
        l1_penalties = jnp.array([0.0, 1.0])

        results = fit_grid_datasets_hyperparams(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            l1_penalties=l1_penalties,
        )

        self.assertTrue(jnp.all(results.ks_dist >= 0))
        self.assertTrue(jnp.all(results.ks_dist < 1))


class TestFitGridFull(absltest.TestCase):
    """Tests for fit_grid_full - 3D grid over datasets x hyperparams x num_terms."""

    def setUp(self):
        """Set up test data with multiple datasets."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x])
        self.batched_y = jnp.stack([data1.y, data2.y])

    def test_3d_grid_shape(self):
        """Test that 3D grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,  # Will be overridden
        )
        l1_penalties = jnp.array([0.0, 1.0])
        num_terms_list = [5, 7, 9]

        results = fit_grid_full(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            l1_penalties=l1_penalties,
            num_terms_list=num_terms_list,
        )

        # Shape: (n_datasets, n_penalties, n_terms, max_terms)
        max_terms = max(num_terms_list)
        self.assertEqual(results.metalog.a.shape, (2, 2, 3, max_terms))
        # Shape: (n_datasets, n_penalties, n_terms)
        self.assertEqual(results.ks_dist.shape, (2, 2, 3))

    def test_3d_grid_coefficients_padded(self):
        """Test that coefficients are properly padded for different num_terms."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0])
        num_terms_list = [5, 7]

        results = fit_grid_full(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            l1_penalties=l1_penalties,
            num_terms_list=num_terms_list,
        )

        # For 5 terms, last 2 coefficients should be zero (padded)
        coeffs_5_terms = results.metalog.a[
            0, 0, 0
        ]  # First dataset, first penalty, 5 terms
        self.assertEqual(coeffs_5_terms.shape, (7,))
        # Last 2 elements should be zero (padding)
        self.assertTrue(jnp.allclose(coeffs_5_terms[5:], jnp.zeros(2), atol=1e-5))


class TestFitGridNumTerms(absltest.TestCase):
    """Tests for fit_grid_num_terms function - vmap over num_terms for single dataset."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        self.data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.base_data = MetalogBaseData(
            x=self.data.x,
            y=self.data.y,
            precomputed_quantiles=self.data.precomputed_quantiles,
        )

    def test_fit_multiple_num_terms_shape(self):
        """Test that fitting with multiple num_terms returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,  # Will be overridden
        )
        num_terms_list = [5, 7, 9]

        results = fit_grid_num_terms(
            data=self.base_data,
            params=params,
            num_terms_list=num_terms_list,
        )

        # Shape: (n_terms, max_terms)
        max_terms = max(num_terms_list)
        self.assertEqual(results.metalog.a.shape, (3, max_terms))
        # Shape: (n_terms,)
        self.assertEqual(results.ks_dist.shape, (3,))

    def test_fit_multiple_num_terms_ks_reasonable(self):
        """Test that all KS distances are reasonable values."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms_list = [5, 7, 9]

        results = fit_grid_num_terms(
            data=self.base_data,
            params=params,
            num_terms_list=num_terms_list,
        )

        self.assertTrue(jnp.all(results.ks_dist >= 0))
        self.assertTrue(jnp.all(results.ks_dist < 1))

    def test_fit_coefficients_properly_padded(self):
        """Test that coefficients are properly padded for different num_terms."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms_list = [5, 9]

        results = fit_grid_num_terms(
            data=self.base_data,
            params=params,
            num_terms_list=num_terms_list,
        )

        # For 5 terms, last 4 coefficients should be zero (padded to max 9)
        coeffs_5_terms = results.metalog.a[0]
        self.assertEqual(coeffs_5_terms.shape, (9,))
        # Last 4 elements should be zero (padding)
        self.assertTrue(jnp.allclose(coeffs_5_terms[5:], jnp.zeros(4), atol=1e-5))

    def test_more_terms_generally_better_ks(self):
        """Test that more terms generally produces better (lower) KS distance."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=3,
        )
        num_terms_list = [3, 5, 7, 9]

        results = fit_grid_num_terms(
            data=self.base_data,
            params=params,
            num_terms_list=num_terms_list,
        )

        # Generally more terms should improve fit (lower KS)
        # Allow some tolerance as this isn't always monotonic
        self.assertLess(float(results.ks_dist[3]), float(results.ks_dist[0]))


class TestFitGridDatasetsNumTerms(absltest.TestCase):
    """Tests for fit_grid_datasets_num_terms - 2D grid over datasets x num_terms."""

    def setUp(self):
        """Set up test data with multiple datasets."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)
        dist3 = beta(a=3.5, b=3.5)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)
        rv3 = dist3.rvs(size=100, random_state=44)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)
        data3 = MetalogInputData.from_values(rv3, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x, data3.x])
        self.batched_y = jnp.stack([data1.y, data2.y, data3.y])

    def test_2d_grid_shape(self):
        """Test that 2D grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,  # Will be overridden
        )
        num_terms_list = [5, 7, 9]

        results = fit_grid_datasets_num_terms(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            num_terms_list=num_terms_list,
        )

        # Shape: (n_datasets, n_terms, max_terms)
        max_terms = max(num_terms_list)
        self.assertEqual(results.metalog.a.shape, (3, 3, max_terms))
        # Shape: (n_datasets, n_terms)
        self.assertEqual(results.ks_dist.shape, (3, 3))

    def test_2d_grid_ks_distances_reasonable(self):
        """Test that all KS distances are reasonable values."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms_list = [5, 7]

        results = fit_grid_datasets_num_terms(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            num_terms_list=num_terms_list,
        )

        self.assertTrue(jnp.all(results.ks_dist >= 0))
        self.assertTrue(jnp.all(results.ks_dist < 1))

    def test_datasets_produce_different_results(self):
        """Test that different datasets produce different coefficients."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms_list = [5, 7]

        results = fit_grid_datasets_num_terms(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            num_terms_list=num_terms_list,
        )

        # Coefficients should differ across datasets
        self.assertFalse(jnp.allclose(results.metalog.a[0], results.metalog.a[1]))
        self.assertFalse(jnp.allclose(results.metalog.a[1], results.metalog.a[2]))

    def test_coefficients_properly_padded(self):
        """Test that coefficients are properly padded for different num_terms."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms_list = [5, 9]

        results = fit_grid_datasets_num_terms(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            num_terms_list=num_terms_list,
        )

        # For 5 terms (index 0), last 4 coefficients should be zero (padded to max 9)
        for dataset_idx in range(3):
            coeffs_5_terms = results.metalog.a[dataset_idx, 0]
            self.assertEqual(coeffs_5_terms.shape, (9,))
            # Last 4 elements should be zero (padding)
            self.assertTrue(jnp.allclose(coeffs_5_terms[5:], jnp.zeros(4), atol=1e-5))


class TestFindBestConfig(absltest.TestCase):
    """Tests for find_best_config function."""

    def test_finds_minimum_in_1d(self):
        """Test finding minimum in 1D KS distance array."""
        ks_dists = jnp.array([0.5, 0.1, 0.3, 0.2])

        best_idx, best_ks = find_best_config(ks_dists)

        self.assertEqual(int(best_idx), 1)
        self.assertAlmostEqual(float(best_ks), 0.1, places=5)

    def test_finds_minimum_in_2d(self):
        """Test finding minimum in 2D KS distance grid."""
        ks_dists = jnp.array(
            [
                [0.5, 0.3, 0.4],
                [0.2, 0.1, 0.15],  # Min at [1, 1]
                [0.6, 0.25, 0.35],
            ]
        )

        best_idx, best_ks = find_best_config(ks_dists)

        # best_idx should be (1, 1) for 2D
        self.assertEqual(best_idx[0], 1)
        self.assertEqual(best_idx[1], 1)
        self.assertAlmostEqual(float(best_ks), 0.1, places=5)

    def test_finds_minimum_in_3d(self):
        """Test finding minimum in 3D KS distance grid."""
        ks_dists = jnp.zeros((2, 3, 4))
        ks_dists = ks_dists.at[:].set(1.0)
        ks_dists = ks_dists.at[1, 2, 3].set(0.05)  # Min at [1, 2, 3]

        best_idx, best_ks = find_best_config(ks_dists)

        self.assertEqual(best_idx[0], 1)
        self.assertEqual(best_idx[1], 2)
        self.assertEqual(best_idx[2], 3)
        self.assertAlmostEqual(float(best_ks), 0.05, places=5)


class TestExtractBestFromGrid(absltest.TestCase):
    """Tests for extract_best_from_grid function."""

    def setUp(self):
        """Set up test data."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x])
        self.batched_y = jnp.stack([data1.y, data2.y])

    def test_extract_from_2d_grid(self):
        """Test extracting best result from 2D grid."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        l1_penalties = jnp.array([0.0, 1.0, 10.0])

        results = fit_grid_datasets_hyperparams(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            l1_penalties=l1_penalties,
        )

        # Extract best for first dataset
        best = extract_best_from_grid(results, dataset_idx=0)

        # Should have single metalog coefficients
        self.assertEqual(best.metalog.a.shape, (7,))
        # KS distance should be scalar
        self.assertEqual(best.ks_dist.shape, ())
        # Should be the minimum for that dataset
        expected_min = jnp.min(results.ks_dist[0])
        self.assertAlmostEqual(float(best.ks_dist), float(expected_min), places=5)

    def test_extract_from_3d_grid(self):
        """Test extracting best result from 3D grid."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0])
        num_terms_list = [5, 7]

        results = fit_grid_full(
            batched_x=self.batched_x,
            batched_y=self.batched_y,
            params=params,
            l1_penalties=l1_penalties,
            num_terms_list=num_terms_list,
        )

        # Extract best for first dataset
        best = extract_best_from_grid(results, dataset_idx=0)

        # Should have single metalog coefficients (max_terms)
        max_terms = max(num_terms_list)
        self.assertEqual(best.metalog.a.shape, (max_terms,))
        # KS distance should be scalar
        self.assertEqual(best.ks_dist.shape, ())
        # Should be the minimum for that dataset
        expected_min = jnp.min(results.ks_dist[0])
        self.assertAlmostEqual(float(best.ks_dist), float(expected_min), places=5)


class TestVmapCompatibility(absltest.TestCase):
    """Tests for vmap compatibility of grid search functions."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rv = dist.rvs(size=100, random_state=42)
        self.data = MetalogInputData.from_values(rv, DEFAULT_Y, False)

    def test_fit_grid_hyperparams_produces_valid_results(self):
        """Test that fit_grid_hyperparams produces valid, different results per penalty."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0, 10.0])

        base_data = MetalogBaseData(
            x=self.data.x,
            y=self.data.y,
            precomputed_quantiles=self.data.precomputed_quantiles,
        )

        results = fit_grid_hyperparams(base_data, params, l1_penalties)

        # Verify output shape
        self.assertEqual(results.metalog.a.shape, (3, 5))
        self.assertEqual(results.ks_dist.shape, (3,))

        # Verify coefficients differ across penalties
        # Higher regularization should produce different coefficients
        self.assertFalse(jnp.allclose(results.metalog.a[0], results.metalog.a[2]))

    def test_extract_best_is_vmappable(self):
        """Test that extract_best_from_grid can be vmapped over datasets."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)

        batched_x = jnp.stack([data1.x, data2.x])
        batched_y = jnp.stack([data1.y, data2.y])

        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0])

        results = fit_grid_datasets_hyperparams(
            batched_x=batched_x,
            batched_y=batched_y,
            params=params,
            l1_penalties=l1_penalties,
        )

        # Vmap extract_best_from_grid over all datasets
        vmapped_extract = jax.vmap(
            lambda idx: extract_best_from_grid(results, dataset_idx=idx)
        )
        dataset_indices = jnp.arange(2)
        all_best = vmapped_extract(dataset_indices)

        self.assertEqual(all_best.metalog.a.shape, (2, 5))
        self.assertEqual(all_best.ks_dist.shape, (2,))


class TestExtractMetalog(absltest.TestCase):
    """Tests for extract_metalog function."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.x = data.x
        self.y = data.y
        self.params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )

    def test_extract_from_1d_grid(self):
        """Test extracting metalog from 1D grid over L1 penalties."""
        from metalog_jax.grid_search import extract_metalog

        l1_penalties = jnp.array([0.0, 0.01, 0.1])
        result = fit_grid(self.x, self.y, self.params, l1_penalties=l1_penalties)

        # Extract metalog at index 0
        metalog = extract_metalog(result, 0)

        # Verify it's a usable Metalog with Python-typed fields
        self.assertIsInstance(metalog.metalog_params.num_terms, int)
        self.assertIsInstance(metalog.metalog_params.lower_bound, float)
        self.assertIsInstance(metalog.metalog_params.upper_bound, float)

        # Verify we can use ppf (JIT-compiled method)
        median = metalog.ppf(jnp.array([0.5]))
        self.assertEqual(median.shape, (1,))
        self.assertGreater(float(median[0]), 0)
        self.assertLess(float(median[0]), 1)

    def test_extract_from_2d_grid(self):
        """Test extracting metalog from 2D grid (L1 x num_terms)."""
        from metalog_jax.grid_search import extract_metalog

        l1_penalties = jnp.array([0.0, 0.01])
        num_terms_list = [5, 7]

        result = fit_grid(
            self.x,
            self.y,
            self.params,
            l1_penalties=l1_penalties,
            num_terms=num_terms_list,
        )

        # Grid shape is (len(l1_penalties), len(num_terms_list)) = (2, 2)
        # Extract metalog at index (1, 1) - second L1, second num_terms (7)
        metalog = extract_metalog(result, 1, 1)

        # Verify it's a usable Metalog with Python-typed fields
        self.assertIsInstance(metalog.metalog_params.num_terms, int)
        self.assertEqual(metalog.metalog_params.num_terms, 7)

        # Verify we can use ppf (JIT-compiled method)
        quantiles = metalog.ppf(jnp.array([0.25, 0.5, 0.75]))
        self.assertEqual(quantiles.shape, (3,))

    def test_extract_from_3d_grid(self):
        """Test extracting metalog from 3D grid (datasets x L1 x num_terms)."""
        from metalog_jax.grid_search import extract_metalog

        # Create two datasets
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)
        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)
        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)

        batched_x = jnp.stack([data1.x, data2.x])
        batched_y = jnp.stack([data1.y, data2.y])

        l1_penalties = jnp.array([0.0, 0.01])
        num_terms_list = [5, 7]

        result = fit_grid(
            batched_x,
            batched_y,
            self.params,
            l1_penalties=l1_penalties,
            num_terms=num_terms_list,
        )

        # Extract metalog for dataset 1, L1 index 0, num_terms index 1
        metalog = extract_metalog(result, 1, 0, 1)

        # Verify it's a usable Metalog with Python-typed fields
        self.assertIsInstance(metalog.metalog_params.num_terms, int)
        self.assertEqual(metalog.metalog_params.num_terms, 7)

        # Verify we can use pdf (another JIT-compiled method)
        pdf_vals = metalog.pdf(jnp.array([0.3, 0.5, 0.7]))
        self.assertEqual(pdf_vals.shape, (3,))
        # PDF values should be positive
        self.assertTrue(jnp.all(pdf_vals > 0))

    def test_extract_best_metalog_is_usable(self):
        """Test that extracted best metalog can be used for all distribution methods."""
        from metalog_jax.grid_search import extract_metalog

        l1_penalties = jnp.array([0.0, 0.01, 0.1])
        num_terms_list = [5, 7, 9]

        result = fit_grid(
            self.x,
            self.y,
            self.params,
            l1_penalties=l1_penalties,
            num_terms=num_terms_list,
        )

        # Find best configuration
        best_idx, _ = find_best_config(result.ks_dist)
        best_l1_idx, best_terms_idx = int(best_idx[0]), int(best_idx[1])

        # Extract best metalog
        best_metalog = extract_metalog(result, best_l1_idx, best_terms_idx)

        # Verify all distribution methods work
        ppf_result = best_metalog.ppf(jnp.array([0.1, 0.5, 0.9]))
        self.assertEqual(ppf_result.shape, (3,))

        pdf_result = best_metalog.pdf(jnp.array([0.2, 0.4, 0.6]))
        self.assertEqual(pdf_result.shape, (3,))

        cdf_result = best_metalog.cdf(jnp.array([0.2, 0.4, 0.6]))
        self.assertEqual(cdf_result.shape, (3,))

        # Check median and mean properties
        _ = best_metalog.median
        _ = best_metalog.mean


# =============================================================================
# Tests for unified fit_grid function
# =============================================================================


class TestFitGridSingleDatasetNoGrid(absltest.TestCase):
    """Test fit_grid with single dataset, no grid search (simplest case)."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.x = data.x
        self.y = data.y

    def test_single_dataset_ols(self):
        """Test single dataset with OLS returns scalar KS distance."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(self.x, self.y, params)

        # Scalar KS distance
        self.assertEqual(result.ks_dist.shape, ())
        # Coefficients shape matches num_terms
        self.assertEqual(result.metalog.a.shape, (7,))
        # KS distance is reasonable
        self.assertGreaterEqual(float(result.ks_dist), 0)
        self.assertLess(float(result.ks_dist), 1)

    def test_single_dataset_lasso_with_params(self):
        """Test single dataset with Lasso and fixed params."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )

        result = fit_grid(self.x, self.y, params)

        self.assertEqual(result.ks_dist.shape, ())
        self.assertEqual(result.metalog.a.shape, (7,))


class TestFitGridSingleDatasetNumTermsGrid(absltest.TestCase):
    """Test fit_grid with single dataset, grid over num_terms."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.x = data.x
        self.y = data.y

    def test_num_terms_grid_shape(self):
        """Test that num_terms grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,  # Will be overridden
        )
        num_terms = [5, 7, 9]

        result = fit_grid(self.x, self.y, params, num_terms=num_terms)

        # Shape: (n_terms,) for KS
        self.assertEqual(result.ks_dist.shape, (3,))
        # Shape: (n_terms, max_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (3, 9))

    def test_num_terms_grid_coefficients_padded(self):
        """Test that coefficients are zero-padded for smaller num_terms."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms = [5, 9]

        result = fit_grid(self.x, self.y, params, num_terms=num_terms)

        # For 5 terms, last 4 should be zero
        coeffs_5 = result.metalog.a[0]
        self.assertTrue(jnp.allclose(coeffs_5[5:], jnp.zeros(4), atol=1e-5))

    def test_more_terms_improves_fit(self):
        """Test that more terms generally improves fit."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=3,
        )
        num_terms = [3, 5, 7, 9]

        result = fit_grid(self.x, self.y, params, num_terms=num_terms)

        # More terms should generally improve fit
        self.assertLess(float(result.ks_dist[3]), float(result.ks_dist[0]))


class TestFitGridSingleDatasetL1Grid(absltest.TestCase):
    """Test fit_grid with single dataset, grid over L1 penalties."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.x = data.x
        self.y = data.y

    def test_l1_grid_shape(self):
        """Test that L1 grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        l1_penalties = jnp.array([0.0, 0.1, 1.0, 10.0])

        result = fit_grid(self.x, self.y, params, l1_penalties=l1_penalties)

        # Shape: (n_penalties,) for KS
        self.assertEqual(result.ks_dist.shape, (4,))
        # Shape: (n_penalties, num_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (4, 7))

    def test_l1_grid_regularization_shrinks_coefficients(self):
        """Test that higher L1 shrinks coefficients."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        l1_penalties = jnp.array([0.0, 10.0, 100.0])

        result = fit_grid(self.x, self.y, params, l1_penalties=l1_penalties)

        # Higher penalty should shrink coefficients
        norms = jnp.linalg.norm(result.metalog.a, axis=1)
        self.assertGreater(float(norms[0]), float(norms[2]) * 0.5)


class TestFitGridSingleDatasetBothGrids(absltest.TestCase):
    """Test fit_grid with single dataset, grid over both L1 and num_terms."""

    def setUp(self):
        """Set up test data."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)
        self.x = data.x
        self.y = data.y

    def test_2d_grid_shape(self):
        """Test that 2D grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0, 10.0])
        num_terms = [5, 7, 9]

        result = fit_grid(
            self.x,
            self.y,
            params,
            l1_penalties=l1_penalties,
            num_terms=num_terms,
        )

        # Shape: (n_penalties, n_terms) for KS
        self.assertEqual(result.ks_dist.shape, (3, 3))
        # Shape: (n_penalties, n_terms, max_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (3, 3, 9))

    def test_2d_grid_all_ks_reasonable(self):
        """Test that all KS distances are reasonable."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0])
        num_terms = [5, 7]

        result = fit_grid(
            self.x,
            self.y,
            params,
            l1_penalties=l1_penalties,
            num_terms=num_terms,
        )

        self.assertTrue(jnp.all(result.ks_dist >= 0))
        self.assertTrue(jnp.all(result.ks_dist < 1))


class TestFitGridBatchedDatasetsNoGrid(absltest.TestCase):
    """Test fit_grid with batched datasets, no hyperparameter grid."""

    def setUp(self):
        """Set up batched test data."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)
        dist3 = beta(a=3.5, b=3.5)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)
        rv3 = dist3.rvs(size=100, random_state=44)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)
        data3 = MetalogInputData.from_values(rv3, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x, data3.x])
        self.batched_y = jnp.stack([data1.y, data2.y, data3.y])

    def test_batched_datasets_shape(self):
        """Test batched datasets returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(self.batched_x, self.batched_y, params)

        # Shape: (n_datasets,) for KS
        self.assertEqual(result.ks_dist.shape, (3,))
        # Shape: (n_datasets, num_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (3, 7))

    def test_batched_datasets_different_results(self):
        """Test that different datasets produce different results."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(self.batched_x, self.batched_y, params)

        # Coefficients should differ across datasets
        self.assertFalse(jnp.allclose(result.metalog.a[0], result.metalog.a[1]))
        self.assertFalse(jnp.allclose(result.metalog.a[1], result.metalog.a[2]))


class TestFitGridBatchedDatasetsNumTermsGrid(absltest.TestCase):
    """Test fit_grid with batched datasets and num_terms grid (2D)."""

    def setUp(self):
        """Set up batched test data."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)
        dist3 = beta(a=3.5, b=3.5)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)
        rv3 = dist3.rvs(size=100, random_state=44)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)
        data3 = MetalogInputData.from_values(rv3, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x, data3.x])
        self.batched_y = jnp.stack([data1.y, data2.y, data3.y])

    def test_2d_grid_shape(self):
        """Test datasets x num_terms returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )
        num_terms = [5, 7, 9]

        result = fit_grid(
            self.batched_x,
            self.batched_y,
            params,
            num_terms=num_terms,
        )

        # Shape: (n_datasets, n_terms) for KS
        self.assertEqual(result.ks_dist.shape, (3, 3))
        # Shape: (n_datasets, n_terms, max_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (3, 3, 9))


class TestFitGridBatchedDatasetsL1Grid(absltest.TestCase):
    """Test fit_grid with batched datasets and L1 grid (2D)."""

    def setUp(self):
        """Set up batched test data."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x])
        self.batched_y = jnp.stack([data1.y, data2.y])

    def test_2d_grid_shape(self):
        """Test datasets x L1 returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )
        l1_penalties = jnp.array([0.0, 1.0, 10.0])

        result = fit_grid(
            self.batched_x,
            self.batched_y,
            params,
            l1_penalties=l1_penalties,
        )

        # Shape: (n_datasets, n_penalties) for KS
        self.assertEqual(result.ks_dist.shape, (2, 3))
        # Shape: (n_datasets, n_penalties, num_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (2, 3, 7))


class TestFitGridFull3D(absltest.TestCase):
    """Test fit_grid with all 3 axes: datasets x L1 x num_terms."""

    def setUp(self):
        """Set up batched test data."""
        dist1 = beta(a=2, b=5)
        dist2 = beta(a=5, b=2)

        rv1 = dist1.rvs(size=100, random_state=42)
        rv2 = dist2.rvs(size=100, random_state=43)

        data1 = MetalogInputData.from_values(rv1, DEFAULT_Y, False)
        data2 = MetalogInputData.from_values(rv2, DEFAULT_Y, False)

        self.batched_x = jnp.stack([data1.x, data2.x])
        self.batched_y = jnp.stack([data1.y, data2.y])

    def test_3d_grid_shape(self):
        """Test full 3D grid returns correct shape."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0])
        num_terms = [5, 7, 9]

        result = fit_grid(
            self.batched_x,
            self.batched_y,
            params,
            l1_penalties=l1_penalties,
            num_terms=num_terms,
        )

        # Shape: (n_datasets, n_penalties, n_terms) for KS
        self.assertEqual(result.ks_dist.shape, (2, 2, 3))
        # Shape: (n_datasets, n_penalties, n_terms, max_terms) for coefficients
        self.assertEqual(result.metalog.a.shape, (2, 2, 3, 9))

    def test_3d_grid_all_ks_reasonable(self):
        """Test that all KS distances are reasonable."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0])
        num_terms = [5, 7]

        result = fit_grid(
            self.batched_x,
            self.batched_y,
            params,
            l1_penalties=l1_penalties,
            num_terms=num_terms,
        )

        self.assertTrue(jnp.all(result.ks_dist >= 0))
        self.assertTrue(jnp.all(result.ks_dist < 1))

    def test_3d_grid_coefficients_padded(self):
        """Test coefficients are properly padded in 3D grid."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0])
        num_terms = [5, 9]

        result = fit_grid(
            self.batched_x,
            self.batched_y,
            params,
            l1_penalties=l1_penalties,
            num_terms=num_terms,
        )

        # For 5 terms (index 0), last 4 should be zero
        coeffs_5 = result.metalog.a[0, 0, 0]  # first dataset, first penalty, 5 terms
        self.assertTrue(jnp.allclose(coeffs_5[5:], jnp.zeros(4), atol=1e-5))


class TestFitGridBoundednessVariants(absltest.TestCase):
    """Test fit_grid with different boundedness settings."""

    def test_unbounded(self):
        """Test with unbounded metalog."""
        from scipy.stats import norm

        dist = norm(loc=50, scale=10)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0,
            upper_bound=0,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(data.x, data.y, params)

        self.assertEqual(result.ks_dist.shape, ())
        self.assertLess(float(result.ks_dist), 0.5)

    def test_strictly_lower_bound(self):
        """Test with lower-bounded metalog."""
        dist = lognorm(s=0.5, loc=0, scale=1)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
            lower_bound=0,
            upper_bound=0,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(data.x, data.y, params)

        self.assertEqual(result.ks_dist.shape, ())
        self.assertLess(float(result.ks_dist), 0.5)

    def test_strictly_upper_bound(self):
        """Test with upper-bounded metalog."""
        # Negative lognormal for upper-bounded
        dist = lognorm(s=0.5, loc=0, scale=1)
        rvs = -dist.rvs(size=100, random_state=42)  # Flip to negative
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.STRICTLY_UPPER_BOUND,
            lower_bound=0,
            upper_bound=0,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(data.x, data.y, params)

        self.assertEqual(result.ks_dist.shape, ())


class TestFitGridPrecomputedQuantiles(absltest.TestCase):
    """Test fit_grid with precomputed quantiles."""

    def test_precomputed_quantiles_single(self):
        """Test with precomputed quantiles (single dataset)."""
        # Precomputed quantiles from a known distribution
        quantiles = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        probs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )

        result = fit_grid(quantiles, probs, params, precomputed_quantiles=True)

        self.assertEqual(result.ks_dist.shape, ())
        self.assertEqual(result.metalog.a.shape, (5,))


class TestFitGridIntegrationWithFindBest(absltest.TestCase):
    """Test fit_grid integration with find_best_config."""

    def test_find_best_on_unified_result(self):
        """Test finding best config from unified fit_grid result."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=3,
        )
        num_terms = [3, 5, 7, 9]

        result = fit_grid(data.x, data.y, params, num_terms=num_terms)

        best_idx, best_ks = find_best_config(result.ks_dist)

        # Best index should be valid
        self.assertGreaterEqual(int(best_idx), 0)
        self.assertLess(int(best_idx), 4)
        # Best KS should be the minimum
        self.assertAlmostEqual(float(best_ks), float(jnp.min(result.ks_dist)), places=5)


class TestFitGridEdgeCases(absltest.TestCase):
    """Test edge cases for fit_grid."""

    def test_single_num_terms_value(self):
        """Test with single num_terms value (degenerate grid)."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.OLS,
            num_terms=7,
        )

        result = fit_grid(data.x, data.y, params, num_terms=[7])

        # Should still have grid dimension of 1
        self.assertEqual(result.ks_dist.shape, (1,))
        self.assertEqual(result.metalog.a.shape, (1, 7))

    def test_single_l1_penalty_value(self):
        """Test with single L1 penalty value (degenerate grid)."""
        dist = beta(a=2, b=5)
        rvs = dist.rvs(size=100, random_state=42)
        data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=7,
        )

        result = fit_grid(data.x, data.y, params, l1_penalties=jnp.array([0.1]))

        # Should still have grid dimension of 1
        self.assertEqual(result.ks_dist.shape, (1,))
        self.assertEqual(result.metalog.a.shape, (1, 7))


class TestGridFeasibleMethod(absltest.TestCase):
    """Grid search with MetalogFitMethod.Feasible: rejected by vmapped grids."""

    def setUp(self):
        """Paper data, whose 5-term OLS fit is infeasible."""
        x = jnp.array([1.0, 2.0, 4.0, 8.0, 12.0])
        y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.data = MetalogBaseData(x=x, y=y, precomputed_quantiles=True)
        self.batched_x = jnp.stack([x, x + 1.0])
        self.batched_y = jnp.stack([y, y])
        self.params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            method=MetalogFitMethod.Feasible,
            num_terms=5,
        )

    def test_fit_grid_rejects_feasible(self):
        """fit_grid raises instead of silently fitting OLS."""
        with self.assertRaises(ValueError) as ctx:
            fit_grid(
                self.batched_x, self.batched_y, self.params, precomputed_quantiles=True
            )
        self.assertIn("fit_grid_datasets()", str(ctx.exception))

    def test_num_terms_grids_reject_feasible(self):
        """fit_grid_num_terms and fit_grid_datasets_num_terms raise ValueError."""
        with self.assertRaises(ValueError):
            fit_grid_num_terms(self.data, self.params, [3, 4, 5])
        with self.assertRaises(ValueError):
            fit_grid_datasets_num_terms(
                self.batched_x, self.batched_y, self.params, [3, 4, 5], True
            )

    def test_fit_grid_datasets_supports_feasible(self):
        """fit_grid_datasets, the suggested alternative, returns feasible fits."""
        result = fit_grid_datasets(
            self.batched_x, self.batched_y, self.params, precomputed_quantiles=True
        )
        self.assertEqual(result.metalog.a.shape, (2, 5))
        for i in range(2):
            expected = best_feasible_fit(
                self.batched_x[i], self.batched_y[i], num_terms=5
            )
            self.assertTrue(jnp.allclose(result.metalog.a[i], expected.a_star))
            self.assertTrue(
                check_feasibility(get_engine(5), result.metalog.a[i]).feasible
            )


class TestGridScoresAtDataProbabilities(absltest.TestCase):
    """Every grid function scores its fits at the dataset's own probability levels."""

    # A probability grid other than DEFAULT_Y
    Y = tuple(round(0.05 * i, 2) for i in range(1, 20))
    L1 = (0.0, 0.01)
    NUM_TERMS = (3, 4)

    def _params(self, method):
        """Return unbounded parameters for the given fit method."""
        return MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=method,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=3,
        )

    def _assert_scored_at_y(self, result, x, y):
        """Check each KS distance against the fit's own quantiles at y."""
        a = result.metalog.a.reshape(-1, result.metalog.a.shape[-1])
        ks = result.ks_dist.reshape(-1)
        for a_i, ks_i in zip(a, ks, strict=True):
            params = self._params(MetalogFitMethod.OLS).replace(num_terms=a_i.shape[0])
            # _ppf, not ppf, so infeasible grid fits are scored rather than rejected
            fitted = Metalog(params, a_i)._ppf(y, a_i.shape[0])
            self.assertAlmostEqual(float(ks_i), float(ks_distance(x, fitted)))

    def test_all_grid_functions_score_at_y(self):
        """Each grid function's KS distances use y, not DEFAULT_Y."""
        y = jnp.asarray(self.Y)
        # Lognormal quantiles aren't an exact metalog, so no fitted value ties with
        # the data and rounding can't change a KS distance
        x = jnp.asarray(lognorm(s=0.5).ppf(self.Y))
        bx, by = jnp.stack([x, x]), jnp.stack([y, y])
        data = MetalogInputData.from_values(x, y, precomputed_quantiles=True)
        ols = self._params(MetalogFitMethod.OLS)
        lasso = self._params(MetalogFitMethod.Lasso)
        l1 = jnp.asarray(self.L1)
        terms = list(self.NUM_TERMS)
        results = {
            "fit_grid": fit_grid(
                x, y, ols, num_terms=terms, precomputed_quantiles=True
            ),
            "fit_grid_num_terms": fit_grid_num_terms(data, ols, terms),
            "fit_grid_datasets_num_terms": fit_grid_datasets_num_terms(
                bx, by, ols, terms, precomputed_quantiles=True
            ),
            "fit_grid_full": fit_grid_full(
                bx, by, lasso, l1, terms, precomputed_quantiles=True
            ),
            "fit_grid_datasets": fit_grid_datasets(
                bx, by, ols, precomputed_quantiles=True
            ),
            "fit_grid_hyperparams": fit_grid_hyperparams(data, lasso, l1),
            "fit_grid_datasets_hyperparams": fit_grid_datasets_hyperparams(
                bx, by, lasso, l1, precomputed_quantiles=True
            ),
        }
        for name, result in results.items():
            with self.subTest(function=name):
                self._assert_scored_at_y(result, x, y)


def _unbounded(method: MetalogFitMethod, num_terms: int = 7) -> MetalogParameters:
    """Return unbounded parameters for the given fit method and term count."""
    return MetalogParameters(
        boundedness=MetalogBoundedness.UNBOUNDED,
        method=method,
        lower_bound=0.0,
        upper_bound=0.0,
        num_terms=num_terms,
    )


def _assert_coefficients_close(actual, expected):
    """Assert two coefficient arrays agree to rounding, relative to their size."""
    expected = np.asarray(expected)
    atol = 1e-9 * np.max(np.abs(expected))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=atol)


class TestFitGridLassoParams(absltest.TestCase):
    """fit_grid fits with lasso_params (#50)."""

    DATA = MetalogInputData.from_values(
        jnp.asarray(np.random.default_rng(0).gamma(2.0, size=300)), DEFAULT_Y, False
    )
    LASSO = _unbounded(MetalogFitMethod.Lasso)
    LAM = 0.1

    def test_lasso_params_set_the_penalty(self):
        """Without l1_penalties, a Lasso grid fits with lasso_params.lam."""
        expected = fit(self.DATA, self.LASSO, LassoParameters(lam=self.LAM)).a
        result = fit_grid(
            self.DATA.x,
            self.DATA.y,
            self.LASSO,
            lasso_params=LassoParameters(lam=self.LAM),
        )
        unpenalized = fit_grid(self.DATA.x, self.DATA.y, self.LASSO)

        _assert_coefficients_close(result.metalog.a, expected)
        self.assertFalse(np.allclose(unpenalized.metalog.a, expected))

    def test_default_is_default_lasso_parameters(self):
        """With lasso_params=None, a Lasso grid fits with DEFAULT_LASSO_PARAMETERS."""
        expected = fit(self.DATA, self.LASSO, DEFAULT_LASSO_PARAMETERS).a
        result = fit_grid(self.DATA.x, self.DATA.y, self.LASSO)
        _assert_coefficients_close(result.metalog.a, expected)

    def test_penalty_grids_use_its_num_iters(self):
        """With l1_penalties, every fit uses lasso_params' num_iters."""
        l1 = jnp.array([self.LAM])
        expected = fit(self.DATA, self.LASSO, LassoParameters(lam=self.LAM)).a
        solved = fit_grid(self.DATA.x, self.DATA.y, self.LASSO, l1_penalties=l1)
        one_step = fit_grid(
            self.DATA.x,
            self.DATA.y,
            self.LASSO,
            l1_penalties=l1,
            lasso_params=LassoParameters(lam=0.0, num_iters=1),
        )

        _assert_coefficients_close(solved.metalog.a[0], expected)
        self.assertFalse(np.allclose(one_step.metalog.a[0], expected))

    def test_num_terms_grids_pass_regression_params(self):
        """The term-count grids fit with regression_params, as fit_grid does."""
        lasso_params = LassoParameters(lam=self.LAM)
        terms = [5, 7]
        expected = fit_grid(
            self.DATA.x,
            self.DATA.y,
            self.LASSO,
            num_terms=terms,
            lasso_params=lasso_params,
        ).metalog.a
        single = fit_grid_num_terms(self.DATA, self.LASSO, terms, lasso_params)
        batched = fit_grid_datasets_num_terms(
            self.DATA.x[None],
            self.DATA.y[None],
            self.LASSO,
            terms,
            regression_params=lasso_params,
        )

        _assert_coefficients_close(single.metalog.a, expected)
        _assert_coefficients_close(batched.metalog.a[0], expected)
        _assert_coefficients_close(
            expected[1], fit(self.DATA, self.LASSO, lasso_params).a
        )

    def test_lasso_params_for_an_ols_grid_raise(self):
        """Lasso settings for an OLS grid without l1_penalties raise, as in fit()."""
        ols = self.LASSO.replace(method=MetalogFitMethod.OLS)
        lasso_params = LassoParameters(lam=self.LAM)
        with self.assertRaisesRegex(TypeError, "lasso_params only affects Lasso fits"):
            fit_grid(self.DATA.x, self.DATA.y, ols, lasso_params=lasso_params)
        with self.assertRaises(TypeError):
            fit_grid_num_terms(self.DATA, ols, [5, 7], lasso_params)

    def test_lasso_params_for_an_ols_penalty_grid(self):
        """With l1_penalties, OLS parameters take lasso_params too."""
        ols = self.LASSO.replace(method=MetalogFitMethod.OLS)
        result = fit_grid(
            self.DATA.x,
            self.DATA.y,
            ols,
            l1_penalties=jnp.array([self.LAM]),
            lasso_params=LassoParameters(lam=0.0),
        )
        expected = fit(self.DATA, self.LASSO, LassoParameters(lam=self.LAM)).a
        _assert_coefficients_close(result.metalog.a[0], expected)


class TestGridFitMethodLabel(absltest.TestCase):
    """Grids label each fit with the method that made it (#50)."""

    DATA = MetalogInputData.from_values(
        jnp.asarray(np.random.default_rng(0).gamma(2.0, size=300)), DEFAULT_Y, False
    )
    OLS = _unbounded(MetalogFitMethod.OLS)
    L1 = jnp.array([0.0, 0.1])
    TERMS = (3, 5)

    def test_penalty_grids_label_fits_lasso(self):
        """Grids over L1 penalties fit with Lasso, so their fits say Lasso."""
        x, y = self.DATA.x, self.DATA.y
        results = {
            "fit_grid": fit_grid(x, y, self.OLS, l1_penalties=self.L1),
            "fit_grid with num_terms": fit_grid(
                x, y, self.OLS, l1_penalties=self.L1, num_terms=self.TERMS
            ),
            "fit_grid_full": fit_grid_full(
                x[None], y[None], self.OLS, self.L1, self.TERMS
            ),
        }
        for name, result in results.items():
            with self.subTest(function=name):
                index = (0,) * result.ks_dist.ndim
                metalog = extract_metalog(result, *index)
                self.assertEqual(metalog.method, MetalogFitMethod.Lasso)
                self.assertEqual(
                    Metalog.loads(metalog.dumps()).method, MetalogFitMethod.Lasso
                )

    def test_ols_grids_label_fits_ols(self):
        """Without penalties, an OLS grid's fits say OLS."""
        result = fit_grid(self.DATA.x, self.DATA.y, self.OLS, num_terms=self.TERMS)
        self.assertEqual(extract_metalog(result, 0).method, MetalogFitMethod.OLS)

    def test_fit_grid_full_rejects_feasible(self):
        """fit_grid_full raises for Feasible rather than labelling Lasso fits so."""
        feasible = self.OLS.replace(method=MetalogFitMethod.Feasible)
        with self.assertRaisesRegex(ValueError, "fit_grid_datasets"):
            fit_grid_full(
                self.DATA.x[None], self.DATA.y[None], feasible, self.L1, self.TERMS
            )


class TestGridPrecomputedQuantiles(absltest.TestCase):
    """precomputed_quantiles=False fits raw samples by their quantiles (#50)."""

    # 400 samples per dataset, quantiles at the 105 DEFAULT_Y levels
    SAMPLES = jnp.asarray(np.random.default_rng(1).gamma(2.0, size=(2, 400)))
    Y = jnp.broadcast_to(DEFAULT_Y, (2, DEFAULT_Y.shape[0]))
    QUANTILES = jnp.stack(
        [MetalogInputData.from_values(s, DEFAULT_Y, False).x for s in SAMPLES]
    )
    OLS = _unbounded(MetalogFitMethod.OLS)
    LASSO = _unbounded(MetalogFitMethod.Lasso)
    L1 = jnp.array([0.0, 0.01])
    TERMS = (5, 7)

    def _grids(self, x, **flag):
        """Fit x with each grid function that takes arrays, passing flag through."""
        y, ols, lasso = self.Y, self.OLS, self.LASSO
        on_infeasible = {"on_infeasible": "flag"}
        return {
            "fit_grid": fit_grid(x[0], y[0], ols, **flag),
            "fit_grid batched": fit_grid(
                x, y, lasso, l1_penalties=self.L1, num_terms=self.TERMS, **flag
            ),
            "fit_grid_datasets_num_terms": fit_grid_datasets_num_terms(
                x, y, ols, self.TERMS, **flag
            ),
            "fit_grid_full": fit_grid_full(x, y, lasso, self.L1, self.TERMS, **flag),
            "fit_grid_datasets": fit_grid_datasets(x, y, ols, **flag, **on_infeasible),
            "fit_grid_datasets_hyperparams": fit_grid_datasets_hyperparams(
                x, y, lasso, self.L1, **flag, **on_infeasible
            ),
        }

    def _assert_same_grids(self, actual, expected):
        """Assert each grid's coefficients and KS distances agree."""
        for name, result in expected.items():
            with self.subTest(function=name):
                _assert_coefficients_close(actual[name].metalog.a, result.metalog.a)
                np.testing.assert_array_equal(actual[name].ks_dist, result.ks_dist)

    def test_raw_samples_are_fitted_by_their_quantiles(self):
        """False computes each dataset's quantiles at y, as from_values does."""
        self._assert_same_grids(
            self._grids(self.SAMPLES, precomputed_quantiles=False),
            self._grids(self.QUANTILES, precomputed_quantiles=True),
        )

    def test_quantiles_are_the_default(self):
        """Without the flag, x is taken as quantiles at y."""
        self._assert_same_grids(
            self._grids(self.QUANTILES),
            self._grids(self.QUANTILES, precomputed_quantiles=True),
        )

    def test_ks_dist_is_float64(self):
        """Every grid's KS distances are float64, the inputs' precision."""
        data = MetalogInputData.from_values(self.QUANTILES[0], DEFAULT_Y, True)
        results = self._grids(self.QUANTILES) | {
            "fit_grid_num_terms": fit_grid_num_terms(data, self.OLS, self.TERMS),
            "fit_grid_hyperparams": fit_grid_hyperparams(
                data, self.LASSO, self.L1, on_infeasible="flag"
            ),
        }
        for name, result in results.items():
            with self.subTest(function=name):
                self.assertEqual(result.ks_dist.dtype, jnp.float64)


class TestKsDistanceDtype(absltest.TestCase):
    """ks_distance returns the inputs' floating-point type (#50)."""

    X = (0.0, 1.0, 2.0)
    Y = (0.5, 1.5, 2.5)

    def test_float64_inputs(self):
        """float64 samples give a float64 distance, to float64 rounding."""
        d = ks_distance(jnp.asarray(self.X), jnp.asarray(self.Y))
        self.assertEqual(d.dtype, jnp.float64)
        # float32 would be about 1e-8 off
        self.assertAlmostEqual(float(d), 1 / 3, delta=1e-15)

    def test_float32_inputs(self):
        """float32 samples give a float32 distance."""
        d = ks_distance(
            jnp.asarray(self.X, dtype=jnp.float32), jnp.asarray(self.Y, jnp.float32)
        )
        self.assertEqual(d.dtype, jnp.float32)

    def test_integer_inputs(self):
        """Integer samples give a float64 distance."""
        d = ks_distance(jnp.arange(3), jnp.arange(1, 4))
        self.assertEqual(d.dtype, jnp.float64)


if __name__ == "__main__":
    absltest.main()
