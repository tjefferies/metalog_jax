"""Integration tests for the grid search module.

Tests fit_grid, the one grid search function, which fits metalog distributions
across datasets, L1 penalties and term counts, and the functions that select from
its results.
"""
# Copyright: Travis Jefferies 2026

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify
from scipy.stats import beta, gamma, lognorm

from metalog_jax import grid_search
from metalog_jax.base import (
    MetalogBase,
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
    MAX_TERMS,
    extract_best_from_grid,
    extract_metalog,
    find_best_config,
    fit_grid,
    make_batch,
    unvmap,
)
from metalog_jax.metalog import GridResult, Metalog, fit
from metalog_jax.regression import DEFAULT_LASSO_PARAMETERS, LassoParameters
from metalog_jax.regression.base import has_full_column_rank
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

        results = fit_grid(
            self.batched_x, self.batched_y, params, l1_penalties=l1_penalties
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

        results = fit_grid(
            self.batched_x,
            self.batched_y,
            params,
            l1_penalties=l1_penalties,
            num_terms=num_terms_list,
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

    def test_penalty_grid_produces_valid_results(self):
        """Test that a penalty grid produces valid, different results per penalty."""
        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0,
            upper_bound=1,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        l1_penalties = jnp.array([0.0, 1.0, 10.0])

        results = fit_grid(self.data.x, self.data.y, params, l1_penalties=l1_penalties)

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

        results = fit_grid(batched_x, batched_y, params, l1_penalties=l1_penalties)

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
    """fit_grid fits MetalogFitMethod.Feasible over datasets and term counts."""

    def setUp(self):
        """Paper data, whose 5-term OLS fit is infeasible."""
        x = jnp.array([1.0, 2.0, 4.0, 8.0, 12.0])
        y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.batched_x = jnp.stack([x, x + 1.0])
        self.batched_y = jnp.stack([y, y])
        self.params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            method=MetalogFitMethod.Feasible,
            num_terms=5,
        )

    def test_datasets_are_best_feasible_fits(self):
        """A batch of Feasible fits is each dataset's best feasible fit."""
        result = fit_grid(self.batched_x, self.batched_y, self.params)
        self.assertEqual(result.metalog.a.shape, (2, 5))
        self.assertEqual(result.feasible.tolist(), [True, True])
        for i in range(2):
            expected = best_feasible_fit(
                self.batched_x[i], self.batched_y[i], num_terms=5
            )
            self.assertTrue(jnp.allclose(result.metalog.a[i], expected.a_star))
            self.assertTrue(
                check_feasibility(get_engine(5), result.metalog.a[i]).feasible
            )

    def test_term_counts_are_padded_fits(self):
        """Each term count's fits are its own grid's, zero-padded to the largest."""
        terms = [3, 5, 4]
        result = fit_grid(self.batched_x, self.batched_y, self.params, num_terms=terms)
        self.assertEqual(result.metalog.a.shape, (2, 3, 5))
        self.assertEqual(result.ks_dist.shape, (2, 3))
        self.assertTrue(bool(jnp.all(result.feasible)))
        for j, k in enumerate(terms):
            single = fit_grid(
                self.batched_x, self.batched_y, self.params.replace(num_terms=k)
            )
            np.testing.assert_array_equal(result.metalog.a[:, j, :k], single.metalog.a)
            np.testing.assert_array_equal(result.metalog.a[:, j, k:], 0.0)
            np.testing.assert_array_equal(result.ks_dist[:, j], single.ks_dist)

    def test_one_dataset(self):
        """One dataset's term-count grid has the term axis only."""
        result = fit_grid(
            self.batched_x[0], self.batched_y[0], self.params, num_terms=[3, 5]
        )
        self.assertEqual(result.metalog.a.shape, (2, 5))
        self.assertEqual(result.feasible.tolist(), [True, True])
        best = extract_metalog(result, 1)
        self.assertEqual(best.method, MetalogFitMethod.Feasible)
        self.assertTrue(bool(jnp.isfinite(best.ppf(jnp.array([0.5]))[0])))

    def test_repeated_term_counts(self):
        """A repeated term count gives the same fits in each of its entries."""
        result = fit_grid(
            self.batched_x, self.batched_y, self.params, num_terms=[5, 3, 5]
        )
        np.testing.assert_array_equal(result.metalog.a[:, 0], result.metalog.a[:, 2])

    def test_l1_penalties_raise(self):
        """Feasible takes no penalty, so an L1 axis raises TypeError."""
        with self.assertRaisesRegex(TypeError, "takes no penalty"):
            fit_grid(
                self.batched_x,
                self.batched_y,
                self.params,
                l1_penalties=jnp.array([0.0, 0.1]),
            )

    def test_lasso_params_raise(self):
        """Feasible takes no Lasso settings, as in fit()."""
        with self.assertRaisesRegex(TypeError, "lasso_params only affects Lasso"):
            fit_grid(
                self.batched_x,
                self.batched_y,
                self.params,
                lasso_params=LassoParameters(lam=0.1),
            )


class TestGridScoresAtDataProbabilities(absltest.TestCase):
    """Every grid scores its fits at the dataset's own probability levels."""

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

    def test_every_grid_scores_at_y(self):
        """Every configuration's KS distances use y, not DEFAULT_Y."""
        y = jnp.asarray(self.Y)
        # Lognormal quantiles aren't an exact metalog, so no fitted value ties with
        # the data and rounding can't change a KS distance
        x = jnp.asarray(lognorm(s=0.5).ppf(self.Y))
        bx, by = jnp.stack([x, x]), jnp.stack([y, y])
        ols = self._params(MetalogFitMethod.OLS)
        lasso = self._params(MetalogFitMethod.Lasso)
        l1 = jnp.asarray(self.L1)
        terms = list(self.NUM_TERMS)
        feasible = self._params(MetalogFitMethod.Feasible)
        results = {
            "terms": fit_grid(x, y, ols, num_terms=terms),
            "datasets x terms": fit_grid(bx, by, ols, num_terms=terms),
            "datasets x penalties x terms": fit_grid(
                bx, by, lasso, l1_penalties=l1, num_terms=terms
            ),
            "datasets": fit_grid(bx, by, ols),
            "penalties": fit_grid(x, y, lasso, l1_penalties=l1),
            "datasets x penalties": fit_grid(bx, by, lasso, l1_penalties=l1),
            "Feasible datasets x terms": fit_grid(bx, by, feasible, num_terms=terms),
        }
        for name, result in results.items():
            with self.subTest(function=name):
                self._assert_scored_at_y(result, x, y)


def _unbounded(
    method: MetalogFitMethod,
    num_terms: int = 7,
    term_order: TermOrder = TermOrder.METALOG_2,
) -> MetalogParameters:
    """Return unbounded parameters for the given fit method, term count and order."""
    return MetalogParameters(
        boundedness=MetalogBoundedness.UNBOUNDED,
        method=method,
        lower_bound=0.0,
        upper_bound=0.0,
        num_terms=num_terms,
        term_order=term_order,
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

    def test_num_terms_grids_use_lasso_params(self):
        """Term-count grids, of one dataset or a batch, fit with lasso_params."""
        lasso_params = LassoParameters(lam=self.LAM)
        terms = [5, 7]
        single = fit_grid(
            self.DATA.x,
            self.DATA.y,
            self.LASSO,
            num_terms=terms,
            lasso_params=lasso_params,
        ).metalog.a
        batched = fit_grid(
            self.DATA.x[None],
            self.DATA.y[None],
            self.LASSO,
            num_terms=terms,
            lasso_params=lasso_params,
        )

        _assert_coefficients_close(batched.metalog.a[0], single)
        _assert_coefficients_close(
            single[1], fit(self.DATA, self.LASSO, lasso_params).a
        )

    def test_lasso_params_for_an_ols_grid_raise(self):
        """Lasso settings for an OLS grid without l1_penalties raise, as in fit()."""
        ols = self.LASSO.replace(method=MetalogFitMethod.OLS)
        lasso_params = LassoParameters(lam=self.LAM)
        with self.assertRaisesRegex(TypeError, "lasso_params only affects Lasso fits"):
            fit_grid(self.DATA.x, self.DATA.y, ols, lasso_params=lasso_params)
        with self.assertRaises(TypeError):
            fit_grid(
                self.DATA.x,
                self.DATA.y,
                ols,
                num_terms=[5, 7],
                lasso_params=lasso_params,
            )

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
            "fit_grid batched": fit_grid(
                x[None], y[None], self.OLS, l1_penalties=self.L1, num_terms=self.TERMS
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

    def test_feasible_grids_label_fits_feasible(self):
        """A Feasible grid's fits say Feasible, and it can't take penalties."""
        feasible = self.OLS.replace(method=MetalogFitMethod.Feasible)
        result = fit_grid(self.DATA.x, self.DATA.y, feasible, num_terms=self.TERMS)
        self.assertEqual(extract_metalog(result, 0).method, MetalogFitMethod.Feasible)
        with self.assertRaises(TypeError):
            fit_grid(self.DATA.x, self.DATA.y, feasible, l1_penalties=self.L1)


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
        """Fit x in each grid configuration, passing flag through."""
        y, ols, lasso = self.Y, self.OLS, self.LASSO
        feasible = ols.replace(method=MetalogFitMethod.Feasible)
        return {
            "one dataset": fit_grid(x[0], y[0], ols, **flag),
            "datasets x penalties x terms": fit_grid(
                x, y, lasso, l1_penalties=self.L1, num_terms=self.TERMS, **flag
            ),
            "datasets x terms": fit_grid(x, y, ols, num_terms=self.TERMS, **flag),
            "datasets": fit_grid(x, y, ols, **flag),
            "datasets x penalties": fit_grid(x, y, lasso, l1_penalties=self.L1, **flag),
            "Feasible datasets x terms": fit_grid(
                x, y, feasible, num_terms=self.TERMS, **flag
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
        x, y = self.QUANTILES[0], DEFAULT_Y
        results = self._grids(self.QUANTILES) | {
            "terms": fit_grid(x, y, self.OLS, num_terms=self.TERMS),
            "penalties": fit_grid(x, y, self.LASSO, l1_penalties=self.L1),
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


class TestGridRank(parameterized.TestCase):
    """OLS and Feasible grids raise for a rank-deficient fit, as fit() does (#55)."""

    # Seven symmetric probabilities: the 2016 order's 7-term basis is singular there
    # (Baucells et al. 2025, Proposition 2), and the 2.0 order's isn't
    SYMMETRIC_Y = jnp.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95])
    ASYMMETRIC_Y = jnp.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95])
    X = jnp.asarray(gamma(a=2).ppf(np.asarray(SYMMETRIC_Y)))

    def assert_raises_for(self, mask, run):
        """Check that run raises the rank error, showing the full_rank mask."""
        with self.assertRaisesRegex(
            checkify.JaxRuntimeError, r"rank deficient.*full_rank = " + mask
        ):
            run()

    @parameterized.parameters(MetalogFitMethod.OLS, MetalogFitMethod.Feasible)
    def test_2016_order_raises_for_its_singular_term_count(self, method):
        """Only the 7-term fit of the 2016 order is singular, and the mask says so."""
        p = _unbounded(method, term_order=TermOrder.KEELIN_2016)
        self.assert_raises_for(
            r"\[ True  True False\]",
            lambda: fit_grid(self.X, self.SYMMETRIC_Y, p, num_terms=[5, 6, 7]),
        )
        # fit() raises for the same fit
        data = MetalogInputData.from_values(self.X, self.SYMMETRIC_Y, True)
        with self.assertRaisesRegex(checkify.JaxRuntimeError, "rank deficient"):
            fit(data, p)

    @parameterized.parameters(MetalogFitMethod.OLS, MetalogFitMethod.Feasible)
    def test_2_0_order_fits_the_same_grid(self, method):
        """The default order has full rank there, so the grid fits."""
        result = fit_grid(
            self.X, self.SYMMETRIC_Y, _unbounded(method), num_terms=[5, 6, 7]
        )
        self.assertEqual(result.ks_dist.shape, (3,))

    def test_lasso_is_not_checked(self):
        """Lasso is regularized, so its singular fits return, as in fit()."""
        p = _unbounded(MetalogFitMethod.Lasso, term_order=TermOrder.KEELIN_2016)
        result = fit_grid(self.X, self.SYMMETRIC_Y, p, num_terms=[5, 7])
        self.assertEqual(result.ks_dist.shape, (2,))

    def test_more_terms_than_quantiles_raise(self):
        """With 7 quantiles, an 8-term fit isn't unique."""
        self.assert_raises_for(
            r"\[ True False\]",
            lambda: fit_grid(
                self.X,
                self.SYMMETRIC_Y,
                _unbounded(MetalogFitMethod.OLS),
                num_terms=[5, 8],
            ),
        )

    def test_mask_has_the_grid_shape(self):
        """With datasets, the mask is datasets x terms, from each dataset's own y."""
        x = jnp.stack([self.X, self.X, self.X])
        y = jnp.stack([self.SYMMETRIC_Y, self.ASYMMETRIC_Y, self.SYMMETRIC_Y])
        p = _unbounded(MetalogFitMethod.OLS, term_order=TermOrder.KEELIN_2016)
        self.assert_raises_for(
            r"\[\[ True False\]\s+\[ True  True\]\s+\[ True False\]\]",
            lambda: fit_grid(x, y, p, num_terms=[6, 7]),
        )
        self.assert_raises_for(
            r"\[False  True False\]", lambda: fit_grid(x, y, p.replace(num_terms=7))
        )

    def test_jitted_grids_report_rank_through_checkify(self):
        """Under jit, y is traced, so every row is checked and checkify reports it."""
        terms = [5, 7]
        for term_order in (TermOrder.METALOG_2, TermOrder.KEELIN_2016):
            p = _unbounded(MetalogFitMethod.OLS, term_order=term_order)
            grid = checkify.checkify(
                jax.jit(lambda x, y, p=p: fit_grid(x, y, p, num_terms=terms))
            )
            err, result = grid(self.X, self.SYMMETRIC_Y)
            with self.subTest(term_order=term_order.name):
                if term_order == TermOrder.KEELIN_2016:
                    self.assertIn("full_rank = [ True False]", err.get())
                else:
                    self.assertIsNone(err.get())
                    eager = fit_grid(self.X, self.SYMMETRIC_Y, p, num_terms=terms)
                    np.testing.assert_array_equal(result.ks_dist, eager.ks_dist)

    @parameterized.parameters(TermOrder.METALOG_2, TermOrder.KEELIN_2016)
    def test_grid_raises_when_fit_does(self, term_order):
        """For n = k symmetric probabilities, a grid raises exactly when fit() does."""
        for k in range(3, 16):
            with self.subTest(num_terms=k):
                y = jnp.linspace(0.5 / k, 1 - 0.5 / k, k)
                x = jnp.asarray(gamma(a=2).ppf(np.asarray(y)))
                p = _unbounded(MetalogFitMethod.OLS, k, term_order)
                design = MetalogBase._get_target(y, k, term_order)
                if bool(has_full_column_rank(design)):
                    fit_grid(x, y, p)
                else:
                    self.assert_raises_for(
                        r"False", lambda x=x, y=y, p=p: fit_grid(x, y, p)
                    )


class TestFitGridArguments(absltest.TestCase):
    """fit_grid is the one grid function, with keyword-only arguments (#55)."""

    X = jnp.asarray(gamma(a=2).ppf(np.asarray(DEFAULT_Y)))
    REMOVED = (
        "fit_grid_datasets",
        "fit_grid_hyperparams",
        "fit_grid_datasets_hyperparams",
        "fit_grid_num_terms",
        "fit_grid_datasets_num_terms",
        "fit_grid_full",
    )

    def test_fit_grid_is_the_only_grid_function(self):
        """The six specialized grid functions are gone."""
        for name in self.REMOVED:
            with self.subTest(function=name):
                self.assertFalse(hasattr(grid_search, name))

    def test_everything_after_params_is_keyword_only(self):
        """Arguments after params are keyword-only, so new ones don't break callers."""
        signature = inspect.signature(fit_grid).parameters
        kinds = {name: p.kind for name, p in signature.items()}
        self.assertEqual(list(kinds)[:3], ["x", "y", "params"])
        for name in list(kinds)[3:]:
            with self.subTest(argument=name):
                self.assertEqual(kinds[name], inspect.Parameter.KEYWORD_ONLY)

    def test_on_infeasible_defaults_to_flag(self):
        """Grids return every fit by default, with the infeasible ones marked."""
        signature = inspect.signature(fit_grid).parameters
        self.assertEqual(signature["on_infeasible"].default, "flag")

    def test_ols_and_lasso_grids_fit_at_most_max_terms(self):
        """The masked design has MAX_TERMS columns, so more terms raise ValueError."""
        for method in (MetalogFitMethod.OLS, MetalogFitMethod.Lasso):
            with self.subTest(method=method.name):
                with self.assertRaisesRegex(ValueError, f"at most {MAX_TERMS} terms"):
                    fit_grid(
                        self.X,
                        DEFAULT_Y,
                        _unbounded(method),
                        num_terms=[5, MAX_TERMS + 1],
                    )


if __name__ == "__main__":
    absltest.main()
