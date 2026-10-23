# Copyright: Travis Jefferies 2026
"""Regression tests for issue #40: grids mark infeasible fits, and selection skips them."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify

from metalog_jax.base import (
    MetalogBaseData,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
)
from metalog_jax.feasibility import check_feasibility, get_engine
from metalog_jax.grid_search import (
    extract_best_from_grid,
    find_best_config,
    fit_grid,
    fit_grid_datasets,
    fit_grid_datasets_hyperparams,
    fit_grid_datasets_num_terms,
    fit_grid_full,
    fit_grid_hyperparams,
    fit_grid_num_terms,
)
from metalog_jax.metalog import GridResult, Metalog
from metalog_jax.utils import DEFAULT_Y

UNBOUNDED = MetalogBoundedness.UNBOUNDED


def quantiles(samples):
    """Return the samples' quantiles at DEFAULT_Y."""
    return MetalogInputData.from_values(jnp.asarray(samples), DEFAULT_Y, False).x


def params(method, num_terms, boundedness=UNBOUNDED, lower=0.0, upper=0.0):
    """Return MetalogParameters for a grid."""
    return MetalogParameters(
        boundedness=boundedness,
        method=method,
        lower_bound=lower,
        upper_bound=upper,
        num_terms=num_terms,
    )


def certified(a, num_terms):
    """Whether check_feasibility certifies the first num_terms coefficients."""
    return bool(check_feasibility(get_engine(num_terms), a[:num_terms]).feasible)


class GridFeasibilityTest(parameterized.TestCase):
    """Every grid function marks which fits are valid distributions."""

    # The issue's batch: dataset 1's 9-term OLS fit is infeasible, the rest are fine
    GOOD = quantiles(np.random.default_rng(0).normal(size=500))
    BAD = quantiles(np.random.default_rng(0).lognormal(1, 0.5, 200))
    X = jnp.stack([GOOD, BAD, GOOD])
    Y = jnp.stack([DEFAULT_Y] * 3)
    OLS = params(MetalogFitMethod.OLS, 9)
    TERMS = (5, 9)
    EXPECTED = ((True, True), (True, False), (True, True))

    # Quantiles of 3-term metalogs: |a3| / a2 = 1.7 is past THETA_3 = 1.667, so an
    # unpenalized Lasso fit is infeasible and larger penalties shrink it back
    LOGIT = jnp.log(DEFAULT_Y / (1 - DEFAULT_Y))
    LASSO_GOOD = 10 + 2 * LOGIT + 1.0 * (DEFAULT_Y - 0.5) * LOGIT
    LASSO_BAD = 10 + 2 * LOGIT + 3.4 * (DEFAULT_Y - 0.5) * LOGIT
    LASSO = params(MetalogFitMethod.Lasso, 3)
    L1 = jnp.array([0.0, 1.0, 3.0, 10.0])
    LASSO_EXPECTED = ((True, True, True, True), (False, False, True, True))

    def assert_feasible_is_certified(self, result, num_terms):
        """Check feasible against check_feasibility of each fit's own terms."""
        a = np.asarray(result.metalog.a).reshape(-1, result.metalog.a.shape[-1])
        terms = np.broadcast_to(num_terms, result.feasible.shape).reshape(-1)
        expected = [
            certified(jnp.asarray(a_i), int(k)) for a_i, k in zip(a, terms, strict=True)
        ]
        self.assertEqual(np.asarray(result.feasible).reshape(-1).tolist(), expected)

    def test_masked_grids(self):
        """The four masked grids mark dataset 1's 9-term fit infeasible."""
        bad = MetalogBaseData(x=self.BAD, y=DEFAULT_Y, precomputed_quantiles=True)
        terms = list(self.TERMS)
        results = {
            "fit_grid": fit_grid(
                self.X, self.Y, self.OLS, num_terms=terms, precomputed_quantiles=True
            ),
            "fit_grid_datasets_num_terms": fit_grid_datasets_num_terms(
                self.X, self.Y, self.OLS, terms, precomputed_quantiles=True
            ),
        }
        for name, result in results.items():
            with self.subTest(function=name):
                self.assertEqual(
                    result.feasible.tolist(), [list(row) for row in self.EXPECTED]
                )
                self.assert_feasible_is_certified(result, np.array(self.TERMS))
        result = fit_grid_num_terms(bad, self.OLS, terms)
        self.assertEqual(result.feasible.tolist(), [True, False])
        self.assert_feasible_is_certified(result, np.array(self.TERMS))

    def test_full_grid(self):
        """fit_grid_full marks the unpenalized fits of the infeasible metalog."""
        x = jnp.stack([self.LASSO_GOOD, self.LASSO_BAD])
        y = jnp.stack([DEFAULT_Y] * 2)
        result = fit_grid_full(
            x, y, self.LASSO, self.L1, [3, 5], precomputed_quantiles=True
        )
        self.assertEqual(result.feasible.shape, (2, 4, 2))
        self.assertEqual(
            result.feasible[:, :, 0].tolist(),
            [list(row) for row in self.LASSO_EXPECTED],
        )
        self.assert_feasible_is_certified(result, np.array([3, 5]))

    def test_fit_grid_without_axes(self):
        """A single fit has a scalar feasible flag."""
        bad = fit_grid(self.BAD, DEFAULT_Y, self.OLS, precomputed_quantiles=True)
        good = fit_grid(self.GOOD, DEFAULT_Y, self.OLS, precomputed_quantiles=True)
        self.assertEqual(
            (bad.feasible.shape, bool(bad.feasible), bool(good.feasible)),
            ((), False, True),
        )

    @parameterized.parameters(
        MetalogBoundedness.STRICTLY_LOWER_BOUND,
        MetalogBoundedness.STRICTLY_UPPER_BOUND,
        MetalogBoundedness.BOUNDED,
    )
    def test_bounded_grids(self, boundedness):
        """Feasibility of the unbounded M(y) decides every boundedness type."""
        p = params(MetalogFitMethod.OLS, 9, boundedness, 0.0, 30.0)
        terms = [3, 5, 7, 9]
        result = fit_grid(
            self.BAD, DEFAULT_Y, p, num_terms=terms, precomputed_quantiles=True
        )
        self.assert_feasible_is_certified(result, np.array(terms))
        self.assertEqual(result.feasible.tolist(), [True, True, False, False])


class OnInfeasibleTest(parameterized.TestCase):
    """The grids that call fit() raise for an infeasible fit, or flag it."""

    X, Y, OLS = GridFeasibilityTest.X, GridFeasibilityTest.Y, GridFeasibilityTest.OLS
    LASSO, L1 = GridFeasibilityTest.LASSO, GridFeasibilityTest.L1
    LASSO_X = jnp.stack([GridFeasibilityTest.LASSO_GOOD, GridFeasibilityTest.LASSO_BAD])
    LASSO_EXPECTED = GridFeasibilityTest.LASSO_EXPECTED

    def grids(self, on_infeasible):
        """Run the three fit()-based grids on batches with an infeasible fit."""
        bad = MetalogBaseData(
            x=self.LASSO_X[1], y=DEFAULT_Y, precomputed_quantiles=True
        )
        y = jnp.stack([DEFAULT_Y] * 2)
        return {
            "fit_grid_datasets": lambda: fit_grid_datasets(
                self.X,
                self.Y,
                self.OLS,
                precomputed_quantiles=True,
                on_infeasible=on_infeasible,
            ),
            "fit_grid_hyperparams": lambda: fit_grid_hyperparams(
                bad, self.LASSO, self.L1, on_infeasible=on_infeasible
            ),
            "fit_grid_datasets_hyperparams": lambda: fit_grid_datasets_hyperparams(
                self.LASSO_X,
                y,
                self.LASSO,
                self.L1,
                precomputed_quantiles=True,
                on_infeasible=on_infeasible,
            ),
        }

    def test_flag_returns_every_fit(self):
        """on_infeasible='flag' returns every fit with the infeasible ones marked."""
        expected = {
            "fit_grid_datasets": [True, False, True],
            "fit_grid_hyperparams": list(self.LASSO_EXPECTED[1]),
            "fit_grid_datasets_hyperparams": [list(row) for row in self.LASSO_EXPECTED],
        }
        for name, run in self.grids("flag").items():
            with self.subTest(function=name):
                self.assertEqual(run().feasible.tolist(), expected[name])

    def test_raise_names_the_infeasible_fits(self):
        """The default raises one error for the batch, showing the feasibility mask."""
        masks = {
            "fit_grid_datasets": r"\[ True False  True\]",
            "fit_grid_hyperparams": r"\[False False  True  True\]",
            "fit_grid_datasets_hyperparams": r"\[\[ True  True  True  True\]\s+\[False False  True  True\]\]",
        }
        for name, run in self.grids("raise").items():
            with self.subTest(function=name):
                with self.assertRaisesRegex(
                    checkify.JaxRuntimeError, "feasible = " + masks[name]
                ):
                    run()

    def test_raise_is_a_value_error(self):
        """checkify.JaxRuntimeError is a ValueError, as before."""
        with self.assertRaises(ValueError):
            self.grids("raise")["fit_grid_datasets"]()

    def test_feasible_batches_still_return(self):
        """With every fit feasible, the default returns, all marked feasible."""
        x = jnp.stack([GridFeasibilityTest.GOOD] * 2)
        result = fit_grid_datasets(x, self.Y[:2], self.OLS, precomputed_quantiles=True)
        self.assertEqual(result.feasible.tolist(), [True, True])

    def test_flag_keeps_the_fits(self):
        """Flagged fits are the coefficients and KS distances of the masked grid."""
        flagged = fit_grid_datasets(
            self.X, self.Y, self.OLS, precomputed_quantiles=True, on_infeasible="flag"
        )
        masked = fit_grid_datasets_num_terms(
            self.X, self.Y, self.OLS, [9], precomputed_quantiles=True
        )
        np.testing.assert_allclose(
            flagged.metalog.a, masked.metalog.a[:, 0], rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(flagged.ks_dist, masked.ks_dist[:, 0])

    def test_feasible_method_fits_are_feasible(self):
        """MetalogFitMethod.Feasible fits of the same batch are all valid."""
        feasible = params(MetalogFitMethod.Feasible, 9)
        result = fit_grid_datasets(self.X, self.Y, feasible, precomputed_quantiles=True)
        self.assertEqual(result.feasible.tolist(), [True, True, True])

    @parameterized.parameters("warn", None, True)
    def test_invalid_option_raises(self, on_infeasible):
        """on_infeasible must be 'raise' or 'flag', checked before fitting."""
        for name, run in self.grids(on_infeasible).items():
            with self.subTest(function=name):
                with self.assertRaisesRegex(
                    ValueError, "on_infeasible must be 'raise' or 'flag'"
                ):
                    run()


class FeasibleSelectionTest(absltest.TestCase):
    """find_best_config and extract_best_from_grid skip infeasible fits."""

    def setUp(self):
        """Fit the issue's grid and flag fit_grid_datasets on the same batch."""
        super().setUp()
        g = GridFeasibilityTest
        self.grid = fit_grid_datasets_num_terms(
            g.X, g.Y, g.OLS, list(g.TERMS), precomputed_quantiles=True
        )
        self.flagged = fit_grid_datasets(
            g.X, g.Y, g.OLS, precomputed_quantiles=True, on_infeasible="flag"
        )

    def test_extract_best_skips_the_infeasible_fit(self):
        """Dataset 1's best is its 5-term fit, though the 9-term fit has a lower KS."""
        self.assertLess(float(self.grid.ks_dist[1, 1]), float(self.grid.ks_dist[1, 0]))
        best = extract_best_from_grid(self.grid, 1)
        self.assertTrue(bool(best.feasible))
        self.assertAlmostEqual(float(best.ks_dist), float(self.grid.ks_dist[1, 0]))
        self.assertAlmostEqual(float(best.ks_dist), 0.0476, places=4)
        np.testing.assert_array_equal(best.metalog.a, self.grid.metalog.a[1, 0])
        # The selected fit is usable
        model = Metalog(GridFeasibilityTest.OLS, best.metalog.a)
        self.assertTrue(bool(jnp.isfinite(model.ppf(jnp.array([0.5]))[0])))

    def test_extract_best_keeps_feasible_lower_ks(self):
        """Datasets whose best fit is feasible keep it."""
        for i in (0, 2):
            best = extract_best_from_grid(self.grid, i)
            self.assertAlmostEqual(float(best.ks_dist), float(self.grid.ks_dist[i, 1]))

    def test_extract_best_raises_when_no_fit_is_feasible(self):
        """A dataset with no feasible fit raises a ValueError naming it."""
        with self.assertRaisesRegex(
            ValueError, "No configuration for dataset 1 is feasible"
        ):
            extract_best_from_grid(self.flagged, 1)

    def test_extract_best_under_vmap(self):
        """Under vmap, a dataset without a feasible fit gets an infinite KS distance."""
        best = jax.vmap(lambda i: extract_best_from_grid(self.flagged, i))(
            jnp.arange(3)
        )
        self.assertEqual(best.feasible.tolist(), [True, False, True])
        self.assertEqual(np.isinf(best.ks_dist).tolist(), [False, True, False])

    def test_find_best_config_with_mask(self):
        """With a mask, infeasible fits count as infinitely far."""
        ks = jnp.array([0.1, 0.05, 0.2])
        feasible = jnp.array([True, False, True])
        idx, best_ks = find_best_config(ks, feasible)
        self.assertEqual((int(idx), float(best_ks)), (0, 0.1))
        idx, best_ks = find_best_config(ks)  # without a mask, as before
        self.assertEqual((int(idx), float(best_ks)), (1, 0.05))

    def test_find_best_config_on_a_grid(self):
        """On the issue's grid, the best feasible fit is dataset 0's or 2's 9-term fit."""
        idx, best_ks = find_best_config(self.grid.ks_dist, self.grid.feasible)
        self.assertTrue(bool(self.grid.feasible[tuple(idx)]))
        self.assertAlmostEqual(float(best_ks), float(self.grid.ks_dist[0, 1]))

    def test_find_best_config_raises_when_no_fit_is_feasible(self):
        """No feasible fit at all raises; under jit, best_ks is infinite instead."""
        ks, none = jnp.array([0.1, 0.2]), jnp.array([False, False])
        with self.assertRaisesRegex(
            ValueError, "No configuration in the grid is feasible"
        ):
            find_best_config(ks, none)
        _, best_ks = jax.jit(find_best_config)(ks, none)
        self.assertTrue(bool(jnp.isinf(best_ks)))

    def test_grid_result_without_feasible(self):
        """A GridResult built without feasible selects as before."""
        result = GridResult(metalog=self.grid.metalog, ks_dist=self.grid.ks_dist)
        self.assertIsNone(result.feasible)
        best = extract_best_from_grid(result, 1)
        self.assertIsNone(best.feasible)
        self.assertAlmostEqual(float(best.ks_dist), float(self.grid.ks_dist[1, 1]))


if __name__ == "__main__":
    absltest.main()
