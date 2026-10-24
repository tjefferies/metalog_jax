"""Integration tests for Lasso metalog fits: exact solutions, intercept and grids.

Regression tests for issue #51: fit_lasso ran momentum gradient descent with a fixed
step, so its default settings stopped far from the solution (2.21 from OLS at
lam = 0 for a 5-term fit), larger steps diverged into a misleading density error,
it never reported convergence, and it penalized the intercept a1. Each test checks
against a reference that doesn't use the solver: least squares, scikit-learn's
coordinate descent, or the Lasso's optimality conditions computed in NumPy.
"""
# Copyright: Travis Jefferies 2026

import warnings

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso as SklearnLasso

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
)
from metalog_jax.grid_search import fit_grid, fit_grid_full, fit_grid_hyperparams
from metalog_jax.metalog import Metalog, _fit_coefficients, _get_fit_function, fit
from metalog_jax.regression import LassoParameters, fit_lasso
from metalog_jax.utils import DEFAULT_Y

LASSO = MetalogFitMethod.Lasso


def _params(boundedness, lower, upper, num_terms, method=LASSO):
    """MetalogParameters for the given support and number of terms."""
    return MetalogParameters(
        boundedness=boundedness,
        method=method,
        lower_bound=lower,
        upper_bound=upper,
        num_terms=num_terms,
    )


class LassoFitTestBase(parameterized.TestCase):
    """The issue's log-normal data, and a support of each boundedness type."""

    DATA = MetalogInputData.from_values(
        jnp.asarray(np.random.default_rng(0).lognormal(1, 0.5, 200)),
        DEFAULT_Y,
        precomputed_quantiles=False,
    )
    # The samples run from about 0.8 to 12, inside every support
    SUPPORTS = (
        ("unbounded", MetalogBoundedness.UNBOUNDED, 0.0, 0.0),
        ("strictly_lower_bound", MetalogBoundedness.STRICTLY_LOWER_BOUND, 0.0, 0.0),
        ("strictly_upper_bound", MetalogBoundedness.STRICTLY_UPPER_BOUND, 0.0, 30.0),
        ("bounded", MetalogBoundedness.BOUNDED, 0.0, 30.0),
    )

    def _design(self, boundedness, lower, upper, num_terms):
        """The design matrix and transformed quantiles that fit regresses."""
        p = _params(boundedness, lower, upper, num_terms)
        X = Metalog.get_target(data=self.DATA, metalog_params=p)
        q = Metalog.get_quantiles(data=self.DATA, metalog_params=p)
        return X, q

    @staticmethod
    def _metalog_lasso(X, q, lam):
        """The Lasso fit fit() uses: the intercept a1 unpenalized."""
        return _get_fit_function(LASSO, LassoParameters(lam=lam))(X, q)


class LassoSolutionTest(LassoFitTestBase):
    """Lasso fits are the exact minimizers of the Lasso objective."""

    @parameterized.product(support=LassoFitTestBase.SUPPORTS, num_terms=(3, 5, 9, 12))
    def test_unpenalized_fit_is_least_squares(self, support, num_terms):
        """At lam = 0 the fit is the least-squares fit, the old one 2.21 away."""
        _, boundedness, lower, upper = support
        X, q = self._design(boundedness, lower, upper, num_terms)

        model = self._metalog_lasso(X, q, 0.0)

        ols = np.linalg.lstsq(np.asarray(X), np.asarray(q), rcond=None)[0]
        self.assertTrue(bool(model.converged))
        np.testing.assert_allclose(
            model.weights, ols, rtol=0, atol=1e-10 * np.max(np.abs(ols))
        )

    @parameterized.product(
        support=LassoFitTestBase.SUPPORTS[::3] + LassoFitTestBase.SUPPORTS[1:2],
        num_terms=(5, 9),
        lam=(0.001, 0.01, 0.1),
    )
    def test_penalized_fit_matches_scikit_learn(self, support, num_terms, lam):
        """For lam > 0 the fit matches scikit-learn's Lasso with fit_intercept.

        scikit-learn minimizes the same objective, (1/2n)||y - Xw||² + lam ||w||₁,
        and fit_intercept leaves the intercept unpenalized, as metalog fits do.
        Its coordinate descent runs to tol 1e-14.
        """
        _, boundedness, lower, upper = support
        X, q = self._design(boundedness, lower, upper, num_terms)
        X_np, q_np = np.asarray(X), np.asarray(q)

        model = self._metalog_lasso(X, q, lam)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            reference = SklearnLasso(
                alpha=lam, fit_intercept=True, tol=1e-14, max_iter=1_000_000
            ).fit(X_np[:, 1:], q_np)
        expected = np.concatenate([[reference.intercept_], reference.coef_])

        def objective(w):
            residual = X_np @ w - q_np
            return residual @ residual / (2 * len(q_np)) + lam * np.sum(np.abs(w[1:]))

        np.testing.assert_allclose(
            model.weights, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected))
        )
        self.assertLessEqual(
            objective(np.asarray(model.weights)), objective(expected) + 1e-14
        )

    @parameterized.product(lam=(0.001, 0.01, 0.1), num_terms=(5, 9, 12))
    def test_optimality_conditions_hold(self, lam, num_terms):
        """The KKT conditions hold, computed in NumPy from X and the weights.

        With g = Xᵀ(Xw - q) / n: g_1 = 0 for the unpenalized intercept,
        g_j = -lam sign(w_j) for nonzero coefficients, and |g_j| <= lam for zeros.
        """
        X, q = self._design(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, num_terms)
        X_np, q_np = np.asarray(X), np.asarray(q)

        w = np.asarray(self._metalog_lasso(X, q, lam).weights)

        g = X_np.T @ (X_np @ w - q_np) / len(q_np)
        scale = np.max(np.abs(X_np.T @ q_np)) / len(q_np)
        slack = 1e-9 * scale
        self.assertLess(abs(g[0]), slack)
        nonzero = w[1:] != 0
        np.testing.assert_allclose(
            g[1:][nonzero], -lam * np.sign(w[1:][nonzero]), rtol=0, atol=slack
        )
        self.assertTrue(np.all(np.abs(g[1:][~nonzero]) <= lam + slack))

    def test_intercept_is_unpenalized(self):
        """Shifting the data moves only a1, by the shift.

        With a1 penalized, as before, the penalty depended on the data's
        location, so the shrinkage of the other coefficients did too.
        """
        X, q = self._design(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5)

        base = self._metalog_lasso(X, q, 0.05).weights
        shifted = self._metalog_lasso(X, q + 100.0, 0.05).weights

        np.testing.assert_allclose(shifted[0], base[0] + 100.0, rtol=1e-12)
        np.testing.assert_allclose(shifted[1:], base[1:], rtol=0, atol=1e-10)

    def test_fit_lasso_penalizes_every_column_by_default(self):
        """Without intercept=True, fit_lasso is plain Lasso, as without fit_intercept."""
        X, q = self._design(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5)
        X_np, q_np = np.asarray(X), np.asarray(q)

        model = fit_lasso(X, q, LassoParameters(lam=0.05))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            reference = SklearnLasso(
                alpha=0.05, fit_intercept=False, tol=1e-14, max_iter=1_000_000
            ).fit(X_np, q_np)
        np.testing.assert_allclose(
            model.weights, reference.coef_, rtol=0, atol=1e-8 * np.max(np.abs(q_np))
        )


class LassoDiagnosticsTest(LassoFitTestBase):
    """Lasso fits report convergence, and fit raises a clear error otherwise."""

    def test_reports_convergence_and_steps(self):
        """A fit converges in a handful of steps: about one per active coefficient."""
        X, q = self._design(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 9)

        unpenalized = self._metalog_lasso(X, q, 0.0)
        penalized = self._metalog_lasso(X, q, 0.01)

        self.assertTrue(bool(unpenalized.converged))
        self.assertEqual(int(unpenalized.iterations), 1)
        self.assertTrue(bool(penalized.converged))
        self.assertLessEqual(int(penalized.iterations), 2 * 9)

    def test_step_limit_raises_a_clear_error(self):
        """Stopping at num_iters raises before the density check, saying so."""
        p = _params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5)

        with self.assertRaisesRegex(
            checkify.JaxRuntimeError, "Lasso stopped after 1 steps without meeting"
        ):
            fit(self.DATA, p, LassoParameters(lam=0.01, num_iters=1))

    def test_non_finite_data_raise_a_clear_error(self):
        """Data below a lower bound give NaN quantiles, reported as such.

        The old solver returned NaN coefficients, which fit then reported as a
        non-positive density.
        """
        p = _params(MetalogBoundedness.STRICTLY_LOWER_BOUND, 5.0, 0.0, 5)

        with self.assertRaisesRegex(
            checkify.JaxRuntimeError, "The Lasso coefficients aren't finite"
        ):
            fit(self.DATA, p)

    def test_issue_example(self):
        """The issue's 5-term fit: default Lasso settings now give the OLS fit."""
        ols = fit(
            self.DATA,
            _params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5, MetalogFitMethod.OLS),
        )
        lasso = fit(self.DATA, _params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5))

        np.testing.assert_allclose(lasso.a, ols.a, rtol=1e-12)
        np.testing.assert_allclose(
            lasso.a, [2.7297, 0.3354, 0.243, 2.6946, 2.4629], atol=5e-5
        )


class LassoGridTest(LassoFitTestBase):
    """The grids' Lasso fits, which use their own masked designs, match fit's."""

    L1 = jnp.array([0.0, 0.01, 0.1])

    def _expected(self, num_terms):
        """The coefficients fit() gives for each penalty, at num_terms terms."""
        p = _params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, num_terms)
        return np.stack(
            [
                np.asarray(
                    _fit_coefficients(self.DATA, p, LassoParameters(lam=float(lam))).a
                )
                for lam in self.L1
            ]
        )

    def test_fit_grid_and_fit_grid_hyperparams(self):
        """The l1 grids' fits are fit's fits."""
        p = _params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5)
        expected = self._expected(5)

        grid = fit_grid(
            self.DATA.x,
            self.DATA.y,
            p,
            l1_penalties=self.L1,
            precomputed_quantiles=True,
        )
        hyper = fit_grid_hyperparams(self.DATA, p, self.L1, on_infeasible="flag")

        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(grid.metalog.a, expected, rtol=0, atol=1e-9 * scale)
        np.testing.assert_allclose(hyper.metalog.a, expected, rtol=0, atol=1e-9 * scale)

    def test_fit_grid_full_pads_with_zeros(self):
        """fit_grid_full's padded design gives fit's fits, with zeros past them."""
        p = _params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0, 5)
        x, y = self.DATA.x[None, :], self.DATA.y[None, :]

        result = fit_grid_full(x, y, p, self.L1, [3, 5], precomputed_quantiles=True)

        for j, k in enumerate((3, 5)):
            with self.subTest(num_terms=k):
                expected = self._expected(k)
                a = np.asarray(result.metalog.a[0, :, j])
                np.testing.assert_allclose(
                    a[:, :k], expected, rtol=0, atol=1e-9 * np.max(np.abs(expected))
                )
                np.testing.assert_array_equal(a[:, k:], 0.0)


if __name__ == "__main__":
    absltest.main()
