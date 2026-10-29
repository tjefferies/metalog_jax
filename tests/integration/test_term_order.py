"""Integration tests for the Metalog 2.0 term order (issue #58).

Each test checks a model in ``TermOrder.METALOG_2`` against an independent reference:
a NumPy basis written from the paper's list of terms, the feasibility module's own
design matrix and solver, autodiff of the quantile function, quadrature, or the same
distribution written in the 2016 order.
"""
# Copyright: Travis Jefferies 2026

import json
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify
from scipy.integrate import quad
from scipy.stats import gamma

from metalog_jax.base import (
    MetalogBase,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
)
from metalog_jax.feasibility import (
    TermOrder,
    best_feasible_fit,
    check_feasibility,
    design_matrix,
    get_engine,
    term_basis,
)
from metalog_jax.grid_search import (
    MAX_TERMS,
    _get_target_vmap,
    _masked_ppf,
    extract_metalog,
    fit_grid,
    fit_grid_datasets,
)
from metalog_jax.metalog import Metalog, fit
from metalog_jax.regression.base import has_full_column_rank
from metalog_jax.utils import DEFAULT_Y

KEELIN_2016 = TermOrder.KEELIN_2016
METALOG_2 = TermOrder.METALOG_2

# Probabilities from deep in both tails to the middle
Y = np.concatenate([[1e-6, 1e-4], np.linspace(0.001, 0.999, 199), [1 - 1e-4]])

# 400 gamma(2) samples, and bounds around them for each boundedness type
SAMPLES = np.random.default_rng(0).gamma(2.0, size=400)
BOUNDS = {
    MetalogBoundedness.UNBOUNDED: (0.0, 0.0),
    MetalogBoundedness.STRICTLY_LOWER_BOUND: (0.0, 0.0),
    MetalogBoundedness.STRICTLY_UPPER_BOUND: (0.0, float(SAMPLES.max()) + 1.0),
    MetalogBoundedness.BOUNDED: (0.0, float(SAMPLES.max()) + 1.0),
}


def paper_basis(y: np.ndarray, num_terms: int, order: TermOrder) -> np.ndarray:
    """The basis matrix, written from each paper's list of terms.

    Keelin (2016, Definition 1): 1, L, xL, x, then x^2, x^2 L, x^3, x^3 L, ...
    Baucells et al. (2025, Section 2): 1, L, xL, x, x^2, x^2 L, x^3 L, x^3, x^4,
    x^4 L, x^5 L, x^5, ... With x = y - 0.5 and L = ln(y / (1 - y)).

    Args:
        y: Probabilities, shape (n,).
        num_terms: Number of terms k.
        order: Which paper's list.

    Returns:
        The (n, k) basis matrix.
    """
    x, logit = y - 0.5, np.log(y / (1 - y))
    first_four = [np.ones_like(y), logit, x * logit, x]
    columns = []
    for j in range(1, num_terms + 1):
        if j <= 4:
            columns.append(first_four[j - 1])
            continue
        power = (j - 1) // 2
        if order == METALOG_2:
            # Within each pair of equal powers, the one from the 2.0 list's
            # repeating pattern 1, L, L, 1 (by j mod 4) gets the logit
            with_logit = j % 4 in (2, 3)
        else:
            with_logit = j % 2 == 0
        columns.append(x**power * logit if with_logit else x**power)
    return np.stack(columns, axis=1)


def params(
    boundedness: MetalogBoundedness,
    num_terms: int,
    order: TermOrder,
    method: MetalogFitMethod = MetalogFitMethod.Feasible,
) -> MetalogParameters:
    """Return parameters for the gamma samples with the given settings."""
    lower, upper = BOUNDS[boundedness]
    return MetalogParameters(
        boundedness=boundedness,
        method=method,
        lower_bound=lower,
        upper_bound=upper,
        num_terms=num_terms,
        term_order=order,
    )


@cache
def fitted(
    boundedness: MetalogBoundedness,
    num_terms: int,
    order: TermOrder,
    method: MetalogFitMethod = MetalogFitMethod.Feasible,
) -> Metalog:
    """Fit the gamma samples, once per configuration."""
    data = MetalogInputData.from_values(jnp.asarray(SAMPLES), DEFAULT_Y, False)
    return fit(data, params(boundedness, num_terms, order, method))


def unchecked_quantile(m: Metalog, y: jnp.ndarray) -> jnp.ndarray:
    """The quantile function without ppf's input checks, so autodiff can trace it."""
    return Metalog.inverse_quantile_transform(
        m._ppf(y, m.num_terms), m.boundedness, m.lower_bound, m.upper_bound
    )


class TermBasisTest(parameterized.TestCase):
    """term_basis gives each order's terms, as the papers list them (#58)."""

    # (power of x, has the logit) for terms 1 to 12
    KEELIN_2016_TERMS = (
        (0, False), (0, True), (1, True), (1, False), (2, False), (2, True),
        (3, False), (3, True), (4, False), (4, True), (5, False), (5, True),
    )  # fmt: skip
    METALOG_2_TERMS = (
        (0, False), (0, True), (1, True), (1, False), (2, False), (2, True),
        (3, True), (3, False), (4, False), (4, True), (5, True), (5, False),
    )  # fmt: skip

    def test_each_order_lists_the_papers_terms(self):
        """The first 12 terms of each order are the papers' lists."""
        self.assertEqual(term_basis(12, KEELIN_2016), self.KEELIN_2016_TERMS)
        self.assertEqual(term_basis(12, METALOG_2), self.METALOG_2_TERMS)

    def test_orders_agree_up_to_six_terms(self):
        """The orders differ only from the 7th term on."""
        for k in range(2, 7):
            with self.subTest(num_terms=k):
                self.assertEqual(term_basis(k, KEELIN_2016), term_basis(k, METALOG_2))

    def test_orders_span_different_functions_only_at_7_11_15(self):
        """The orders hold the same functions unless k is 7, 11, 15, ..."""
        for k in range(2, MAX_TERMS + 1):
            with self.subTest(num_terms=k):
                same = set(term_basis(k, KEELIN_2016)) == set(term_basis(k, METALOG_2))
                self.assertEqual(same, k < 7 or k % 4 != 3)


class Metalog2EvaluationTest(parameterized.TestCase):
    """A Metalog 2.0 model's basis, quantile function and density are right (#58)."""

    def test_design_matrix_matches_the_papers_basis(self):
        """_get_target builds each order's basis, as the papers list it."""
        y = jnp.asarray(Y)
        for order in TermOrder:
            for k in range(2, 15):
                with self.subTest(order=order.name, num_terms=k):
                    expected = paper_basis(Y, k, order)
                    np.testing.assert_allclose(
                        MetalogBase._get_target(y, k, order),
                        expected,
                        rtol=1e-14,
                        atol=1e-14 * np.max(np.abs(expected)),
                    )

    def test_design_matrix_matches_the_feasibility_module(self):
        """_get_target equals the feasibility module's design_matrix."""
        y = jnp.asarray(Y)
        for k in (7, 8, 11, 12, 30):
            with self.subTest(num_terms=k):
                np.testing.assert_allclose(
                    MetalogBase._get_target(y, k, METALOG_2),
                    design_matrix(get_engine(k, METALOG_2), y),
                    rtol=1e-14,
                )

    def test_grid_design_and_quantiles_match(self):
        """The grids' masked design and quantile function follow the order too."""
        y = jnp.asarray(Y)
        a = jnp.asarray(np.random.default_rng(1).normal(size=MAX_TERMS))
        for k in (5, 7, 8, 11, 12):
            with self.subTest(num_terms=k):
                design = _get_target_vmap(y, jnp.asarray(k), METALOG_2)
                expected = MetalogBase._get_target(y, k, METALOG_2)
                np.testing.assert_allclose(design[:, :k], expected, rtol=1e-14)
                np.testing.assert_array_equal(design[:, k:], 0.0)
                padded = jnp.zeros(MAX_TERMS).at[:k].set(a[:k])
                np.testing.assert_allclose(
                    _masked_ppf(padded, y, jnp.asarray(k), METALOG_2),
                    expected @ a[:k],
                    rtol=1e-12,
                    atol=1e-12 * float(jnp.max(jnp.abs(expected @ a[:k]))),
                )

    @parameterized.parameters(*MetalogBoundedness)
    def test_ppf_is_the_papers_basis_times_a(self, boundedness):
        """The quantile function is the 2.0 basis times a, mapped to the support."""
        for k in (7, 12):
            with self.subTest(num_terms=k):
                m = fitted(boundedness, k, METALOG_2)
                unbounded = paper_basis(Y, k, METALOG_2) @ np.asarray(m.a)
                expected = Metalog.inverse_quantile_transform(
                    jnp.asarray(unbounded), boundedness, m.lower_bound, m.upper_bound
                )
                np.testing.assert_allclose(
                    m.ppf(jnp.asarray(Y)), expected, rtol=1e-12, atol=1e-12
                )

    @parameterized.parameters(*MetalogBoundedness)
    def test_pdf_is_the_reciprocal_of_the_quantile_derivative(self, boundedness):
        """The density is 1 / Q'(y), with Q' from autodiff of the quantile function."""
        y = jnp.asarray(Y[2:-1])  # Away from where Q' overflows
        for k in (7, 12):
            with self.subTest(num_terms=k):
                m = fitted(boundedness, k, METALOG_2)
                slope = jax.vmap(jax.grad(lambda p, m=m: unchecked_quantile(m, p)))(y)
                np.testing.assert_allclose(m.pdf(y) * slope, 1.0, rtol=1e-10)


class TermOrderConversionTest(parameterized.TestCase):
    """with_term_order rewrites a model in the other order (#58)."""

    @parameterized.product(
        boundedness=tuple(MetalogBoundedness), num_terms=(5, 6, 8, 9, 10, 12)
    )
    def test_conversion_keeps_the_distribution(self, boundedness, num_terms):
        """ppf, pdf, cdf, the moments and the mode don't change."""
        m = fitted(boundedness, num_terms, KEELIN_2016)
        converted = m.with_term_order(METALOG_2)
        self.assertEqual(converted.term_order, METALOG_2)
        y = jnp.asarray(Y[1:-1])
        q = jnp.asarray(np.quantile(SAMPLES, [0.01, 0.25, 0.5, 0.75, 0.99]))
        for name, a, b in (
            ("ppf", converted.ppf(y), m.ppf(y)),
            ("pdf", converted.pdf(y), m.pdf(y)),
            ("cdf", converted.cdf(q), m.cdf(q)),
        ):
            with self.subTest(method=name):
                np.testing.assert_allclose(a, b, rtol=1e-11)
        for name in ("mean", "var", "skewness", "kurtosis", "mode"):
            with self.subTest(method=name):
                expected = float(getattr(m, name))
                self.assertAlmostEqual(
                    float(getattr(converted, name)),
                    expected,
                    delta=1e-10 * max(1.0, abs(expected)),
                )

    @parameterized.product(
        boundedness=tuple(MetalogBoundedness), num_terms=(8, 9, 10, 12)
    )
    def test_conversion_matches_a_fit_in_the_other_order(self, boundedness, num_terms):
        """A 2016 fit, converted, is the Feasible fit made in the 2.0 order."""
        converted = fitted(boundedness, num_terms, KEELIN_2016).with_term_order(
            METALOG_2
        )
        direct = fitted(boundedness, num_terms, METALOG_2)
        # The Feasible solver stops within its tolerance of the same optimum
        np.testing.assert_allclose(
            converted.a, direct.a, rtol=0, atol=1e-9 * np.max(np.abs(direct.a))
        )

    def test_round_trip_is_exact(self):
        """Converting to 2.0 and back gives the same coefficients, bit for bit."""
        m = fitted(MetalogBoundedness.UNBOUNDED, 12, KEELIN_2016)
        back = m.with_term_order(METALOG_2).with_term_order(KEELIN_2016)
        self.assertEqual(back.metalog_params, m.metalog_params)
        np.testing.assert_array_equal(back.a, m.a)

    def test_the_same_order_returns_the_model(self):
        """Converting to the model's own order returns it unchanged."""
        m = fitted(MetalogBoundedness.UNBOUNDED, 8, METALOG_2)
        self.assertIs(m.with_term_order(METALOG_2), m)

    def test_a_batch_is_reordered_along_its_last_axis(self):
        """Stacked models convert as each would on its own."""
        models = [
            fitted(MetalogBoundedness.UNBOUNDED, 9, KEELIN_2016),
            fitted(MetalogBoundedness.STRICTLY_LOWER_BOUND, 9, KEELIN_2016).replace(
                metalog_params=params(MetalogBoundedness.UNBOUNDED, 9, KEELIN_2016)
            ),
        ]
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)
        converted = batch.with_term_order(METALOG_2)
        for i, m in enumerate(models):
            np.testing.assert_array_equal(
                converted.a[i], m.with_term_order(METALOG_2).a
            )

    def test_7_11_and_15_terms_raise(self):
        """Where the orders' last terms differ, conversion raises and says to refit."""
        for k in (7, 11, 15):
            with self.subTest(num_terms=k):
                m = Metalog(
                    params(MetalogBoundedness.UNBOUNDED, k, KEELIN_2016),
                    jnp.zeros(k).at[1].set(1.0),
                )
                with self.assertRaisesRegex(ValueError, "Refit the data"):
                    m.with_term_order(METALOG_2)

    def test_a_non_enum_order_raises(self):
        """with_term_order takes a TermOrder."""
        m = fitted(MetalogBoundedness.UNBOUNDED, 8, KEELIN_2016)
        with self.assertRaisesRegex(ValueError, "must be a TermOrder"):
            m.with_term_order(2)


class SymmetricPercentilesTest(absltest.TestCase):
    """k symmetric percentiles fit with k terms in the 2.0 order (#58).

    Baucells et al. (2025), Proposition 2: for probabilities symmetric about 0.5,
    the Metalog 2.0 basis matrix has full rank for every k <= n, and the 2016 order's
    is rank deficient for n = k in {7, 11, 15, ...}.
    """

    # An expert's 5th to 95th percentiles of a right-skewed quantity
    PROBABILITIES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    PERCENTILES = (1.0, 1.6, 2.4, 3.2, 4.3, 6.0, 7.5)

    def _data(self) -> MetalogInputData:
        """The percentiles as precomputed quantiles."""
        return MetalogInputData.from_values(
            jnp.asarray(self.PERCENTILES), jnp.asarray(self.PROBABILITIES), True
        )

    def test_basis_rank_follows_proposition_2(self):
        """At n = k = 7, 11 and 15, only the 2016 basis is rank deficient."""
        for k in (7, 11, 15):
            with self.subTest(num_terms=k):
                y = jnp.linspace(0.05, 0.95, k)
                rank_2016 = has_full_column_rank(
                    MetalogBase._get_target(y, k, KEELIN_2016)
                )
                rank_2 = has_full_column_rank(MetalogBase._get_target(y, k, METALOG_2))
                self.assertFalse(bool(rank_2016))
                self.assertTrue(bool(rank_2))

    def test_2016_fits_raise(self):
        """The 2016 order can't fit 7 symmetric percentiles with 7 terms."""
        for method in (MetalogFitMethod.OLS, MetalogFitMethod.Feasible):
            with self.subTest(method=method.name):
                p = params(MetalogBoundedness.UNBOUNDED, 7, KEELIN_2016, method)
                with self.assertRaisesRegex(checkify.JaxRuntimeError, "rank deficient"):
                    fit(self._data(), p)

    def test_metalog2_ols_passes_through_every_percentile(self):
        """In the 2.0 order, the 7-term OLS fit interpolates the 7 percentiles."""
        p = params(MetalogBoundedness.UNBOUNDED, 7, METALOG_2, MetalogFitMethod.OLS)
        m = fit(self._data(), p)
        np.testing.assert_allclose(
            m.ppf(jnp.asarray(self.PROBABILITIES)), self.PERCENTILES, atol=1e-12
        )

    def test_metalog2_feasible_fit_is_the_solvers(self):
        """The Feasible fit is best_feasible_fit's in the 2.0 order, and certified."""
        p = params(MetalogBoundedness.UNBOUNDED, 7, METALOG_2)
        m = fit(self._data(), p)
        expected = best_feasible_fit(
            jnp.asarray(self.PERCENTILES),
            jnp.asarray(self.PROBABILITIES),
            7,
            order=METALOG_2,
        )
        np.testing.assert_array_equal(m.a, expected.a_star)
        self.assertTrue(bool(check_feasibility(get_engine(7, METALOG_2), m.a).feasible))


class Metalog2MomentsTest(absltest.TestCase):
    """Exact moments and the mode read the coefficients in the model's order (#58)."""

    def test_moments_match_quadrature(self):
        """At 7 terms, where no 2016 model is the same, moments match quadrature."""
        m = fitted(MetalogBoundedness.UNBOUNDED, 7, METALOG_2)
        a = np.asarray(m.a)

        def raw_moment(r: int) -> float:
            """E[X^r] = the integral of Q(y)^r over (0, 1)."""
            value, _ = quad(
                lambda y: float(paper_basis(np.array([y]), 7, METALOG_2)[0] @ a) ** r,
                0,
                1,
                limit=400,
                epsabs=0,
                epsrel=1e-10,
            )
            return value

        mean, second, third, fourth = (raw_moment(r) for r in (1, 2, 3, 4))
        var = second - mean**2
        skewness = (third - 3 * mean * var - mean**3) / var**1.5
        kurtosis = (
            fourth - 4 * mean * third + 6 * mean**2 * second - 3 * mean**4
        ) / var**2
        for name, expected in (
            ("mean", mean),
            ("var", var),
            ("skewness", skewness),
            ("kurtosis", kurtosis),
        ):
            with self.subTest(moment=name):
                self.assertAlmostEqual(
                    float(getattr(m, name)), expected, delta=1e-8 * abs(expected)
                )

    def test_mode_is_where_the_density_peaks(self):
        """The mode's density is at least the density anywhere on a fine grid."""
        m = fitted(MetalogBoundedness.UNBOUNDED, 7, METALOG_2)
        y = jnp.linspace(1e-4, 1 - 1e-4, 20001)
        mode_density = m.pdf(m.cdf(jnp.atleast_1d(m.mode)))[0]
        self.assertGreaterEqual(float(mode_density), float(jnp.max(m.pdf(y))) - 1e-9)


class TermOrderSerializationTest(absltest.TestCase):
    """term_order saves and loads, and 2016 JSON doesn't change (#58)."""

    def test_metalog2_round_trip(self):
        """A 2.0 model saves its order and loads back equal."""
        m = fitted(MetalogBoundedness.BOUNDED, 8, METALOG_2)
        saved = json.loads(m.dumps())
        self.assertEqual(saved["metalog_params"]["term_order"], int(METALOG_2))
        loaded = Metalog.loads(m.dumps())
        self.assertEqual(loaded, m)
        self.assertEqual(loaded.term_order, METALOG_2)

    def test_2016_json_has_no_term_order(self):
        """A 2016 model writes the same keys as before, so older versions load it."""
        m = fitted(MetalogBoundedness.BOUNDED, 8, KEELIN_2016)
        self.assertEqual(
            list(json.loads(m.dumps())["metalog_params"]),
            ["boundedness", "method", "lower_bound", "upper_bound", "num_terms"],
        )

    def test_json_without_term_order_loads_in_the_2016_order(self):
        """JSON saved before term_order existed loads as the 2016 order."""
        old = (
            '{"a": [1.0, 2.0, 0.5], "metalog_params": {"boundedness": 1, '
            '"method": 1, "lower_bound": 0.0, "upper_bound": 0.0, "num_terms": 3}}'
        )
        self.assertEqual(Metalog.loads(old).term_order, KEELIN_2016)

    def test_unknown_term_order_raises(self):
        """An unknown term_order value is rejected."""
        saved = json.loads(fitted(MetalogBoundedness.UNBOUNDED, 8, METALOG_2).dumps())
        saved["metalog_params"]["term_order"] = 9
        with self.assertRaises(ValueError):
            Metalog.loads(json.dumps(saved))

    def test_parameters_take_only_a_term_order(self):
        """MetalogParameters rejects an int for term_order."""
        with self.assertRaisesRegex(ValueError, "must be a TermOrder"):
            MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=8,
                term_order=2,
            )


class TermOrderDisplayTest(absltest.TestCase):
    """Printed terms follow the model's order (#58)."""

    A = jnp.arange(1.0, 9.0)

    def _model(self, order: TermOrder) -> Metalog:
        """An 8-term model with coefficients 1 to 8, which display doesn't check."""
        return Metalog(params(MetalogBoundedness.UNBOUNDED, 8, order), self.A)

    def test_plain_text(self):
        """Plain text shows x^3 L before x^3 in the 2.0 order, after it in 2016's."""
        self.assertIn(
            "+ 7 (y - 0.5)^3 ln(y/(1-y)) + 8 (y - 0.5)^3", str(self._model(METALOG_2))
        )
        self.assertIn(
            "+ 7 (y - 0.5)^3 + 8 (y - 0.5)^3 ln(y/(1-y))",
            str(self._model(KEELIN_2016)),
        )

    def test_latex(self):
        """Jupyter's LaTeX follows the order too."""
        latex = self._model(METALOG_2)._repr_latex_()
        self.assertLess(
            latex.index(r"7\,(y - 0.5)^{3}\,\ell(y)"), latex.index(r"8\,(y - 0.5)^{3}")
        )


class TermOrderGridTest(absltest.TestCase):
    """Grids fit, check and extract models in the parameters' order (#58)."""

    # Gamma(2) quantiles at DEFAULT_Y: the 7-term 2.0 OLS fit is feasible, and the
    # same coefficients read in the 2016 order are not
    X = jnp.asarray(gamma(a=2).ppf(np.asarray(DEFAULT_Y)))
    OLS_2 = params(MetalogBoundedness.UNBOUNDED, 3, METALOG_2, MetalogFitMethod.OLS)

    def test_term_count_grid_is_least_squares_on_the_papers_basis(self):
        """Each term count's fit is lstsq on the 2.0 basis."""
        terms = [5, 7, 8, 11]
        result = fit_grid(self.X, DEFAULT_Y, self.OLS_2, num_terms=terms)
        for i, k in enumerate(terms):
            with self.subTest(num_terms=k):
                expected = np.linalg.lstsq(
                    paper_basis(np.asarray(DEFAULT_Y), k, METALOG_2),
                    np.asarray(self.X),
                    rcond=None,
                )[0]
                np.testing.assert_allclose(
                    result.metalog.a[i, :k],
                    expected,
                    rtol=0,
                    atol=1e-9 * np.max(np.abs(expected)),
                )

    def test_feasibility_mask_uses_the_term_order(self):
        """The feasible mask comes from the 2.0 engine, which differs from 2016's here."""
        result = fit_grid(self.X, DEFAULT_Y, self.OLS_2.replace(num_terms=7))
        a = result.metalog.a
        self.assertTrue(bool(check_feasibility(get_engine(7, METALOG_2), a).feasible))
        # The case only tells the engines apart because 2016's disagrees
        self.assertFalse(
            bool(check_feasibility(get_engine(7, KEELIN_2016), a).feasible)
        )
        self.assertTrue(bool(result.feasible))

    def test_extract_metalog_keeps_the_term_order(self):
        """A model extracted from a grid has the grid's order."""
        result = fit_grid(self.X, DEFAULT_Y, self.OLS_2, num_terms=[5, 7])
        self.assertEqual(extract_metalog(result, 1).term_order, METALOG_2)

    def test_feasible_batch_matches_the_solver(self):
        """fit_grid_datasets with Feasible fits in the 2.0 order."""
        feasible = self.OLS_2.replace(method=MetalogFitMethod.Feasible, num_terms=7)
        result = fit_grid_datasets(
            self.X[None], DEFAULT_Y[None], feasible, on_infeasible="flag"
        )
        expected = best_feasible_fit(self.X, DEFAULT_Y, 7, order=METALOG_2)
        np.testing.assert_allclose(result.metalog.a[0], expected.a_star, rtol=1e-12)


class TermOrderTransformsTest(absltest.TestCase):
    """2.0 models pass through jit and vmap (#58)."""

    def test_jit_matches_eager(self):
        """A jitted ppf gives the eager values, in both orders."""
        y = jnp.asarray(Y[1:-1])
        ppf = checkify.checkify(jax.jit(lambda m, p: m.ppf(p)))
        for order in TermOrder:
            with self.subTest(order=order.name):
                m = fitted(MetalogBoundedness.BOUNDED, 9, order)
                err, values = ppf(m, y)
                err.throw()
                np.testing.assert_array_equal(values, m.ppf(y))

    def test_vmap_over_stacked_models(self):
        """A vmap over stacked 2.0 models matches each model's values."""
        y = jnp.asarray(Y[1:-1])
        models = [
            fitted(MetalogBoundedness.UNBOUNDED, 8, METALOG_2),
            fitted(MetalogBoundedness.UNBOUNDED, 8, KEELIN_2016).with_term_order(
                METALOG_2
            ),
        ]
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)
        values = jax.vmap(lambda m: m._ppf(y, 8))(batch)
        for i, m in enumerate(models):
            np.testing.assert_allclose(values[i], m._ppf(y, 8), rtol=1e-14)


if __name__ == "__main__":
    absltest.main()
