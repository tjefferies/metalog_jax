"""Property-based tests for issue #29: invariants checked on generated inputs.

Each test states an invariant that holds for every feasible metalog, whatever its
boundedness, fit method or number of terms, and Hypothesis checks it on generated
models, probabilities, data and SPT triplets (see strategies.py). A failure shrinks
to a minimal counterexample; pin it with ``@example`` and, if it's a real bug, add a
regression test in tests/unit or tests/integration.
"""
# Copyright: Travis Jefferies 2026

import io
import re

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st
from jax.experimental import checkify
from scipy import stats
from scipy.integrate import trapezoid
from scipy.special import expit

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    MetalogRandomVariableParameters,
    SPTMetalogParameters,
)
from metalog_jax.grid_search import (
    find_best_config,
    fit_grid,
    fit_grid_datasets,
    fit_grid_datasets_hyperparams,
    fit_grid_datasets_num_terms,
    fit_grid_full,
    fit_grid_hyperparams,
    fit_grid_num_terms,
)
from metalog_jax.metalog import THETA_3, Metalog, SPTMetalog, fit, fit_spt_metalog
from metalog_jax.regression import fit_ordinary_least_squares
from metalog_jax.utils import (
    DEFAULT_Y,
    DEFAULT_Y_FULL,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
    ks_distance,
)
from tests.property import strategies

# A fifth of the profile's examples, for properties that fit several models per
# example, and for exact identities, which fail on any input if they fail at all
FEW = settings(max_examples=max(5, settings().max_examples // 5))


def _distribution_methods(m, y):
    """Return the model's ppf and pdf at y, and its cdf at ppf(y)."""
    x = m.ppf(y)
    return x, m.pdf(y), m.cdf(x)


def _methods_and_draws(m, y, rv_params):
    """Return _distribution_methods and a sample for each rvs parameters object."""
    return (*_distribution_methods(m, y), *(m.rvs(p) for p in rv_params))


class MetalogPropertiesTest(parameterized.TestCase):
    """Invariants of every feasible metalog's distribution methods."""

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_ppf_is_strictly_increasing(self, m, y):
        """Quantiles from ppf strictly increase with the probability."""
        x = np.asarray(m.ppf(y))
        np.testing.assert_array_less(x[:-1], x[1:])

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_ppf_stays_inside_the_support(self, m, y):
        """Quantiles from ppf are finite and strictly inside the bounds in use."""
        x = np.asarray(m.ppf(y))
        self.assertTrue(np.all(np.isfinite(x)))
        if strategies.uses_lower(m.boundedness):
            np.testing.assert_array_less(m.lower_bound, x)
        if strategies.uses_upper(m.boundedness):
            np.testing.assert_array_less(x, m.upper_bound)

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_pdf_is_positive(self, m, y):
        """Densities from pdf are finite and strictly positive."""
        density = np.asarray(m.pdf(y))
        self.assertTrue(np.all(np.isfinite(density)))
        np.testing.assert_array_less(0.0, density)

    @FEW
    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_r_aliases_match(self, m, y):
        """Regression test for issue #14: q, p and d match ppf, cdf and pdf(cdf)."""
        x = m.ppf(y)
        c = m.cdf(x)
        np.testing.assert_array_equal(m.q(y), x)
        np.testing.assert_array_equal(m.p(x), c)
        np.testing.assert_array_equal(m.d(x), m.pdf(c))

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_cdf_inverts_ppf(self, m, y):
        """Regression test for issue #49: cdf(ppf(y)) is y, to rounding.

        cdf used to return the nearest of 9,990 probabilities 0.0001 apart. The
        drawn probabilities are multiples of 0.001, which that grid contains, so
        shift them off it.
        """
        y = y + 3.7e-5
        np.testing.assert_allclose(m.cdf(m.ppf(y)), y, rtol=1e-9)
        np.testing.assert_allclose(m.sf(m.ppf(y)), 1 - y, rtol=1e-9)

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_ppf_is_the_back_transform_of_the_unbounded_ppf(self, m, y):
        """Regression test for issue #14: ppf back-transforms the unbounded ppf."""
        unbounded = m.metalog_params.replace(boundedness=MetalogBoundedness.UNBOUNDED)
        z = m.replace(metalog_params=unbounded).ppf(y)
        expected = strategies.from_unbounded_scale(
            z, m.boundedness, m.lower_bound, m.upper_bound
        )
        # The same closed form in NumPy; exp may differ from XLA's in the last bit
        np.testing.assert_allclose(m.ppf(y), expected, rtol=1e-12)
        # Every term but a1 vanishes at y = 0.5, so a1 is the median on the
        # unbounded scale (Keelin 2016, Section 3.3)
        median = strategies.from_unbounded_scale(
            m.a[0], m.boundedness, m.lower_bound, m.upper_bound
        )
        np.testing.assert_allclose(m.median, median, rtol=1e-12)

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_pdf_is_the_reciprocal_of_the_ppf_slope(self, m, y):
        """pdf(y) times the slope of ppf at y, by autodiff, is 1 (Keelin 2016, eq. 9)."""
        slope = checkify.checkify(jax.vmap(jax.grad(lambda t: m.ppf(t))))
        err, dx = slope(y)
        err.throw()
        np.testing.assert_allclose(m.pdf(y) * dx, 1.0, rtol=1e-9)

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_mirror_image_is_the_distribution_of_minus_x(self, m, y):
        """Mirrored coefficients and bounds give -X: quantiles -ppf(1 - y)."""
        mirrored = strategies.mirror(m)
        x = np.asarray(m.ppf(1 - y))
        # 1 - y and the mirror's own 1 - y can differ in the last bit
        np.testing.assert_allclose(
            mirrored.ppf(y), -x, rtol=1e-9, atol=1e-12 * np.max(np.abs(x))
        )
        np.testing.assert_allclose(mirrored.pdf(y), m.pdf(1 - y), rtol=1e-9)
        self.assertEqual(
            strategies.is_certified(mirrored.a), strategies.is_certified(m.a)
        )
        if m.boundedness == MetalogBoundedness.UNBOUNDED:
            scale = float(jnp.sum(jnp.abs(m.a)))
            np.testing.assert_allclose(mirrored.mean, -m.mean, atol=1e-12 * scale)
            np.testing.assert_allclose(mirrored.var, m.var, rtol=1e-10)
            np.testing.assert_allclose(mirrored.skewness, -m.skewness, atol=1e-9)
            np.testing.assert_allclose(mirrored.kurtosis, m.kurtosis, rtol=1e-9)

    # Padded to 5 or 7 terms; each new term count compiles new moment and mode code
    @given(m=strategies.metalogs(term_counts=(3, 5)), y=strategies.probabilities())
    def test_zero_terms_change_nothing(self, m, y):
        """Padding the coefficients with two zero terms leaves the model unchanged."""
        params = m.metalog_params.replace(num_terms=m.num_terms + 2)
        padded = Metalog(params, jnp.pad(m.a, (0, 2)))
        x = m.ppf(y)
        np.testing.assert_allclose(padded.ppf(y), x, rtol=1e-12)
        np.testing.assert_allclose(padded.pdf(y), m.pdf(y), rtol=1e-12)
        np.testing.assert_array_equal(padded.cdf(x), m.cdf(x))
        if m.boundedness == MetalogBoundedness.UNBOUNDED:
            # Moments and modes come from different polynomial tables for k + 2
            # terms, so they agree to rounding rather than exactly
            tolerance = {"rtol": 1e-9, "atol": 1e-12 * float(jnp.sum(jnp.abs(m.a)))}
            for name in ("mean", "var", "skewness", "kurtosis"):
                np.testing.assert_allclose(
                    getattr(padded, name), getattr(m, name), err_msg=name, **tolerance
                )
            # highest_mode is what mode returns; eager mode takes 0.4 s a call
            for got, want in zip(
                strategies.highest_density_mode(padded.a),
                strategies.highest_density_mode(m.a),
                strict=True,
            ):
                np.testing.assert_allclose(got, want, err_msg="mode", **tolerance)

    @FEW
    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_results_do_not_depend_on_the_coefficient_container(self, m, y):
        """Regression test for issue #9: list, tuple and NumPy coefficients match JAX."""
        a = np.asarray(m.a)
        expected = (m.ppf(y), m.pdf(y))
        for container in (a.tolist(), tuple(a.tolist()), a):
            other = Metalog(m.metalog_params, container)
            for got, want in zip((other.ppf(y), other.pdf(y)), expected, strict=True):
                np.testing.assert_array_equal(
                    got, want, err_msg=type(container).__name__
                )

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_serialization_round_trip(self, m, y):
        """loads(dumps(m)) equals m and gives the same ppf."""
        loaded = Metalog.loads(m.dumps())
        self.assertIsInstance(loaded, Metalog)
        self.assertEqual(loaded, m)
        np.testing.assert_array_equal(loaded.ppf(y), m.ppf(y))

    @FEW
    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_float32_probabilities_match_float64(self, m, y):
        """Regression test for issue #26: float32 probabilities lose no precision."""
        y32 = y.astype(jnp.float32)
        y64 = y32.astype(jnp.float64)
        np.testing.assert_array_equal(m.ppf(y32), m.ppf(y64))
        np.testing.assert_array_equal(m.pdf(y32), m.pdf(y64))


class UnboundedMetalogPropertiesTest(parameterized.TestCase):
    """Invariants of unbounded metalogs: equivariance and exact moments."""

    # Integrate over u = logit(y), where dy = y (1 - y) du and the integrand decays
    # like exp(-|u|): the trapezoid rule converges geometrically, and the mass
    # beyond |u| = 30 is about 1e-13 of the scale
    LOGIT_GRID = np.linspace(-30.0, 30.0, 6001)

    @FEW
    @given(
        m=strategies.metalogs(MetalogBoundedness.UNBOUNDED),
        y=strategies.probabilities(),
        c=st.floats(-100, 100),
        k=st.floats(0.01, 100),
    )
    def test_location_and_scale_equivariance(self, m, y, c, k):
        """Adding c to a1 shifts ppf by c; scaling a by k scales ppf by k, pdf by 1/k."""
        x = np.asarray(m.ppf(y))
        # Rounding is relative to the terms' magnitudes (|logit(y)| < 7), not to x
        scale = 10 * float(jnp.sum(jnp.abs(m.a)))
        shifted = m.replace(a=m.a.at[0].add(c))
        np.testing.assert_allclose(
            shifted.ppf(y), x + c, rtol=0, atol=1e-12 * (scale + abs(c))
        )
        scaled = m.replace(a=k * m.a)
        np.testing.assert_allclose(scaled.ppf(y), k * x, rtol=0, atol=1e-12 * k * scale)
        # 1 / M'(y) loses accuracy where M' is much smaller than its terms
        np.testing.assert_allclose(scaled.pdf(y), m.pdf(y) / k, rtol=1e-9)

    @given(m=strategies.metalogs(MetalogBoundedness.UNBOUNDED))
    def test_moments_match_numerical_integrals(self, m):
        """The exact mean, variance, skewness and kurtosis match integrals of ppf."""
        u = self.LOGIT_GRID
        y = expit(u)
        weight = y * (1 - y)
        x = np.asarray(m.ppf(jnp.asarray(y)))
        mean = trapezoid(x * weight, u)
        var, third, fourth = (trapezoid((x - mean) ** p * weight, u) for p in (2, 3, 4))
        scale = float(jnp.sum(jnp.abs(m.a)))
        np.testing.assert_allclose(m.mean, mean, rtol=0, atol=1e-9 * scale)
        np.testing.assert_allclose(m.var, var, rtol=1e-8, atol=1e-9 * scale**2)
        # Standardized moments are scale-free; the quadrature error is relative
        np.testing.assert_allclose(m.skewness, third / var**1.5, rtol=0, atol=1e-7)
        np.testing.assert_allclose(m.kurtosis, fourth / var**2, rtol=1e-7)

    @given(m=strategies.low_order_unbounded_metalogs())
    def test_three_and_four_term_mode_has_a_closed_form(self, m):
        """With 3 or 4 terms the mode is at y = 0.5 - a3 / (4 a2) (Metalog 2.0, 4.3)."""
        # y^2 (1 - y)^2 M''(y) = a2 (2y - 1) + a3 / 2, whose root is the one mode.
        # highest_mode is what mode returns; eager mode takes 0.4 s a call.
        y_mode = 0.5 - 0.25 * float(m.a[2] / m.a[1])
        x, y, found = strategies.highest_density_mode(m.a)
        self.assertTrue(found)
        np.testing.assert_allclose(y, y_mode, rtol=1e-9)
        np.testing.assert_allclose(
            x,
            m.ppf(jnp.array([y_mode]))[0],
            rtol=1e-9,
            atol=1e-12 * float(jnp.sum(jnp.abs(m.a))),
        )

    @given(m=strategies.metalogs(MetalogBoundedness.UNBOUNDED))
    def test_mode_has_the_highest_density(self, m):
        """No grid point has a higher density than highest_mode's (what mode returns)."""
        _, y_mode, found = strategies.highest_density_mode(m.a)
        event(f"interior mode found: {bool(found)}")
        if not found:
            return
        peak = float(m.pdf(jnp.array([float(y_mode)]))[0])
        # Root finding pins the mode to rounding, so allow that much below the grid
        self.assertGreaterEqual(
            peak, float(jnp.max(m.pdf(DEFAULT_Y_FULL))) * (1 - 1e-9)
        )


class KnownDistributionPropertiesTest(parameterized.TestCase):
    """Special cases that are named distributions (Keelin 2016, Tables 2-4)."""

    @FEW
    @given(
        y=strategies.probabilities(),
        location=st.floats(-10, 10),
        scale=st.floats(0.1, 5),
        shape=st.floats(0.1, 1),
        lower=st.floats(-10, 10),
        width=st.floats(0.5, 50),
    )
    def test_special_cases_match_scipy(self, y, location, scale, shape, lower, width):
        """Logistic, uniform, log-logistic and their bounded forms match scipy."""
        upper = lower + width
        b = MetalogBoundedness
        fisk = stats.fisk(c=1 / shape, scale=np.exp(location / 10))
        yy = np.asarray(y)
        # name: (model, the same distribution in scipy)
        cases = {
            "logistic": (
                self._model(b.UNBOUNDED, [location, scale, 0.0]),
                stats.logistic(loc=location, scale=scale),
            ),
            "uniform": (
                self._model(b.UNBOUNDED, [location, 0.0, 0.0, scale]),
                stats.uniform(loc=location - scale / 2, scale=scale),
            ),
            "log-logistic": (
                self._model(b.STRICTLY_LOWER_BOUND, [location / 10, shape, 0.0], lower),
                stats.fisk(c=1 / shape, loc=lower, scale=np.exp(location / 10)),
            ),
            "bounded uniform": (
                self._model(b.BOUNDED, [0.0, 1.0, 0.0], lower, upper),
                stats.uniform(loc=lower, scale=width),
            ),
        }
        for name, (m, reference) in cases.items():
            x = reference.ppf(yy)
            self._assert_matches(m, y, x, reference.pdf(x), name)
        # U - X for a log-logistic X: the upper-bounded form, a mirror image
        m = self._model(
            b.STRICTLY_UPPER_BOUND, [-location / 10, shape, 0.0], 0.0, upper
        )
        x = fisk.ppf(1 - yy)
        self._assert_matches(m, y, upper - x, fisk.pdf(x), "upper log-logistic")
        logistic, uniform = cases["logistic"][0], cases["uniform"][0]
        for m, var, kurtosis in (
            (logistic, np.pi**2 * scale**2 / 3, 4.2),
            (uniform, scale**2 / 12, 1.8),
        ):
            np.testing.assert_allclose(m.mean, location, atol=1e-12 * (1 + scale))
            np.testing.assert_allclose(m.var, var, rtol=1e-12)
            np.testing.assert_allclose(m.skewness, 0.0, atol=1e-12)
            np.testing.assert_allclose(m.kurtosis, kurtosis, rtol=1e-12)

    @staticmethod
    def _model(boundedness, a, lower=0.0, upper=0.0):
        """Return a Metalog with the given coefficients and bounds."""
        params = MetalogParameters(
            boundedness=boundedness,
            method=MetalogFitMethod.OLS,
            lower_bound=lower,
            upper_bound=upper,
            num_terms=len(a),
        )
        return Metalog(params, jnp.asarray(a, dtype=jnp.float64))

    @staticmethod
    def _assert_matches(m, y, x, density, name):
        """Compare the model's ppf and pdf at y with reference values."""
        np.testing.assert_allclose(
            m.ppf(y), x, rtol=1e-10, atol=1e-12 * np.max(np.abs(x)), err_msg=name
        )
        np.testing.assert_allclose(m.pdf(y), density, rtol=1e-10, err_msg=name)


class FeasibilityPropertiesTest(parameterized.TestCase):
    """Invariants of check_feasibility and the Feasible fit method."""

    # Log-spaced tails down to 1e-12 from each end, and a fine grid in between
    TAILS = np.logspace(-12, -2, 300)
    DENSE_Y = np.unique(
        np.concatenate([TAILS, np.linspace(0.01, 0.99, 3000), 1 - TAILS])
    )

    @FEW
    @given(
        pair=strategies.coefficient_pairs(),
        # The second weight is positive, so the combination is never zero
        weights=st.tuples(st.floats(0, 10), st.floats(0.01, 10)),
        y=strategies.probabilities(),
    )
    def test_feasible_set_is_a_convex_cone(self, pair, weights, y):
        """Nonnegative combinations of feasible metalogs are feasible (Proposition 6)."""
        combined = weights[0] * pair[0] + weights[1] * pair[1]
        self.assertTrue(strategies.is_certified(combined))
        # The quantile function is linear in a, so the combination's ppf combines too
        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            method=MetalogFitMethod.OLS,
            lower_bound=0.0,
            upper_bound=0.0,
            num_terms=combined.shape[0],
        )
        ppfs = [Metalog(params, a).ppf(y) for a in (*pair, combined)]
        scale = 10 * sum(
            w * float(jnp.sum(jnp.abs(a))) for w, a in zip(weights, pair, strict=True)
        )
        np.testing.assert_allclose(
            ppfs[2], weights[0] * ppfs[0] + weights[1] * ppfs[1], atol=1e-12 * scale
        )

    @given(a=strategies.coefficient_vectors())
    def test_check_feasibility_matches_the_tails_and_a_dense_grid(self, a):
        """Tails follow s(0) and s(1) (Proposition 5); certified M rises on a grid."""
        report = strategies.feasibility_report(a)
        event(f"certified: {bool(report.feasible)}")
        # M'(y) ~ s(0) / y near 0 and s(1) / (1 - y) near 1, so a clearly
        # nonzero s decides the tail; leave values within rounding of 0 out
        margin = 1e-9 * float(np.sum(np.abs(a)))
        for s_end, tail_feasible in zip(
            strategies.s_at_ends(a),
            (report.tail_feasible_zero, report.tail_feasible_one),
            strict=True,
        ):
            if abs(s_end) > margin:
                self.assertEqual(bool(tail_feasible), s_end > 0)
        if report.feasible:
            m = strategies.basis(self.DENSE_Y, a.shape[0]) @ a
            # |logit| <= 28 on the grid, so rounding is within 1e-12 of 28 sum|a|
            steps = np.diff(m)
            self.assertGreaterEqual(
                float(steps.min()), -1e-12 * 28 * float(np.sum(np.abs(a)))
            )

    @given(m=strategies.fitted_metalogs())
    def test_inflection_points_obey_proposition_1(self, m):
        """At most 2 floor((k-1)/2) - 1 inflection points and floor((k-1)/2) modes."""
        report = strategies.feasibility_report(m.a)
        roots = int(np.sum(~np.isnan(np.asarray(report.roots))))
        modes = int(np.sum(np.asarray(report.is_mode)))
        half = (m.num_terms - 1) // 2
        event(f"{m.num_terms} terms, {modes} modes")
        self.assertLessEqual(roots, max(2 * half - 1, 0))
        self.assertLessEqual(modes, half)

    @given(inputs=strategies.fit_inputs(method=MetalogFitMethod.Feasible))
    def test_feasible_fit_is_certified_and_matches_feasible_ols(self, inputs):
        """A Feasible fit is certified, and equals the OLS fit when that is feasible."""
        data, params = inputs
        a = fit(data, params).a
        self.assertTrue(strategies.is_certified(a))
        target = Metalog.get_target(data, params)
        ols = fit_ordinary_least_squares(target, Metalog.get_quantiles(data, params))
        ols_feasible = strategies.is_certified(ols.weights)
        event(f"OLS fit feasible: {ols_feasible}")
        if ols_feasible:
            # Both solve the same least-squares problem; only rounding differs
            np.testing.assert_allclose(
                a, ols.weights, rtol=1e-9, atol=1e-9 * float(jnp.max(jnp.abs(a)))
            )


class FitPropertiesTest(parameterized.TestCase):
    """Invariants of fitting: identifiability and equivariance."""

    @given(m=strategies.metalogs())
    def test_refitting_a_metalogs_quantiles_recovers_it(self, m):
        """OLS, Feasible and grid fits to a metalog's own quantiles return its a."""
        x = m.ppf(DEFAULT_Y)
        data = MetalogInputData.from_values(x, DEFAULT_Y, precomputed_quantiles=True)
        ols = m.metalog_params.replace(method=MetalogFitMethod.OLS)
        refits = {
            "OLS": fit(data, ols).a,
            "Feasible": fit(data, ols.replace(method=MetalogFitMethod.Feasible)).a,
            "fit_grid": fit_grid(
                x, DEFAULT_Y, ols, num_terms=[m.num_terms], precomputed_quantiles=True
            ).metalog.a[0],
        }
        # The quantiles reach y = 0.001 and 0.999, where the log and logit transforms
        # of values near a bound lose up to about 1e-7 of relative precision
        for name, a in refits.items():
            np.testing.assert_allclose(
                a, m.a, rtol=0, atol=1e-6 * float(jnp.max(jnp.abs(m.a))), err_msg=name
            )

    @given(draws=st.data(), shift=st.floats(-100, 100), scale=st.floats(0.01, 100))
    def test_fits_follow_affine_maps_of_the_data(self, draws, shift, scale):
        """Shifting and scaling the data shifts and scales the fit (Keelin 2016, Table 2)."""
        # The bounded (logit) fit has no such simple map
        b = draws.draw(st.sampled_from(strategies.BOUNDEDNESS[:3]))
        method = draws.draw(
            st.sampled_from((MetalogFitMethod.OLS, MetalogFitMethod.Feasible))
        )
        data, params = draws.draw(strategies.fit_inputs(b, method))
        lower, upper = params.lower_bound, params.upper_bound
        x = data.x
        if b == MetalogBoundedness.UNBOUNDED:
            moved = shift + scale * x
        elif b == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            moved = lower + scale * (x - lower)  # ln(x - lower) gains ln(scale)
        else:
            moved = upper - scale * (upper - x)  # -ln(upper - x) loses ln(scale)
        moved_data = MetalogInputData.from_values(moved, data.y, True)
        try:
            a = fit(data, params).a
        except checkify.JaxRuntimeError:
            # Feasibility is scale-free, so an infeasible OLS fit stays infeasible
            with self.assertRaises(checkify.JaxRuntimeError):
                fit(moved_data, params)
            return
        if b == MetalogBoundedness.UNBOUNDED:
            expected = (scale * a).at[0].add(shift)
        else:
            sign = 1.0 if b == MetalogBoundedness.STRICTLY_LOWER_BOUND else -1.0
            expected = a.at[0].add(sign * np.log(scale))
        # OLS is exact to rounding; the Feasible solver converges to about 1e-7
        tolerance = 1e-9 if params.method == MetalogFitMethod.OLS else 1e-6
        np.testing.assert_allclose(
            fit(moved_data, params).a,
            expected,
            rtol=0,
            atol=tolerance * float(jnp.max(jnp.abs(expected))),
        )


class TransformPropertiesTest(parameterized.TestCase):
    """Regression tests for issue #25: jax.jit and jax.vmap match eager calls."""

    RV_SIZE = 25
    # Each combination of static fields compiles once, so draw fewer term counts
    NUM_TERMS = (3, 5)
    # Compiled and eager code can round differently, including in the iterative
    # Lasso and Feasible solvers, so allow more than float64 epsilon
    RTOL = 1e-9
    # Jitted once at class level, so examples that share static fields (boundedness,
    # method and num_terms) share one compilation
    JITTED_METHODS = staticmethod(jax.jit(checkify.checkify(_methods_and_draws)))
    JITTED_SPT_FIT = staticmethod(
        jax.jit(
            checkify.checkify(
                lambda q, p: fit_spt_metalog(q, p, precomputed_quantiles=True).a
            )
        )
    )

    def _assert_all_close(self, actual, expected):
        """Compare tuples or lists of arrays with RTOL."""
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=self.RTOL)

    @given(
        m=strategies.metalogs(term_counts=NUM_TERMS),
        y=strategies.probabilities(),
        seed=st.integers(0, 10_000),
        trial=st.integers(1, 10_000),
    )
    def test_jit_with_model_and_rvs_parameters_matches_eager(self, m, y, seed, trial):
        """Jitted methods with the model and rvs parameters as arguments match eager."""
        # No distribution method reads the fit method, but as a static field it
        # would compile once per method, so fix it
        m = m.replace(
            metalog_params=m.metalog_params.replace(method=MetalogFitMethod.OLS)
        )
        rv_params = tuple(
            MetalogRandomVariableParameters(prng_params=prng, size=self.RV_SIZE)
            for prng in (
                JaxUniformDistributionParameters(seed=seed),
                HDRPRNGParameters(trial=trial),
            )
        )
        err, jitted = self.JITTED_METHODS(m, y, rv_params)
        err.throw()
        self._assert_all_close(jitted, _methods_and_draws(m, y, rv_params))

    @FEW
    @given(
        models=strategies.metalog_batches(term_counts=NUM_TERMS),
        y=strategies.probabilities(),
    )
    def test_vmap_over_stacked_models_matches_loop(self, models, y):
        """Vmapping methods over models that share static fields matches a loop."""
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)
        batched = jax.vmap(lambda m: _distribution_methods(m, y))(stacked)
        looped = [_distribution_methods(m, y) for m in models]
        self._assert_all_close(
            batched, [np.stack(v) for v in zip(*looped, strict=True)]
        )

    @FEW
    @given(
        batch=strategies.sample_batches(term_counts=NUM_TERMS),
        y=strategies.probabilities(),
    )
    def test_vmap_over_raw_data_matches_loop(self, batch, y):
        """Vmapping from_values, fit and ppf over raw samples matches a loop."""
        x, params = batch

        def fit_samples(samples):
            """Build the input data, fit it and return the coefficients and ppf."""
            data = MetalogInputData.from_values(samples, DEFAULT_Y, False)
            m = fit(data, params)
            return m.a, m.ppf(y)

        try:
            looped = [fit_samples(jnp.asarray(s)) for s in x]
        except checkify.JaxRuntimeError:
            # An OLS or Lasso fit was infeasible, which vmap would also report
            assume(False)
        batched = jax.vmap(fit_samples)(jnp.asarray(x))
        self._assert_all_close(
            batched, [np.stack(v) for v in zip(*looped, strict=True)]
        )

    @given(triplet=strategies.spt_triplets(feasible=True))
    def test_jit_spt_fit_with_parameters_argument_matches_eager(self, triplet):
        """A jitted SPT fit with the triplet and parameters as arguments matches eager."""
        q, params = triplet
        q = jnp.asarray(q)
        err, a = self.JITTED_SPT_FIT(q, params)
        err.throw()
        eager = fit_spt_metalog(q, params, precomputed_quantiles=True).a
        np.testing.assert_allclose(a, eager, rtol=self.RTOL)


class SPTPropertiesTest(parameterized.TestCase):
    """Regression tests for issues #20, #23 and #24: SPT fits match Keelin (2016)."""

    @staticmethod
    def _three_term_ratio(q, params):
        """|a3| / a2 of the 3-term metalog through q, or None if q leaves the support.

        Proposition 1 (Keelin 2016) on the metalog's unbounded scale gives
        a2 = (z2 - z0) / (2L) and a3 = (z2 + z0 - 2 z1) / ((1 - 2 alpha) L), and the
        metalog is feasible exactly when the ratio is below THETA_3 (Proposition 2).
        """
        b = params.boundedness
        if strategies.uses_lower(b) and not q[0] > params.lower_bound:
            return None
        if strategies.uses_upper(b) and not q[2] < params.upper_bound:
            return None
        z = strategies.to_unbounded_scale(q, b, params.lower_bound, params.upper_bound)
        return (
            2 * abs(z[2] + z[0] - 2 * z[1]) / ((1 - 2 * params.alpha) * (z[2] - z[0]))
        )

    @staticmethod
    def _data_with_quantiles(q, params):
        """101 data points whose alpha, 0.5 and 1 - alpha quantiles are q."""
        # alpha is a multiple of 0.01, so linear interpolation hits indices
        # 100 alpha, 50 and 100 (1 - alpha) exactly
        i = round(100 * params.alpha)
        # End points inside the support, beyond the outer quantiles
        low, high = q[0] - (q[2] - q[0]), q[2] + (q[2] - q[0])
        if strategies.uses_lower(params.boundedness):
            low = (params.lower_bound + q[0]) / 2
        if strategies.uses_upper(params.boundedness):
            high = (q[2] + params.upper_bound) / 2
        return np.interp(
            np.arange(101), [0, i, 50, 100 - i, 100], [low, q[0], q[1], q[2], high]
        )

    @given(triplet=strategies.spt_triplets(feasible=True))
    def test_spt_fit_reproduces_its_anchors(self, triplet):
        """The fit passes through the triplet at alpha, 0.5 and 1 - alpha."""
        q, params = triplet
        m = fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True)
        y = jnp.array([params.alpha, 0.5, 1 - params.alpha])
        # Only rounding in the transform and Proposition 1 separates them
        np.testing.assert_allclose(
            m.ppf(y), q, rtol=1e-9, atol=1e-12 * (np.max(np.abs(q)) + 1)
        )

    @given(triplet=strategies.spt_triplets(feasible=True))
    def test_serialization_round_trip(self, triplet):
        """Regression test for issue #31: SPTMetalog.loads(dumps(m)) equals m."""
        q, params = triplet
        m = fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True)
        loaded = SPTMetalog.loads(m.dumps())
        self.assertIsInstance(loaded.metalog_params, SPTMetalogParameters)
        self.assertEqual(loaded, m)
        y = jnp.array([params.alpha, 0.5, 1 - params.alpha])
        np.testing.assert_array_equal(loaded.ppf(y), m.ppf(y))

    @given(triplet=strategies.spt_triplets(feasible=True))
    def test_mirrored_triplet_gives_mirrored_coefficients(self, triplet):
        """Fitting -q in reverse with mirrored bounds gives (-a1, a2, -a3)."""
        q, params = triplet
        a = fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True).a
        lower, upper = strategies.mirror_bounds(
            params.boundedness, params.lower_bound, params.upper_bound
        )
        mirrored = params.replace(
            boundedness=strategies.MIRROR_BOUNDEDNESS[params.boundedness],
            lower_bound=lower,
            upper_bound=upper,
        )
        b = fit_spt_metalog(jnp.asarray(-q[::-1]), mirrored, precomputed_quantiles=True)
        np.testing.assert_allclose(
            b.a,
            strategies.mirror_coefficients(a),
            rtol=1e-9,
            atol=1e-12 * float(jnp.max(jnp.abs(a))),
        )

    @given(
        triplet=st.one_of(
            strategies.spt_triplets(), strategies.spt_out_of_bounds_triplets()
        )
    )
    def test_spt_fit_rejects_exactly_the_invalid_triplets(self, triplet):
        """Infeasible or out-of-support triplets raise; others give finite fits."""
        q, params = triplet
        ratio = self._three_term_ratio(q, params)
        if ratio is None:
            event("outside the support")
        else:
            # Rounding can decide a triplet within float64 error of the boundary
            # either way
            assume(abs(ratio / THETA_3 - 1) > 1e-6)
            event("feasible" if ratio < THETA_3 else "infeasible")
        if ratio is None or ratio > THETA_3:
            with self.assertRaises(AssertionError):
                fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True)
        else:
            m = fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True)
            self.assertTrue(np.all(np.isfinite(m.a)))

    @given(triplet=strategies.spt_triplets(feasible=True))
    def test_spt_fit_paths_agree(self, triplet):
        """Precomputed quantiles, data with those quantiles and a 3-term OLS fit agree."""
        q, params = triplet
        expected = fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True).a
        tol = {"rtol": 1e-9, "atol": 1e-9 * float(jnp.max(jnp.abs(expected)))}
        data = jnp.asarray(self._data_with_quantiles(q, params))
        np.testing.assert_allclose(fit_spt_metalog(data, params).a, expected, **tol)
        # Three terms through three points: least squares interpolates exactly
        y = jnp.array([params.alpha, 0.5, 1 - params.alpha])
        ols_params = MetalogParameters(
            boundedness=params.boundedness,
            method=MetalogFitMethod.OLS,
            lower_bound=params.lower_bound,
            upper_bound=params.upper_bound,
            num_terms=3,
        )
        ols = fit(MetalogInputData.from_values(jnp.asarray(q), y, True), ols_params)
        np.testing.assert_allclose(ols.a, expected, **tol)


class GridSearchPropertiesTest(parameterized.TestCase):
    """Regression tests for issues #26 and #40: grid fits' scores and feasibility."""

    L1 = (0.0, 0.01)
    NUM_TERMS = (3, 9)

    def _assert_scored_at_y(self, result, x, y, params):
        """Check each KS distance against the fit's own quantiles at y."""
        a = result.metalog.a.reshape(-1, result.metalog.a.shape[-1])
        ks = result.ks_dist.reshape(-1)
        for a_i, ks_i in zip(a, ks, strict=True):
            m = Metalog(params.replace(num_terms=a_i.shape[0]), a_i)
            # _ppf, not ppf, so infeasible grid fits are scored rather than rejected
            fitted = Metalog.inverse_quantile_transform(
                m._ppf(y, a_i.shape[0]), m.boundedness, m.lower_bound, m.upper_bound
            )
            self.assertAlmostEqual(float(ks_i), float(ks_distance(x, fitted)))

    def _grids(self, dataset, boundedness, data):
        """Run all seven grid functions on a generated dataset, flagging infeasible fits.

        Returns:
            The OLS parameters, and (result, number of terms of each fit) pairs.
        """
        x, y = dataset
        x = jnp.asarray(x)
        lower, upper = data.draw(strategies.bounds_around(boundedness, x))
        ols, lasso = (
            MetalogParameters(
                boundedness=boundedness,
                method=method,
                lower_bound=lower,
                upper_bound=upper,
                num_terms=3,
            )
            for method in (MetalogFitMethod.OLS, MetalogFitMethod.Lasso)
        )
        input_data = MetalogInputData.from_values(x, y, precomputed_quantiles=True)
        bx, by = jnp.stack([x, x]), jnp.stack([y, y])
        l1 = jnp.asarray(self.L1)
        terms = list(self.NUM_TERMS)
        flag = {"on_infeasible": "flag"}
        grids = (
            (fit_grid(x, y, ols, num_terms=terms, precomputed_quantiles=True), terms),
            (fit_grid_num_terms(input_data, ols, terms), terms),
            (
                fit_grid_datasets_num_terms(
                    bx, by, ols, terms, precomputed_quantiles=True
                ),
                terms,
            ),
            (
                fit_grid_full(bx, by, lasso, l1, terms, precomputed_quantiles=True),
                terms,
            ),
            (fit_grid_datasets(bx, by, ols, precomputed_quantiles=True, **flag), 3),
            (fit_grid_hyperparams(input_data, lasso, l1, **flag), 3),
            (
                fit_grid_datasets_hyperparams(
                    bx, by, lasso, l1, precomputed_quantiles=True, **flag
                ),
                3,
            ),
        )
        return x, y, ols, grids

    @FEW
    @given(
        dataset=strategies.grid_datasets(),
        boundedness=st.sampled_from(strategies.BOUNDEDNESS),
        data=st.data(),
    )
    def test_grid_functions_score_at_the_data_probabilities(
        self, dataset, boundedness, data
    ):
        """Every grid function's KS distance uses its fit's quantiles at the data's y."""
        x, y, ols, grids = self._grids(dataset, boundedness, data)
        for result, _ in grids:
            self._assert_scored_at_y(result, x, y, ols)

    @FEW
    @given(
        dataset=strategies.grid_datasets(),
        boundedness=st.sampled_from(strategies.BOUNDEDNESS),
        data=st.data(),
    )
    def test_feasible_marks_exactly_the_valid_fits(self, dataset, boundedness, data):
        """The feasible mask is check_feasibility of each fit; selection picks only those."""
        _, _, _, grids = self._grids(dataset, boundedness, data)
        for result, terms in grids:
            a = np.asarray(result.metalog.a).reshape(-1, result.metalog.a.shape[-1])
            counts = np.broadcast_to(terms, result.feasible.shape).reshape(-1)
            # Each fit at its own term count, not zero-padded as the grid checks it
            expected = [
                strategies.is_certified(a_i[:k])
                for a_i, k in zip(a, counts, strict=True)
            ]
            feasible = np.asarray(result.feasible).reshape(-1)
            self.assertEqual(feasible.tolist(), expected)
            event(f"{int(feasible.sum())} of {feasible.size} grid fits feasible")
            if not feasible.any():
                with self.assertRaises(ValueError):
                    find_best_config(result.ks_dist, result.feasible)
                continue
            idx, best_ks = find_best_config(result.ks_dist, result.feasible)
            self.assertTrue(
                bool(np.asarray(result.feasible)[tuple(np.atleast_1d(idx))])
            )
            ks = np.asarray(result.ks_dist).reshape(-1)
            self.assertEqual(float(best_ks), float(ks[feasible].min()))


class RvsPropertiesTest(parameterized.TestCase):
    """Regression tests for issue #26: rvs uses every HDR parameter."""

    @FEW
    @given(m=strategies.metalogs(), stream=strategies.hdr_streams())
    def test_hdr_rvs_continues_the_stream(self, m, stream):
        """Draws from trial k + 1 continue the draws from trial 1 at index k."""
        n, k, fields = stream

        def draw(trial, size):
            """Return size HDR draws starting at the given trial."""
            prng = HDRPRNGParameters(trial=trial, **fields)
            return m.rvs(MetalogRandomVariableParameters(prng_params=prng, size=size))

        np.testing.assert_array_equal(draw(1, n)[k:], draw(k + 1, n - k))


# A coefficient is followed by a space and its basis function, or ends the equation;
# the 0.5 in "(y - 0.5)" is followed by ")", so it doesn't match
_COEFFICIENT = re.compile(r"(?:^|\s([+-])\s)(-?[\d.]+(?:e[+-]?\d+)?)(?=\s|$)")


def _printed(model, **options):
    """Return what model.print(**options) writes."""
    buffer = io.StringIO()
    model.print(file=buffer, **options)
    return buffer.getvalue()


def _as_python(equation):
    """Rewrite a printed Q(y) as a Python expression in y, log and exp."""
    lines = equation.splitlines()
    expression = " ".join(line.strip() for line in lines).removeprefix("Q(y) = ")
    # A space between two operands, as in "2.5 ln(...)", is a multiplication
    expression = re.sub(r"(?<=[\w)]) (?=[\w(])", " * ", expression)
    return expression.replace("^", "**").replace("ln(", "log(")


class DisplayPropertiesTest(parameterized.TestCase):
    """Regression tests for issue #41: the printed equation is the model."""

    def assert_printed_equation_is_the_model(self, m, y):
        """Evaluate Q(y), printed at 17 digits with M(y) substituted, against ppf."""
        equation = _printed(m, header=False, m=False, verbose=True, precision=17)
        namespace = {
            "__builtins__": {},
            "log": np.log,
            "exp": np.exp,
            "y": np.asarray(y),
        }
        with np.errstate(over="ignore", invalid="ignore"):
            printed = eval(_as_python(equation), namespace)
        expected = np.asarray(m.ppf(y))
        # Relative to the quantiles' scale: an unbounded Q(y) can cross 0
        scale = np.max(np.abs(expected[np.isfinite(expected)]), initial=0.0)
        np.testing.assert_allclose(printed, expected, rtol=1e-12, atol=1e-12 * scale)

    @given(m=strategies.printable_metalogs())
    def test_every_form_prints_and_repr_is_exact(self, m):
        """No form raises, and repr's coefficients parse back to a exactly."""
        forms = (str(m), repr(m), m._repr_latex_(), _printed(m))
        self.assertTrue(all(isinstance(form, str) for form in forms))
        equation = repr(m).rpartition(" = ")[2]
        parsed = [
            float(number) * (-1.0 if sign == "-" else 1.0)
            for sign, number in _COEFFICIENT.findall(equation)
        ]
        self.assertEqual(parsed, m.a.tolist())

    @given(m=strategies.printable_metalogs())
    def test_layout(self, m):
        """M(y) prints one line per coefficient, and the LaTeX is balanced."""
        self.assertLen(_printed(m, header=False, q=False).splitlines(), len(m.a))
        latex = m._repr_latex_()
        self.assertEqual(latex.count("{"), latex.count("}"))
        self.assertEqual(latex.count(r"\begin{"), latex.count(r"\end{"))

    @given(m=strategies.metalogs(), y=strategies.probabilities())
    def test_printed_equation_is_the_model(self, m, y):
        """A Metalog's printed Q(y) is its quantile function."""
        self.assert_printed_equation_is_the_model(m, y)

    @given(triplet=strategies.spt_triplets(feasible=True), y=strategies.probabilities())
    def test_printed_spt_equation_is_the_model(self, triplet, y):
        """An SPTMetalog's printed Q(y) is its quantile function."""
        q, params = triplet
        m = fit_spt_metalog(jnp.asarray(q), params, precomputed_quantiles=True)
        self.assert_printed_equation_is_the_model(m, y)


if __name__ == "__main__":
    absltest.main()
