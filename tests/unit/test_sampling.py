"""Unit tests for cdf, rvs, the moments and the mode of bounded metalogs.

Regression tests for issue #49: these used a fixed table of 9,990 probabilities,
0.0001 apart, so cdf was quantized and clamped, rvs draws were discrete and
truncated, and bounded moments and modes were biased. Each test checks against a
reference that doesn't use the code under test: the inverse relation with ppf,
scipy's log-logistic distribution (an exact 2-term metalog), the HDR generator
itself, binomial counts, a KS test, or a numerical optimizer.
"""
# Copyright: Travis Jefferies 2026

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify
from scipy import optimize, stats

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    MetalogRandomVariableParameters,
)
from metalog_jax.metalog import Metalog, fit
from metalog_jax.utils import (
    DEFAULT_Y,
    DEFAULT_Y_FULL,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
    hdrprng,
)

LOWER = MetalogBoundedness.STRICTLY_LOWER_BOUND
UPPER = MetalogBoundedness.STRICTLY_UPPER_BOUND


def _params(boundedness, lower=0.0, upper=0.0, num_terms=2, method=None):
    """MetalogParameters, OLS unless another method is given."""
    return MetalogParameters(
        boundedness=boundedness,
        method=MetalogFitMethod.OLS if method is None else method,
        lower_bound=lower,
        upper_bound=upper,
        num_terms=num_terms,
    )


class SamplingTestBase(parameterized.TestCase):
    """Fits of every boundedness type to log-normal samples."""

    DATA = MetalogInputData.from_values(
        jnp.asarray(np.random.default_rng(1).lognormal(0.0, 0.5, 2000)),
        DEFAULT_Y,
        precomputed_quantiles=False,
    )
    # The samples run from about 0.2 to 6, inside every support
    FIT_CASES = (
        ("unbounded", MetalogBoundedness.UNBOUNDED, 0.0, 0.0),
        ("strictly_lower_bound", LOWER, 0.0, 0.0),
        ("strictly_upper_bound", UPPER, 0.0, 10.0),
        ("bounded", MetalogBoundedness.BOUNDED, 0.0, 10.0),
    )

    def _fit(self, boundedness, lower, upper):
        """A 6-term Feasible fit to the class data."""
        return fit(
            self.DATA,
            _params(boundedness, lower, upper, 6, MetalogFitMethod.Feasible),
        )


class CdfTest(SamplingTestBase):
    """cdf inverts ppf exactly, over all of (0, 1), for any input shape."""

    # Down to 1e-100 and up to 1 - 1e-12, far past the old table's 0.001 and 0.9999
    Y = (1e-100, 1e-15, 1e-5, 1e-4, 0.123456, 0.5, 0.99999, 1 - 1e-12)

    @parameterized.named_parameters(*SamplingTestBase.FIT_CASES)
    def test_cdf_inverts_ppf(self, boundedness, lower, upper):
        """cdf(ppf(y)) = y and sf(ppf(y)) = 1 - y, to rounding.

        The table version returned cdf(ppf(1e-5)) = 0.001 and
        cdf(ppf(0.123456)) = 0.1235.
        """
        m = self._fit(boundedness, lower, upper)
        y = jnp.asarray(self.Y)
        x = m.ppf(y)

        np.testing.assert_allclose(m.cdf(x), y, rtol=1e-12)
        np.testing.assert_allclose(m.sf(x), 1 - y, rtol=1e-9)
        np.testing.assert_allclose(m.logsf(x), np.log1p(-np.asarray(y)), rtol=1e-9)
        np.testing.assert_array_equal(m.p(x), m.cdf(x))

    @parameterized.named_parameters(
        ("lower_bound", LOWER, 1.0, 0.0, [0.5, 1.0], [0.0, 0.0]),
        ("upper_bound", UPPER, 0.0, 9.0, [9.0, 12.0], [1.0, 1.0]),
        ("bounded", MetalogBoundedness.BOUNDED, 1.0, 9.0, [0.0, 1.0, 9.0, 9.5], None),
    )
    def test_outside_the_support(self, boundedness, lower, upper, q, expected):
        """At or beyond a bound the CDF is 0 or 1, and the density d is 0."""
        m = Metalog(_params(boundedness, lower, upper), jnp.array([0.0, 0.5]))
        if expected is None:
            expected = [0.0, 0.0, 1.0, 1.0]

        np.testing.assert_array_equal(m.cdf(jnp.asarray(q)), expected)
        np.testing.assert_array_equal(m.sf(jnp.asarray(q)), 1 - np.asarray(expected))
        np.testing.assert_array_equal(m.d(jnp.asarray(q)), 0.0)

    def test_input_shapes_and_types(self):
        """A 0-d array, Python and NumPy numbers and an n-d array are all accepted.

        The table version rejected a 0-d JAX array, such as an element of ppf's
        output, and anything but 1-d JAX arrays and Python scalars.
        """
        m = self._fit(MetalogBoundedness.UNBOUNDED, 0.0, 0.0)
        median = m.ppf(jnp.array([0.5]))[0]

        np.testing.assert_allclose(m.cdf(median), 0.5, rtol=1e-12)
        self.assertEqual(m.cdf(median).shape, ())
        self.assertEqual(m.cdf(float(median)).shape, ())
        self.assertEqual(m.cdf(np.float64(median)).shape, ())
        self.assertEqual(m.cdf(2).dtype, jnp.float64)
        self.assertEqual(m.cdf(np.full((2, 3), float(median))).shape, (2, 3))
        self.assertTrue(np.isnan(m.cdf(jnp.nan)))

    def test_cdf_under_jit_and_vmap(self):
        """The CDF works inside jit and vmap, with the same values.

        cdf checks feasibility with checkify, so jit needs checkify.checkify.
        """
        m = self._fit(LOWER, 0.0, 0.0)
        x = m.ppf(jnp.asarray(self.Y))

        err, jitted = checkify.checkify(jax.jit(lambda q: m.cdf(q)))(x)
        err.throw()
        np.testing.assert_allclose(jitted, m.cdf(x), rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(m.cdf)(x), m.cdf(x), rtol=1e-12)


class RvsTest(SamplingTestBase):
    """rvs draws are continuous, reach the tails, and follow the distribution."""

    SIZE = 100_000

    def _draws(self, m, seed=0):
        """SIZE draws with jax.random."""
        prng = JaxUniformDistributionParameters(seed=seed)
        return np.asarray(
            m.rvs(MetalogRandomVariableParameters(prng_params=prng, size=self.SIZE))
        )

    @parameterized.named_parameters(*SamplingTestBase.FIT_CASES)
    def test_draws_are_continuous_and_reach_the_tails(self, boundedness, lower, upper):
        """No repeated values, and the tails beyond the old table are sampled.

        The table version drew only its 9,990 values, so never below Q(0.001) or
        above Q(0.9999). The counts there are binomial(SIZE, p); 5 standard
        deviations either way gives a false failure about once in 2 million runs.
        """
        m = self._fit(boundedness, lower, upper)
        draws = self._draws(m)
        low, high = (float(v) for v in m.ppf(jnp.array([0.001, 0.9999])))

        self.assertEqual(np.unique(draws).size, self.SIZE)
        for count, p in ((np.sum(draws < low), 0.001), (np.sum(draws > high), 1e-4)):
            mean, sd = self.SIZE * p, np.sqrt(self.SIZE * p * (1 - p))
            self.assertLess(abs(count - mean), 5 * sd)

    @parameterized.named_parameters(*SamplingTestBase.FIT_CASES)
    def test_draws_follow_the_cdf(self, boundedness, lower, upper):
        """A KS test of the draws against cdf doesn't reject at the 0.1% level.

        The seed is fixed, so the test is deterministic.
        """
        m = self._fit(boundedness, lower, upper)
        draws = self._draws(m, seed=1)[:20_000]

        result = stats.kstest(draws, lambda v: np.asarray(m.cdf(jnp.asarray(v))))
        self.assertGreater(result.pvalue, 1e-3)

    def test_hdr_draws_are_the_quantiles_of_hdr_uniforms(self):
        """Draw i is ppf of HDR trial trial + i, which hdrprng puts in (0, 1)."""
        m = self._fit(MetalogBoundedness.BOUNDED, 0.0, 10.0)
        prng = HDRPRNGParameters(trial=7, variable=3, entity=2)

        draws = m.rvs(MetalogRandomVariableParameters(prng_params=prng, size=5))

        u = jnp.array([hdrprng(prng.replace(trial=7 + i)) for i in range(5)])
        np.testing.assert_array_equal(draws, m.ppf(u))


class BoundedMomentsTest(parameterized.TestCase):
    """Moments of semi-bounded metalogs, exact for the 2-term log-logistic.

    A lower-bounded 2-term metalog has X = L + exp(a1) (y / (1 - y))^a2, the
    log-logistic (scipy.stats.fisk) with c = 1 / a2 and scale exp(a1). Its k-th
    moment is finite only for k a2 < 1.
    """

    @parameterized.parameters(0.1, 0.2)
    def test_lower_bound_matches_log_logistic(self, a2):
        """Mean, variance, skewness, kurtosis and mode match scipy's closed forms."""
        a1, lower = 0.3, 1.0
        m = Metalog(_params(LOWER, lower), jnp.array([a1, a2]))
        ref = stats.fisk(1 / a2, loc=lower, scale=np.exp(a1))
        mean, var, skew, excess = ref.stats("mvsk")
        c = 1 / a2
        mode = lower + np.exp(a1) * ((c - 1) / (c + 1)) ** (1 / c)

        np.testing.assert_allclose(m.mean, mean, rtol=1e-12)
        np.testing.assert_allclose(m.var, var, rtol=1e-12)
        np.testing.assert_allclose(m.std, np.sqrt(var), rtol=1e-12)
        np.testing.assert_allclose(m.skewness, skew, rtol=1e-10)
        np.testing.assert_allclose(m.kurtosis, excess + 3, rtol=1e-10)
        np.testing.assert_allclose(m.mode, mode, rtol=1e-12)

    def test_upper_bound_mirrors_the_lower_bound(self):
        """U - X is log-logistic for an upper-bounded 2-term metalog."""
        m = Metalog(_params(UPPER, upper=5.0), jnp.array([-0.3, 0.2]))
        ref = stats.fisk(5.0, scale=np.exp(0.3))
        mean, var, skew, excess = ref.stats("mvsk")

        np.testing.assert_allclose(m.mean, 5.0 - mean, rtol=1e-12)
        np.testing.assert_allclose(m.var, var, rtol=1e-12)
        np.testing.assert_allclose(m.skewness, -skew, rtol=1e-10)
        np.testing.assert_allclose(m.kurtosis, excess + 3, rtol=1e-10)

    @parameterized.named_parameters(
        ("no_third_moment", LOWER, 0.4, [True, True, False, False], np.inf),
        ("no_variance", LOWER, 0.6, [True, False, False, False], np.inf),
        ("no_mean", LOWER, 1.2, [False, False, False, False], np.inf),
        ("no_mean_upper_bound", UPPER, 1.2, [False, False, False, False], -np.inf),
    )
    def test_moments_that_do_not_exist(self, boundedness, a2, exists, infinite_mean):
        """Missing moments are inf or NaN, as in scipy.stats.pareto.

        The table version returned finite averages of draws for all of them.
        """
        m = Metalog(_params(boundedness, upper=5.0), jnp.array([0.3, a2]))
        mean, var, skew, kurt = (
            float(v) for v in (m.mean, m.var, m.skewness, m.kurtosis)
        )

        self.assertEqual(np.isfinite([mean, var, skew, kurt]).tolist(), exists)
        if not exists[0]:
            self.assertEqual(mean, infinite_mean)
        if not exists[1]:
            self.assertEqual(var, np.inf)
            self.assertEqual(float(m.std), np.inf)
        self.assertEqual(
            [np.isnan(skew), np.isnan(kurt)], [not exists[2], not exists[3]]
        )

    def test_issue_example(self):
        """The issue's 6-term lognormal fit: the mean is 1.6453, not 0.31% low."""
        samples = np.random.default_rng(1).lognormal(0.0, 1.0, 2000)
        data = MetalogInputData.from_values(
            jnp.asarray(samples), DEFAULT_Y, precomputed_quantiles=False
        )
        m = fit(data, _params(LOWER, num_terms=6, method=MetalogFitMethod.Feasible))

        # The issue's reference: quadrature of Q on the logit scale (scipy quad)
        np.testing.assert_allclose(m.mean, 1.6453233014, rtol=1e-9)


class BoundedModeTest(SamplingTestBase):
    """The mode of a bounded or semi-bounded metalog, refined past the old grid."""

    @parameterized.named_parameters(*SamplingTestBase.FIT_CASES[1:])
    def test_mode_maximizes_the_density(self, boundedness, lower, upper):
        """The mode is where scipy's bounded optimizer puts the density's maximum.

        The table version returned the best of 9,990 probabilities, 0.0001 apart.
        """
        m = self._fit(boundedness, lower, upper)
        y_grid = DEFAULT_Y_FULL[jnp.argmax(m.pdf(DEFAULT_Y_FULL))]

        best = optimize.minimize_scalar(
            lambda y: -float(m.pdf(jnp.array([y]))[0]),
            bounds=(float(y_grid) - 2e-4, float(y_grid) + 2e-4),
            method="bounded",
            options={"xatol": 1e-12},
        )

        expected = float(m.ppf(jnp.array([best.x]))[0])
        np.testing.assert_allclose(m.mode, expected, rtol=1e-6)
        self.assertGreaterEqual(float(m.d(m.mode)), -best.fun * (1 - 1e-9))

    def test_density_rising_to_the_bound(self):
        """A log-logistic with c < 1 has its density rise to L, so the mode is L.

        To rounding: L + exp(M) at the edge of the search is about L + 1e-21. The
        table version returned Q(0.001), 2.00003 here.
        """
        m = Metalog(_params(LOWER, lower=2.0), jnp.array([0.0, 1.5]))

        self.assertGreaterEqual(float(m.mode), 2.0)
        self.assertLess(float(m.mode) - 2.0, 1e-15)


if __name__ == "__main__":
    absltest.main()
