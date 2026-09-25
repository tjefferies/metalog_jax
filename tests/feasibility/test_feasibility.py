# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.feasibility (Metalog 2.0: a*, Algorithm 1, exact moments).

References:
    Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
    Properties of the Metalog Distribution. Darden Business School Working
    Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
)
from metalog_jax.feasibility import (
    TermOrder,
    best_feasible_fit,
    check_feasibility,
    feasibility_function,
    get_engine,
    project_onto_polyhedron,
    raw_moment,
    summary_stats,
)
from metalog_jax.metalog import fit

jax.config.update("jax_enable_x64", True)

M2 = TermOrder.METALOG_2
FIG1_A = jnp.array(
    [23.5, 35.02, -17.01, -81.6, 117.5, -77.2]
)  # paper, Figure 1 (6-metalog)
README_X, README_Y = jnp.array([1.0, 2, 4, 8, 12]), jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
PAPER_X, PAPER_Y = (
    jnp.array([8.0, 12, 19, 20, 35, 40, 45]),
    (jnp.arange(1, 8) - 0.5) / 7,
)


def dense_min_g(engine, a, n=400_001):
    """Brute-force min of G = y(1-y)M' on a dense logit grid plus both corners."""
    u = jnp.concatenate([jnp.array([-1000.0, 1000.0]), jnp.linspace(-40.0, 40.0, n)])
    return float(jnp.min(jax.vmap(lambda uu: feasibility_function(engine, a, uu))(u)))


def quantile(engine, a, y):
    """Evaluate M(y) = mu(y - 1/2) + s(y - 1/2) * logit(y) with NumPy."""
    mu, s = engine.split(a)
    x = y - 0.5
    return np.polyval(np.asarray(mu)[::-1], x) + np.polyval(
        np.asarray(s)[::-1], x
    ) * np.log(y / (1 - y))


class TestAlgorithm1:
    """Air-tight inflection points (Algorithm 1) and the feasibility report."""

    def test_figure1_modes_and_antimode(self):
        """Figure 1 6-metalog: two modes around one anti-mode."""
        rep = check_feasibility(get_engine(6, M2), FIG1_A)
        roots = np.asarray(rep.roots)[~np.isnan(np.asarray(rep.roots))]
        np.testing.assert_allclose(
            roots, [0.19016242, 0.61548439, 0.93005277], atol=1e-7
        )
        np.testing.assert_array_equal(np.asarray(rep.is_mode)[:3], [True, False, True])
        assert bool(rep.feasible)

    def test_polynomial_only_metalog(self):
        """A metalog with s == 0 still yields the exact roots of M''."""
        # s == 0: M'' has roots at exactly 0.2 and 0.7
        rep = check_feasibility(
            get_engine(9, M2), jnp.array([0.0, 0, 0, 5, -0.36, 0, 0, 0.2, 1])
        )
        roots = np.asarray(rep.roots)[~np.isnan(np.asarray(rep.roots))]
        np.testing.assert_allclose(roots, [0.2, 0.7], atol=1e-9)

    def test_detects_infeasibility_near_the_edge(self):
        """A negative slope at y = 1e-7 is caught."""
        # negative-slope inflection at y = 1e-7 (below a 1e-6 root filter)
        rep = check_feasibility(get_engine(3), jnp.array([0.0, 1000.0, 1999.9996]))
        assert not bool(rep.feasible)
        np.testing.assert_allclose(np.nanmin(np.asarray(rep.roots)), 1e-7, rtol=1e-6)

    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
    def test_scale_invariant(self, scale):
        """Roots do not depend on the units of the coefficients."""
        rep = check_feasibility(get_engine(6, M2), FIG1_A * scale)
        roots = np.asarray(rep.roots)[~np.isnan(np.asarray(rep.roots))]
        np.testing.assert_allclose(
            roots, [0.19016242, 0.61548439, 0.93005277], atol=1e-7
        )


class TestMoments:
    """Exact moments (Lemma 1, Proposition 3)."""

    @pytest.mark.parametrize(
        "k,order,a",
        [(4, TermOrder.KEELIN_2016, [22.62, 5.64, 3.19, 35.51]), (6, M2, FIG1_A)],
    )
    def test_match_quadrature(self, k, order, a):
        """Mean, variance, skewness and kurtosis agree with adaptive quadrature."""
        eng, a = get_engine(k, order), jnp.array(a)
        st = summary_stats(eng, a)
        f = lambda y: quantile(eng, a, y)  # noqa: E731
        mean = quad(f, 0, 1, limit=400)[0]
        c = [quad(lambda y: (f(y) - mean) ** p, 0, 1, limit=400)[0] for p in (2, 3, 4)]
        np.testing.assert_allclose(float(st["mean"]), mean, rtol=1e-9)
        np.testing.assert_allclose(float(st["variance"]), c[0], rtol=1e-8)
        np.testing.assert_allclose(float(st["skewness"]), c[1] / c[0] ** 1.5, rtol=1e-6)
        np.testing.assert_allclose(float(st["kurtosis"]), c[2] / c[0] ** 2, rtol=1e-6)

    def test_two_term_metalog_is_logistic(self):
        """The 2-term metalog is the logistic: skewness 0 and Pearson kurtosis 4.2."""
        st = summary_stats(get_engine(2), jnp.array([3.0, 2.0]))
        np.testing.assert_allclose(float(st["mean"]), 3.0, rtol=1e-14)
        np.testing.assert_allclose(
            float(st["variance"]), 4.0 * np.pi**2 / 3, rtol=1e-14
        )
        np.testing.assert_allclose(float(st["skewness"]), 0.0, atol=1e-14)
        np.testing.assert_allclose(float(st["kurtosis"]), 4.2, rtol=1e-13)

    def test_mean_coefficients_equation5(self):
        """The gradient of the mean reproduces Equation (5)."""
        g = jax.grad(lambda a: raw_moment(get_engine(9, M2), a, 1))(jnp.ones(9))
        np.testing.assert_allclose(
            g, [1, 0, 0.5, 0, 1 / 12, 0, 1 / 12, 0, 1 / 80], atol=1e-14
        )


class TestAStar:
    """Best feasible fit a*."""

    def test_projection_qp(self):
        """The interior-point projection solves a one-constraint problem exactly."""
        z, lam = project_onto_polyhedron(
            jnp.array([1.0, 2, 3]),
            jnp.array([[1.0, 0, 0], [0, 0, 0]]),
            jnp.array([5.0, 0]),
            jnp.array([True, False]),
        )
        np.testing.assert_allclose(z, [5, 2, 3], atol=1e-10)
        np.testing.assert_allclose(lam[0], 4, atol=1e-8)

    def test_readme_example(self):
        """Reference README data: OLS is infeasible, a* matches the reference RSS."""
        res = best_feasible_fit(README_X, README_Y, 4)
        assert not bool(check_feasibility(get_engine(4), res.a_ols).feasible)
        assert bool(res.feasible) and bool(res.converged)
        np.testing.assert_allclose(
            float(res.rss_star), 0.72307, rtol=1e-4
        )  # reference code: 0.72307

    def test_paper_data(self):
        """Paper data: k=5 matches the reference; k=6 OLS is already feasible."""
        res5 = best_feasible_fit(PAPER_X, PAPER_Y, 5, order=M2)
        np.testing.assert_allclose(float(res5.rss_star), 39.381, rtol=1e-4)
        res6 = best_feasible_fit(PAPER_X, PAPER_Y, 6, order=M2)  # OLS already feasible
        assert int(res6.iterations) == 0
        np.testing.assert_allclose(res6.a_star, res6.a_ols, rtol=1e-12)

    def test_feasible_where_reference_is_not(self):
        """The reference returns an infeasible a* here; ours is feasible."""
        x = jnp.array([2.129, 3.014, 5.265, 10.774, 11.268, 21.431, 23.617, 73.921])
        y = jnp.array(
            [0.16833, 0.27694, 0.50937, 0.63406, 0.73212, 0.81683, 0.83328, 0.86514]
        )
        eng = get_engine(3)
        ref_a = jnp.array(
            [3.97529561, 12.62056867, 25.24110525]
        )  # reference find_a_star output
        assert dense_min_g(eng, ref_a) < -1.0  # negative density near y = 0.083
        res = best_feasible_fit(x, y, 3)
        assert bool(res.feasible) and dense_min_g(eng, res.a_star) > 0

    @pytest.mark.parametrize("seed", range(4))
    def test_random_fits_are_feasible_and_no_better_than_ols(self, seed):
        """Random data: certified feasible, dense-grid feasible, RSS >= OLS."""
        rng = np.random.default_rng(seed)
        n = 9
        y = jnp.array(np.sort(rng.uniform(0.02, 0.98, n)))
        x = jnp.array(np.sort(rng.lognormal(0, 1, n) * 10))
        res = best_feasible_fit(x, y, 6)
        assert bool(res.feasible) and bool(res.converged)
        assert float(res.rss_star) >= float(res.rss_ols) - 1e-9
        assert dense_min_g(get_engine(6), res.a_star) > -1e-9 * float(
            jnp.max(jnp.abs(x))
        )

    def test_vmap_over_datasets(self):
        """A vmapped fit equals per-dataset fits."""
        xs = jnp.stack([README_X, README_X * 2 + 1, README_X**1.5])
        batched = jax.vmap(lambda xx: best_feasible_fit(xx, README_Y, 4).a_star)(xs)
        for i in range(3):
            np.testing.assert_allclose(
                batched[i], best_feasible_fit(xs[i], README_Y, 4).a_star, rtol=1e-9
            )


class TestFitIntegration:
    """MetalogFitMethod.Feasible through the public fit() API."""

    @pytest.mark.parametrize(
        "bnd", [MetalogBoundedness.UNBOUNDED, MetalogBoundedness.STRICTLY_LOWER_BOUND]
    )
    def test_feasible_method_repairs_ols(self, bnd):
        """OLS raises on invalid PDFs; Feasible returns a valid metalog."""
        data = MetalogInputData.from_values(
            README_X, README_Y, precomputed_quantiles=True
        )

        def params(m):
            return MetalogParameters(
                boundedness=bnd, method=m, lower_bound=0.0, upper_bound=0.0, num_terms=4
            )

        with pytest.raises(Exception, match="non-positive"):
            fit(data, params(MetalogFitMethod.OLS))
        m = fit(data, params(MetalogFitMethod.Feasible))
        assert bool(check_feasibility(get_engine(4), m.a).feasible)
        assert float(jnp.min(m.pdf(jnp.linspace(1e-6, 1 - 1e-6, 10001)))) > 0


class TestMetalogMomentProperties:
    """Metalog moment properties: exact when unbounded, Monte Carlo when bounded."""

    X = jnp.array([2.1, 3.5, 4.2, 5.8, 6.1, 7.3, 8.9, 12.4, 15.2, 18.7])

    @staticmethod
    def _by_quadrature(m, lim=30.0, n=60001):
        """Mean, variance, skewness, kurtosis by the trapezoid rule in u = logit(y).

        With y = sigmoid(u), dy = y(1-y) du and the integrand decays like
        exp(-|u|), so the rule converges very fast; only ``ppf`` is used.
        """
        u = np.linspace(-lim, lim, n)
        w = np.exp(-np.logaddexp(0, -u) - np.logaddexp(0, u))
        q = np.asarray(m.ppf(jnp.asarray(1 / (1 + np.exp(-u)))), dtype=np.float64)
        h = u[1] - u[0]
        mean = h * np.sum(q * w)
        c = q - mean
        var = h * np.sum(c**2 * w)
        return mean, var, h * np.sum(c**3 * w) / var**1.5, h * np.sum(c**4 * w) / var**2

    @staticmethod
    def _assert_moments(m, ref, rtol=1e-9):
        mean, var, skew, kurt = ref
        np.testing.assert_allclose(float(m.mean), mean, rtol=rtol)
        np.testing.assert_allclose(float(m.var), var, rtol=rtol)
        np.testing.assert_allclose(float(m.std), np.sqrt(var), rtol=rtol)
        np.testing.assert_allclose(float(m.skewness), skew, rtol=100 * rtol, atol=1e-10)
        np.testing.assert_allclose(float(m.kurtosis), kurt, rtol=100 * rtol)

    @pytest.mark.parametrize(
        "k,method", [(3, MetalogFitMethod.OLS), (5, MetalogFitMethod.Feasible)]
    )
    def test_unbounded_moments_are_exact(self, k, method):
        """Unbounded mean/var/std/skewness/kurtosis agree with quadrature of the quantile function."""
        from metalog_jax.utils import DEFAULT_Y

        data = MetalogInputData.from_values(
            self.X, DEFAULT_Y, precomputed_quantiles=False
        )
        m = fit(
            data,
            MetalogParameters(
                boundedness=MetalogBoundedness.UNBOUNDED,
                method=method,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=k,
            ),
        )
        self._assert_moments(m, self._by_quadrature(m))

    def test_unbounded_spt_moments_are_exact(self):
        """The 3-term SPT metalog shares the basis, so its moments are exact too."""
        from metalog_jax.base import SPTMetalogParameters
        from metalog_jax.metalog import fit_spt_metalog

        params = SPTMetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            alpha=0.1,
        )
        m = fit_spt_metalog(self.X, params)
        self._assert_moments(m, self._by_quadrature(m))

    def test_bounded_moments_still_use_monte_carlo(self):
        """Semi-bounded metalogs keep the seeded 20,000-draw estimate."""
        from metalog_jax.base.parameters import MetalogRandomVariableParameters
        from metalog_jax.utils import DEFAULT_Y, JaxUniformDistributionParameters

        data = MetalogInputData.from_values(
            self.X, DEFAULT_Y, precomputed_quantiles=False
        )
        m = fit(
            data,
            MetalogParameters(
                boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
                method=MetalogFitMethod.OLS,
                lower_bound=0.0,
                upper_bound=0.0,
                num_terms=4,
            ),
        )
        rv = m.rvs(
            MetalogRandomVariableParameters(
                prng_params=JaxUniformDistributionParameters(seed=0), size=20_000
            )
        )
        assert float(m.mean) == float(jnp.mean(rv))
        assert float(m.var) == float(jnp.var(rv))
        assert float(m.std) == float(jnp.std(rv))
        c = rv - jnp.mean(rv)
        m2 = jnp.mean(c**2)
        np.testing.assert_allclose(
            float(m.skewness), float(jnp.mean(c**3) / m2**1.5), rtol=1e-12
        )
        np.testing.assert_allclose(
            float(m.kurtosis), float(jnp.mean(c**4) / m2**2), rtol=1e-12
        )
