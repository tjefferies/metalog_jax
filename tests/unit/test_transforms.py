"""Unit tests for running metalog_jax under JAX transforms.

These tests check jax.jit and jax.vmap over probabilities, models and parameters
against eager results, for every boundedness type and fit method. Value checks
inside transforms use checkify, so jitted calls are wrapped in checkify.checkify.
"""
# Copyright: Travis Jefferies 2026

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import checkify

from metalog_jax import feasibility
from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    MetalogRandomVariableParameters,
    SPTMetalogParameters,
)
from metalog_jax.feasibility import get_engine
from metalog_jax.metalog import Metalog, fit, fit_spt_metalog
from metalog_jax.utils import (
    DEFAULT_Y,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
)


class _CompileCounter:
    """Count XLA compilations, from the event jax.monitoring records for each.

    jax 0.8.0, the oldest supported version, can't unregister a listener, so one
    counter is registered for the whole test session.
    """

    EVENT = "/jax/core/compile/backend_compile_duration"

    def __init__(self):
        """Register the listener."""
        self.count = 0
        jax.monitoring.register_event_duration_secs_listener(self._record)

    def _record(self, event, duration, **kwargs):
        """Count a compilation event."""
        del duration, kwargs
        if event == self.EVENT:
            self.count += 1


_COMPILES = _CompileCounter()


class TransformTestBase(parameterized.TestCase):
    """Shared fixtures and helpers for the transform tests."""

    SAMPLES = (1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1, 10.5, 11.0, 12.4)
    PROBABILITIES = (0.25, 0.5, 0.75)
    VALUES = (3.0, 5.0, 8.0)
    NUM_TERMS = 3
    BOUNDEDNESS_CASES = (
        ("unbounded", MetalogBoundedness.UNBOUNDED, 0.0, 0.0),
        ("strictly_lower_bound", MetalogBoundedness.STRICTLY_LOWER_BOUND, 0.0, 0.0),
        ("strictly_upper_bound", MetalogBoundedness.STRICTLY_UPPER_BOUND, 0.0, 15.0),
        ("bounded", MetalogBoundedness.BOUNDED, 0.0, 15.0),
    )
    FIT_CASES = tuple(
        (f"{name}_{method.name.lower()}", boundedness, lower, upper, method)
        for name, boundedness, lower, upper in BOUNDEDNESS_CASES
        for method in (
            MetalogFitMethod.OLS,
            MetalogFitMethod.Lasso,
            MetalogFitMethod.Feasible,
        )
    )
    # Compiled and eager code can round differently, including in the iterative
    # Lasso and feasible solvers, so allow more than float64 epsilon
    RTOL = 1e-9
    # Bounded and semi-bounded moments average 20,000 fixed-seed draws; compiled
    # code sums the same draws in a different order
    MOMENT_RTOL = 1e-7

    def _data(self, scale=1.0):
        """Return MetalogInputData for the class samples, optionally scaled."""
        samples = scale * jnp.asarray(self.SAMPLES)
        return MetalogInputData.from_values(
            samples, DEFAULT_Y, precomputed_quantiles=False
        )

    def _params(self, boundedness, lower, upper, method=MetalogFitMethod.Feasible):
        """Return MetalogParameters for the class number of terms."""
        return MetalogParameters(
            boundedness=boundedness,
            method=method,
            lower_bound=lower,
            upper_bound=upper,
            num_terms=self.NUM_TERMS,
        )

    def _methods(self):
        """Return (name, function of model and input, input) for each method."""
        y = jnp.asarray(self.PROBABILITIES)
        x = jnp.asarray(self.VALUES)
        return (
            ("ppf", lambda m, v: m.ppf(v), y),
            ("pdf", lambda m, v: m.pdf(v), y),
            ("logpdf", lambda m, v: m.logpdf(v), y),
            ("q", lambda m, v: m.q(v), y),
            ("cdf", lambda m, v: m.cdf(v), x),
            ("sf", lambda m, v: m.sf(v), x),
            ("p", lambda m, v: m.p(v), x),
            ("d", lambda m, v: m.d(v), x),
        )

    def _checked_jit(self, fn):
        """Jit fn under checkify, raise any check failure, and return its output."""
        jitted = checkify.checkify(jax.jit(fn))

        def run(*args):
            """Call the jitted function and raise its errors."""
            err, out = jitted(*args)
            err.throw()
            return out

        return run


class JitTest(TransformTestBase):
    """Regression tests for issue #25: jax.jit over models and parameters."""

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES)
    def test_jit_over_probabilities_matches_eager(self, boundedness, lower, upper):
        """Jitted methods with the model closed over match eager calls."""
        m = fit(self._data(), self._params(boundedness, lower, upper))
        for name, method, arg in self._methods():
            with self.subTest(method=name):
                jitted = self._checked_jit(lambda v, method=method: method(m, v))
                np.testing.assert_allclose(jitted(arg), method(m, arg), rtol=self.RTOL)

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES)
    def test_jit_with_model_argument_matches_eager(self, boundedness, lower, upper):
        """Jitted methods and moments with the model as an argument match eager."""
        m = fit(self._data(), self._params(boundedness, lower, upper))
        for name, method, arg in self._methods():
            with self.subTest(method=name):
                jitted = self._checked_jit(method)
                np.testing.assert_allclose(
                    jitted(m, arg), method(m, arg), rtol=self.RTOL
                )
        for name in ("mean", "var", "std"):
            with self.subTest(method=name):
                jitted = self._checked_jit(lambda mm, name=name: getattr(mm, name))
                np.testing.assert_allclose(
                    jitted(m), getattr(m, name), rtol=self.MOMENT_RTOL
                )

    @parameterized.named_parameters(*TransformTestBase.FIT_CASES)
    def test_jit_fit_with_parameters_argument_matches_eager(
        self, boundedness, lower, upper, method
    ):
        """A jitted fit with the parameters as an argument matches the eager fit."""
        data = self._data()
        params = self._params(boundedness, lower, upper, method)
        jitted = self._checked_jit(lambda d, p: fit(d, p).a)
        np.testing.assert_allclose(
            jitted(data, params), fit(data, params).a, rtol=self.RTOL
        )

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES)
    def test_jit_fit_spt_metalog_with_parameters_argument(
        self, boundedness, lower, upper
    ):
        """A jitted SPT fit with the parameters as an argument matches eager."""
        samples = jnp.asarray(self.SAMPLES)
        params = SPTMetalogParameters(
            boundedness=boundedness, alpha=0.1, lower_bound=lower, upper_bound=upper
        )
        jitted = self._checked_jit(lambda d, p: fit_spt_metalog(d, p).a)
        np.testing.assert_allclose(
            jitted(samples, params),
            fit_spt_metalog(samples, params).a,
            rtol=self.RTOL,
        )

    def test_jitted_feasible_fit_leaves_engine_cache_usable(self):
        """A first Feasible fit under jit doesn't cache tracers in the engine."""
        get_engine.cache_clear()
        data = self._data()
        params = self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0)
        jitted = self._checked_jit(lambda d, p: fit(d, p).a)(data, params)
        np.testing.assert_allclose(jitted, fit(data, params).a, rtol=self.RTOL)

    def test_compiles_once_per_static_fields(self):
        """Models that share their static fields share one compilation."""
        traces = []
        y = jnp.asarray(self.PROBABILITIES)
        jitted = self._checked_jit(lambda m: (traces.append(1), m.ppf(y))[1])
        m = fit(self._data(), self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0))
        jitted(m)
        jitted(m.replace(a=1.01 * m.a))
        self.assertLen(traces, 1)
        # A different boundedness type selects different code, so it compiles again
        lower = self._params(MetalogBoundedness.STRICTLY_LOWER_BOUND, 0.0, 0.0)
        jitted(fit(self._data(), lower))
        self.assertLen(traces, 2)

    def test_checkified_jit_reports_invalid_probabilities(self):
        """A jitted ppf returns an error for probabilities outside (0, 1)."""
        m = fit(self._data(), self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0))
        err, _ = checkify.checkify(jax.jit(lambda v: m.ppf(v)))(jnp.array([0.0, 0.5]))
        self.assertIn("must be strictly greater than 0", err.get())
        with self.assertRaises(checkify.JaxRuntimeError):
            err.throw()

    def test_eager_invalid_probabilities_raise_value_error(self):
        """Eager calls still raise ValueError for probabilities outside (0, 1)."""
        m = fit(self._data(), self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0))
        with self.assertRaises(ValueError):
            m.ppf(jnp.array([0.5, 1.0]))


class VmapTest(TransformTestBase):
    """Regression tests for issue #25: jax.vmap over stacked models and bounds."""

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES)
    def test_vmap_over_stacked_models_matches_loop(self, boundedness, lower, upper):
        """Vmapping methods over stacked models matches calling each model."""
        params = self._params(boundedness, lower, upper)
        models = [fit(self._data(scale), params) for scale in (1.0, 1.05, 1.1)]
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)
        for name, method, arg in self._methods():
            with self.subTest(method=name):
                batched = jax.vmap(lambda m, method=method, arg=arg: method(m, arg))(
                    stacked
                )
                looped = [method(m, arg) for m in models]
                np.testing.assert_allclose(batched, looped, rtol=self.RTOL)

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES[1:])
    def test_vmap_over_bounds_matches_loop(self, boundedness, lower, upper):
        """Vmapping ppf over a traced bound matches each bound in turn."""
        m = fit(self._data(), self._params(boundedness, lower, upper))
        y = jnp.asarray(self.PROBABILITIES)
        # Shift whichever bound the boundedness type uses
        field = (
            "upper_bound"
            if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND
            else "lower_bound"
        )
        shifts = jnp.array([0.0, -0.5, -1.0])

        def ppf_with_shift(shift):
            """Return ppf with the bound moved by shift."""
            bound = getattr(m.metalog_params, field) + shift
            params = m.metalog_params.replace(**{field: bound})
            return m.replace(metalog_params=params).ppf(y)

        np.testing.assert_allclose(
            jax.vmap(ppf_with_shift)(shifts),
            [ppf_with_shift(float(s)) for s in shifts],
            rtol=self.RTOL,
        )

    def test_vmap_rejects_invalid_probabilities(self):
        """Vmapping ppf over a batch with an invalid row raises JaxRuntimeError."""
        m = fit(self._data(), self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0))
        batch = jnp.array([[0.25, 0.5], [0.0, 0.5]])
        with self.assertRaises(checkify.JaxRuntimeError):
            jax.vmap(lambda v: m.ppf(v))(batch)

    def test_stacking_mixed_boundedness_fails(self):
        """Models with different boundedness types can't be stacked."""
        unbounded = self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0)
        lower = self._params(MetalogBoundedness.STRICTLY_LOWER_BOUND, 0.0, 0.0)
        models = [fit(self._data(), unbounded), fit(self._data(), lower)]
        # JAX's wording varies by version: "pytree structure error" in recent
        # releases, "Mismatch custom dataclass node data" in 0.8
        with self.assertRaisesRegex(ValueError, "pytree|node data"):
            jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)


class RvsTest(TransformTestBase):
    """Regression tests for issue #25: rvs under jax.jit and jax.vmap."""

    SIZE = 50

    def _rv_params(self, name, value=None):
        """Return rvs parameters for the "jax" or "hdr" generator."""
        if name == "jax":
            prng = JaxUniformDistributionParameters(seed=7 if value is None else value)
        else:
            prng = HDRPRNGParameters(trial=1 if value is None else value)
        return MetalogRandomVariableParameters(prng_params=prng, size=self.SIZE)

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES)
    def test_jit_rvs_matches_eager(self, boundedness, lower, upper):
        """Jitted rvs, with the model or rvs parameters as arguments, matches eager."""
        m = fit(self._data(), self._params(boundedness, lower, upper))
        for name in ("jax", "hdr"):
            rv_params = self._rv_params(name)
            eager = m.rvs(rv_params)
            with self.subTest(prng=name, argument="model"):
                jitted = self._checked_jit(lambda mm, p=rv_params: mm.rvs(p))
                np.testing.assert_allclose(jitted(m), eager, rtol=self.RTOL)
            with self.subTest(prng=name, argument="rvs parameters"):
                jitted = self._checked_jit(lambda mm, p: mm.rvs(p))
                np.testing.assert_allclose(jitted(m, rv_params), eager, rtol=self.RTOL)

    @parameterized.named_parameters(*TransformTestBase.BOUNDEDNESS_CASES)
    def test_vmap_rvs_over_stacked_models_matches_loop(self, boundedness, lower, upper):
        """Vmapping rvs over stacked models matches sampling each model."""
        params = self._params(boundedness, lower, upper)
        models = [fit(self._data(scale), params) for scale in (1.0, 1.05)]
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)
        rv_params = self._rv_params("jax")
        np.testing.assert_allclose(
            jax.vmap(lambda m: m.rvs(rv_params))(stacked),
            [m.rvs(rv_params) for m in models],
            rtol=self.RTOL,
        )

    def test_vmap_rvs_over_seeds_and_trials_matches_loop(self):
        """Vmapping rvs over JAX seeds or HDR trials matches a loop."""
        m = fit(self._data(), self._params(MetalogBoundedness.UNBOUNDED, 0.0, 0.0))
        for name in ("jax", "hdr"):
            with self.subTest(prng=name):
                batched = jax.vmap(
                    lambda v, name=name: m.rvs(self._rv_params(name, v))
                )(jnp.array([1, 2, 3]))
                looped = [m.rvs(self._rv_params(name, v)) for v in (1, 2, 3)]
                np.testing.assert_allclose(batched, looped, rtol=self.RTOL)


class FromValuesTest(TransformTestBase):
    """Regression tests for issue #25: MetalogInputData.from_values in transforms."""

    def _batch(self):
        """Return three sample arrays of the same length."""
        samples = jnp.asarray(self.SAMPLES)
        return jnp.stack([samples, 1.05 * samples, samples + 0.5])

    def test_vmap_from_values_matches_loop(self):
        """Vmapping from_values over raw samples matches a loop."""
        batch = self._batch()

        def quantiles(samples):
            """Return the quantiles from_values computes."""
            data = MetalogInputData.from_values(
                samples, DEFAULT_Y, precomputed_quantiles=False
            )
            return data.x

        np.testing.assert_allclose(
            jax.vmap(quantiles)(batch), [quantiles(s) for s in batch], rtol=self.RTOL
        )

    @parameterized.named_parameters(*TransformTestBase.FIT_CASES)
    def test_vmap_from_values_and_fit_matches_loop(
        self, boundedness, lower, upper, method
    ):
        """Vmapping from_values and fit over raw samples matches a loop."""
        params = self._params(boundedness, lower, upper, method)

        def fit_samples(samples):
            """Build the input data and fit it."""
            data = MetalogInputData.from_values(
                samples, DEFAULT_Y, precomputed_quantiles=False
            )
            return fit(data, params).a

        batch = self._batch()
        np.testing.assert_allclose(
            jax.vmap(fit_samples)(batch),
            [fit_samples(s) for s in batch],
            rtol=self.RTOL,
        )

    def test_checkified_jit_from_values_matches_eager(self):
        """A jitted from_values, functionalized with checkify, matches eager."""
        samples = jnp.asarray(self.SAMPLES)
        jitted = self._checked_jit(
            lambda s: MetalogInputData.from_values(s, DEFAULT_Y, False).x
        )
        eager = MetalogInputData.from_values(samples, DEFAULT_Y, False).x
        np.testing.assert_allclose(jitted(samples), eager, rtol=self.RTOL)

    def test_vmap_rejects_non_ascending_inputs(self):
        """Unsorted probabilities or quantiles in a vmapped batch raise errors."""
        y = jnp.asarray(self.PROBABILITIES)
        bad_y = jnp.stack([y, y[::-1]])
        with self.subTest(argument="probabilities"):
            with self.assertRaises(checkify.JaxRuntimeError):
                jax.vmap(lambda yy: MetalogInputData.from_values(yy, yy, False).x)(
                    bad_y
                )
        x = jnp.asarray(self.VALUES)
        bad_x = jnp.stack([x, x[::-1]])
        with self.subTest(argument="precomputed quantiles"):
            with self.assertRaises(checkify.JaxRuntimeError):
                jax.vmap(lambda xx: MetalogInputData.from_values(xx, y, True).x)(bad_x)


class StaticFieldsTest(TransformTestBase):
    """Tests that the fields selecting code paths are static, not pytree leaves."""

    # JSON written by dumps() before the fields became static
    SAVED_JSON = (
        (
            "unbounded",
            MetalogBoundedness.UNBOUNDED,
            0.0,
            0.0,
            '{"a": [0.5, 1.0, 0.25], "metalog_params": {"boundedness": 1, '
            '"method": 1, "lower_bound": 0.0, "upper_bound": 0.0, "num_terms": 3}}',
        ),
        (
            "strictly_lower_bound",
            MetalogBoundedness.STRICTLY_LOWER_BOUND,
            0.0,
            0.0,
            '{"a": [0.5, 1.0, 0.25], "metalog_params": {"boundedness": 2, '
            '"method": 1, "lower_bound": 0.0, "upper_bound": 0.0, "num_terms": 3}}',
        ),
        (
            "strictly_upper_bound",
            MetalogBoundedness.STRICTLY_UPPER_BOUND,
            0.0,
            15.0,
            '{"a": [0.5, 1.0, 0.25], "metalog_params": {"boundedness": 3, '
            '"method": 1, "lower_bound": 0.0, "upper_bound": 15.0, "num_terms": 3}}',
        ),
        (
            "bounded",
            MetalogBoundedness.BOUNDED,
            0.0,
            15.0,
            '{"a": [0.5, 1.0, 0.25], "metalog_params": {"boundedness": 4, '
            '"method": 1, "lower_bound": 0.0, "upper_bound": 15.0, "num_terms": 3}}',
        ),
    )

    def test_rvs_parameter_leaves_are_the_prng_parameters(self):
        """Size and max_draws are static; only the PRNG parameters are leaves."""
        rv_params = MetalogRandomVariableParameters(
            prng_params=JaxUniformDistributionParameters(seed=3), size=10
        )
        self.assertEqual(jax.tree.leaves(rv_params), [3])

    def test_only_bounds_and_alpha_are_leaves(self):
        """Pytree leaves are the bounds (and alpha for SPT), not the static fields."""
        params = self._params(MetalogBoundedness.BOUNDED, 1.0, 15.0)
        self.assertEqual(jax.tree.leaves(params), [1.0, 15.0])
        spt = SPTMetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            alpha=0.1,
            lower_bound=1.0,
            upper_bound=15.0,
        )
        self.assertEqual(jax.tree.leaves(spt), [1.0, 15.0, 0.1])

    @parameterized.named_parameters(*SAVED_JSON)
    def test_saved_json_is_unchanged(self, boundedness, lower, upper, saved):
        """Earlier JSON loads to the same model, and dumps() writes it unchanged."""
        params = MetalogParameters(
            boundedness=boundedness,
            method=MetalogFitMethod.OLS,
            lower_bound=lower,
            upper_bound=upper,
            num_terms=3,
        )
        m = Metalog(params, jnp.array([0.5, 1.0, 0.25]))
        loaded = Metalog.loads(saved)
        self.assertEqual(loaded, m)
        self.assertIs(loaded.boundedness, boundedness)
        self.assertEqual(m.dumps(), saved)


class CompileOnceTest(TransformTestBase):
    """Regression tests for issue #48: calling a method again doesn't recompile.

    pdf, rvs and Lasso fits used to define their jitted functions on every call, so
    every call recompiled them. pdf took about 15 ms a call, and ppf, cdf and fit
    paid it too, through assert_fitted.
    """

    def _assert_no_recompile(self, call):
        """After a first call, which may compile, two more calls compile nothing."""
        jax.block_until_ready(call())
        before = _COMPILES.count
        for _ in range(2):
            jax.block_until_ready(call())
        self.assertEqual(_COMPILES.count - before, 0)

    @parameterized.named_parameters(*TransformTestBase.FIT_CASES)
    def test_fit_and_methods_compile_once(self, boundedness, lower, upper, method):
        """Fitting and each method compile once for a model, not on every call."""
        data = self._data()
        params = self._params(boundedness, lower, upper, method)
        m = fit(data, params)
        calls = {
            "fit": lambda: fit(data, params).a,
            **{name: partial(f, m, v) for name, f, v in self._methods()},
            "cdf of a float": lambda: m.cdf(5.0),
            "mode": lambda: m.mode,
            "mean": lambda: m.mean,
            "rvs with jax.random": lambda: m.rvs(
                MetalogRandomVariableParameters(
                    prng_params=JaxUniformDistributionParameters(seed=0), size=10
                )
            ),
            "rvs with HDR": lambda: m.rvs(
                MetalogRandomVariableParameters(
                    prng_params=HDRPRNGParameters(), size=10
                )
            ),
        }
        for name, call in calls.items():
            with self.subTest(name):
                self._assert_no_recompile(call)

    @parameterized.parameters("check_feasibility", "is_feasible", "highest_mode")
    def test_feasibility_functions_are_jitted(self, name):
        """They're jitted with the engine static; eagerly they took about 2 s a call."""
        fn = getattr(feasibility, name)
        engine = get_engine(5)
        a = jnp.array([1.0, 1.0, 0.1, 0.1, 0.1])

        fn.lower(engine, a)  # only jitted functions have lower

        self._assert_no_recompile(lambda: fn(engine, a))


if __name__ == "__main__":
    absltest.main()
