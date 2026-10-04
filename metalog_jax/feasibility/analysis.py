# Copyright: Travis Jefferies 2026
"""Feasibility, modes and exact moments of an (unbounded) metalog.

All functions are ``jit``/``vmap``/``grad`` compatible once the engine is fixed.
For bounded metalogs these statements apply to the underlying unbounded metalog
fitted to the transformed quantiles; feasibility is preserved by the monotone
back-transform, but the moment formulas are not.

References:
    Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
    Properties of the Metalog Distribution. Darden Business School Working
    Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
"""

from functools import partial
from math import comb

import chex
import jax
import jax.numpy as jnp
from flax import struct

from metalog_jax.feasibility._exact import i_matrix
from metalog_jax.feasibility.engine import (
    EPS,
    Engine,
    _horner,
    _horner_abs,
    _poly_der,
    inflection_points,
)


@struct.dataclass
class FeasibilityReport:
    """Result of :func:`check_feasibility`.

    Attributes:
        feasible: ``M'(y) >= 0`` on (0, 1) (interior and both tails).
        interior_feasible: ``M' >= 0`` at every root of ``M''``.
        tail_feasible_zero: Proposition 5 condition at y -> 0.
        tail_feasible_one: Proposition 5 condition at y -> 1.
        roots: Roots of ``M''`` as probabilities, NaN-padded.
        slopes: ``M'`` at each root, NaN-padded.
        is_mode: True where the root is a mode of the density (``M''' >= 0``).
    """

    feasible: chex.Array
    interior_feasible: chex.Array
    tail_feasible_zero: chex.Array
    tail_feasible_one: chex.Array
    roots: chex.Array
    slopes: chex.Array
    is_mode: chex.Array


def feasibility_function(
    engine: Engine, a: chex.Array, u: chex.Numeric
) -> chex.Numeric:
    """``G(y) = y(1-y) M'(y)`` at ``y = sigmoid(u)`` (``u = -/+inf`` gives s(0), s(1))."""
    mu, s = engine.split(a)
    return engine.level(1, mu, s, u)


def zero_test_width(tol: float = 0.0) -> chex.Numeric:
    """Width of the zero tests, in machine epsilons of the terms' magnitude.

    A value counts as zero when ``|v| <= width * eps * magnitude``, where
    ``magnitude`` is the sum of the absolute values of the terms that produced ``v``
    (a running bound on its rounding error). ``tol`` adds slack relative to the same
    magnitude; ``tol = 0`` decides every sign exactly up to rounding.
    """
    return 64.0 + tol / EPS


def _sign(v: chex.Numeric, mag: chex.Numeric, width: chex.Numeric) -> chex.Numeric:
    """Sign of ``v``, or 0 when ``|v|`` is within ``width * eps * mag``."""
    return jnp.where(jnp.abs(v) <= width * EPS * mag, 0.0, jnp.sign(v))


def tail_feasibility(
    engine: Engine, a: chex.Array, tol: float = 0.0
) -> tuple[chex.Array, chex.Array]:
    """Proposition 5 tail conditions at y -> 0 and y -> 1.

    Each of ``s``, ``s'`` and ``mu'`` at the edge counts as zero only within its own
    rounding error (see :func:`zero_test_width`), so a small but resolved ``s(0) > 0``
    is tail feasible whatever the scale of ``a``.

    Args:
        engine: Engine for the coefficient layout.
        a: Coefficients, shape (k,).
        tol: Extra zero-test slack, relative to the terms' magnitude.

    Returns:
        ``(feasible at y -> 0, feasible at y -> 1)``.
    """
    width = zero_test_width(tol)
    mu, s = engine.split(a)
    ds, dmu = _poly_der(s, 1), _poly_der(mu, 1)
    out = []
    for x0, sgn in ((-0.5, -1.0), (0.5, 1.0)):
        s0 = _sign(_horner(s, x0), _horner_abs(s, x0), width)
        s1 = _sign(_horner(ds, x0), _horner_abs(ds, x0), width)
        m1 = _sign(_horner(dmu, x0), _horner_abs(dmu, x0), width)
        zero = s0 == 0
        out.append((s0 > 0) | (zero & (sgn * s1 > 0)) | (zero & (s1 == 0) & (m1 >= 0)))
    return out[0], out[1]


def check_feasibility(
    engine: Engine, a: chex.Array, tol: float = 0.0
) -> FeasibilityReport:
    """Air-tight feasibility test (Proposition 5 with Algorithm 1).

    Unlike a grid test this cannot miss a negative-density region: ``M'`` is checked
    at every root of ``M''`` (its local extrema) plus both tails. Every sign is
    decided exactly up to rounding (see :func:`zero_test_width`), so the verdict does
    not depend on the scale of ``a``.

    Args:
        engine: Engine for the coefficient layout.
        a: Coefficients, shape (k,).
        tol: Extra zero-test slack, relative to the terms' magnitude (0: exact up
            to rounding).

    Returns:
        FeasibilityReport.

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    width = zero_test_width(tol)
    scale = jnp.maximum(jnp.max(jnp.abs(a)), 1e-300)
    an = a / scale  # keeps Algorithm 1 in range; the sign tests are scale-free
    mu, s = engine.split(an)
    u, m = inflection_points(engine, an)
    g = jax.vmap(lambda uu: engine.level(1, mu, s, uu))(u)
    g_sign = jax.vmap(lambda uu: engine.level_sign(1, mu, s, uu, width))(u)
    w = jax.nn.sigmoid(u) * jax.nn.sigmoid(-u)
    m3 = jax.vmap(lambda uu: engine.level_sign(3, mu, s, uu))(u)
    interior = jnp.all(jnp.where(m, g_sign >= 0, True))
    t0, t1 = tail_feasibility(engine, an, tol)
    nan = jnp.nan
    return FeasibilityReport(
        feasible=interior & t0 & t1,
        interior_feasible=interior,
        tail_feasible_zero=t0,
        tail_feasible_one=t1,
        roots=jnp.where(m, jax.nn.sigmoid(u), nan),
        slopes=jnp.where(m, g / w * scale, nan),
        is_mode=m & (m3 >= 0),
    )


def is_feasible(engine: Engine, a: chex.Array, tol: float = 0.0) -> chex.Array:
    """Shorthand for ``check_feasibility(engine, a, tol).feasible``."""
    return check_feasibility(engine, a, tol).feasible


def highest_mode(
    engine: Engine, a: chex.Array
) -> tuple[chex.Numeric, chex.Numeric, chex.Array]:
    """Highest-density interior mode of a feasible unbounded metalog (Algorithm 1).

    The density is ``1 / M'(y)``, so among the modes (roots of ``M''`` where the
    third derivative is non-negative) the highest has the smallest slope. A binding
    mode of a best feasible fit, where ``M'(y)`` is close to 0, has the highest
    density.

    Args:
        engine: Engine for the coefficient layout.
        a: Coefficients of a feasible unbounded metalog, shape (k,).

    Returns:
        ``(x, y, found)``: the mode's quantile ``M(y)``, its probability ``y``, and
        whether an interior mode exists (a uniform metalog, for example, has none).

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    scale = jnp.maximum(jnp.max(jnp.abs(a)), 1e-300)
    u, m = inflection_points(engine, a / scale)
    mu, s = engine.split(a)
    w = jnp.maximum(jax.nn.sigmoid(u) * jax.nn.sigmoid(-u), 1e-300)
    slope = jax.vmap(lambda uu: engine.level(1, mu, s, uu))(u) / w
    up = jax.vmap(lambda uu: engine.level_sign(3, mu, s, uu))(u) >= 0
    is_mode = m & up
    i = jnp.argmin(jnp.where(is_mode, slope, jnp.inf))
    x = 0.5 * jnp.tanh(0.5 * u[i])
    value = _horner(mu, x) + _horner(s, x) * u[i]
    return value, jax.nn.sigmoid(u[i]), jnp.any(is_mode)


# ----------------------------------------------------------------------------------
# Exact moments (Lemma 1 / Proposition 3)
# ----------------------------------------------------------------------------------
def _poly_pow(c: chex.Array, n: int, length: int) -> chex.Array:
    out = jnp.zeros(length).at[0].set(1.0)
    for _ in range(n):
        out = jnp.convolve(out, c)[:length]
    return out


def moment_table(engine: Engine, max_order: int = 4) -> chex.Array:
    """Float table ``I(m, u)`` large enough for moments up to ``max_order``."""
    mmax = max_order * max(engine.num_mu, engine.num_s)
    return jnp.array(i_matrix(mmax, max_order))


def raw_moment(
    engine: Engine, a: chex.Array, p: int, table: chex.Array | None = None
) -> chex.Numeric:
    """Exact ``E[X^p] = int_0^1 M(y)^p dy`` for an unbounded metalog (Proposition 3).

    Expands ``(mu + s L)^p = sum_n C(p, n) mu^(p-n) s^n L^n`` by polynomial
    convolution and integrates each monomial with the exact ``I(m, n)``.

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    table = moment_table(engine, p) if table is None else table
    mu, s = engine.split(a)
    length = table.shape[0]
    total = 0.0
    for n in range(p + 1):
        poly = jnp.convolve(_poly_pow(mu, p - n, length), _poly_pow(s, n, length))[
            :length
        ]
        total = total + comb(p, n) * poly @ table[:, n]
    return total


@partial(jax.jit, static_argnames=("engine",))
def mean_and_variance(
    engine: Engine, a: chex.Array
) -> tuple[chex.Numeric, chex.Numeric]:
    """Exact mean and variance of an unbounded metalog (Lemma 1, Proposition 3).

    Args:
        engine: Engine for the coefficient layout.
        a: Coefficients, shape (k,).

    Returns:
        ``(mean, variance)``.

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    table = moment_table(engine, 2)
    mean = raw_moment(engine, a, 1, table)
    return mean, raw_moment(engine, a.at[engine.mu_idx[0]].add(-mean), 2, table)


@partial(jax.jit, static_argnames=("engine",))
def summary_stats(engine: Engine, a: chex.Array) -> dict[str, chex.Numeric]:
    """Exact mean, variance, standard deviation, skewness and kurtosis.

    Central moments are the raw moments of the metalog shifted by its mean, which
    avoids the cancellation of expanding ``E[(X - m)^p]``.

    Args:
        engine: Engine for the coefficient layout.
        a: Coefficients of a feasible unbounded metalog.

    Returns:
        Dict with keys ``mean``, ``variance``, ``std``, ``skewness``, ``kurtosis``.
        The kurtosis is Pearson's (not excess): 3 for a normal, 4.2 for the
        logistic (the 2-term metalog).

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    table = moment_table(engine, 4)
    mean = raw_moment(engine, a, 1, table)
    shifted = a.at[engine.mu_idx[0]].add(-mean)
    m2, m3, m4 = (raw_moment(engine, shifted, p, table) for p in (2, 3, 4))
    sd = jnp.sqrt(m2)
    return dict(
        mean=mean, variance=m2, std=sd, skewness=m3 / sd**3, kurtosis=m4 / sd**4
    )
