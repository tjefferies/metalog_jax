# Copyright: Travis Jefferies 2026
"""Static-shape engine for Metalog 2.0 feasibility (Algorithm 1 of Baucells et al.).

A k-term metalog is written as ``M(y) = mu(x) + s(x) * logit(y)`` with ``x = y - 1/2``.
The engine splits a coefficient vector into the ascending coefficients of ``mu`` and
``s`` and evaluates the scaled derivative levels ``G_j(y) = (y(1-y))^j M^(j)(y)`` in
logit coordinates ``u = logit(y)``, which keeps the tails (y within 1e-300 of 0 or 1)
well conditioned.

Two term orderings are supported:

* ``TermOrder.KEELIN_2016`` - the ordering used by ``metalog_jax`` (Keelin 2016):
  1, L, xL, x, x^2, x^2 L, x^3, x^3 L, ...
* ``TermOrder.METALOG_2`` - the Metalog 2.0 ordering of the paper:
  1, L, xL, x, x^2, x^2 L, x^3 L, x^3, x^4, ...

They coincide for k <= 5.

All loops are over static sizes, so every public function here is ``jit``/``vmap``
compatible.

References:
    Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
    Properties of the Metalog Distribution. Darden Business School Working
    Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
"""

from enum import IntEnum, auto
from functools import lru_cache

import chex
import jax
import jax.numpy as jnp

from metalog_jax.feasibility._exact import rational_part_tables

EPS = float(jnp.finfo(jnp.float64).eps)
TINY = 1e-300
UMAX = 709.0  # |logit| bracket used by the cascade
BISECTION_STEPS = 200


class TermOrder(IntEnum):
    """Ordering of the metalog basis terms."""

    KEELIN_2016 = auto()
    METALOG_2 = auto()


def term_layout(num_terms: int, order: TermOrder) -> tuple[list[int], list[int]]:
    """Coefficient indices of the mu- and s-polynomials, sorted by power of (y - 1/2).

    Args:
        num_terms: Number of metalog terms k.
        order: Basis ordering.

    Returns:
        ``(mu_idx, s_idx)`` where ``a[mu_idx[p]]`` multiplies ``x^p`` and
        ``a[s_idx[p]]`` multiplies ``x^p * logit(y)``.
    """
    mu, s = [], []
    base = [("mu", 0), ("s", 0), ("s", 1), ("mu", 1)]
    for j in range(1, num_terms + 1):
        if order == TermOrder.METALOG_2:
            kind, power = ("mu" if j % 4 <= 1 else "s"), (j - 1) // 2
        elif j <= 4:
            kind, power = base[j - 1]
        else:
            kind, power = ("mu" if j % 2 == 1 else "s"), (j - 1) // 2
        (mu if kind == "mu" else s).append((power, j - 1))
    return [ix for _, ix in sorted(mu)], [ix for _, ix in sorted(s)]


def _poly_der(c: chex.Array, j: int) -> chex.Array:
    """j-th derivative of ascending coefficients, keeping the length (zero padded)."""
    n = c.shape[0]
    for _ in range(j):
        c = jnp.concatenate([c[1:] * jnp.arange(1, n), jnp.zeros(1)])
    return c


def _horner(c: chex.Array, x: chex.Numeric) -> chex.Numeric:
    n = c.shape[0]
    return jax.lax.fori_loop(0, n, lambda i, acc: acc * x + c[n - 1 - i], 0.0 * x)


def _horner_abs(c: chex.Array, x: chex.Numeric) -> chex.Numeric:
    return _horner(jnp.abs(c), jnp.abs(x))


class Engine:
    """Trace-time tables and level evaluators for one ``(num_terms, order)`` pair.

    Engines are hashable by identity and cached by :func:`get_engine`, so they can be
    passed as static arguments to ``jax.jit``.
    """

    def __init__(self, num_terms: int, order: TermOrder = TermOrder.KEELIN_2016):
        """Build the exact tables for a k-term metalog.

        Args:
            num_terms: Number of metalog terms (>= 2).
            order: Basis ordering.
        """
        if num_terms < 2:
            raise ValueError("num_terms must be >= 2")
        self.num_terms, self.order = num_terms, order
        self.mu_idx, self.s_idx = term_layout(num_terms, order)
        self.num_mu, self.num_s = len(self.mu_idx), len(self.s_idx)
        # Lemma 4: (y(1-y))^i M^(i) is a polynomial of degree i-1 for i >= max(#mu, #s)
        self.istar = max(self.num_mu, self.num_s, 2)
        self._rational = {
            j: tuple(jnp.array(t) for t in rational_part_tables(j, self.num_s))
            for j in range(1, max(self.istar, 3) + 1)  # level 3 classifies modes
        }

    def split(self, a: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Ascending coefficients ``(mu, s)`` of the location and scale polynomials."""
        return a[jnp.array(self.mu_idx)], a[jnp.array(self.s_idx)]

    def _parts(self, j, mu, s, u):
        y, z = jax.nn.sigmoid(u), jax.nn.sigmoid(-u)
        x = 0.5 * jnp.tanh(0.5 * u)
        pm, ps = _poly_der(mu, j), _poly_der(s, j)
        rows_y, rows_z = self._rational[j]
        hy, hz = s @ rows_y, s @ rows_z
        trans = _horner(pm, x) + _horner(ps, x) * u
        trans_mag = _horner_abs(pm, x) + _horner_abs(ps, x) * jnp.abs(u)
        rat = jnp.where(y < 0.5, _horner(hy, y), _horner(hz, z))
        rat_mag = jnp.where(y < 0.5, _horner_abs(hy, y), _horner_abs(hz, z))
        logw = -jax.nn.softplus(-u) - jax.nn.softplus(u)  # log(y(1-y)), stable
        return trans, trans_mag, rat, rat_mag, logw

    def level(
        self, j: int, mu: chex.Array, s: chex.Array, u: chex.Numeric
    ) -> chex.Numeric:
        """Value of ``G_j = (y(1-y))^j M^(j)(y)`` at ``y = sigmoid(u)``.

        ``G_1(y) = y(1-y) M'(y)`` is the paper's feasibility function; its limits at
        ``u = -inf, +inf`` are ``s(0)`` and ``s(1)``.
        """
        trans, _, rat, _, logw = self._parts(j, mu, s, u)
        return jnp.exp(j * logw) * trans + rat

    def level_sign(
        self,
        j: int,
        mu: chex.Array,
        s: chex.Array,
        u: chex.Numeric,
        kappa: float = 64.0,
    ):
        """Sign of ``M^(j)`` at ``sigmoid(u)`` with a running-error zero test.

        Both parts are rescaled by their joint log-magnitude before combining, so
        the sign is exact up to rounding even when ``(y(1-y))^j`` underflows.
        Returns 0 where the value is indistinguishable from zero (e.g. levels that
        vanish identically).
        """
        trans, trans_mag, rat, rat_mag, logw = self._parts(j, mu, s, u)
        lp = j * logw + jnp.log(jnp.maximum(trans_mag, TINY))
        lr = jnp.log(jnp.maximum(rat_mag, TINY))
        c = jnp.maximum(lp, lr)
        fp, fr = jnp.exp(j * logw - c), jnp.exp(-c)
        v = trans * fp + rat * fr
        mag = trans_mag * fp + rat_mag * fr
        dead = (mag <= 0) | (jnp.abs(v) <= kappa * EPS * mag)
        return jnp.where(dead, 0.0, jnp.sign(v))

    def tail_sign(
        self, j: int, mu: chex.Array, s: chex.Array, at_one: bool
    ) -> chex.Numeric:
        """Lemma 2: sign of ``lim M^(j)`` as y -> 0 (``at_one=False``) or y -> 1."""
        pt = 0.5 if at_one else -0.5
        tol = 64 * EPS * (jnp.sum(jnp.abs(s)) + jnp.sum(jnp.abs(mu)) + TINY)
        sign, found = 0.0, False
        for m in range(0, j + 1):  # first non-vanishing s^(m) at the edge dominates
            v = _horner(_poly_der(s, m), pt)
            fac = 1.0 if at_one else (-1.0) ** (j + 1 - m)
            nz = jnp.abs(v) > tol
            sign = jnp.where(found | ~nz, sign, jnp.sign(fac * v))
            found = found | nz
        muj = _horner(_poly_der(mu, j), pt)
        return jnp.where(found, sign, jnp.where(jnp.abs(muj) > tol, jnp.sign(muj), 0.0))


@lru_cache(maxsize=None)
def get_engine(num_terms: int, order: TermOrder = TermOrder.KEELIN_2016) -> Engine:
    """Cached :class:`Engine` (stable identity keeps ``jit`` caches warm)."""
    return Engine(num_terms, order)


# ----------------------------------------------------------------------------------
# Algorithm 1: air-tight roots of M'' through the derivative cascade
# ----------------------------------------------------------------------------------
class _PolyLevel:
    """Adapter exposing the top polynomial levels ``p^(m)`` through the level interface."""

    def __init__(self, cy, cz, m):
        self.cy, self.cz, self.m = cy, cz, m

    def level_sign(self, j, mu, s, u, kappa=64.0):
        y, z = jax.nn.sigmoid(u), jax.nn.sigmoid(-u)
        py, pz = (
            _poly_der(self.cy, self.m),
            _poly_der(self.cz, self.m) * (-1.0) ** self.m,
        )
        v = jnp.where(y < 0.5, _horner(py, y), _horner(pz, z))
        mag = jnp.where(y < 0.5, _horner_abs(py, y), _horner_abs(pz, z))
        return jnp.where(
            (mag <= 0) | (jnp.abs(v) <= kappa * EPS * mag), 0.0, jnp.sign(v)
        )

    def tail_sign(self, j, mu, s, at_one):
        c = _poly_der(self.cz if at_one else self.cy, self.m) * (
            ((-1.0) ** self.m) if at_one else 1.0
        )
        tol = 64 * EPS * jnp.sum(jnp.abs(c)) + TINY
        nz = jnp.abs(c) > tol
        return jnp.where(jnp.any(nz), jnp.sign(c[jnp.argmax(nz)]), 0.0)


def _bisect(lv, j, mu, s, lo, hi, sign_lo):
    """Bisection in logit space on a sign-changing bracket (exact up to rounding)."""

    def body(_, st):
        a, b = st
        mid = 0.5 * (a + b)
        right = lv.level_sign(j, mu, s, mid) == sign_lo
        return jnp.where(right, mid, a), jnp.where(right, b, mid)

    a, b = jax.lax.fori_loop(0, BISECTION_STEPS, body, (lo, hi))
    return 0.5 * (a + b)


def _cascade_step(lv, j, mu, s, prev_u, prev_mask):
    """Roots of level j from the sorted roots of level j+1 (paper, Algorithm 1 lines 5-12)."""
    R = prev_u.shape[0]
    pts = jnp.concatenate(
        [jnp.array([-UMAX]), jnp.where(prev_mask, prev_u, UMAX), jnp.array([UMAX])]
    )
    sg = jax.vmap(lambda uu: lv.level_sign(j, mu, s, uu))(pts[1:-1])
    sg = jnp.concatenate(
        [
            lv.tail_sign(j, mu, s, False)[None],
            jnp.where(prev_mask, sg, 0.0),
            lv.tail_sign(j, mu, s, True)[None],
        ]
    )
    last = jnp.sum(prev_mask)  # the interval ending at the y = 1 edge
    idx = jnp.arange(R + 1)
    lo_s = sg[:-1]
    hi_s = jnp.where(idx == last, sg[-1], sg[1:])
    hi_u = jnp.where(idx == last, UMAX, pts[1:])
    has = (lo_s * hi_s < 0) & (idx <= last)
    roots = jax.vmap(lambda a, b, sl: _bisect(lv, j, mu, s, a, b, sl))(
        pts[:-1], hi_u, lo_s
    )
    hits = (
        sg[1:-1] == 0.0
    ) & prev_mask  # a root of level j+1 that is also a root of level j
    cand = jnp.sort(
        jnp.concatenate(
            [jnp.where(has, roots, jnp.inf), jnp.where(hits, prev_u, jnp.inf)]
        )
    )[: R + 1]
    return jnp.where(jnp.isfinite(cand), cand, UMAX), jnp.isfinite(cand)


def inflection_points(engine: Engine, a: chex.Array) -> tuple[chex.Array, chex.Array]:
    """All roots of ``M''`` on (0, 1), i.e. the modes and anti-modes (Algorithm 1).

    The highest level ``(y(1-y))^i M^(i)`` is an exact polynomial of degree ``i-1``
    (Lemma 4). Its roots are bracketed by running the same cascade over its own
    polynomial derivatives, so no eigenvalue solver is needed, and the roots are
    then carried down one level at a time to ``M''`` using Lemma 2 edge signs.

    Args:
        engine: Engine for the coefficient layout.
        a: Coefficients, shape (k,).

    Returns:
        ``(u, mask)`` - roots in logit coordinates (``y = sigmoid(u)``), sorted and
        padded with ``UMAX``, and a boolean mask of valid entries. The static capacity
        is ``engine.istar + 1`` (Proposition 1 bounds the count well below it).

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    mu, s = engine.split(a)
    i = engine.istar
    rows_y, rows_z = engine._rational[i]
    cy, cz = (s @ rows_y)[:i], (s @ rows_z)[:i]  # Lemma 4: degree <= i-1
    u, m = jnp.zeros((0,)), jnp.zeros((0,), bool)
    for mm in range(i - 2, -1, -1):
        u, m = _cascade_step(_PolyLevel(cy, cz, mm), None, mu, s, u, m)
    for j in range(i - 1, 1, -1):
        u, m = _cascade_step(engine, j, mu, s, u, m)
    return u, m
