# Copyright: Travis Jefferies 2026
"""Best feasible metalog fit ``a*`` (semi-infinite QP solved by the exchange method).

Solves ``min_a ||x - Y a||^2`` subject to ``G(y) = y(1-y) M'(y) >= epsilon`` for all
y in [0, 1], where ``G(0) = s(0)`` and ``G(1) = s(1)``. The constraint is linear in
``a``, so each violated point becomes one linear cut. The loop alternates between
(i) finding the most violated points of the current fit and (ii) re-solving a small
QP with all cuts so far, and stops when an air-tight check finds no violation. This
mirrors ``find_a_star`` in the authors' reference code, with these differences:

* The QP is whitened with a QR factorization of Y (Hessian becomes the identity) and
  solved by a fixed-iteration Mehrotra predictor-corrector interior-point method.
* Shapes are static (a masked buffer of ``max_cuts`` cuts), so the whole fit runs
  under ``jit`` and can be ``vmap``-ed over many datasets.
* Violations are found on a logit grid (reaching y ~ 1e-16 from either edge) with
  golden-section refinement, then Algorithm 1 plus the Proposition 5 tail tests as
  an exact fallback and final certificate.
* ``epsilon`` and ``tol`` are relative to the spread of x, so results do not depend
  on the units of the data.

Note: "a*" is the paper's name for the optimal coefficient vector; this is not the
A* graph-search algorithm.

References:
    Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
    Properties of the Metalog Distribution. Darden Business School Working
    Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    Xu, Z. S. metalog_algorithm (reference implementation), licensed under
    CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); adapted and ported
    to JAX, see NOTICE. https://github.com/Stephenxuu/metalog_algorithm
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct

from metalog_jax.feasibility.analysis import tail_feasibility
from metalog_jax.feasibility.engine import (
    Engine,
    TermOrder,
    get_engine,
    inflection_points,
)
from metalog_jax.regression.base import RegressionModel

UCORNER = 1000.0  # sigmoid(+-1000) is exactly 1/0 in float64, so G there is s(1)/s(0)
_GOLD = 0.6180339887498949


@struct.dataclass
class FeasibleFitResult:
    """Result of :func:`best_feasible_fit`.

    Attributes:
        a_star: Best feasible coefficients.
        a_ols: Unconstrained least-squares coefficients.
        rss_star: Residual sum of squares of ``a_star``.
        rss_ols: Residual sum of squares of ``a_ols``.
        iterations: Number of QP solves (0 when OLS is already feasible).
        converged: The exchange loop found no remaining violation.
        feasible: Air-tight certificate (Algorithm 1 + tails) for ``a_star``.
        cut_points: Probabilities where cuts were placed, NaN-padded.
    """

    a_star: chex.Array
    a_ols: chex.Array
    rss_star: chex.Array
    rss_ols: chex.Array
    iterations: chex.Array
    converged: chex.Array
    feasible: chex.Array
    cut_points: chex.Array


@struct.dataclass
class FeasibleModel(RegressionModel):
    """Regression-style model returned by :func:`fit_feasible` (weights = ``a*``)."""

    pass


def design_matrix(engine: Engine, y: chex.Array) -> chex.Array:
    """Metalog basis matrix of shape (n, k) for the engine's term ordering."""
    x, L = y - 0.5, jnp.log(y / (1 - y))
    cols = [None] * engine.num_terms
    for p, ix in enumerate(engine.mu_idx):
        cols[ix] = x**p
    for p, ix in enumerate(engine.s_idx):
        cols[ix] = x**p * L
    return jnp.stack(cols, axis=1)


def _G(engine, a, u):
    mu, s = engine.split(a)
    return engine.level(1, mu, s, u)


def _cut_rows(engine, u):
    """Rows c(u) with G(u) = c(u) @ a."""
    zero = jnp.zeros(engine.num_terms)
    return jax.vmap(lambda uu: jax.jacfwd(lambda a: _G(engine, a, uu))(zero))(u)


def project_onto_polyhedron(z0, A, b, mask, iters: int = 40):
    """``argmin 0.5||z - z0||^2`` s.t. ``A z >= b`` on rows where ``mask`` is True.

    Mehrotra predictor-corrector interior-point method with a fixed iteration
    budget. The iterate is frozen once converged or if a step becomes non-finite
    (the barrier Hessian overflows as active slacks reach zero).
    """
    nrm = jnp.linalg.norm(A, axis=1)
    keep = mask & (nrm > 0)
    safe = jnp.where(keep, nrm, 1.0)
    A = jnp.where(keep[:, None], A / safe[:, None], 0.0)
    b = jnp.where(keep, b / safe, -1.0)  # inactive rows: 0 >= -1, always slack
    m = b.shape[0]
    z = z0
    s = jnp.maximum(A @ z - b, 1.0)
    lam = jnp.ones(m)

    def newton(z, s, lam, rc):
        rd = z - z0 - A.T @ lam
        rp = A @ z - s - b
        K = jnp.eye(z.shape[0]) + A.T @ ((lam / s)[:, None] * A)
        dz = jnp.linalg.solve(K, -rd + A.T @ ((rc - lam * rp) / s))
        ds = A @ dz + rp
        return dz, ds, (rc - lam * ds) / s

    def max_step(v, dv):
        return jnp.minimum(
            1.0, jnp.min(jnp.where(dv < 0, -v / jnp.where(dv < 0, dv, -1.0), jnp.inf))
        )

    def body(_, st):
        z, s, lam = st
        mu = s @ lam / m
        rd0 = jnp.max(jnp.abs(z - z0 - A.T @ lam))
        rp0 = jnp.max(jnp.abs(jnp.where(keep, A @ z - s - b, 0.0)))
        conv = (mu < 1e-15) & (rd0 < 1e-13) & (rp0 < 1e-13)
        dz, ds, dl = newton(z, s, lam, -lam * s)  # predictor
        aff = jnp.minimum(max_step(s, ds), max_step(lam, dl))
        mu_aff = (s + aff * ds) @ (lam + aff * dl) / m
        sigma = (mu_aff / jnp.maximum(mu, 1e-300)) ** 3
        dz, ds, dl = newton(z, s, lam, -lam * s - ds * dl + sigma * mu)  # corrector
        al = jnp.minimum(0.99 * jnp.minimum(max_step(s, ds), max_step(lam, dl)), 1.0)
        zn, sn, ln = (
            z + al * dz,
            jnp.maximum(s + al * ds, 1e-300),
            jnp.maximum(lam + al * dl, 1e-300),
        )
        bad = conv | ~(
            jnp.all(jnp.isfinite(zn))
            & jnp.all(jnp.isfinite(sn))
            & jnp.all(jnp.isfinite(ln))
        )
        return jnp.where(bad, z, zn), jnp.where(bad, s, sn), jnp.where(bad, lam, ln)

    z, _, lam = jax.lax.fori_loop(0, iters, body, (z, s, lam))
    return z, lam


def _golden_min(f, lo, hi, iters: int = 90):
    def body(_, st):
        a, b = st
        c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
        left = f(c) <= f(d)
        return jnp.where(left, a, c), jnp.where(left, d, b)

    a, b = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return 0.5 * (a + b)


def _violations(engine, a, grid, n_new, tol):
    """Up to ``n_new`` violated points (logit u) of G and a validity mask."""
    g = jax.vmap(lambda uu: _G(engine, a, uu))(grid)
    left = jnp.concatenate([jnp.array([jnp.inf]), g[:-1]])
    right = jnp.concatenate([g[1:], jnp.array([jnp.inf])])
    score = jnp.where((g <= left) & (g <= right) & (g < -tol), g, jnp.inf)
    _, idx = jax.lax.top_k(-score, n_new)
    lo = jnp.concatenate([grid[:1], grid[:-1]])[idx]
    hi = jnp.concatenate([grid[1:], grid[-1:]])[idx]
    u_grid = jax.vmap(lambda l, h: _golden_min(lambda uu: _G(engine, a, uu), l, h))(
        lo, hi
    )
    g_grid = jax.vmap(lambda uu: _G(engine, a, uu))(u_grid)
    m_grid = jnp.isfinite(score[idx]) & (g_grid < -tol)
    # exact fallback: roots of M'' where M' < 0, and infeasible tails
    ur, mr = inflection_points(engine, a)
    bad_root = mr & (jax.vmap(lambda uu: _G(engine, a, uu))(ur) < -tol)
    t0, t1 = tail_feasibility(engine, a, tol)
    u_fb = jnp.concatenate([ur, jnp.array([-UCORNER, UCORNER])])
    m_fb = jnp.concatenate([bad_root, jnp.array([~t0, ~t1])])
    pad = max(0, n_new - u_fb.shape[0])
    u_fb = jnp.concatenate([u_fb, jnp.zeros(pad)])[:n_new]
    m_fb = jnp.concatenate([m_fb, jnp.zeros(pad, bool)])[:n_new]
    use_grid = jnp.any(m_grid)
    return jnp.where(use_grid, u_grid, u_fb), jnp.where(use_grid, m_grid, m_fb)


@partial(jax.jit, static_argnames=("engine", "max_cuts", "n_new", "max_iter", "n_grid"))
def _solve(
    engine, Y, x, epsilon, tol, max_cuts, n_new, max_iter, n_grid
) -> FeasibleFitResult:
    x_mid = jnp.median(x)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x - x_mid)), 1e-300)
    xn = (x - x_mid) / x_scale  # G is linear in a and blind to the intercept
    Q, R = jnp.linalg.qr(Y)
    z0 = Q.T @ xn

    def to_a(z):
        return jax.scipy.linalg.solve_triangular(R, z, lower=False)

    grid = jnp.concatenate(
        [jnp.array([-UCORNER]), jnp.linspace(-36.0, 36.0, n_grid), jnp.array([UCORNER])]
    )
    a_ols = to_a(z0)

    def cond(st):
        return (~st[4]) & (st[3] < max_iter)

    def body(st):
        a, U, M, it, _ = st
        un, mn = _violations(engine, a, grid, n_new, tol)
        pos = jnp.sum(M) + jnp.cumsum(mn) - 1
        slot = jnp.where(mn & (pos < max_cuts), pos, max_cuts)
        U = U.at[slot].set(un, mode="drop")
        M = M.at[slot].set(True, mode="drop")
        A = jax.scipy.linalg.solve_triangular(
            R, _cut_rows(engine, U).T, lower=False, trans="T"
        ).T
        z, _ = project_onto_polyhedron(z0, A, jnp.full(max_cuts, epsilon), M)
        return jnp.where(jnp.any(mn), to_a(z), a), U, M, it + 1, ~jnp.any(mn)

    init = (
        a_ols,
        jnp.zeros(max_cuts),
        jnp.zeros(max_cuts, bool),
        jnp.array(0),
        jnp.array(False),
    )
    a, U, M, it, done = jax.lax.while_loop(cond, body, init)

    ur, mr = inflection_points(engine, a)
    roots_ok = jnp.all(
        jnp.where(mr, jax.vmap(lambda uu: _G(engine, a, uu))(ur) >= -tol, True)
    )
    t0, t1 = tail_feasibility(engine, a, tol)

    shift = jnp.zeros(engine.num_terms).at[engine.mu_idx[0]].set(x_mid)
    a_star, a_ls = a * x_scale + shift, a_ols * x_scale + shift
    rss = lambda c: jnp.sum((x - Y @ c) ** 2)  # noqa: E731
    return FeasibleFitResult(
        a_star=a_star,
        a_ols=a_ls,
        rss_star=rss(a_star),
        rss_ols=rss(a_ls),
        iterations=jnp.where(done, it - 1, it),
        converged=done,
        feasible=roots_ok & t0 & t1,
        cut_points=jnp.where(M, jax.nn.sigmoid(U), jnp.nan),
    )


def best_feasible_fit(
    x: chex.Array,
    y: chex.Array,
    num_terms: int,
    order: TermOrder = TermOrder.KEELIN_2016,
    epsilon: float = 1e-6,
    tol: float = 1e-9,
    max_cuts: int = 64,
    n_new: int = 8,
    max_iter: int = 60,
    n_grid: int = 2049,
) -> FeasibleFitResult:
    """Fit the best feasible k-term metalog to quantile/probability pairs.

    Args:
        x: Quantile values, shape (n,).
        y: Cumulative probabilities in (0, 1), shape (n,).
        num_terms: Number of metalog terms k (2 <= k <= n).
        order: Basis ordering (``KEELIN_2016`` matches ``metalog_jax``; use
            ``METALOG_2`` to reproduce the paper and the reference code).
        epsilon: Margin in ``G(y) >= epsilon``, relative to ``max|x - median(x)|``.
        tol: Violation tolerance, in the same relative units.
        max_cuts: Capacity of the cut buffer (static).
        n_new: Maximum cuts added per iteration (static).
        max_iter: Maximum number of exchange iterations (static).
        n_grid: Number of logit grid points used to locate violations (static).

    Returns:
        FeasibleFitResult. Check ``feasible`` (the certificate) and ``converged``.

    Example:
        >>> import jax.numpy as jnp
        >>> from metalog_jax.feasibility import best_feasible_fit
        >>> x = jnp.array([1.0, 2.0, 4.0, 8.0, 12.0])
        >>> y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> res = best_feasible_fit(x, y, num_terms=4)
        >>> bool(res.feasible)
        True

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    engine = get_engine(int(num_terms), TermOrder(order))
    x, y = jnp.asarray(x, jnp.float64), jnp.asarray(y, jnp.float64)
    return _solve(
        engine,
        design_matrix(engine, y),
        x,
        epsilon,
        tol,
        max_cuts,
        n_new,
        max_iter,
        n_grid,
    )


def fit_feasible(X: chex.Array, y: chex.Array) -> FeasibleModel:
    """Regression-dispatch entry point: best feasible fit from a metalog design matrix.

    ``X`` must be a ``metalog_jax`` design matrix (Keelin 2016 term ordering), as
    built by ``MetalogBase.get_target``; ``y`` holds the (possibly transformed)
    quantiles. Used by ``fit`` when ``method=MetalogFitMethod.Feasible``.

    Args:
        X: Design matrix of shape (n_samples, num_terms).
        y: Target quantiles of shape (n_samples,).

    Returns:
        FeasibleModel whose ``weights`` are the best feasible coefficients.

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """
    engine = get_engine(int(X.shape[1]), TermOrder.KEELIN_2016)
    res = _solve(
        engine,
        jnp.asarray(X, jnp.float64),
        jnp.asarray(y, jnp.float64),
        1e-6,
        1e-9,
        64,
        8,
        60,
        2049,
    )
    return FeasibleModel(weights=res.a_star)
