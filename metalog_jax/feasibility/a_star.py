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
  an exact fallback. A point is a violation when G(y) < 0 beyond rounding, the same
  sign test :func:`check_feasibility` uses, and the final certificate is
  :func:`check_feasibility` itself.
* As in the paper, each iteration adds every flagged point: all confirmed grid
  minima (at least ``floor((k + 1) / 2) + 1`` are refined, the most G can have) or
  every inflection point with ``M' < 0`` plus any infeasible tail.
* ``epsilon`` is relative to the spread of x, so results do not depend on the units
  of the data.

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

from collections.abc import Callable
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import checkify

from metalog_jax.feasibility.analysis import (
    check_feasibility,
    tail_feasibility,
    zero_test_width,
)
from metalog_jax.feasibility.engine import (
    DEFAULT_TERM_ORDER,
    Engine,
    TermOrder,
    get_engine,
    inflection_points,
)
from metalog_jax.regression.base import RegressionModel, has_full_column_rank

NOT_CERTIFIED_MESSAGE = (
    "MetalogFitMethod.Feasible could not certify its fit as feasible: the exchange "
    "method stopped (iteration or cut limit) with M'(y) < 0 somewhere."
)
UCORNER = 1000.0  # sigmoid(+-1000) is exactly 1/0 in float64, so G there is s(1)/s(0)
_GOLD = 0.6180339887498949

# Defaults of the exchange method, shared by best_feasible_fit, fit_feasible and
# solve_batch (see best_feasible_fit for each one's meaning)
_EPSILON = 1e-6
_TOL = 0.0
_MAX_CUTS = 64
_N_NEW = 8
_MAX_ITER = 60
_N_GRID = 2049

# A batched solve runs the exchange loop in rounds of this many passes, and drops the
# fits that have converged between rounds, so a fit costs about its own passes rather
# than the slowest fit's. Most fits that need cuts converge in one iteration, which
# takes two passes: one to cut and one to confirm.
_ROUND_PASSES = 2
# Rounds are never compacted below this many fits. Each new round size compiles the
# exchange loop again, which costs far more than a few padded fits, and the last
# rounds' sizes would otherwise vary from batch to batch.
_MIN_ROUND = 64


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
        feasible: Air-tight certificate for ``a_star``: :func:`check_feasibility`
            (Algorithm 1 + Proposition 5, exact up to rounding).
        full_rank: The basis matrix has full column rank, so ``a_ols`` and ``a_star``
            are unique (Proposition 2 describes when it does not).
        cut_points: Probabilities where cuts were placed, NaN-padded.
    """

    a_star: chex.Array
    a_ols: chex.Array
    rss_star: chex.Array
    rss_ols: chex.Array
    iterations: chex.Array
    converged: chex.Array
    feasible: chex.Array
    full_rank: chex.Array
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


def _G(engine: Engine, a: chex.Array, u: chex.Numeric) -> chex.Numeric:
    """Return ``G(y) = y(1-y) M'(y)`` at ``y = sigmoid(u)``.

    Args:
        engine: Engine for the coefficient layout.
        a: Metalog coefficients, shape (num_terms,).
        u: Logit of the probability, ``log(y / (1 - y))``.

    Returns:
        G at ``sigmoid(u)``. A fit is feasible when G > 0 for every y in (0, 1).
    """
    mu, s = engine.split(a)
    return engine.level(1, mu, s, u)


def _G_sign(engine, a, u, width):
    """Sign of G (and M') at sigmoid(u), 0 within rounding (see zero_test_width)."""
    mu, s = engine.split(a)
    return engine.level_sign(1, mu, s, u, width)


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


def _golden_min(
    f: Callable[[chex.Numeric], chex.Numeric],
    lo: chex.Numeric,
    hi: chex.Numeric,
    iters: int = 90,
) -> chex.Numeric:
    """Return the minimizer of ``f`` on ``[lo, hi]`` by golden-section search.

    Args:
        f: Scalar function to minimize, assumed unimodal on the bracket.
        lo: Lower end of the bracket.
        hi: Upper end of the bracket.
        iters: Number of steps, fixed so the search runs under ``jit``.

    Returns:
        Midpoint of the final bracket.
    """

    def body(_, st):
        a, b = st
        c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
        left = f(c) <= f(d)
        return jnp.where(left, a, c), jnp.where(left, d, b)

    a, b = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return 0.5 * (a + b)


def _violations(engine, a, grid, n_new, tol):
    """Points (logit u) where G(y) < 0 and a validity mask (Algorithm 2, Lines 5-6).

    Line 5: the grid's negative local minima of G, most negative first, refined by
    golden section. At least ``istar + 1 = floor((k + 1) / 2) + 1`` are refined, the
    most G can have: G = y(1-y) M' is itself a metalog-type function with one more
    term in each polynomial, so Proposition 1 bounds its critical points. Line 6, if
    none is confirmed: every inflection point with ``M' < 0`` (Algorithm 1), plus any
    infeasible tail (Proposition 5). As in the paper, all of them are returned. A
    point counts only when G is negative beyond rounding, the test
    :func:`check_feasibility` uses.
    """
    width = zero_test_width(tol)
    g = jax.vmap(lambda uu: _G(engine, a, uu))(grid)
    left = jnp.concatenate([jnp.array([jnp.inf]), g[:-1]])
    right = jnp.concatenate([g[1:], jnp.array([jnp.inf])])
    score = jnp.where((g <= left) & (g <= right) & (g < 0), g, jnp.inf)
    _, idx = jax.lax.top_k(-score, max(n_new, engine.istar + 1))
    lo = jnp.concatenate([grid[:1], grid[:-1]])[idx]
    hi = jnp.concatenate([grid[1:], grid[-1:]])[idx]
    u_grid = jax.vmap(lambda l, h: _golden_min(lambda uu: _G(engine, a, uu), l, h))(
        lo, hi
    )
    neg_grid = jax.vmap(lambda uu: _G_sign(engine, a, uu, width))(u_grid) < 0
    m_grid = jnp.isfinite(score[idx]) & neg_grid
    # exact fallback: roots of M'' where M' < 0, and infeasible tails
    ur, mr = inflection_points(engine, a)
    bad_root = mr & (jax.vmap(lambda uu: _G_sign(engine, a, uu, width))(ur) < 0)
    t0, t1 = tail_feasibility(engine, a, tol)
    u_fb = jnp.concatenate([ur, jnp.array([-UCORNER, UCORNER])])
    m_fb = jnp.concatenate([bad_root, jnp.array([~t0, ~t1])])
    # One static width fits both paths, so no candidate is ever truncated
    n_out = max(u_grid.shape[0], u_fb.shape[0])

    def pad(v, fill):
        return jnp.concatenate([v, jnp.full(n_out - v.shape[0], fill, v.dtype)])

    u_grid, m_grid = pad(u_grid, 0.0), pad(m_grid, False)
    u_fb, m_fb = pad(u_fb, 0.0), pad(m_fb, False)
    use_grid = jnp.any(m_grid)
    return jnp.where(use_grid, u_grid, u_fb), jnp.where(use_grid, m_grid, m_fb)


def _normalize(x: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Center x on its median and scale it to unit maximum deviation.

    G is linear in a and blind to the intercept, so the exchange method works on the
    normalized quantiles and rescales its result.

    Args:
        x: Quantile values, shape (n,).

    Returns:
        tuple: The normalized quantiles, the median and the scale.
    """
    x_mid = jnp.median(x)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x - x_mid)), 1e-300)
    return (x - x_mid) / x_scale, x_mid, x_scale


def _logit_grid(n_grid: int) -> chex.Array:
    """The logit grid that locates violations, with both corners (static size)."""
    return jnp.concatenate(
        [jnp.array([-UCORNER]), jnp.linspace(-36.0, 36.0, n_grid), jnp.array([UCORNER])]
    )


def _initial_state(R: chex.Array, z0: chex.Array, max_cuts: int) -> tuple:
    """The exchange loop's state before its first pass: the OLS fit and no cuts.

    Args:
        R: Upper triangular factor of the basis matrix, shape (k, k).
        z0: Whitened normalized quantiles, ``Q.T @ xn``, shape (k,).
        max_cuts: Capacity of the cut buffer.

    Returns:
        tuple: ``(a, U, M, it, done)``: the coefficients, the cut points (logit)
        and their mask, the number of passes so far, and whether no violation
        remains.
    """
    return (
        jax.scipy.linalg.solve_triangular(R, z0, lower=False),
        jnp.zeros(max_cuts),
        jnp.zeros(max_cuts, bool),
        jnp.array(0),
        jnp.array(False),
    )


def _exchange(
    engine: Engine,
    R: chex.Array,
    z0: chex.Array,
    state: tuple,
    epsilon: float,
    tol: float,
    max_cuts: int,
    n_new: int,
    n_grid: int,
    stop: chex.Numeric,
) -> tuple:
    """Run exchange passes until no violation remains or ``stop`` passes are done.

    Each pass adds cuts where ``G(y) < 0`` and projects the OLS fit onto them. The
    state can be resumed with a larger ``stop``, which batched fits use to run in
    rounds.

    Args:
        engine: Engine for the coefficient layout.
        R: Upper triangular factor of the basis matrix, shape (k, k).
        z0: Whitened normalized quantiles, shape (k,).
        state: ``(a, U, M, it, done)``, as :func:`_initial_state` returns it.
        epsilon: Margin in ``G(y) >= epsilon``, relative to ``max|x - median(x)|``.
        tol: Extra slack for the violation sign test.
        max_cuts: Capacity of the cut buffer (static).
        n_new: Minimum number of grid minima refined per pass (static).
        n_grid: Number of logit grid points used to locate violations (static).
        stop: Total number of passes at which to stop.

    Returns:
        tuple: The state after the passes.
    """
    grid = _logit_grid(n_grid)

    def to_a(z):
        return jax.scipy.linalg.solve_triangular(R, z, lower=False)

    def cond(st):
        return (~st[4]) & (st[3] < stop)

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

    return jax.lax.while_loop(cond, body, state)


def _unnormalize(
    engine: Engine, a: chex.Array, x_mid: chex.Numeric, x_scale: chex.Numeric
) -> chex.Array:
    """Map coefficients fitted to normalized quantiles back to the units of x."""
    shift = jnp.zeros(engine.num_terms).at[engine.mu_idx[0]].set(x_mid)
    return a * x_scale + shift


@partial(jax.jit, static_argnames=("engine", "max_cuts", "n_new", "max_iter", "n_grid"))
def _solve(
    engine: Engine,
    Y: chex.Array,
    x: chex.Array,
    epsilon: float,
    tol: float,
    max_cuts: int,
    n_new: int,
    max_iter: int,
    n_grid: int,
) -> FeasibleFitResult:
    """Run the exchange method for ``a*`` (Algorithm 2) on one dataset.

    Starting from the OLS fit, each iteration adds cuts where ``G(y) < 0`` and
    projects the OLS fit onto them, until no violation remains.

    Args:
        engine: Engine for the coefficient layout.
        Y: Basis matrix, shape (n, num_terms).
        x: Quantile values, shape (n,).
        epsilon: Margin in ``G(y) >= epsilon``, relative to ``max|x - median(x)|``.
        tol: Extra slack for the violation and certificate sign tests.
        max_cuts: Capacity of the cut buffer (static).
        n_new: Minimum number of grid minima refined per iteration (static).
        max_iter: Maximum number of exchange iterations (static).
        n_grid: Number of logit grid points used to locate violations (static).

    Returns:
        FeasibleFitResult, with coefficients in the units of ``x``.
    """
    xn, x_mid, x_scale = _normalize(x)
    Q, R = jnp.linalg.qr(Y)
    z0 = Q.T @ xn
    init = _initial_state(R, z0, max_cuts)
    a, U, M, it, done = _exchange(
        engine, R, z0, init, epsilon, tol, max_cuts, n_new, n_grid, max_iter
    )
    a_star = _unnormalize(engine, a, x_mid, x_scale)
    a_ls = _unnormalize(engine, init[0], x_mid, x_scale)
    rss = lambda c: jnp.sum((x - Y @ c) ** 2)
    return FeasibleFitResult(
        a_star=a_star,
        a_ols=a_ls,
        rss_star=rss(a_star),
        rss_ols=rss(a_ls),
        iterations=jnp.where(done, it - 1, it),
        converged=done,
        feasible=check_feasibility(engine, a, tol).feasible,
        full_rank=has_full_column_rank(Y),
        cut_points=jnp.where(M, jax.nn.sigmoid(U), jnp.nan),
    )


def best_feasible_fit(
    x: chex.Array,
    y: chex.Array,
    num_terms: int,
    order: TermOrder = DEFAULT_TERM_ORDER,
    epsilon: float = _EPSILON,
    tol: float = _TOL,
    max_cuts: int = _MAX_CUTS,
    n_new: int = _N_NEW,
    max_iter: int = _MAX_ITER,
    n_grid: int = _N_GRID,
) -> FeasibleFitResult:
    """Fit the best feasible k-term metalog to quantile/probability pairs.

    Args:
        x: Quantile values, shape (n,).
        y: Cumulative probabilities in (0, 1), shape (n,).
        num_terms: Number of metalog terms k (2 <= k <= n).
        order: Basis ordering. The default, ``METALOG_2``, is the paper's and the
            reference code's; ``KEELIN_2016`` is the ordering of Keelin (2016).
        epsilon: Margin in ``G(y) >= epsilon``, relative to ``max|x - median(x)|``.
        tol: Extra slack for the violation and certificate sign tests, relative to
            the magnitude of G's terms. With 0 (the default) a violation is any
            ``G(y) < 0`` beyond rounding, as in the paper.
        max_cuts: Capacity of the cut buffer (static).
        n_new: Minimum number of grid minima refined per iteration (static); at
            least ``floor((k + 1) / 2) + 1`` are always refined.
        max_iter: Maximum number of exchange iterations (static).
        n_grid: Number of logit grid points used to locate violations (static).

    Returns:
        FeasibleFitResult. Check ``feasible`` (the certificate), ``converged`` and
        ``full_rank`` (with n = k symmetric probabilities the Keelin 2016 basis is
        singular for 7, 11, 15, ... terms; the default ``METALOG_2`` isn't).

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


def fit_feasible(
    X: chex.Array, y: chex.Array, order: TermOrder = DEFAULT_TERM_ORDER
) -> FeasibleModel:
    """Regression-dispatch entry point: best feasible fit from a metalog design matrix.

    ``X`` must be a ``metalog_jax`` design matrix in the term order ``order``, as
    built by ``MetalogBase.get_target``; ``y`` holds the (possibly transformed)
    quantiles. Used by ``fit`` when ``method=MetalogFitMethod.Feasible``, which
    passes the parameters' ``term_order``.

    Args:
        X: Design matrix of shape (n_samples, num_terms).
        y: Target quantiles of shape (n_samples,).
        order: The term order of X's columns.

    Returns:
        FeasibleModel whose ``weights`` are the best feasible coefficients.

    Raises:
        checkify.JaxRuntimeError: If the result cannot be certified feasible (the
            exchange loop hit its iteration or cut limit). ``vmap`` compatible.

    References:
        Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
        Properties of the Metalog Distribution. Darden Business School Working
        Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    """  # noqa: DOC502
    engine = get_engine(int(X.shape[1]), TermOrder(order))
    res = _solve(
        engine,
        jnp.asarray(X, jnp.float64),
        jnp.asarray(y, jnp.float64),
        _EPSILON,
        _TOL,
        _MAX_CUTS,
        _N_NEW,
        _MAX_ITER,
        _N_GRID,
    )
    checkify.check(res.feasible, NOT_CERTIFIED_MESSAGE)
    return FeasibleModel(weights=res.a_star)


@struct.dataclass
class BatchFitResult:
    """Result of :func:`solve_batch`, one entry per dataset.

    Attributes:
        a_star: Best feasible coefficients, shape (n_datasets, k).
        feasible: Each fit's certificate from :func:`check_feasibility`.
        iterations: Each fit's number of QP solves, 0 when its OLS fit is feasible.
    """

    a_star: chex.Array
    feasible: chex.Array
    iterations: chex.Array


def _round_size(n: int) -> int:
    """Round a batch size up to _MIN_ROUND, or to one of four sizes per power of two.

    Rounds compact the unconverged fits into a smaller batch, and each new size
    compiles once. Four sizes per power of two waste at most a quarter of a round on
    padding while keeping the number of compilations small, and the floor keeps the
    last, smallest rounds to one size.

    Args:
        n: The number of fits, at least 1.

    Returns:
        int: The padded size, at least n.
    """
    if n <= _MIN_ROUND:
        return _MIN_ROUND
    quarter = 1 << ((n - 1).bit_length() - 3)
    return -(-n // quarter) * quarter


def _pad_index(index: np.ndarray, size: int, fill: int | None = None) -> np.ndarray:
    """Pad an index array to ``size`` with ``fill``, by default its last entry."""
    fill = index[-1] if fill is None else fill
    return np.concatenate([index, np.full(size - index.size, fill)])


@partial(jax.jit, static_argnames=("shared", "max_cuts"))
def _start_batch(Y: chex.Array, X: chex.Array, shared: bool, max_cuts: int) -> tuple:
    """Whiten each dataset's least-squares problem and start its exchange loop.

    Args:
        Y: Basis matrix, shape (m, k) if ``shared``, else (n_datasets, m, k).
        X: Quantiles, shape (n_datasets, m).
        shared: Whether every dataset has the same basis matrix (static).
        max_cuts: Capacity of the cut buffer (static).

    Returns:
        tuple: ``(R, z0, x_mid, x_scale, state)``, with R of shape (k, k) if
        ``shared``, and a batched exchange state.
    """
    xn, x_mid, x_scale = jax.vmap(_normalize)(X)
    if shared:
        Q, R = jnp.linalg.qr(Y)
        z0 = xn @ Q
        state = jax.vmap(lambda z: _initial_state(R, z, max_cuts))(z0)
    else:
        Q, R = jax.vmap(jnp.linalg.qr)(Y)
        z0 = jnp.einsum("nmk,nm->nk", Q, xn)
        state = jax.vmap(lambda r, z: _initial_state(r, z, max_cuts))(R, z0)
    return R, z0, x_mid, x_scale, state


@partial(jax.jit, static_argnums=0)
def _certify_rows(engine: Engine, a: chex.Array, tol: float) -> chex.Array:
    """Return check_feasibility(engine, row, tol).feasible for each row of a."""
    return jax.vmap(lambda row: check_feasibility(engine, row, tol).feasible)(a)


def _certify(
    engine: Engine, a: chex.Array, index: np.ndarray, tol: float
) -> np.ndarray:
    """Certify the rows of ``a`` at ``index``, in a batch padded to a round size."""
    padded = _pad_index(index, _round_size(index.size))
    return np.asarray(_certify_rows(engine, _gather(a, padded), tol))[: index.size]


@partial(
    jax.jit,
    static_argnames=("engine", "shared", "max_cuts", "n_new", "n_grid", "max_iter"),
)
def _run_round(
    engine: Engine,
    R: chex.Array,
    z0: chex.Array,
    state: tuple,
    passes: chex.Numeric,
    epsilon: float,
    tol: float,
    shared: bool,
    max_cuts: int,
    n_new: int,
    n_grid: int,
    max_iter: int,
) -> tuple:
    """Run up to ``passes`` more exchange passes on each fit of a batch.

    Args:
        engine: Engine for the coefficient layout (static).
        R: Upper triangular factors, shape (k, k) if ``shared``, else (n, k, k).
        z0: Whitened normalized quantiles, shape (n, k).
        state: Batched exchange state.
        passes: The number of passes to run.
        epsilon: Margin in ``G(y) >= epsilon``.
        tol: Extra slack for the violation sign test.
        shared: Whether R is shared by every fit (static).
        max_cuts: Capacity of the cut buffer (static).
        n_new: Minimum number of grid minima refined per pass (static).
        n_grid: Number of logit grid points (static).
        max_iter: Maximum number of passes in all (static).

    Returns:
        tuple: The batched state after the round.
    """

    def run(r, z, st):
        stop = jnp.minimum(st[3] + passes, max_iter)
        return _exchange(engine, r, z, st, epsilon, tol, max_cuts, n_new, n_grid, stop)

    return jax.vmap(run, in_axes=(None if shared else 0, 0, 0))(R, z0, state)


@jax.jit
def _gather(tree, index: chex.Array):
    """Take the entries of every leaf of ``tree`` at ``index`` along axis 0."""
    return jax.tree.map(lambda v: v[index], tree)


@jax.jit
def _scatter(tree, index: chex.Array, update):
    """Write ``update`` into every leaf of ``tree`` at ``index`` along axis 0.

    An index past the end drops its entry, so a round's padding writes nothing.
    """
    return jax.tree.map(lambda v, u: v.at[index].set(u, mode="drop"), tree, update)


def solve_batch(
    engine: Engine,
    Y: chex.Array,
    X: chex.Array,
    shared: bool,
    epsilon: float = _EPSILON,
    tol: float = _TOL,
    max_cuts: int = _MAX_CUTS,
    n_new: int = _N_NEW,
    max_iter: int = _MAX_ITER,
    n_grid: int = _N_GRID,
) -> BatchFitResult:
    """Best feasible fits of a batch of datasets, with triage and rounds.

    Under ``jax.vmap``, the exchange loop of :func:`best_feasible_fit` runs every fit
    until the slowest has converged, and every fit pays for that. This solves a batch
    in two steps instead:

    * **Triage:** every dataset's OLS fit is checked with :func:`check_feasibility`.
      A feasible OLS fit is its own best feasible fit, and needs no exchange loop.
    * **Rounds:** the other fits run the exchange loop a few passes at a time. The
      fits that have converged drop out between rounds, so the rest run in a
      smaller batch.

    The fits are those of :func:`best_feasible_fit`, with the same iteration counts
    and certificates. Their coefficients agree to the solver's precision, not bit for
    bit: the interior-point projection amplifies rounding, so a different batch,
    compilation or CPU can move them by about 1e-8 of the largest. Under ``jax.jit``
    the batch can't be compacted, so every fit runs in one round, as with
    ``jax.vmap(best_feasible_fit)``.

    Args:
        engine: Engine for the coefficient layout.
        Y: Basis matrix, shape (m, k) if ``shared``, else (n_datasets, m, k).
        X: Quantiles (possibly transformed), shape (n_datasets, m).
        shared: Whether every dataset has the same basis matrix, so that it's
            factored once.
        epsilon: Margin in ``G(y) >= epsilon``, as for best_feasible_fit.
        tol: Extra slack for the sign tests, as for best_feasible_fit.
        max_cuts: Capacity of the cut buffer, as for best_feasible_fit.
        n_new: Minimum number of grid minima refined per pass, as for
            best_feasible_fit.
        max_iter: Maximum number of exchange passes, as for best_feasible_fit.
        n_grid: Number of logit grid points, as for best_feasible_fit.

    Returns:
        BatchFitResult: Each dataset's coefficients, certificate and iterations.
    """
    X = jnp.asarray(X, jnp.float64)
    R, z0, x_mid, x_scale, state = _start_batch(Y, X, shared, max_cuts)
    options = dict(
        epsilon=epsilon,
        tol=tol,
        shared=shared,
        max_cuts=max_cuts,
        n_new=n_new,
        n_grid=n_grid,
        max_iter=max_iter,
    )
    if isinstance(X, jax.core.Tracer):  # traced: no compaction, one round for all
        state = _run_round(engine, R, z0, state, max_iter, **options)
        a, _, _, it, done = state
        feasible = _certify_rows(engine, a, tol)
    else:
        state, feasible = _triage_and_rounds(engine, R, z0, state, options)
        a, _, _, it, done = state
    iterations = jnp.where(done, it - 1, it)
    a_star = jax.vmap(lambda c, m, s: _unnormalize(engine, c, m, s))(a, x_mid, x_scale)
    return BatchFitResult(a_star=a_star, feasible=feasible, iterations=iterations)


def _triage_and_rounds(
    engine: Engine, R: chex.Array, z0: chex.Array, state: tuple, options: dict
) -> tuple[tuple, chex.Array]:
    """Mark the feasible OLS fits done, then solve the rest in shrinking rounds.

    Args:
        engine: Engine for the coefficient layout.
        R: Upper triangular factors, shared or one per fit.
        z0: Whitened normalized quantiles, shape (n, k).
        state: Batched exchange state, starting at the OLS fits.
        options: The keyword arguments of :func:`_run_round` after ``passes``.

    Returns:
        tuple: The final batched state, and each fit's certificate.
    """
    tol, shared, max_iter = options["tol"], options["shared"], options["max_iter"]
    n = z0.shape[0]
    triaged = _certify(engine, state[0], np.arange(n), tol)
    # A triaged fit ends as the exchange loop would: one pass that found no violation
    a, U, M, it, _ = state
    state = (a, U, M, jnp.where(triaged, 1, it), jnp.asarray(triaged))
    feasible = triaged.copy()
    active = solved = np.flatnonzero(~triaged)
    while active.size:
        # Pad the round by repeating its last fit, and drop the padding's update
        size = _round_size(active.size)
        take, put = _pad_index(active, size), _pad_index(active, size, fill=n)
        r = R if shared else _gather(R, take)
        z, st = _gather((z0, state), take)
        update = _run_round(engine, r, z, st, _ROUND_PASSES, **options)
        state = _scatter(state, put, update)
        it, done = (np.asarray(v)[: active.size] for v in update[3:])
        active = active[~done & (it < max_iter)]
    if solved.size:
        feasible[solved] = _certify(engine, state[0], solved, tol)
    return state, jnp.asarray(feasible)
