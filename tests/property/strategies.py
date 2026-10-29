"""Hypothesis strategies and shared helpers for the property-based tests.

JAX compiles once per array shape and combination of static fields (boundedness,
fit method and number of terms), so strategies draw term counts from a short fixed
list and probabilities as fixed-size grids. Each new shape or term count would cost
a compilation.
"""
# Copyright: Travis Jefferies 2026

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from jax.experimental import checkify
from scipy import stats

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    SPTMetalogParameters,
)
from metalog_jax.feasibility import (
    DEFAULT_TERM_ORDER,
    TermOrder,
    check_feasibility,
    get_engine,
    highest_mode,
)
from metalog_jax.metalog import THETA_3, Metalog, fit
from metalog_jax.utils import DEFAULT_Y

BOUNDEDNESS = tuple(MetalogBoundedness)
FIT_METHODS = (MetalogFitMethod.Feasible, MetalogFitMethod.OLS, MetalogFitMethod.Lasso)
NUM_TERMS = (3, 5, 7, 9)
SAMPLE_SIZE = 200
GRID_SIZE = 16


MIRROR_BOUNDEDNESS = {
    MetalogBoundedness.UNBOUNDED: MetalogBoundedness.UNBOUNDED,
    MetalogBoundedness.STRICTLY_LOWER_BOUND: MetalogBoundedness.STRICTLY_UPPER_BOUND,
    MetalogBoundedness.STRICTLY_UPPER_BOUND: MetalogBoundedness.STRICTLY_LOWER_BOUND,
    MetalogBoundedness.BOUNDED: MetalogBoundedness.BOUNDED,
}


def uses_lower(boundedness):
    """Whether the boundedness type has a lower bound."""
    return boundedness in (
        MetalogBoundedness.STRICTLY_LOWER_BOUND,
        MetalogBoundedness.BOUNDED,
    )


def uses_upper(boundedness):
    """Whether the boundedness type has an upper bound."""
    return boundedness in (
        MetalogBoundedness.STRICTLY_UPPER_BOUND,
        MetalogBoundedness.BOUNDED,
    )


def to_unbounded_scale(q, boundedness, lower, upper):
    """Map quantiles to the metalog's unbounded scale (Keelin 2016, Section 4)."""
    q = np.asarray(q, dtype=np.float64)
    if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
        return np.log(q - lower)
    if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
        return -np.log(upper - q)
    if boundedness == MetalogBoundedness.BOUNDED:
        return np.log((q - lower) / (upper - q))
    return q


def from_unbounded_scale(z, boundedness, lower, upper):
    """Map values of the unbounded metalog back to the support (Keelin 2016, Section 4)."""
    z = np.asarray(z, dtype=np.float64)
    if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
        return lower + np.exp(z)
    if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
        return upper - np.exp(-z)
    if boundedness == MetalogBoundedness.BOUNDED:
        return (lower + upper * np.exp(z)) / (1 + np.exp(z))
    return z


def term_power(j, order=DEFAULT_TERM_ORDER):
    """(power of y - 0.5, whether it multiplies logit(y)) for term j (1-based).

    Both orders start 1, logit(y), (y - 0.5) logit(y), (y - 0.5). Then term j has
    power (j - 1) // 2. Keelin (2016), Definition 1, puts the logit on even terms:
    x^2, x^2 L, x^3, x^3 L, ... The Metalog 2.0 order (Baucells et al. 2025,
    Section 2) puts it on terms 2 and 3 mod 4: x^2, x^2 L, x^3 L, x^3, x^4, ...
    """
    if j <= 4:
        return ((0, False), (0, True), (1, True), (1, False))[j - 1]
    if order == TermOrder.METALOG_2:
        return (j - 1) // 2, j % 4 in (2, 3)
    return ((j - 1) // 2, False) if j % 2 else (j // 2 - 1, True)


def basis(y, num_terms, order=DEFAULT_TERM_ORDER):
    """The metalog basis at y, shape (len(y), num_terms), built in NumPy."""
    y = np.asarray(y, dtype=np.float64)
    logit = np.log(y / (1 - y))
    columns = []
    for j in range(1, num_terms + 1):
        power, with_logit = term_power(j, order)
        columns.append((y - 0.5) ** power * (logit if with_logit else 1.0))
    return np.stack(columns, axis=1)


def s_at_ends(a, order=DEFAULT_TERM_ORDER):
    """s(0) and s(1), the logit polynomial at each end of (0, 1)."""
    ends = [0.0, 0.0]
    for j, coefficient in enumerate(np.asarray(a), start=1):
        power, with_logit = term_power(j, order)
        if with_logit:
            ends[0] += coefficient * (-0.5) ** power
            ends[1] += coefficient * 0.5**power
    return ends


def mirror_coefficients(a, order=DEFAULT_TERM_ORDER):
    """Coefficients of the metalog of -X, whose quantile function is -M(1 - y).

    y -> 1 - y negates y - 0.5 and logit(y), so term j changes sign by
    (-1)^power, and once more for logit terms; the leading minus sign flips them all.
    """
    signs = []
    for j in range(1, len(a) + 1):
        power, with_logit = term_power(j, order)
        signs.append((-1.0) ** (power + 1 + with_logit))
    return jnp.asarray(signs) * jnp.asarray(a)


def mirror_bounds(boundedness, lower, upper):
    """Bounds of -X: (-upper, -lower), with 0 for a bound the mirror doesn't use."""
    mirrored = MIRROR_BOUNDEDNESS[boundedness]
    return (
        -upper if uses_lower(mirrored) else 0.0,
        -lower if uses_upper(mirrored) else 0.0,
    )


def mirror(m):
    """The metalog of -X for a metalog m of X."""
    lower, upper = mirror_bounds(m.boundedness, m.lower_bound, m.upper_bound)
    params = m.metalog_params.replace(
        boundedness=MIRROR_BOUNDEDNESS[m.boundedness],
        lower_bound=lower,
        upper_bound=upper,
    )
    return Metalog(params, mirror_coefficients(m.a, m.term_order))


def feasibility_report(a):
    """check_feasibility's report for the coefficients."""
    a = jnp.asarray(a, dtype=jnp.float64)
    return check_feasibility(get_engine(a.shape[0]), a)


def highest_density_mode(a):
    """highest_mode's (quantile, probability, found) for unbounded coefficients."""
    a = jnp.asarray(a, dtype=jnp.float64)
    return highest_mode(get_engine(a.shape[0]), a)


def is_certified(a):
    """Whether check_feasibility certifies the coefficients feasible."""
    return bool(feasibility_report(a).feasible)


def fit_certified(data, params):
    """Fit, replacing an OLS or Lasso fit that isn't certified feasible by Feasible."""
    if params.method != MetalogFitMethod.Feasible:
        try:
            m = fit(data, params)
        except checkify.JaxRuntimeError:  # infeasible on DEFAULT_Y or rank deficient
            m = None
        if m is not None and is_certified(m.a):
            return m
        params = params.replace(method=MetalogFitMethod.Feasible)
    return fit(data, params)


@st.composite
def probabilities(draw, size=GRID_SIZE):
    """Sorted, distinct probabilities in [0.001, 0.999], at least 0.001 apart.

    One per equal slice of 1..999 thousandths, so they cover both tails without
    the rejections a unique-list strategy needs.
    """
    edges = [i * 999 // size for i in range(size + 1)]
    k = [draw(st.integers(lo + 1, hi)) for lo, hi in pairwise(edges)]
    return jnp.asarray(k, dtype=jnp.float64) / 1000


@st.composite
def hand_set_bounds(draw, boundedness):
    """Bounds for hand-set coefficients; a bound the type doesn't use is 0."""
    lower = draw(st.floats(-10, 10))
    upper = lower + draw(st.floats(0.5, 50))
    if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
        upper = draw(st.floats(-10, 10))
    return (
        lower if uses_lower(boundedness) else 0.0,
        upper if uses_upper(boundedness) else 0.0,
    )


@st.composite
def three_term_coefficients(draw, boundedness=MetalogBoundedness.UNBOUNDED):
    """Feasible 3-term coefficients: |a3| / a2 < THETA_3 (Keelin 2016, Proposition 2)."""
    if boundedness == MetalogBoundedness.UNBOUNDED:
        a1, a2 = draw(st.floats(-100, 100)), draw(st.floats(0.1, 10))
    else:
        # Keeps |M(y)| below about 22 for y in [0.001, 0.999], so the back-transform
        # stays distinguishable from the bounds in float64
        a1, a2 = draw(st.floats(-3, 3)), draw(st.floats(0.05, 1.5))
    return [a1, a2, draw(st.floats(-0.99, 0.99)) * THETA_3 * a2]


@st.composite
def hand_set_metalogs(draw, boundedness=None):
    """A feasible 3-term Metalog with hand-set coefficients."""
    if boundedness is None:
        boundedness = draw(st.sampled_from(BOUNDEDNESS))
    lower, upper = draw(hand_set_bounds(boundedness))
    params = MetalogParameters(
        boundedness=boundedness,
        method=draw(st.sampled_from(FIT_METHODS)),
        lower_bound=lower,
        upper_bound=upper,
        num_terms=3,
    )
    a = draw(three_term_coefficients(boundedness))
    return Metalog(params, jnp.asarray(a, dtype=jnp.float64))


@st.composite
def samples(draw):
    """Seeded draws from a family below, shifted and scaled.

    Normal, lognormal, gamma, beta, or a two-component normal mixture, whose fits
    can have several modes.
    """
    families = ("normal", "lognormal", "gamma", "beta", "mixture")
    family = draw(st.sampled_from(families))
    shape = draw(st.floats(0, 1))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    if family == "normal":
        x = rng.normal(size=SAMPLE_SIZE)
    elif family == "lognormal":
        x = rng.lognormal(sigma=0.2 + 0.8 * shape, size=SAMPLE_SIZE)
    elif family == "gamma":
        x = rng.gamma(1 + 9 * shape, size=SAMPLE_SIZE)
    elif family == "beta":
        x = rng.beta(1.5 + 4 * shape, 5.5 - 4 * shape, size=SAMPLE_SIZE)
    else:
        second = rng.random(SAMPLE_SIZE) < 0.2 + 0.4 * shape
        x = rng.normal(size=SAMPLE_SIZE) + second * (3 + 3 * shape)
    return draw(st.floats(-50, 50)) + draw(st.floats(0.1, 20)) * x


@st.composite
def bounds_around(draw, boundedness, x):
    """Bounds that contain the data x; a bound the type doesn't use is 0."""
    low, high = float(np.min(x)), float(np.max(x))
    width = high - low
    lower = low - draw(st.floats(0.01, 2)) * width
    upper = high + draw(st.floats(0.01, 2)) * width
    return (
        lower if uses_lower(boundedness) else 0.0,
        upper if uses_upper(boundedness) else 0.0,
    )


@st.composite
def fit_inputs(
    draw, boundedness=None, method=None, num_terms=None, term_counts=NUM_TERMS
):
    """Input data from samples() and parameters whose bounds contain it."""
    if boundedness is None:
        boundedness = draw(st.sampled_from(BOUNDEDNESS))
    if method is None:
        method = draw(st.sampled_from(FIT_METHODS))
    if num_terms is None:
        num_terms = draw(st.sampled_from(term_counts))
    x = draw(samples())
    lower, upper = draw(bounds_around(boundedness, x))
    params = MetalogParameters(
        boundedness=boundedness,
        method=method,
        lower_bound=lower,
        upper_bound=upper,
        num_terms=num_terms,
    )
    data = MetalogInputData.from_values(
        jnp.asarray(x), DEFAULT_Y, precomputed_quantiles=False
    )
    return data, params


@st.composite
def fitted_metalogs(draw, boundedness=None, term_counts=NUM_TERMS):
    """A fitted Metalog: Feasible, or an OLS or Lasso fit certified feasible."""
    return fit_certified(*draw(fit_inputs(boundedness, term_counts=term_counts)))


def metalogs(boundedness=None, term_counts=NUM_TERMS):
    """Hand-set 3-term or fitted metalogs (shrinking prefers hand-set).

    Hand-set metalogs have 3 terms, so they're drawn only when 3 is a term count.
    """
    fitted = fitted_metalogs(boundedness, term_counts)
    if 3 not in term_counts:
        return fitted
    return st.one_of(hand_set_metalogs(boundedness), fitted)


@st.composite
def coefficient_pairs(draw):
    """Two certified-feasible coefficient vectors with the same number of terms."""
    k = draw(st.sampled_from(NUM_TERMS))
    return tuple(draw(metalogs(term_counts=(k,))).a for _ in range(2))


@st.composite
def low_order_unbounded_metalogs(draw):
    """Feasible unbounded 3- or 4-term metalogs, whose mode has a closed form.

    For 4 terms, feasibility is |a3| < 2 a2 and a4 >= a3 ln((2 a2 + a3) /
    (2 a2 - a3)) - 4 a2 (Baucells et al. 2025, Proposition 6).
    """
    a = draw(three_term_coefficients())
    if draw(st.booleans()):
        a2 = a[1]
        a3 = draw(st.floats(-0.95, 0.95)) * 2 * a2
        floor = a3 * np.log((2 * a2 + a3) / (2 * a2 - a3)) - 4 * a2
        a = [a[0], a2, a3, floor + draw(st.floats(0.01, 5)) * a2]
    params = MetalogParameters(
        boundedness=MetalogBoundedness.UNBOUNDED,
        method=MetalogFitMethod.OLS,
        lower_bound=0.0,
        upper_bound=0.0,
        num_terms=len(a),
    )
    return Metalog(params, jnp.asarray(a, dtype=jnp.float64))


@st.composite
def coefficient_vectors(draw):
    """Coefficients on both sides of the feasibility boundary, 3 to 9 terms.

    a2 = 1 sets the scale (feasibility is scale-free), and the higher terms have a
    spread from 0.01 to 10: small spreads are feasible and large ones mostly not.
    """
    k = draw(st.sampled_from((3, 4, 5, 6, 7, 8, 9)))
    # A power of 10 times a factor, so every decade is drawn as often
    spread = 10 ** draw(st.integers(-2, 0)) * draw(st.floats(1, 10))
    rest = draw(st.lists(st.floats(-1, 1), min_size=k - 2, max_size=k - 2))
    return np.array([draw(st.floats(-10, 10)), 1.0, *(spread * np.array(rest))])


@st.composite
def printable_metalogs(draw):
    """A Metalog of any boundedness with coefficient_vectors(), feasible or not."""
    boundedness = draw(st.sampled_from(BOUNDEDNESS))
    lower, upper = draw(hand_set_bounds(boundedness))
    a = draw(coefficient_vectors())
    params = MetalogParameters(
        boundedness=boundedness,
        method=draw(st.sampled_from(FIT_METHODS)),
        lower_bound=lower,
        upper_bound=upper,
        num_terms=len(a),
    )
    return Metalog(params, jnp.asarray(a, dtype=jnp.float64))


@st.composite
def metalog_batches(draw, size=3, term_counts=NUM_TERMS):
    """Fitted metalogs for jax.vmap: shared static fields, different data and bounds."""
    boundedness = draw(st.sampled_from(BOUNDEDNESS))
    method = draw(st.sampled_from(FIT_METHODS))
    num_terms = draw(st.sampled_from(term_counts))
    inputs = [draw(fit_inputs(boundedness, method, num_terms)) for _ in range(size)]
    models = [fit_certified(data, params) for data, params in inputs]
    if len({m.method for m in models}) > 1:
        # A fallback to Feasible changed the static fields, so refit them all
        feasible = MetalogFitMethod.Feasible
        models = [fit(data, params.replace(method=feasible)) for data, params in inputs]
    return models


@st.composite
def sample_batches(draw, size=3, term_counts=NUM_TERMS):
    """Raw samples for jax.vmap, shape (size, SAMPLE_SIZE), and parameters for all."""
    boundedness = draw(st.sampled_from(BOUNDEDNESS))
    x = np.stack([draw(samples()) for _ in range(size)])
    lower, upper = draw(bounds_around(boundedness, x))
    params = MetalogParameters(
        boundedness=boundedness,
        method=draw(st.sampled_from(FIT_METHODS)),
        lower_bound=lower,
        upper_bound=upper,
        num_terms=draw(st.sampled_from(term_counts)),
    )
    return x, params


def alphas():
    """SPT alphas in [0.01, 0.45], on a 0.01 grid so data can hit the quantiles."""
    return st.integers(1, 45).map(lambda j: j / 100)


@st.composite
def spt_triplets(draw, boundedness=None, feasible=False):
    """An SPT triplet and its parameters, drawn on the metalog's unbounded scale.

    The median's position r between the outer quantiles on that scale decides
    feasibility: k_alpha < r < 1 - k_alpha (Keelin 2016, Propositions 2-4). r spans
    (0, 1), so feasible and infeasible triplets are both drawn; ``feasible=True``
    keeps r inside the feasible range, away from its edges.
    """
    if boundedness is None:
        boundedness = draw(st.sampled_from(BOUNDEDNESS))
    alpha = draw(alphas())
    lower, upper = draw(hand_set_bounds(boundedness))
    if boundedness == MetalogBoundedness.UNBOUNDED:
        z0, width = draw(st.floats(-100, 100)), draw(st.floats(0.01, 100))
    else:
        z0, width = draw(st.floats(-4, 2)), draw(st.floats(0.05, 4))
    if feasible:
        k_alpha = 0.5 * (1 - THETA_3 * (0.5 - alpha))
        r = k_alpha + draw(st.floats(0.01, 0.99)) * (1 - 2 * k_alpha)
    else:
        r = draw(st.floats(0.001, 0.999))
    z = np.array([z0, z0 + r * width, z0 + width])
    params = SPTMetalogParameters(
        boundedness=boundedness, alpha=alpha, lower_bound=lower, upper_bound=upper
    )
    return from_unbounded_scale(z, boundedness, lower, upper), params


@st.composite
def spt_out_of_bounds_triplets(draw):
    """An SPT triplet with a quantile at or beyond a bound its parameters use."""
    boundedness = draw(st.sampled_from(BOUNDEDNESS[1:]))
    q, params = draw(spt_triplets(boundedness))
    sides = [
        side
        for side, used in (("lower", uses_lower), ("upper", uses_upper))
        if used(boundedness)
    ]
    t = draw(st.floats(0, 0.99)) * (q[2] - q[0])
    if draw(st.sampled_from(sides)) == "lower":
        return q, params.replace(lower_bound=float(q[0] + t))
    return q, params.replace(upper_bound=float(q[2] - t))


@st.composite
def grid_datasets(draw, size=20):
    """Lognormal or gamma quantiles at a size-point grid of probabilities.

    Neither family is a metalog, so no fitted quantile ties with the data and
    rounding can't change a KS distance.
    """
    y = draw(probabilities(size))
    shape = draw(st.floats(0.2, 1))
    if draw(st.booleans()):
        x = stats.lognorm(s=shape).ppf(np.asarray(y))
    else:
        x = stats.gamma(a=1 + 9 * shape).ppf(np.asarray(y))
    return draw(st.floats(-50, 50)) + draw(st.floats(0.1, 20)) * x, y


@st.composite
def hdr_streams(draw):
    """A stream of n HDR draws, a split point 0 < k < n, and the other HDR fields."""
    n = draw(st.integers(2, 30))
    k = draw(st.integers(1, n - 1))
    fields = ("variable", "entity", "time", "agent")
    return n, k, {name: draw(st.integers(0, 1_000_000)) for name in fields}
