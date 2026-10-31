"""Grid search methods for the Metalog JAX library.

This module fits metalog distributions across datasets, L1 penalties and term counts
in one call, using JAX's vectorization primitives for efficient parallel computation
on CPU, GPU, or TPU devices.

Key functions:
    - fit_grid: Grid search over any combination of datasets, L1 penalties and term
      counts, with any fit method
    - find_best_config: Find best configuration in a grid
    - extract_best_from_grid: Extract best result from grid for a dataset
    - extract_metalog: Extract one fit as a ready-to-use Metalog

Every grid result says which fits are valid distributions: ``GridResult.feasible``
is True where ``M'(y) >= 0`` on all of (0, 1), decided exactly by
``check_feasibility`` (Baucells et al. 2025, Proposition 5 with Algorithm 1). OLS and
Lasso fits can be infeasible, and an infeasible fit's KS distance doesn't compare two
distributions, so find_best_config and extract_best_from_grid skip infeasible fits.
fit_grid returns every fit, with the infeasible ones marked; pass
``on_infeasible="raise"`` to raise for them instead.

fit_grid reads x as quantiles at y unless ``precomputed_quantiles=False``, which
computes each dataset's quantiles from raw samples first, as
MetalogInputData.from_values does.
"""
# Copyright: Travis Jefferies 2026

from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

from metalog_jax.base import (
    MetalogBase,
    MetalogBaseData,
    MetalogParameters,
)
from metalog_jax.base.enums import MetalogFitMethod
from metalog_jax.feasibility import Engine, check_feasibility, get_engine
from metalog_jax.feasibility.engine import DEFAULT_TERM_ORDER, TermOrder, term_basis
from metalog_jax.metalog import GridResult, Metalog, SPTMetalog, _fit_coefficients
from metalog_jax.regression import LassoParameters
from metalog_jax.regression.base import RANK_DEFICIENT_MESSAGE
from metalog_jax.regression.lasso import DEFAULT_LASSO_PARAMETERS, fit_lasso
from metalog_jax.regression.ols import fit_ordinary_least_squares
from metalog_jax.utils import ks_distance

# Maximum number of terms supported for dynamic num_terms in vectorized operations.
# Pre-allocating to this size allows num_terms to be traced (non-static) for vmap.
# It limits the OLS and Lasso grids; Feasible fits each term count separately.
MAX_TERMS = 30

# What fit_grid does when a fit is infeasible
OnInfeasible = Literal["raise", "flag"]
_ON_INFEASIBLE = ("raise", "flag")

_INFEASIBLE_MESSAGE = (
    "Some grid fits are infeasible: M'(y) < 0 somewhere in (0, 1), so they aren't "
    "valid distributions. They're the False entries of feasible = {feasible}. Pass "
    "on_infeasible='flag' to get every fit, with GridResult.feasible marking these."
)

_GRID_RANK_MESSAGE = (
    RANK_DEFICIENT_MESSAGE
    + " The rank-deficient fits are the False entries of full_rank = {full_rank}."
)


@partial(jax.jit, static_argnums=0)
def _check_rows(engine: Engine, a: jnp.ndarray) -> jnp.ndarray:
    """Return check_feasibility(engine, row).feasible for each row of a."""
    return jax.vmap(lambda row: check_feasibility(engine, row).feasible)(a)


def _with_feasibility(result: GridResult) -> GridResult:
    """Fill in ``result.feasible`` with check_feasibility of every fit.

    Grids zero-pad each fit's coefficients to the largest term count, which doesn't
    change its quantile function, so that count's engine checks every fit. The fits
    are checked in one jitted call. Their count is rounded up, to a power of two or
    past 1024 to a multiple of 1024, so that grids of similar sizes share a
    compilation without padding a large grid by much.

    Args:
        result: A grid result, with coefficients of shape (*batch, num_terms).

    Returns:
        GridResult: The result, with ``feasible`` of shape ``batch``.
    """
    a = result.metalog.a
    rows = a.reshape(-1, a.shape[-1])
    n = rows.shape[0]
    padded = 1 << max(n - 1, 0).bit_length() if n <= 1024 else -(-n // 1024) * 1024
    rows = jnp.pad(rows, ((0, padded - n), (0, 0)), mode="edge")
    engine = get_engine(a.shape[-1], result.metalog.term_order)
    feasible = _check_rows(engine, rows)[:n]
    return result.replace(feasible=feasible.reshape(a.shape[:-1]))


def _as_quantiles(
    x: jnp.ndarray, y: jnp.ndarray, precomputed_quantiles: bool
) -> jnp.ndarray:
    """Return each dataset's quantiles at y, computing them from samples if needed.

    Args:
        x: Quantiles at y if ``precomputed_quantiles``, else raw samples. Shape (n,)
            for one dataset or (n_datasets, n) for a batch.
        y: Probability levels, shape (m,) or (n_datasets, m).
        precomputed_quantiles: Whether x already holds the quantiles at y.

    Returns:
        x unchanged if ``precomputed_quantiles``, else each dataset's quantiles at y,
        computed with ``jnp.quantile`` as MetalogInputData.from_values does. Shape
        (m,) or (n_datasets, m).
    """
    if precomputed_quantiles:
        return x
    if x.ndim == 2:
        return jax.vmap(jnp.quantile)(x, y)
    return jnp.quantile(x, y)


def _check_on_infeasible(on_infeasible: OnInfeasible) -> None:
    """Raise ValueError unless ``on_infeasible`` is "raise" or "flag".

    Args:
        on_infeasible: The value passed to fit_grid.

    Raises:
        ValueError: For any other value.
    """
    if on_infeasible not in _ON_INFEASIBLE:
        raise ValueError(
            f"on_infeasible must be 'raise' or 'flag', not {on_infeasible!r}"
        )


def _score_fit(metalog: Metalog, x: jnp.ndarray, y: jnp.ndarray) -> GridResult:
    """Score a fit by its KS distance at y, without checking its density.

    ppf checks the density, and under vmap one infeasible fit would raise for the
    whole batch. The grid marks infeasible fits instead.

    Args:
        metalog: A fitted metalog, possibly infeasible.
        x: The dataset's quantiles at y.
        y: The dataset's probability levels.

    Returns:
        GridResult: The fit and its KS distance.
    """
    y = jnp.asarray(y, dtype=jnp.result_type(y, metalog.a))
    fitted = Metalog.inverse_quantile_transform(
        metalog._ppf(y, metalog.num_terms),
        metalog.boundedness,
        metalog.lower_bound,
        metalog.upper_bound,
    )
    return GridResult(metalog=metalog, ks_dist=ks_distance(x, fitted))


def _require_a_feasible_fit(feasible: jnp.ndarray, where: str) -> None:
    """Raise if no entry of ``feasible`` is True, when it can be tested.

    Under jit or vmap the mask is traced and can't be tested in Python, so the
    selection functions return an infinite KS distance instead.

    Args:
        feasible: Feasibility of the fits to choose from.
        where: Which fits they are, for the message, such as "for dataset 2".

    Raises:
        ValueError: If every fit is infeasible.
    """
    try:
        found = bool(jnp.any(feasible))
    except jax.errors.ConcretizationTypeError:  # traced
        return
    if not found:
        raise ValueError(
            f"No configuration {where} is feasible: every fit has M'(y) < 0 "
            "somewhere in (0, 1). Try fewer terms, larger L1 penalties, or "
            "MetalogFitMethod.Feasible."
        )


def _later_terms(term_order: TermOrder) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The powers of (y - 0.5) of terms 5 to MAX_TERMS, and which multiply the logit.

    A term's basis function depends only on its position, not on the number of
    terms, so the first k of these are a k-term metalog's terms 5 to k.

    Args:
        term_order: The order of the basis terms.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: The powers (int) and the logit mask
            (bool), each of shape (MAX_TERMS - 4,).
    """
    powers, has_logit = zip(*term_basis(MAX_TERMS, term_order)[4:], strict=True)
    return jnp.asarray(powers), jnp.asarray(has_logit)


@partial(jax.jit, static_argnames=("term_order",))
def _get_target_vmap(
    y: jnp.ndarray,
    num_terms: jnp.ndarray,
    term_order: TermOrder = DEFAULT_TERM_ORDER,
) -> jnp.ndarray:
    """Construct the design matrix with dynamic num_terms for vmap compatibility.

    Unlike MetalogBase._get_target which uses static_argnames for efficient
    single-fit JIT compilation, this version supports traced num_terms for
    vmap operations over different term counts.

    Always allocates MAX_TERMS columns and masks unused columns to zero.

    Args:
        y: Probability values, shape (m,).
        num_terms: Number of metalog terms (traced for vmap).
        term_order: The order of the basis terms (static).

    Returns:
        Design matrix of shape (m, MAX_TERMS) with columns beyond num_terms
        masked to zero.
    """
    m = y.shape[0]
    delta = y - 0.5
    log_t = jnp.log(y / (1 - y))

    # The first four terms are the same in every order
    col0 = jnp.ones(m)
    col1 = log_t
    col2 = delta * log_t
    col3 = delta

    # Columns 4+ (terms 5 to MAX_TERMS)
    powers, has_logit = _later_terms(term_order)
    delta_powers = jnp.power(delta[:, None], powers)
    additional_cols = jnp.where(has_logit, log_t[:, None] * delta_powers, delta_powers)

    # Stack all columns
    result = jnp.column_stack([col0, col1, col2, col3, additional_cols])

    # Mask unused columns
    col_indices = jnp.arange(MAX_TERMS)
    col_mask = (col_indices < num_terms).astype(result.dtype)

    return result * col_mask[None, :]


def _masked_ppf(
    a: jnp.ndarray,
    y: jnp.ndarray,
    num_terms: jnp.ndarray,
    term_order: TermOrder = DEFAULT_TERM_ORDER,
) -> jnp.ndarray:
    """Evaluate the unbounded metalog quantile function with a traced term count.

    Same basis as MetalogBase._ppf, but num_terms can be traced, so terms beyond it
    are masked rather than sliced off. This keeps shapes fixed under ``jax.vmap``.

    Args:
        a: Coefficients, shape (MAX_TERMS,), zero beyond num_terms.
        y: Probability levels in (0, 1).
        num_terms: Number of active terms (traced for vmap).
        term_order: The order of the basis terms.

    Returns:
        Values of the unbounded quantile function at y.
    """
    delta_term = y - 0.5
    log_term = jnp.log(y / (1 - y))

    a_padded = jnp.zeros(MAX_TERMS)
    a_padded = a_padded.at[: a.shape[0]].set(a)

    ppf = a_padded[0] + (a_padded[1] * log_term)
    ppf = ppf + jnp.where(num_terms > 2, a_padded[2] * delta_term * log_term, 0.0)
    ppf = ppf + jnp.where(num_terms > 3, a_padded[3] * delta_term, 0.0)

    terms = jnp.arange(5, MAX_TERMS + 1)
    active_mask = terms <= num_terms
    powers, has_logit = _later_terms(term_order)
    coeffs = a_padded[4:MAX_TERMS]

    contrib = coeffs * jnp.power(delta_term[:, None], powers)
    contrib = jnp.where(has_logit, contrib * log_term[:, None], contrib)
    return ppf + jnp.sum(jnp.where(active_mask, contrib, 0.0), axis=1)


def _grid_result(
    weights: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_terms: jnp.ndarray,
    params: MetalogParameters,
    max_terms_output: int,
) -> GridResult:
    """Score one grid fit and package it as a GridResult.

    Zeroes the coefficients beyond num_terms, evaluates the fit at the dataset's
    probability levels y, maps the result back to the distribution's support and
    takes its KS distance from x.

    Args:
        weights: Fitted coefficients, shape (MAX_TERMS,).
        x: The dataset's quantiles at y.
        y: The dataset's probability levels.
        num_terms: Number of active terms (traced for vmap).
        params: Metalog parameters for the grid.
        max_terms_output: Number of coefficients to keep in the result.

    Returns:
        GridResult with the fitted metalog and its KS distance.
    """
    coeff_mask = (jnp.arange(MAX_TERMS) < num_terms).astype(weights.dtype)
    masked_weights = weights * coeff_mask
    fitted_quantiles = Metalog.inverse_quantile_transform(
        _masked_ppf(masked_weights, y, num_terms, params.term_order),
        params.boundedness,
        params.lower_bound,
        params.upper_bound,
    )
    result_params = params.replace(num_terms=max_terms_output)
    result_metalog = Metalog(
        metalog_params=result_params, a=masked_weights[:max_terms_output]
    )
    return GridResult(metalog=result_metalog, ks_dist=ks_distance(x, fitted_quantiles))


def _build_fit_single_config(
    params: MetalogParameters,
    max_terms_output: int,
    lasso_params: LassoParameters | None,
):
    """Build a fit function for a single configuration.

    Returns a function that fits a single (x, y, l1, n_terms) configuration.
    This is extracted to reduce cyclomatic complexity of fit_grid.

    Args:
        params: Metalog parameters for the grid, with the method the fits use.
        max_terms_output: Number of coefficients to keep in each result.
        lasso_params: Lasso settings, whose ``lam`` each fit replaces with its
            penalty; None to fit by OLS.

    Returns:
        A function of (x, y, l1_penalty, n_terms) that returns a GridResult.
    """
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound
    boundedness = params.boundedness

    def fit_single_config(
        x_i: jnp.ndarray,
        y_i: jnp.ndarray,
        l1_penalty: jnp.ndarray,
        n_terms: jnp.ndarray,
    ) -> GridResult:
        """Fit a single configuration (traceable for vmap)."""
        quantiles = Metalog.quantile_transform(
            x_i, boundedness, lower_bound, upper_bound
        )
        target = _get_target_vmap(
            y=y_i, num_terms=n_terms, term_order=params.term_order
        )

        if lasso_params is None:
            model = fit_ordinary_least_squares(target, quantiles)
        else:
            lasso_p = lasso_params.replace(lam=l1_penalty)
            model = fit_lasso(target, quantiles, lasso_p, intercept=True)

        return _grid_result(model.weights, x_i, y_i, n_terms, params, max_terms_output)

    return fit_single_config


def _apply_grid_vmaps(
    fit_fn,
    x: jnp.ndarray,
    y: jnp.ndarray,
    l1_array: jnp.ndarray,
    num_terms_array: jnp.ndarray,
    has_datasets: bool,
    has_l1: bool,
    has_num_terms: bool,
) -> GridResult:
    """Apply the appropriate vmap structure based on active grid axes.

    This helper dispatches to the correct vmap nesting pattern based on
    which axes are active (datasets, l1, num_terms).
    """
    # Encode the 8 cases as a 3-bit pattern for dispatch
    case = (int(has_datasets) << 2) | (int(has_l1) << 1) | int(has_num_terms)

    if case == 0b000:  # No grids, single dataset
        return fit_fn(x, y, l1_array[0], num_terms_array[0])

    if case == 0b001:  # num_terms only
        return jax.vmap(lambda nt: fit_fn(x, y, l1_array[0], nt))(num_terms_array)

    if case == 0b010:  # l1 only
        return jax.vmap(lambda l1: fit_fn(x, y, l1, num_terms_array[0]))(l1_array)

    if case == 0b011:  # l1 x num_terms

        def fit_for_l1(l1):
            return jax.vmap(lambda nt: fit_fn(x, y, l1, nt))(num_terms_array)

        return jax.vmap(fit_for_l1)(l1_array)

    if case == 0b100:  # datasets only
        return jax.vmap(lambda xi, yi: fit_fn(xi, yi, l1_array[0], num_terms_array[0]))(
            x, y
        )

    if case == 0b101:  # datasets x num_terms

        def fit_for_dataset(xi, yi):
            return jax.vmap(lambda nt: fit_fn(xi, yi, l1_array[0], nt))(num_terms_array)

        return jax.vmap(fit_for_dataset)(x, y)

    if case == 0b110:  # datasets x l1

        def fit_for_dataset(xi, yi):
            return jax.vmap(lambda l1: fit_fn(xi, yi, l1, num_terms_array[0]))(l1_array)

        return jax.vmap(fit_for_dataset)(x, y)

    # case == 0b111: datasets x l1 x num_terms
    def fit_for_dataset(xi, yi):
        def fit_for_l1(l1):
            return jax.vmap(lambda nt: fit_fn(xi, yi, l1, nt))(num_terms_array)

        return jax.vmap(fit_for_l1)(l1_array)

    return jax.vmap(fit_for_dataset)(x, y)


def _grid_method(
    params: MetalogParameters,
    l1_penalties: jnp.ndarray | None,
    lasso_params: LassoParameters | None,
    on_infeasible: OnInfeasible,
) -> MetalogFitMethod:
    """Check fit_grid's arguments, and return the method its fits use.

    Args:
        params: The grid's parameters.
        l1_penalties: The penalty axis, or None.
        lasso_params: The Lasso settings, or None.
        on_infeasible: What to do with infeasible fits.

    Returns:
        MetalogFitMethod: Lasso if ``params.method`` is Lasso or ``l1_penalties`` is
        given, else ``params.method``.

    Raises:
        ValueError: If ``on_infeasible`` isn't "raise" or "flag".
        TypeError: If ``l1_penalties`` is given with MetalogFitMethod.Feasible, or
            ``lasso_params`` for a grid that doesn't fit with Lasso.
    """  # noqa: DOC503
    _check_on_infeasible(on_infeasible)
    if l1_penalties is not None and params.method == MetalogFitMethod.Feasible:
        raise TypeError(
            "l1_penalties makes a Lasso grid, but params.method is "
            "MetalogFitMethod.Feasible, which takes no penalty; pass params with "
            "MetalogFitMethod.OLS or MetalogFitMethod.Lasso, or l1_penalties=None."
        )
    if l1_penalties is not None or params.method == MetalogFitMethod.Lasso:
        return MetalogFitMethod.Lasso
    if lasso_params is not None:
        raise TypeError(
            "lasso_params only affects Lasso fits; pass params with "
            "MetalogFitMethod.Lasso or l1_penalties, or pass lasso_params=None."
        )
    return params.method


def _check_max_terms(terms: list[int], method: MetalogFitMethod) -> None:
    """Raise if an OLS or Lasso grid has more terms than its masked design holds.

    Args:
        terms: The term counts.
        method: The method the fits use.

    Raises:
        ValueError: If ``method`` isn't Feasible and a term count is more than
            MAX_TERMS.
    """
    if method != MetalogFitMethod.Feasible and max(terms) > MAX_TERMS:
        raise ValueError(
            f"OLS and Lasso grids fit at most {MAX_TERMS} terms, not {max(terms)}; "
            "MetalogFitMethod.Feasible fits any number."
        )


@partial(jax.jit, static_argnames=("max_terms", "term_order"))
def _full_rank(
    rows: jnp.ndarray, terms: jnp.ndarray, max_terms: int, term_order: TermOrder
) -> jnp.ndarray:
    """Whether each row's design matrix has full column rank, for each term count.

    A term's basis function depends only on its position, so a k-term design matrix
    is the first k columns of the max_terms one. The rank uses numpy's default SVD
    tolerance, as fit() does.

    Args:
        rows: Probability levels, shape (n_rows, m).
        terms: Term counts, shape (n_terms,).
        max_terms: The largest term count (static).
        term_order: The order of the basis terms (static).

    Returns:
        jnp.ndarray: Boolean array of shape (n_rows, n_terms).
    """

    def row_full_rank(y_i: jnp.ndarray) -> jnp.ndarray:
        design = MetalogBase._get_target(y_i, max_terms, term_order)
        columns = jnp.arange(max_terms)

        def has_rank(k: jnp.ndarray) -> jnp.ndarray:
            return jnp.linalg.matrix_rank(design * (columns < k)) == k

        return jax.vmap(has_rank)(terms)

    return jax.vmap(row_full_rank)(rows)


def _check_full_rank(
    y: jnp.ndarray, terms: list[int], term_order: TermOrder, has_num_terms: bool
) -> None:
    """Raise if any fit's least-squares problem has no unique solution.

    fit() raises for a rank-deficient design matrix with OLS and Feasible, and so
    does every fit of a grid. The rank depends only on the probability levels and
    the term count, so each distinct row of y is checked once; datasets usually
    share their levels.

    Args:
        y: Probability levels, shape (m,) or (n_datasets, m).
        terms: The term counts.
        term_order: The order of the basis terms.
        has_num_terms: Whether the grid has a term-count axis.

    Raises:
        checkify.JaxRuntimeError: If a design matrix is rank deficient. The message
            shows which fits are, in the grid's shape.
    """  # noqa: DOC502
    rows = y.reshape(-1, y.shape[-1])
    try:
        rows, index = np.unique(np.asarray(rows), axis=0, return_inverse=True)
    except jax.errors.TracerArrayConversionError:  # traced
        index = np.arange(rows.shape[0])
    full_rank = _full_rank(
        jnp.asarray(rows), jnp.asarray(terms), max(terms), term_order
    )
    full_rank = full_rank[index.reshape(-1)].reshape(*y.shape[:-1], len(terms))
    if not has_num_terms:
        full_rank = full_rank[..., 0]
    checkify.check(jnp.all(full_rank), _GRID_RANK_MESSAGE, full_rank=full_rank)


def _fit_masked(
    x: jnp.ndarray,
    y: jnp.ndarray,
    params: MetalogParameters,
    terms: list[int],
    has_num_terms: bool,
    l1_penalties: jnp.ndarray | None,
    lasso_params: LassoParameters | None,
) -> GridResult:
    """Fit an OLS or Lasso grid in one vmapped call, with a traced term count.

    Args:
        x: Each dataset's quantiles at y, shape (n,) or (n_datasets, n).
        y: Probability levels, shape (m,) or (n_datasets, m).
        params: Parameters with the method the fits use, OLS or Lasso.
        terms: The term counts.
        has_num_terms: Whether the grid has a term-count axis.
        l1_penalties: The penalty axis, or None.
        lasso_params: The Lasso settings, or None for DEFAULT_LASSO_PARAMETERS.

    Returns:
        GridResult: The fits, with ``feasible`` from check_feasibility.
    """
    max_terms = max(terms)
    if lasso_params is None:
        lasso_params = DEFAULT_LASSO_PARAMETERS
    has_l1 = l1_penalties is not None
    l1_array = (
        jnp.asarray(l1_penalties) if has_l1 else jnp.asarray(lasso_params.lam)[None]
    )
    use_lasso = params.method == MetalogFitMethod.Lasso
    fit_fn = _build_fit_single_config(
        params, max_terms, lasso_params if use_lasso else None
    )
    result = _apply_grid_vmaps(
        fit_fn, x, y, l1_array, jnp.array(terms), x.ndim == 2, has_l1, has_num_terms
    )
    return _with_feasibility(result)


def _fit_feasible(
    x: jnp.ndarray,
    y: jnp.ndarray,
    params: MetalogParameters,
    terms: list[int],
    has_num_terms: bool,
) -> GridResult:
    """Fit a grid with MetalogFitMethod.Feasible, one term count at a time.

    The feasible solver needs the number of terms fixed at trace time, so each
    distinct term count is fitted in its own call, vmapped over the datasets and
    compiled once. The fits are zero-padded to the largest count, which doesn't
    change their quantile functions, and stacked along the last axis, as in the OLS
    and Lasso grids. The solver certifies every fit with check_feasibility, and
    raises if it can't, so every fit is marked feasible.

    Args:
        x: Each dataset's quantiles at y, shape (n,) or (n_datasets, n).
        y: Probability levels, shape (m,) or (n_datasets, m).
        params: Parameters with MetalogFitMethod.Feasible.
        terms: The term counts.
        has_num_terms: Whether the grid has a term-count axis.

    Returns:
        GridResult: The fits, every one marked feasible.

    Raises:
        checkify.JaxRuntimeError: If the solver can't certify a fit.
    """  # noqa: DOC502
    max_terms = max(terms)
    padded_params = params.replace(num_terms=max_terms)

    def fit_terms(k: int) -> GridResult:
        k_params = params.replace(num_terms=k)

        def fit_one(x_i: jnp.ndarray, y_i: jnp.ndarray) -> GridResult:
            data = MetalogBaseData(x=x_i, y=y_i, precomputed_quantiles=True)
            result = _score_fit(_fit_coefficients(data, k_params), x_i, y_i)
            a = jnp.pad(result.metalog.a, (0, max_terms - k))
            return result.replace(metalog=Metalog(metalog_params=padded_params, a=a))

        return jax.vmap(fit_one)(x, y) if x.ndim == 2 else fit_one(x, y)

    fits = {k: fit_terms(k) for k in dict.fromkeys(terms)}
    if has_num_terms:
        axis = x.ndim - 1
        result = jax.tree.map(
            lambda *leaves: jnp.stack(leaves, axis=axis), *(fits[k] for k in terms)
        )
    else:
        result = fits[terms[0]]
    return result.replace(feasible=jnp.ones(result.ks_dist.shape, dtype=bool))


def fit_grid(
    x: jnp.ndarray,
    y: jnp.ndarray,
    params: MetalogParameters,
    *,
    num_terms: Sequence[int] | None = None,
    l1_penalties: jnp.ndarray | None = None,
    precomputed_quantiles: bool = True,
    lasso_params: LassoParameters | None = None,
    on_infeasible: OnInfeasible = "flag",
) -> GridResult:
    """Fit a grid of metalogs over datasets, L1 penalties and term counts.

    Detects which axes to search from its inputs:

    - **Datasets:** if x is 2-D, one fit per row.
    - **Term counts:** if ``num_terms`` is given, one fit per count.
    - **L1 penalties:** if ``l1_penalties`` is given, one Lasso fit per penalty.

    Output shape depends on active axes, ordered datasets, l1_penalties, num_terms:
    from () for one fit to (n_datasets, n_penalties, n_terms) for all three. For
    example, (n_datasets, n_terms) for datasets x num_terms.

    Every fit uses ``params.method``, and Lasso when ``l1_penalties`` is given; the
    fits' ``method`` says which. OLS and Lasso grids fit every configuration in one
    vmapped call, with up to MAX_TERMS terms. MetalogFitMethod.Feasible needs the
    number of terms fixed when it compiles, so it fits each term count in its own
    vmapped call over the datasets. A term-count grid zero-pads each fit's
    coefficients to the largest count, which doesn't change its quantile function.

    Each fit is checked as fit() checks it:

    - **Rank:** for OLS and Feasible, the grid raises if any configuration's basis
      matrix is rank deficient, so that its fit isn't unique. Lasso is regularized
      and isn't checked.
    - **Feasibility:** ``feasible`` says whether each fit is a valid distribution,
      decided exactly by check_feasibility. Feasible's fits always are. An
      infeasible fit is returned and marked, or raises with
      ``on_infeasible="raise"``.

    Args:
        x: Each dataset's quantiles at y, or raw samples if
            ``precomputed_quantiles`` is False. Shape (n,) for a single dataset or
            (batch, n) for batched.
        y: Probability levels. Shape (m,) for single dataset or (batch, m) for
            batched.
        params: MetalogParameters configuration. The num_terms field is ignored
            if num_terms argument is provided.
        num_terms: Optional sequence of term counts to search over.
        l1_penalties: Optional array of L1 penalties to search over (implies Lasso).
        precomputed_quantiles: Whether x holds quantiles at y (the default). If
            False, x holds raw samples, and each dataset's quantiles at y are
            computed first, as MetalogInputData.from_values does.
        lasso_params: Settings for the Lasso fits; DEFAULT_LASSO_PARAMETERS if None.
            Its ``lam`` is the penalty when there's no ``l1_penalties`` axis, and its
            ``num_iters`` and ``tol`` apply to every Lasso fit.
        on_infeasible: What to do with infeasible fits. "flag", the default,
            returns every fit, with ``feasible`` False for the infeasible ones;
            "raise" raises for the grid, naming them.

    Returns:
        GridResult with metalog fits, KS distances and ``feasible``, whether each fit
        is a valid distribution. Shape depends on active axes. Pass ``feasible`` to
        find_best_config so that it skips infeasible fits.

    Raises:
        ValueError: If ``on_infeasible`` isn't "raise" or "flag", or an OLS or Lasso
            grid has more than MAX_TERMS terms.
        TypeError: If ``l1_penalties`` is given with MetalogFitMethod.Feasible, or
            ``lasso_params`` for a grid that doesn't fit with Lasso.
        checkify.JaxRuntimeError: For OLS and Feasible, if a configuration's basis
            matrix is rank deficient, or if Feasible can't certify a fit; and if
            ``on_infeasible`` is "raise" and a fit is infeasible. Each message shows
            which fits, in the grid's shape.
    """  # noqa: DOC502
    method = _grid_method(params, l1_penalties, lasso_params, on_infeasible)
    params = params.replace(method=method)
    has_num_terms = num_terms is not None
    terms = [int(k) for k in num_terms] if has_num_terms else [params.num_terms]
    _check_max_terms(terms, method)
    x = _as_quantiles(x, y, precomputed_quantiles)
    if method != MetalogFitMethod.Lasso:  # Lasso is regularized, as in fit()
        _check_full_rank(y, terms, params.term_order, has_num_terms)
    if method == MetalogFitMethod.Feasible:
        result = _fit_feasible(x, y, params, terms, has_num_terms)
    else:
        result = _fit_masked(
            x, y, params, terms, has_num_terms, l1_penalties, lasso_params
        )
    if on_infeasible == "raise":
        checkify.check(
            jnp.all(result.feasible), _INFEASIBLE_MESSAGE, feasible=result.feasible
        )
    return result


def stack_leaves(*leaves: jnp.ndarray) -> jnp.ndarray:
    """Stack leaf values from multiple PyTrees along a new batch dimension.

    This helper function is used with jax.tree.map to transform a list of
    dataclass instances into a single batched dataclass where each field
    is stacked along axis 0. It enables efficient vectorized operations by
    converting multiple independent structures into a batched structure.

    Args:
        *leaves: Variable number of leaf arrays from corresponding positions
            in multiple PyTrees (e.g., the x field from multiple MetalogInputData
            instances). Each leaf should be a JAX array or scalar value.

    Returns:
        A single JAX array with the leaves stacked along a new first dimension,
        shape (n_leaves, ...) where n_leaves is the number of input leaves and
        ... represents the original shape of each leaf.
    """
    return jnp.stack(leaves)


def make_batch(data: list[MetalogBaseData]) -> MetalogBaseData:
    """Convert a list of MetalogBaseData instances into a single batched instance.

    This function transforms a list of individual MetalogBaseData instances into a
    single MetalogBaseData instance where each field is batched along the first
    dimension. This enables vectorized operations across multiple datasets using
    JAX's vmap.

    Args:
        data: List of MetalogBaseData instances to batch together. Each instance
            should have the same structure (same fields), but can have different
            values. All instances should have arrays with compatible shapes.

    Returns:
        A single MetalogBaseData instance where each field is stacked along axis 0.
        If the input list has n elements and each instance has a field with shape
        (m,), the output will have that field with shape (n, m).
    """
    return jax.tree.map(stack_leaves, *data)


def unvmap(batched_out: Any) -> list[Any]:
    """Convert batched vmap output into a list of individual outputs.

    This function reverses the batching operation performed by jax.vmap by converting
    a single batched PyTree into a list of individual PyTrees, one per batch element.
    Each element in the output list has the same structure as the batched output but
    without the batch dimension.

    Args:
        batched_out: The batched output from `jax.vmap(f)(...)`. Can be a JAX array,
            tuple, dict, dataclass, or any PyTree structure where each leaf has a
            batch dimension as its first axis.

    Returns:
        A list of length n_batch where each element is a PyTree with the same structure
        as batched_out but with the batch dimension removed.
    """
    batch_size = jax.tree_util.tree_leaves(batched_out)[0].shape[0]
    return [
        jax.tree.map(
            lambda x, i=i: x[i].item() if x[i].shape == () else jnp.array(x[i]),
            batched_out,
        )
        for i in range(batch_size)
    ]


def pad_metalog_coeffs(
    metalog: Metalog | SPTMetalog, max_terms: int
) -> Metalog | SPTMetalog:
    """Pad metalog coefficient array to max_terms length with zeros.

    Since different term counts produce different coefficient array lengths,
    we pad shorter arrays with zeros to enable stacking into a consistent
    batched structure.

    Args:
        metalog: Metalog or SPTMetalog instance to pad.
        max_terms: Target length for the coefficient array.

    Returns:
        Metalog instance with coefficients padded to max_terms.
    """
    current_terms = metalog.a.shape[-1]
    if current_terms == max_terms:
        return metalog
    pad_width = max_terms - current_terms
    padded_a = jnp.pad(metalog.a, (0, pad_width), mode="constant")
    return metalog.replace(a=padded_a)


def find_best_config(
    ks_dists: jnp.ndarray, feasible: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the best configuration (minimum KS distance) in a grid.

    Works with grids of any dimensionality (1D, 2D, 3D, etc.). Pass the grid's
    ``feasible`` mask so that infeasible fits, whose KS distances don't compare two
    distributions, can't be chosen: their KS distances count as infinite.

    Args:
        ks_dists: Array of KS distances with shape (d1, d2, ..., dn).
        feasible: Optional mask of the same shape, such as ``GridResult.feasible``.

    Returns:
        Tuple of (best_indices, best_ks) where:
            - best_indices: Indices of the minimum, shape (n,) for n dimensions
            - best_ks: Minimum KS distance value (scalar); infinite under jit or
              vmap if no fit is feasible

    Raises:
        ValueError: If ``feasible`` is given and no fit is feasible.
    """  # noqa: DOC502
    if feasible is not None:
        _require_a_feasible_fit(feasible, "in the grid")
        ks_dists = jnp.where(feasible, ks_dists, jnp.inf)
    flat_idx = jnp.argmin(ks_dists)
    best_idx = jnp.unravel_index(flat_idx, ks_dists.shape)
    best_ks = ks_dists[best_idx]

    # Convert tuple of arrays to stacked array for easier handling
    if len(best_idx) == 1:
        return best_idx[0], best_ks
    return jnp.array(best_idx), best_ks


def extract_best_from_grid(
    grid_results: GridResult,
    dataset_idx: int,
) -> GridResult:
    """Extract the best result from a grid for a specific dataset.

    Finds the feasible configuration with minimum KS distance for the given
    dataset and extracts the corresponding metalog and KS distance. Infeasible fits,
    marked by ``grid_results.feasible``, can't be chosen.

    Args:
        grid_results: GridResult from fit_grid with a datasets axis.
            - For 2D grid: ks_dist shape (n_datasets, n_penalties)
            - For 3D grid: ks_dist shape (n_datasets, n_penalties, n_terms)
        dataset_idx: Index of the dataset to extract best result for.

    Returns:
        GridResult with:
            - metalog: Best metalog for this dataset
            - ks_dist: Corresponding KS distance (scalar); infinite under jit or vmap
              if none of the dataset's fits is feasible
            - feasible: Whether that fit is feasible, or None if the grid has no
              ``feasible`` mask

    Raises:
        ValueError: If none of the dataset's fits is feasible.
    """  # noqa: DOC502
    # Get KS distances for this dataset, with infeasible fits ruled out
    dataset_ks = grid_results.ks_dist[dataset_idx]
    feasible = grid_results.feasible
    if feasible is not None:
        _require_a_feasible_fit(feasible[dataset_idx], f"for dataset {dataset_idx}")
        dataset_ks = jnp.where(feasible[dataset_idx], dataset_ks, jnp.inf)

    # Find best configuration within this dataset's grid
    flat_idx = jnp.argmin(dataset_ks)
    best_idx = jnp.unravel_index(flat_idx, dataset_ks.shape)

    # Extract the best metalog
    # Build full index: (dataset_idx, *best_idx)
    full_idx = (dataset_idx, *best_idx)
    best_metalog = jax.tree.map(
        lambda x: x[full_idx],
        grid_results.metalog,
    )
    best_ks = dataset_ks[tuple(best_idx)]
    best_feasible = None if feasible is None else feasible[full_idx]
    return GridResult(metalog=best_metalog, ks_dist=best_ks, feasible=best_feasible)


def extract_metalog(grid_result: GridResult, *indices: int) -> Metalog:
    """Extract a usable Metalog from grid results at given indices.

    After grid search, metalog fields become JAX arrays due to vmap stacking.
    This function extracts a single metalog at the specified indices and converts
    all scalar fields back to Python types, making the metalog compatible with
    JIT-compiled methods like ppf, pdf, cdf, etc.

    Args:
        grid_result: GridResult from fit_grid.
        *indices: Integer indices to extract. The number of indices should match
            the dimensionality of the grid:
            - 1D grid (e.g., L1 penalties): single index like `extract_metalog(result, 0)`
            - 2D grid (e.g., L1 x num_terms): two indices like `extract_metalog(result, 0, 1)`
            - 3D grid (e.g., datasets x L1 x num_terms): three indices

    Returns:
        Metalog instance with Python-typed fields, ready to use with ppf, pdf,
        cdf, rvs, and other distribution methods.

    Examples:
        Extract the best metalog from a 1D grid search over L1 penalties:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
            >>> from metalog_jax.base import MetalogInputData, MetalogParameters
            >>> from metalog_jax.utils import DEFAULT_Y
            >>>
            >>> samples = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1, 10.5])
            >>> data = MetalogInputData.from_values(
            ...     samples, DEFAULT_Y, precomputed_quantiles=False
            ... )
            >>> params = MetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     method=MetalogFitMethod.Lasso,
            ...     num_terms=5, lower_bound=0.0, upper_bound=0.0
            ... )
            >>> l1_vals = jnp.array([0.0, 0.01, 0.1])
            >>> result = fit_grid(data.x, data.y, params, l1_penalties=l1_vals)
            >>> # Pass feasible so that infeasible fits, such as OLS's at 0.0, are skipped
            >>> best_idx, best_ks = find_best_config(result.ks_dist, result.feasible)
            >>> best_metalog = extract_metalog(result, int(best_idx))
            >>> median = best_metalog.ppf(jnp.array([0.5]))

        Extract the best metalog from a 2D grid search:

            >>> result = fit_grid(data.x, data.y, params,
            ...                   l1_penalties=l1_vals, num_terms=[5, 7, 9])
            >>> result.ks_dist.shape
            (3, 3)
            >>> best_idx, best_ks = find_best_config(result.ks_dist, result.feasible)
            >>> best_l1_idx, best_terms_idx = int(best_idx[0]), int(best_idx[1])
            >>> best_metalog = extract_metalog(result, best_l1_idx, best_terms_idx)

    See Also:
        find_best_config: Find indices of best configuration in a grid.
        extract_best_from_grid: Extract best result for a specific dataset.
    """
    from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod

    # Extract the metalog pytree at the given indices
    extracted = jax.tree.map(lambda x: x[indices], grid_result.metalog)

    # Convert scalar JAX arrays back to Python types for JIT compatibility
    params = MetalogParameters(
        boundedness=MetalogBoundedness(int(extracted.metalog_params.boundedness)),
        lower_bound=float(extracted.metalog_params.lower_bound),
        upper_bound=float(extracted.metalog_params.upper_bound),
        method=MetalogFitMethod(int(extracted.metalog_params.method)),
        num_terms=int(extracted.metalog_params.num_terms),
        term_order=extracted.metalog_params.term_order,
    )
    return Metalog(metalog_params=params, a=extracted.a)
