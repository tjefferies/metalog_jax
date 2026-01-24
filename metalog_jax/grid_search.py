"""Grid search methods for the Metalog JAX library.

This module provides generalized vmap-based grid search functions for fitting
metalog distributions across multiple datasets, hyperparameters, and term counts.

The functions are designed to work with JAX's vectorization primitives for
efficient parallel computation on CPU, GPU, or TPU devices.

Key functions:
    - fit_grid: Unified grid search over any combination of axes
    - fit_grid_datasets: 1D grid over datasets (shared params)
    - fit_grid_hyperparams: 1D grid over hyperparameters for a single dataset
    - fit_grid_num_terms: 1D grid over num_terms for a single dataset
    - fit_grid_datasets_hyperparams: 2D grid over datasets x hyperparameters
    - fit_grid_datasets_num_terms: 2D grid over datasets x num_terms
    - fit_grid_full: 3D grid over datasets x hyperparameters x num_terms
    - find_best_config: Find best configuration in a grid
    - extract_best_from_grid: Extract best result from grid for a dataset
"""
# Copyright: Travis Jefferies 2026

from typing import Any, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from metalog_jax.base import (
    MetalogBaseData,
    MetalogParameters,
)
from metalog_jax.base.enums import MetalogFitMethod
from metalog_jax.metalog import GridResult, Metalog, SPTMetalog, fit
from metalog_jax.regression import LassoParameters, RegularizedParameters
from metalog_jax.regression.lasso import fit_lasso
from metalog_jax.regression.ols import fit_ordinary_least_squares
from metalog_jax.utils import DEFAULT_Y, ks_distance

# Maximum number of terms supported for dynamic num_terms in vectorized operations.
# Pre-allocating to this size allows num_terms to be traced (non-static) for vmap.
MAX_TERMS = 30


@jax.jit
def _get_target_vmap(y: jnp.ndarray, num_terms: jnp.ndarray) -> jnp.ndarray:
    """Construct the design matrix with dynamic num_terms for vmap compatibility.

    Unlike MetalogBase._get_target which uses static_argnames for efficient
    single-fit JIT compilation, this version supports traced num_terms for
    vmap operations over different term counts.

    Always allocates MAX_TERMS columns and masks unused columns to zero.

    Args:
        y: Probability values, shape (m,).
        num_terms: Number of metalog terms (traced for vmap).

    Returns:
        Design matrix of shape (m, MAX_TERMS) with columns beyond num_terms
        masked to zero.
    """
    m = y.shape[0]
    delta = y - 0.5
    log_t = jnp.log(y / (1 - y))

    # Build all MAX_TERMS columns
    col0 = jnp.ones(m)
    col1 = log_t
    col2 = delta * log_t
    col3 = delta

    # Columns 4+ (terms 5 to MAX_TERMS)
    terms = jnp.arange(5, MAX_TERMS + 1)
    odd_mask = terms % 2 == 1
    even_mask = terms % 2 == 0
    odd_powers = terms // 2
    even_powers = (terms - 1) // 2

    delta_powers_odd = jnp.power(delta[:, None], odd_powers)
    delta_powers_even = jnp.power(delta[:, None], even_powers)

    odd_cols = jnp.where(odd_mask, delta_powers_odd, 0.0)
    even_cols = jnp.where(even_mask, log_t[:, None] * delta_powers_even, 0.0)
    additional_cols = odd_cols + even_cols

    # Stack all columns
    result = jnp.column_stack([col0, col1, col2, col3, additional_cols])

    # Mask unused columns
    col_indices = jnp.arange(MAX_TERMS)
    col_mask = (col_indices < num_terms).astype(result.dtype)

    return result * col_mask[None, :]


@jax.jit
def _get_quantiles_vmap(
    x: jnp.ndarray,
    boundedness_value: int,
    lower_bound: float,
    upper_bound: float,
) -> jnp.ndarray:
    """Transform quantiles based on boundedness for vmap compatibility.

    Unlike MetalogBase.get_quantiles which uses Python conditionals,
    this version uses jax.lax.switch for traced boundedness values.

    Args:
        x: Quantile values.
        boundedness_value: Integer value of MetalogBoundedness enum.
        lower_bound: Lower bound value.
        upper_bound: Upper bound value.

    Returns:
        Transformed quantiles.
    """

    def unbounded_transform(x_val):
        return x_val

    def lower_bound_transform(x_val):
        return jnp.log(x_val - lower_bound)

    def upper_bound_transform(x_val):
        return -jnp.log(upper_bound - x_val)

    def bounded_transform(x_val):
        return jnp.log((x_val - lower_bound) / (upper_bound - x_val))

    return jax.lax.switch(
        boundedness_value - 1,
        [
            unbounded_transform,
            lower_bound_transform,
            upper_bound_transform,
            bounded_transform,
        ],
        x,
    )


def _build_fit_single_config(
    params: MetalogParameters,
    max_terms_output: int,
    use_lasso: bool,
    learning_rate: float,
    num_iters: int,
    tol: float,
    momentum: float,
):
    """Build a fit function for a single configuration.

    Returns a function that fits a single (x, y, l1, n_terms) configuration.
    This is extracted to reduce cyclomatic complexity of fit_grid.
    """
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound
    boundedness_value = params.boundedness.value

    def fit_single_config(
        x_i: jnp.ndarray,
        y_i: jnp.ndarray,
        l1_penalty: jnp.ndarray,
        n_terms: jnp.ndarray,
    ) -> GridResult:
        """Fit a single configuration (traceable for vmap)."""
        quantiles = _get_quantiles_vmap(
            x_i, boundedness_value, lower_bound, upper_bound
        )
        target = _get_target_vmap(y=y_i, num_terms=n_terms)

        if use_lasso:
            lasso_p = LassoParameters(
                lam=l1_penalty,
                learning_rate=learning_rate,
                num_iters=num_iters,
                tol=tol,
                momentum=momentum,
            )
            model = fit_lasso(target, quantiles, lasso_p)
        else:
            model = fit_ordinary_least_squares(target, quantiles)

        coeff_indices = jnp.arange(MAX_TERMS)
        coeff_mask = (coeff_indices < n_terms).astype(model.weights.dtype)
        masked_weights = model.weights * coeff_mask
        padded_weights = masked_weights[:max_terms_output]

        delta_term = y_i - 0.5
        log_term = jnp.log(y_i / (1 - y_i))

        a_padded = jnp.zeros(MAX_TERMS)
        a_padded = a_padded.at[: masked_weights.shape[0]].set(masked_weights)

        ppf = a_padded[0] + (a_padded[1] * log_term)
        ppf = ppf + jnp.where(n_terms > 2, a_padded[2] * delta_term * log_term, 0.0)
        ppf = ppf + jnp.where(n_terms > 3, a_padded[3] * delta_term, 0.0)

        terms = jnp.arange(5, MAX_TERMS + 1)
        active_mask = terms <= n_terms
        odd_mask = terms % 2 == 1
        even_mask = terms % 2 == 0
        odd_powers = (terms - 1) // 2
        even_powers = terms // 2 - 1
        coeffs = a_padded[4:MAX_TERMS]

        delta_powers_odd = jnp.power(delta_term[:, None], odd_powers)
        delta_powers_even = jnp.power(delta_term[:, None], even_powers)

        odd_contrib = jnp.where(active_mask & odd_mask, coeffs * delta_powers_odd, 0.0)
        even_contrib = jnp.where(
            active_mask & even_mask, coeffs * delta_powers_even * log_term[:, None], 0.0
        )

        raw_ppf = ppf + jnp.sum(odd_contrib + even_contrib, axis=1)

        def unbounded_transform(ppf_val):
            return ppf_val

        def lower_bound_transform(ppf_val):
            return lower_bound + jnp.exp(ppf_val)

        def upper_bound_transform(ppf_val):
            return upper_bound - jnp.exp(-1 * ppf_val)

        def bounded_transform(ppf_val):
            return (lower_bound + upper_bound * jnp.exp(ppf_val)) / (
                1 + jnp.exp(ppf_val)
            )

        fitted_quantiles = jax.lax.switch(
            boundedness_value - 1,
            [
                unbounded_transform,
                lower_bound_transform,
                upper_bound_transform,
                bounded_transform,
            ],
            raw_ppf,
        )

        ks_dist = ks_distance(x_i, fitted_quantiles)
        result_params = params.replace(num_terms=max_terms_output)
        result_metalog = Metalog(metalog_params=result_params, a=padded_weights)

        return GridResult(metalog=result_metalog, ks_dist=ks_dist)

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


def fit_grid(
    x: jnp.ndarray,
    y: jnp.ndarray,
    params: MetalogParameters,
    *,
    num_terms: Optional[Sequence[int]] = None,
    l1_penalties: Optional[jnp.ndarray] = None,
    precomputed_quantiles: bool = False,
    lasso_params: Optional[LassoParameters] = None,
    learning_rate: float = 0.01,
    num_iters: int = 500,
    tol: float = 1e-6,
    momentum: float = 0.9,
) -> GridResult:
    """Unified grid search over any combination of axes.

    Automatically detects which axes to search based on inputs:
    - If x.ndim == 2: grid over datasets (batch dimension)
    - If num_terms provided: grid over num_terms values
    - If l1_penalties provided: grid over L1 penalties (uses Lasso)

    Output shape depends on active axes (ordered: datasets, l1_penalties, num_terms):
    - (): single dataset, no grids
    - (n_datasets,): only datasets batched
    - (n_penalties,): only l1_penalties grid
    - (n_terms,): only num_terms grid
    - (n_datasets, n_penalties): datasets x l1_penalties
    - (n_datasets, n_terms): datasets x num_terms
    - (n_penalties, n_terms): l1_penalties x num_terms
    - (n_datasets, n_penalties, n_terms): all three axes

    Args:
        x: Quantile values. Shape (n,) for single dataset or (batch, n) for batched.
        y: Probability levels. Shape (n,) for single dataset or (batch, n) for batched.
        params: MetalogParameters configuration. The num_terms field is ignored
            if num_terms argument is provided.
        num_terms: Optional sequence of term counts to search over.
        l1_penalties: Optional array of L1 penalties to search over (implies Lasso).
        precomputed_quantiles: Whether x values are precomputed quantiles.
        lasso_params: Optional fixed LassoParameters (used when l1_penalties is None
            but params.method is Lasso).
        learning_rate: Learning rate for Lasso. Default: 0.01.
        num_iters: Max iterations for Lasso. Default: 500.
        tol: Convergence tolerance for Lasso. Default: 1e-6.
        momentum: Momentum for Lasso optimizer. Default: 0.9.

    Returns:
        GridResult with metalog fits and KS distances. Shape depends on active axes.
    """
    # Detect which axes are active
    has_datasets = x.ndim == 2
    has_l1 = l1_penalties is not None
    has_num_terms = num_terms is not None

    # Prepare arrays
    num_terms_array = (
        jnp.array(num_terms) if has_num_terms else jnp.array([params.num_terms])
    )
    max_terms_output = int(num_terms_array.max()) if has_num_terms else params.num_terms
    l1_array = l1_penalties if has_l1 else jnp.array([0.0])

    # Determine if we should use Lasso
    use_lasso = has_l1 or params.method == MetalogFitMethod.Lasso

    # Build the fit function
    fit_fn = _build_fit_single_config(
        params, max_terms_output, use_lasso, learning_rate, num_iters, tol, momentum
    )

    # Apply the appropriate vmap structure
    return _apply_grid_vmaps(
        fit_fn, x, y, l1_array, num_terms_array, has_datasets, has_l1, has_num_terms
    )


def stack_leaves(*leaves):
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
            lambda x: x[i].item() if x[i].shape == () else jnp.array(x[i]), batched_out
        )
        for i in range(batch_size)
    ]


def pad_metalog_coeffs(
    metalog: Union[Metalog, SPTMetalog], max_terms: int
) -> Union[Metalog, SPTMetalog]:
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


def fit_grid_datasets(
    batched_x: jnp.ndarray,
    batched_y: jnp.ndarray,
    params: MetalogParameters,
    fit_params: Optional[RegularizedParameters] = None,
    precomputed_quantiles: bool = False,
) -> GridResult:
    """Fit multiple datasets with shared parameters using vmap.

    This function efficiently fits metalog distributions to multiple datasets
    in parallel using JAX's vmap. All datasets share the same MetalogParameters
    and optional regularization settings.

    Args:
        batched_x: Stacked quantile values, shape (n_datasets, n_samples).
        batched_y: Stacked probability levels, shape (n_datasets, n_samples).
        params: MetalogParameters shared across all fits.
        fit_params: Optional regularization parameters (e.g., LassoParameters).
        precomputed_quantiles: Whether x values are precomputed quantiles.

    Returns:
        GridResult with batched results:
            - metalog.a: Coefficients of shape (n_datasets, num_terms)
            - ks_dist: KS distances of shape (n_datasets,)
    """

    def fit_single_dataset(x: jnp.ndarray, y: jnp.ndarray) -> GridResult:
        data = MetalogBaseData(x=x, y=y, precomputed_quantiles=precomputed_quantiles)
        metalog = fit(data, params, fit_params)
        # Use DEFAULT_Y to avoid traced validation in ppf
        fitted_quantiles = metalog.ppf(DEFAULT_Y)
        ks_dist = ks_distance(x, fitted_quantiles)
        return GridResult(metalog=metalog, ks_dist=ks_dist)

    vmapped_fit = jax.vmap(fit_single_dataset, in_axes=(0, 0))
    return vmapped_fit(batched_x, batched_y)


def fit_grid_hyperparams(
    data: MetalogBaseData,
    params: MetalogParameters,
    l1_penalties: jnp.ndarray,
    learning_rate: float = 0.01,
    num_iters: int = 500,
    tol: float = 1e-6,
    momentum: float = 0.9,
) -> GridResult:
    """Fit a single dataset with a grid of L1 penalty values (Lasso).

    This function performs a 1D grid search over L1 regularization penalties
    for a single dataset using JAX's vmap for efficient parallel computation.

    Args:
        data: MetalogBaseData instance containing the dataset to fit.
        params: MetalogParameters configuration. The method should be Lasso.
        l1_penalties: Array of L1 penalty values to test, shape (n_penalties,).
        learning_rate: Learning rate for Lasso gradient descent. Default: 0.01.
        num_iters: Maximum iterations for Lasso. Default: 500.
        tol: Convergence tolerance for Lasso. Default: 1e-6.
        momentum: Momentum for Lasso optimizer. Default: 0.9.

    Returns:
        GridResult with batched results:
            - metalog.a: Coefficients of shape (n_penalties, num_terms)
            - ks_dist: KS distances of shape (n_penalties,)
    """

    def fit_with_penalty(l1_penalty: jnp.ndarray) -> GridResult:
        lasso_params = LassoParameters(
            lam=l1_penalty,
            learning_rate=learning_rate,
            num_iters=num_iters,
            tol=tol,
            momentum=momentum,
        )
        metalog = fit(data, params, lasso_params)
        fitted_quantiles = metalog.ppf(data.y)
        ks_dist = ks_distance(data.x, fitted_quantiles)
        return GridResult(metalog=metalog, ks_dist=ks_dist)

    vmapped_fit = jax.vmap(fit_with_penalty)
    return vmapped_fit(l1_penalties)


def fit_grid_datasets_hyperparams(
    batched_x: jnp.ndarray,
    batched_y: jnp.ndarray,
    params: MetalogParameters,
    l1_penalties: jnp.ndarray,
    precomputed_quantiles: bool = False,
    learning_rate: float = 0.01,
    num_iters: int = 500,
    tol: float = 1e-6,
    momentum: float = 0.9,
) -> GridResult:
    """Fit multiple datasets with a grid of L1 penalty values (Lasso).

    This function performs a 2D grid search over datasets x L1 regularization
    penalties using nested vmap for efficient parallel computation.

    Args:
        batched_x: Stacked quantile values, shape (n_datasets, n_samples).
        batched_y: Stacked probability levels, shape (n_datasets, n_samples).
        params: MetalogParameters configuration. The method should be Lasso.
        l1_penalties: Array of L1 penalty values to test, shape (n_penalties,).
        precomputed_quantiles: Whether x values are precomputed quantiles.
        learning_rate: Learning rate for Lasso gradient descent. Default: 0.01.
        num_iters: Maximum iterations for Lasso. Default: 500.
        tol: Convergence tolerance for Lasso. Default: 1e-6.
        momentum: Momentum for Lasso optimizer. Default: 0.9.

    Returns:
        GridResult with batched results:
            - metalog.a: Coefficients of shape (n_datasets, n_penalties, num_terms)
            - ks_dist: KS distances of shape (n_datasets, n_penalties)
    """

    def fit_dataset_with_penalty(
        x: jnp.ndarray, y: jnp.ndarray, l1_penalty: jnp.ndarray
    ) -> GridResult:
        lasso_params = LassoParameters(
            lam=l1_penalty,
            learning_rate=learning_rate,
            num_iters=num_iters,
            tol=tol,
            momentum=momentum,
        )
        data = MetalogBaseData(x=x, y=y, precomputed_quantiles=precomputed_quantiles)
        metalog = fit(data, params, lasso_params)
        # Use DEFAULT_Y to avoid traced validation in ppf
        fitted_quantiles = metalog.ppf(DEFAULT_Y)
        ks_dist = ks_distance(x, fitted_quantiles)
        return GridResult(metalog=metalog, ks_dist=ks_dist)

    # First vmap over penalties
    vmapped_over_penalties = jax.vmap(fit_dataset_with_penalty, in_axes=(None, None, 0))

    # Then vmap over datasets
    vmapped_over_both = jax.vmap(
        lambda x, y: vmapped_over_penalties(x, y, l1_penalties),
        in_axes=(0, 0),
    )

    return vmapped_over_both(batched_x, batched_y)


def _pad_coeffs_for_stacking(
    metalog: Union[Metalog, SPTMetalog], max_terms: int, ndim: int
) -> Union[Metalog, SPTMetalog]:
    """Pad metalog coefficients for stacking across term counts.

    Internal helper that handles padding for different batch dimensions.

    Args:
        metalog: Metalog with batched coefficients.
        max_terms: Target length for coefficient arrays.
        ndim: Number of batch dimensions (1 for single dataset, 2 for dataset x penalty).

    Returns:
        Metalog with padded coefficients.
    """
    current_terms = metalog.a.shape[-1]
    if current_terms == max_terms:
        return metalog
    pad_width = max_terms - current_terms
    if ndim == 1:
        padded_a = jnp.pad(metalog.a, ((0, 0), (0, pad_width)), mode="constant")
    elif ndim == 2:
        padded_a = jnp.pad(metalog.a, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    return metalog.replace(a=padded_a)


def _get_fit_function_for_method(
    method: MetalogFitMethod, regression_params: Any = None
):
    """Get the appropriate fit function for a regression method.

    Args:
        method: The MetalogFitMethod enum value.
        regression_params: Optional regression parameters (for Lasso).

    Returns:
        A function that takes (target, quantiles) and returns a regression model.
    """
    if method == MetalogFitMethod.OLS:
        return fit_ordinary_least_squares
    elif method == MetalogFitMethod.Lasso:
        from metalog_jax.regression.lasso import DEFAULT_LASSO_PARAMETERS

        params = regression_params or DEFAULT_LASSO_PARAMETERS
        return lambda target, quantiles: fit_lasso(target, quantiles, params)
    else:
        raise ValueError(f"Unsupported method: {method}")


def fit_grid_num_terms(
    data: MetalogBaseData,
    params: MetalogParameters,
    num_terms_list: list[int],
    regression_params: Any = None,
) -> GridResult:
    """Fit a single dataset with different num_terms values.

    This function performs a 1D grid search over term counts for a single dataset
    using JAX's vmap for efficient parallel computation. Supports OLS and Lasso
    regression methods based on the params.method setting.

    Vectorization strategy:
        - num_terms dimension: Fully vectorized via vmap
        - Uses MAX_TERMS-sized arrays with masking for uniform shapes

    Args:
        data: MetalogBaseData instance containing the dataset to fit.
        params: MetalogParameters configuration. The num_terms field is ignored
            and replaced by values from num_terms_list.
        num_terms_list: List of term counts to test.
        regression_params: Optional regression hyperparameters (e.g., LassoParameters).
            If None, uses defaults for the method.

    Returns:
        GridResult with batched results:
            - metalog.a: Coefficients of shape (n_terms, max_terms)
            - ks_dist: KS distances of shape (n_terms,)

        Where max_terms = max(num_terms_list) and coefficients for smaller term
        counts are zero-padded.
    """
    max_terms_output = max(num_terms_list)
    num_terms_array = jnp.array(num_terms_list)

    # Extract static parameters
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound
    boundedness_value = params.boundedness.value

    # Get fit function based on method
    fit_fn = _get_fit_function_for_method(params.method, regression_params)

    # Get quantiles using vmap-compatible function
    quantiles = _get_quantiles_vmap(data.x, boundedness_value, lower_bound, upper_bound)

    def fit_single_num_terms(num_terms: jnp.ndarray) -> GridResult:
        """Fit with a specific num_terms value (traceable)."""
        # Build design matrix with dynamic num_terms using vmap-compatible function
        target = _get_target_vmap(y=data.y, num_terms=num_terms)

        # Fit regression
        model = fit_fn(target, quantiles)

        # Create coefficient mask to zero out unused terms
        coeff_indices = jnp.arange(MAX_TERMS)
        coeff_mask = (coeff_indices < num_terms).astype(model.weights.dtype)
        masked_weights = model.weights * coeff_mask

        # Pad to max_terms_output for consistent output shape
        padded_weights = masked_weights[:max_terms_output]

        # Compute PPF for KS distance
        delta_term = data.y - 0.5
        log_term = jnp.log(data.y / (1 - data.y))

        a_padded = jnp.zeros(MAX_TERMS)
        a_padded = a_padded.at[: masked_weights.shape[0]].set(masked_weights)

        ppf = a_padded[0] + (a_padded[1] * log_term)
        ppf = ppf + jnp.where(num_terms > 2, a_padded[2] * delta_term * log_term, 0.0)
        ppf = ppf + jnp.where(num_terms > 3, a_padded[3] * delta_term, 0.0)

        terms = jnp.arange(5, MAX_TERMS + 1)
        active_mask = terms <= num_terms
        odd_mask = terms % 2 == 1
        even_mask = terms % 2 == 0
        odd_powers = (terms - 1) // 2
        even_powers = terms // 2 - 1
        coeffs = a_padded[4:MAX_TERMS]

        delta_powers_odd = jnp.power(delta_term[:, None], odd_powers)
        delta_powers_even = jnp.power(delta_term[:, None], even_powers)

        odd_contrib = jnp.where(active_mask & odd_mask, coeffs * delta_powers_odd, 0.0)
        even_contrib = jnp.where(
            active_mask & even_mask, coeffs * delta_powers_even * log_term[:, None], 0.0
        )

        raw_ppf = ppf + jnp.sum(odd_contrib + even_contrib, axis=1)

        # Apply boundedness transformation
        def unbounded_transform(ppf_val):
            return ppf_val

        def lower_bound_transform(ppf_val):
            return lower_bound + jnp.exp(ppf_val)

        def upper_bound_transform(ppf_val):
            return upper_bound - jnp.exp(-1 * ppf_val)

        def bounded_transform(ppf_val):
            return (lower_bound + upper_bound * jnp.exp(ppf_val)) / (
                1 + jnp.exp(ppf_val)
            )

        fitted_quantiles = jax.lax.switch(
            boundedness_value - 1,
            [
                unbounded_transform,
                lower_bound_transform,
                upper_bound_transform,
                bounded_transform,
            ],
            raw_ppf,
        )

        ks_dist = ks_distance(data.x, fitted_quantiles)

        result_params = params.replace(num_terms=max_terms_output)
        result_metalog = Metalog(metalog_params=result_params, a=padded_weights)

        return GridResult(metalog=result_metalog, ks_dist=ks_dist)

    # vmap over num_terms
    vmapped_fit = jax.vmap(fit_single_num_terms)
    return vmapped_fit(num_terms_array)


def fit_grid_datasets_num_terms(
    batched_x: jnp.ndarray,
    batched_y: jnp.ndarray,
    params: MetalogParameters,
    num_terms_list: list[int],
    precomputed_quantiles: bool = False,
    regression_params: Any = None,
) -> GridResult:
    """Fit multiple datasets with different num_terms values.

    This function performs a 2D grid search over datasets x term counts
    using nested vmap for efficient parallel computation. Supports OLS and Lasso
    regression methods based on the params.method setting.

    Vectorization strategy:
        - datasets dimension: Fully vectorized via vmap
        - num_terms dimension: Fully vectorized via vmap
        - Uses MAX_TERMS-sized arrays with masking for uniform shapes

    Args:
        batched_x: Stacked quantile values, shape (n_datasets, n_samples).
        batched_y: Stacked probability levels, shape (n_datasets, n_samples).
        params: MetalogParameters configuration. The num_terms field is ignored
            and replaced by values from num_terms_list.
        num_terms_list: List of term counts to test.
        precomputed_quantiles: Whether x values are precomputed quantiles.
        regression_params: Optional regression hyperparameters (e.g., LassoParameters).
            If None, uses defaults for the method.

    Returns:
        GridResult with batched results:
            - metalog.a: Coefficients of shape (n_datasets, n_terms, max_terms)
            - ks_dist: KS distances of shape (n_datasets, n_terms)

        Where max_terms = max(num_terms_list) and coefficients for smaller term
        counts are zero-padded.
    """
    max_terms_output = max(num_terms_list)
    num_terms_array = jnp.array(num_terms_list)

    # Extract static parameters
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound
    boundedness_value = params.boundedness.value

    # Get fit function based on method
    fit_fn = _get_fit_function_for_method(params.method, regression_params)

    def fit_single_config(
        x: jnp.ndarray, y: jnp.ndarray, num_terms: jnp.ndarray
    ) -> GridResult:
        """Fit a single dataset with a specific num_terms (traceable)."""
        # Get quantiles using vmap-compatible function
        quantiles = _get_quantiles_vmap(x, boundedness_value, lower_bound, upper_bound)

        # Build design matrix with dynamic num_terms using vmap-compatible function
        target = _get_target_vmap(y=y, num_terms=num_terms)

        # Fit regression
        model = fit_fn(target, quantiles)

        # Create coefficient mask to zero out unused terms
        coeff_indices = jnp.arange(MAX_TERMS)
        coeff_mask = (coeff_indices < num_terms).astype(model.weights.dtype)
        masked_weights = model.weights * coeff_mask

        # Pad to max_terms_output for consistent output shape
        padded_weights = masked_weights[:max_terms_output]

        # Compute PPF for KS distance
        delta_term = y - 0.5
        log_term = jnp.log(y / (1 - y))

        a_padded = jnp.zeros(MAX_TERMS)
        a_padded = a_padded.at[: masked_weights.shape[0]].set(masked_weights)

        ppf = a_padded[0] + (a_padded[1] * log_term)
        ppf = ppf + jnp.where(num_terms > 2, a_padded[2] * delta_term * log_term, 0.0)
        ppf = ppf + jnp.where(num_terms > 3, a_padded[3] * delta_term, 0.0)

        terms = jnp.arange(5, MAX_TERMS + 1)
        active_mask = terms <= num_terms
        odd_mask = terms % 2 == 1
        even_mask = terms % 2 == 0
        odd_powers = (terms - 1) // 2
        even_powers = terms // 2 - 1
        coeffs = a_padded[4:MAX_TERMS]

        delta_powers_odd = jnp.power(delta_term[:, None], odd_powers)
        delta_powers_even = jnp.power(delta_term[:, None], even_powers)

        odd_contrib = jnp.where(active_mask & odd_mask, coeffs * delta_powers_odd, 0.0)
        even_contrib = jnp.where(
            active_mask & even_mask, coeffs * delta_powers_even * log_term[:, None], 0.0
        )

        raw_ppf = ppf + jnp.sum(odd_contrib + even_contrib, axis=1)

        # Apply boundedness transformation
        def unbounded_transform(ppf_val):
            return ppf_val

        def lower_bound_transform(ppf_val):
            return lower_bound + jnp.exp(ppf_val)

        def upper_bound_transform(ppf_val):
            return upper_bound - jnp.exp(-1 * ppf_val)

        def bounded_transform(ppf_val):
            return (lower_bound + upper_bound * jnp.exp(ppf_val)) / (
                1 + jnp.exp(ppf_val)
            )

        fitted_quantiles = jax.lax.switch(
            boundedness_value - 1,
            [
                unbounded_transform,
                lower_bound_transform,
                upper_bound_transform,
                bounded_transform,
            ],
            raw_ppf,
        )

        ks_dist = ks_distance(x, fitted_quantiles)

        result_params = params.replace(num_terms=max_terms_output)
        result_metalog = Metalog(metalog_params=result_params, a=padded_weights)

        return GridResult(metalog=result_metalog, ks_dist=ks_dist)

    # Build nested vmap: datasets x num_terms
    # vmap over num_terms (innermost)
    vmapped_over_terms = jax.vmap(fit_single_config, in_axes=(None, None, 0))
    # vmap over datasets (outermost)
    vmapped_over_all = jax.vmap(
        lambda x, y: vmapped_over_terms(x, y, num_terms_array), in_axes=(0, 0)
    )

    return vmapped_over_all(batched_x, batched_y)


def fit_grid_full(
    batched_x: jnp.ndarray,
    batched_y: jnp.ndarray,
    params: MetalogParameters,
    l1_penalties: jnp.ndarray,
    num_terms_list: list[int],
    precomputed_quantiles: bool = False,
    learning_rate: float = 0.01,
    num_iters: int = 500,
    tol: float = 1e-6,
    momentum: float = 0.9,
) -> GridResult:
    """Fit multiple datasets with grids of L1 penalties and term counts (Lasso).

    This function performs a 3D grid search over datasets x L1 penalties x num_terms.

    Vectorization strategy:
        - All three dimensions (datasets, penalties, num_terms) are fully vectorized
          using nested jax.vmap calls
        - Uses MAX_TERMS-sized arrays with masking to enable vmap over num_terms
        - No Python for loops in the computation path

    The implementation uses masked arrays throughout to handle varying num_terms:
        - Design matrix is always MAX_TERMS columns with unused columns masked to 0
        - Coefficients are always MAX_TERMS with unused terms masked to 0
        - PPF computation uses active_mask to only use valid terms

    Args:
        batched_x: Stacked quantile values, shape (n_datasets, n_samples).
        batched_y: Stacked probability levels, shape (n_datasets, n_samples).
        params: Base MetalogParameters configuration. The num_terms field is ignored
            and replaced by values from num_terms_list.
        l1_penalties: Array of L1 penalty values to test, shape (n_penalties,).
        num_terms_list: List of term counts to test.
        precomputed_quantiles: Whether x values are precomputed quantiles.
        learning_rate: Learning rate for Lasso gradient descent. Default: 0.01.
        num_iters: Maximum iterations for Lasso. Default: 500.
        tol: Convergence tolerance for Lasso. Default: 1e-6.
        momentum: Momentum for Lasso optimizer. Default: 0.9.

    Returns:
        GridResult with batched results:
            - metalog.a: Coefficients of shape (n_datasets, n_penalties, n_terms, max_terms)
            - ks_dist: KS distances of shape (n_datasets, n_penalties, n_terms)

        Where max_terms = max(num_terms_list) and coefficients for smaller term
        counts are zero-padded.
    """
    max_terms_output = max(num_terms_list)
    num_terms_array = jnp.array(num_terms_list)

    # Extract static parameters from params for use in the vmapped function
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound
    boundedness_value = params.boundedness.value

    def fit_single_config(
        x: jnp.ndarray, y: jnp.ndarray, l1_penalty: jnp.ndarray, num_terms: jnp.ndarray
    ) -> GridResult:
        """Fit a single configuration with traced num_terms.

        This function is designed to be vmapped over all dimensions.
        Uses MAX_TERMS-sized arrays with masking for uniform shape.
        """
        # Get quantiles using vmap-compatible function
        quantiles = _get_quantiles_vmap(x, boundedness_value, lower_bound, upper_bound)

        # Build design matrix with dynamic num_terms using vmap-compatible function
        target = _get_target_vmap(y=y, num_terms=num_terms)

        # Fit using Lasso regression with MAX_TERMS-sized design matrix
        lasso_params = LassoParameters(
            lam=l1_penalty,
            learning_rate=learning_rate,
            num_iters=num_iters,
            tol=tol,
            momentum=momentum,
        )
        model = fit_lasso(target, quantiles, lasso_params)

        # Create coefficient mask to zero out unused terms
        coeff_indices = jnp.arange(MAX_TERMS)
        coeff_mask = (coeff_indices < num_terms).astype(model.weights.dtype)
        masked_weights = model.weights * coeff_mask

        # Pad to max_terms_output for consistent output shape
        padded_weights = masked_weights[:max_terms_output]

        # Use base params for the metalog - the actual num_terms is encoded
        # in the coefficient values (zeros beyond num_terms)
        # We use max_terms_output for the metalog so shapes are consistent
        result_params = params.replace(num_terms=max_terms_output)

        # Compute PPF for KS distance using the internal _ppf method
        # Build a temporary metalog-like structure with the padded weights
        delta_term = DEFAULT_Y - 0.5
        log_term = jnp.log(DEFAULT_Y / (1 - DEFAULT_Y))

        # Compute raw PPF using the same logic as _ppf but with traced num_terms
        a_padded = jnp.zeros(MAX_TERMS)
        a_padded = a_padded.at[: masked_weights.shape[0]].set(masked_weights)

        ppf = a_padded[0] + (a_padded[1] * log_term)
        ppf = ppf + jnp.where(num_terms > 2, a_padded[2] * delta_term * log_term, 0.0)
        ppf = ppf + jnp.where(num_terms > 3, a_padded[3] * delta_term, 0.0)

        terms = jnp.arange(5, MAX_TERMS + 1)
        active_mask = terms <= num_terms
        odd_mask = terms % 2 == 1
        even_mask = terms % 2 == 0
        odd_powers = (terms - 1) // 2
        even_powers = terms // 2 - 1
        coeffs = a_padded[4:MAX_TERMS]

        delta_powers_odd = jnp.power(delta_term[:, None], odd_powers)
        delta_powers_even = jnp.power(delta_term[:, None], even_powers)

        odd_contrib = jnp.where(active_mask & odd_mask, coeffs * delta_powers_odd, 0.0)
        even_contrib = jnp.where(
            active_mask & even_mask, coeffs * delta_powers_even * log_term[:, None], 0.0
        )

        raw_ppf = ppf + jnp.sum(odd_contrib + even_contrib, axis=1)

        # Apply boundedness transformation using lax.switch
        def unbounded_transform(ppf_val):
            return ppf_val

        def lower_bound_transform(ppf_val):
            return lower_bound + jnp.exp(ppf_val)

        def upper_bound_transform(ppf_val):
            return upper_bound - jnp.exp(-1 * ppf_val)

        def bounded_transform(ppf_val):
            return (lower_bound + upper_bound * jnp.exp(ppf_val)) / (
                1 + jnp.exp(ppf_val)
            )

        # Subtract 1 because enum values start at 1 but lax.switch uses 0-indexing
        fitted_quantiles = jax.lax.switch(
            boundedness_value - 1,
            [
                unbounded_transform,
                lower_bound_transform,
                upper_bound_transform,
                bounded_transform,
            ],
            raw_ppf,
        )

        ks_dist = ks_distance(x, fitted_quantiles)

        # Create a Metalog with the padded weights for the result
        result_metalog = Metalog(metalog_params=result_params, a=padded_weights)

        return GridResult(metalog=result_metalog, ks_dist=ks_dist)

    # Build nested vmap: datasets x penalties x num_terms
    # vmap over num_terms (innermost)
    vmapped_over_terms = jax.vmap(fit_single_config, in_axes=(None, None, None, 0))
    # vmap over penalties
    vmapped_over_penalties = jax.vmap(
        lambda x, y, l1: vmapped_over_terms(x, y, l1, num_terms_array),
        in_axes=(None, None, 0),
    )
    # vmap over datasets (outermost)
    vmapped_over_all = jax.vmap(
        lambda x, y: vmapped_over_penalties(x, y, l1_penalties), in_axes=(0, 0)
    )

    # Execute the fully vmapped function
    # Result shape: (n_datasets, n_penalties, n_terms, ...)
    return vmapped_over_all(batched_x, batched_y)


def find_best_config(ks_dists: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the best configuration (minimum KS distance) in a grid.

    Works with grids of any dimensionality (1D, 2D, 3D, etc.).

    Args:
        ks_dists: Array of KS distances with shape (d1, d2, ..., dn).

    Returns:
        Tuple of (best_indices, best_ks) where:
            - best_indices: Indices of the minimum, shape (n,) for n dimensions
            - best_ks: Minimum KS distance value (scalar)
    """
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

    Finds the configuration with minimum KS distance for the given dataset
    and extracts the corresponding metalog and KS distance.

    Args:
        grid_results: GridResult from fit_grid_datasets_hyperparams or fit_grid_full.
            - For 2D grid: ks_dist shape (n_datasets, n_penalties)
            - For 3D grid: ks_dist shape (n_datasets, n_penalties, n_terms)
        dataset_idx: Index of the dataset to extract best result for.

    Returns:
        GridResult with:
            - metalog: Best metalog for this dataset
            - ks_dist: Corresponding KS distance (scalar)
    """
    # Get KS distances for this dataset
    dataset_ks = grid_results.ks_dist[dataset_idx]

    # Find best configuration within this dataset's grid
    flat_idx = jnp.argmin(dataset_ks)
    best_idx = jnp.unravel_index(flat_idx, dataset_ks.shape)

    # Extract the best metalog
    # Build full index: (dataset_idx, *best_idx)
    full_idx = (dataset_idx,) + best_idx
    best_metalog = jax.tree.map(
        lambda x: x[full_idx],
        grid_results.metalog,
    )
    best_ks = grid_results.ks_dist[full_idx]
    return GridResult(metalog=best_metalog, ks_dist=best_ks)


def extract_metalog(grid_result: GridResult, *indices: int) -> Metalog:
    """Extract a usable Metalog from grid results at given indices.

    After grid search, metalog fields become JAX arrays due to vmap stacking.
    This function extracts a single metalog at the specified indices and converts
    all scalar fields back to Python types, making the metalog compatible with
    JIT-compiled methods like ppf, pdf, cdf, etc.

    Args:
        grid_result: GridResult from fit_grid or related functions.
        *indices: Integer indices to extract. The number of indices should match
            the dimensionality of the grid:
            - 1D grid (e.g., L1 penalties): single index like `extract_metalog(result, 0)`
            - 2D grid (e.g., L1 x num_terms): two indices like `extract_metalog(result, 0, 1)`
            - 3D grid (e.g., datasets x L1 x num_terms): three indices

    Returns:
        Metalog instance with Python-typed fields, ready to use with ppf, pdf,
        cdf, rvs, and other distribution methods.

    Examples:
        Extract best metalog from a 1D grid search over L1 penalties:

            >>> result = fit_grid(data.x, data.y, params, l1_penalties=l1_vals)
            >>> best_idx, best_ks = find_best_config(result.ks_dist)
            >>> best_metalog = extract_metalog(result, int(best_idx))
            >>> median = best_metalog.ppf(jnp.array([0.5]))

        Extract best metalog from a 2D grid search:

            >>> result = fit_grid(data.x, data.y, params,
            ...                   l1_penalties=l1_vals, num_terms=[5, 7, 9])
            >>> best_idx, best_ks = find_best_config(result.ks_dist)
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
    )
    return Metalog(metalog_params=params, a=extracted.a)
