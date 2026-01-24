"""Grid search demonstration."""

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Unified Grid Search with `fit_grid`

    This notebook demonstrates the unified `fit_grid` function which provides a single
    interface for all grid search combinations. The function automatically detects
    which axes to search based on the inputs provided.

    ## Grid Axes

    The `fit_grid` function can search over three axes:

    1. **Datasets**: Multiple datasets to fit in parallel (detected by x.ndim == 2)
    2. **L1 Penalties**: Regularization strength for Lasso regression
    3. **Num Terms**: Number of terms in the metalog distribution

    This gives us 8 possible combinations (2³), all handled by a single function.
    """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    from scipy.stats import beta, gamma, lognorm, norm, weibull_min

    from metalog_jax.base import (
        MetalogBoundedness,
        MetalogFitMethod,
        MetalogInputData,
        MetalogParameters,
    )
    from metalog_jax.grid_search import find_best_config, fit_grid
    from metalog_jax.utils import DEFAULT_Y

    return (
        DEFAULT_Y,
        MetalogBoundedness,
        MetalogFitMethod,
        MetalogInputData,
        MetalogParameters,
        beta,
        find_best_config,
        fit_grid,
        gamma,
        jnp,
        lognorm,
        norm,
        weibull_min,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 1: Single Dataset, No Grid

    The simplest case: fit a single dataset with fixed parameters.
    Returns scalar KS distance and 1D coefficient array.
    """
    )
    return


@app.cell
def _(DEFAULT_Y, MetalogInputData, beta):
    # Generate sample data from a beta distribution
    dist_beta = beta(a=2, b=5)
    samples_beta = dist_beta.rvs(size=200, random_state=42)

    # Create input data
    data_single = MetalogInputData.from_values(samples_beta, DEFAULT_Y, False)
    return data_single, dist_beta, samples_beta


@app.cell
def _(MetalogBoundedness, MetalogFitMethod, MetalogParameters, data_single, fit_grid):
    # Configure parameters
    params_ols = MetalogParameters(
        boundedness=MetalogBoundedness.BOUNDED,
        lower_bound=0,
        upper_bound=1,
        method=MetalogFitMethod.OLS,
        num_terms=7,
    )

    # Fit single dataset - returns scalar results
    result_single = fit_grid(data_single.x, data_single.y, params_ols)

    print(f"KS Distance: {float(result_single.ks_dist):.4f}")
    print(f"Coefficients shape: {result_single.metalog.a.shape}")
    print(f"Coefficients: {result_single.metalog.a}")
    return params_ols, result_single


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 2: Single Dataset, Num Terms Grid

    Search over different numbers of terms to find optimal complexity.
    Returns 1D array of results indexed by num_terms.
    """
    )
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    data_single,
    find_best_config,
    fit_grid,
):
    # Configure base parameters
    params_num_terms = MetalogParameters(
        boundedness=MetalogBoundedness.BOUNDED,
        lower_bound=0,
        upper_bound=1,
        method=MetalogFitMethod.OLS,
        num_terms=3,  # Will be overridden
    )

    # Grid over num_terms
    num_terms_grid = [3, 5, 7, 9, 11]

    result_num_terms = fit_grid(
        data_single.x, data_single.y, params_num_terms, num_terms=num_terms_grid
    )

    print(f"KS Distances shape: {result_num_terms.ks_dist.shape}")
    print(f"Coefficients shape: {result_num_terms.metalog.a.shape}")
    print()
    print("Results by num_terms:")
    for _i, _nt in enumerate(num_terms_grid):
        print(f"  {_nt} terms: KS = {float(result_num_terms.ks_dist[_i]):.4f}")

    # Find best configuration
    best_idx_nt, best_ks_nt = find_best_config(result_num_terms.ks_dist)
    print(
        f"\nBest: {num_terms_grid[int(best_idx_nt)]} terms (KS = {float(best_ks_nt):.4f})"
    )
    return best_idx_nt, best_ks_nt, num_terms_grid, params_num_terms, result_num_terms


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 3: Single Dataset, L1 Penalties Grid

    Search over L1 regularization strengths using Lasso regression.
    Higher penalties produce sparser coefficients, reducing overfitting.
    """
    )
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    data_single,
    find_best_config,
    fit_grid,
    jnp,
):
    # Configure for Lasso
    params_lasso = MetalogParameters(
        boundedness=MetalogBoundedness.BOUNDED,
        lower_bound=0,
        upper_bound=1,
        method=MetalogFitMethod.Lasso,
        num_terms=9,
    )

    # Grid over L1 penalties
    l1_grid = jnp.array([0.0, 0.001, 0.01, 0.1, 1.0])

    result_l1 = fit_grid(
        data_single.x, data_single.y, params_lasso, l1_penalties=l1_grid
    )

    print(f"KS Distances shape: {result_l1.ks_dist.shape}")
    print(f"Coefficients shape: {result_l1.metalog.a.shape}")
    print()
    print("Results by L1 penalty:")
    for _i, _l1 in enumerate(l1_grid):
        coeff_norm = float(jnp.linalg.norm(result_l1.metalog.a[_i]))
        print(
            f"  L1={float(_l1):5.3f}: KS = {float(result_l1.ks_dist[_i]):.4f}, ||a|| = {coeff_norm:.4f}"
        )

    best_l1_idx, best_l1_ks = find_best_config(result_l1.ks_dist)
    print(
        f"\nBest: L1={float(l1_grid[int(best_l1_idx)]):.3f} (KS = {float(best_l1_ks):.4f})"
    )
    return best_l1_idx, best_l1_ks, l1_grid, params_lasso, result_l1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 4: Single Dataset, Both Grids (2D)

    Search over both L1 penalties and num_terms simultaneously.
    Returns a 2D grid of results: (n_penalties, n_terms).
    """
    )
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    data_single,
    find_best_config,
    fit_grid,
    jnp,
):
    # Configure for 2D search
    params_2d = MetalogParameters(
        boundedness=MetalogBoundedness.BOUNDED,
        lower_bound=0,
        upper_bound=1,
        method=MetalogFitMethod.Lasso,
        num_terms=3,
    )

    # Define both grids
    l1_grid_2d = jnp.array([0.0, 0.01, 0.1])
    num_terms_2d = [5, 7, 9]

    result_2d = fit_grid(
        data_single.x,
        data_single.y,
        params_2d,
        l1_penalties=l1_grid_2d,
        num_terms=num_terms_2d,
    )

    print(f"KS Distances shape: {result_2d.ks_dist.shape}")
    print(f"Coefficients shape: {result_2d.metalog.a.shape}")
    print()
    print("2D Grid Results (L1 penalty x num_terms):")
    print("         ", end="")
    for _nt in num_terms_2d:
        print(f"{_nt:>8} terms", end="")
    print()
    for _i, _l1 in enumerate(l1_grid_2d):
        print(f"L1={float(_l1):5.3f}", end="")
        for _j in range(len(num_terms_2d)):
            print(f"     {float(result_2d.ks_dist[_i, _j]):.4f}", end="")
        print()

    # Find best in 2D grid
    best_2d_idx, best_2d_ks = find_best_config(result_2d.ks_dist)
    best_l1_2d = l1_grid_2d[best_2d_idx[0]]
    best_nt_2d = num_terms_2d[best_2d_idx[1]]
    print(
        f"\nBest: L1={float(best_l1_2d):.3f}, {best_nt_2d} terms (KS = {float(best_2d_ks):.4f})"
    )
    return (
        best_2d_idx,
        best_2d_ks,
        best_l1_2d,
        best_nt_2d,
        l1_grid_2d,
        num_terms_2d,
        params_2d,
        result_2d,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 5: Batched Datasets, No Grid

    Fit multiple datasets in parallel with shared parameters.
    Useful when you have many distributions with similar characteristics.
    """
    )
    return


@app.cell
def _(DEFAULT_Y, MetalogInputData, gamma, jnp, lognorm, weibull_min):
    # Generate multiple distributions (all lower-bounded at 0)
    dist_lognorm = lognorm(s=0.5, loc=0, scale=1).rvs(size=200, random_state=42)
    dist_weibull = weibull_min(c=2, scale=2).rvs(size=200, random_state=43)
    dist_gamma = gamma(a=4, scale=1).rvs(size=200, random_state=44)

    # Create input data for each
    data1 = MetalogInputData.from_values(dist_lognorm, DEFAULT_Y, False)
    data2 = MetalogInputData.from_values(dist_weibull, DEFAULT_Y, False)
    data3 = MetalogInputData.from_values(dist_gamma, DEFAULT_Y, False)

    # Stack into batched arrays
    batched_x = jnp.stack([data1.x, data2.x, data3.x])
    batched_y = jnp.stack([data1.y, data2.y, data3.y])

    print(f"Batched x shape: {batched_x.shape}")
    print(f"Batched y shape: {batched_y.shape}")
    return (
        batched_x,
        batched_y,
        data1,
        data2,
        data3,
        dist_gamma,
        dist_lognorm,
        dist_weibull,
    )


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    batched_x,
    batched_y,
    fit_grid,
):
    # Configure shared parameters
    params_batch = MetalogParameters(
        boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
        lower_bound=0,
        upper_bound=0,
        method=MetalogFitMethod.OLS,
        num_terms=7,
    )

    # Fit all datasets in parallel
    result_batch = fit_grid(batched_x, batched_y, params_batch)

    print(f"KS Distances shape: {result_batch.ks_dist.shape}")
    print(f"Coefficients shape: {result_batch.metalog.a.shape}")
    print()
    dist_names = ["Lognormal", "Weibull", "Gamma"]
    print("Results per dataset:")
    for _i, _name in enumerate(dist_names):
        print(f"  {_name}: KS = {float(result_batch.ks_dist[_i]):.4f}")
    return dist_names, params_batch, result_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 6: Batched Datasets, Num Terms Grid (2D)

    Search over num_terms for each dataset in a batch.
    Returns shape (n_datasets, n_terms).
    """
    )
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    batched_x,
    batched_y,
    dist_names,
    find_best_config,
    fit_grid,
):
    params_batch_nt = MetalogParameters(
        boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
        lower_bound=0,
        upper_bound=0,
        method=MetalogFitMethod.OLS,
        num_terms=3,
    )

    num_terms_batch = [5, 7, 9, 11]

    result_batch_nt = fit_grid(
        batched_x, batched_y, params_batch_nt, num_terms=num_terms_batch
    )

    print(f"KS Distances shape: {result_batch_nt.ks_dist.shape}")
    print(f"Coefficients shape: {result_batch_nt.metalog.a.shape}")
    print()
    print("Results (datasets x num_terms):")
    print("              ", end="")
    for _nt in num_terms_batch:
        print(f"{_nt:>8} terms", end="")
    print()
    for _i, _name in enumerate(dist_names):
        print(f"{_name:>12}", end="")
        for _j in range(len(num_terms_batch)):
            print(f"     {float(result_batch_nt.ks_dist[_i, _j]):.4f}", end="")
        print()

    # Find best per dataset
    print("\nBest config per dataset:")
    for _i, _name in enumerate(dist_names):
        _best_idx, _best_ks = find_best_config(result_batch_nt.ks_dist[_i])
        print(
            f"  {_name}: {num_terms_batch[int(_best_idx)]} terms (KS = {float(_best_ks):.4f})"
        )
    return num_terms_batch, params_batch_nt, result_batch_nt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 7: Batched Datasets, L1 Penalties Grid (2D)

    Search over L1 penalties for each dataset in a batch.
    Returns shape (n_datasets, n_penalties).
    """
    )
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    batched_x,
    batched_y,
    dist_names,
    find_best_config,
    fit_grid,
    jnp,
):
    params_batch_l1 = MetalogParameters(
        boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
        lower_bound=0,
        upper_bound=0,
        method=MetalogFitMethod.Lasso,
        num_terms=9,
    )

    l1_batch = jnp.array([0.0, 0.01, 0.1, 1.0])

    result_batch_l1 = fit_grid(
        batched_x, batched_y, params_batch_l1, l1_penalties=l1_batch
    )

    print(f"KS Distances shape: {result_batch_l1.ks_dist.shape}")
    print(f"Coefficients shape: {result_batch_l1.metalog.a.shape}")
    print()
    print("Results (datasets x L1 penalties):")
    print("              ", end="")
    for _l1 in l1_batch:
        print(f"  L1={float(_l1):5.3f}", end="")
    print()
    for _i, _name in enumerate(dist_names):
        print(f"{_name:>12}", end="")
        for _j in range(len(l1_batch)):
            print(f"     {float(result_batch_l1.ks_dist[_i, _j]):.4f}", end="")
        print()

    # Find best per dataset
    print("\nBest L1 per dataset:")
    for _i, _name in enumerate(dist_names):
        _best_idx, _best_ks = find_best_config(result_batch_l1.ks_dist[_i])
        print(
            f"  {_name}: L1={float(l1_batch[int(_best_idx)]):.3f} (KS = {float(_best_ks):.4f})"
        )
    return l1_batch, params_batch_l1, result_batch_l1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Case 8: Full 3D Grid (Batched Datasets x L1 x Num Terms)

    The most comprehensive search: search over all three axes simultaneously.
    Returns shape (n_datasets, n_penalties, n_terms).
    """
    )
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    batched_x,
    batched_y,
    dist_names,
    find_best_config,
    fit_grid,
    jnp,
):
    params_3d = MetalogParameters(
        boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
        lower_bound=0,
        upper_bound=0,
        method=MetalogFitMethod.Lasso,
        num_terms=3,
    )

    l1_3d = jnp.array([0.0, 0.01, 0.1])
    num_terms_3d = [5, 7, 9]

    result_3d = fit_grid(
        batched_x, batched_y, params_3d, l1_penalties=l1_3d, num_terms=num_terms_3d
    )

    print(f"KS Distances shape: {result_3d.ks_dist.shape}")
    print(f"Coefficients shape: {result_3d.metalog.a.shape}")
    print()

    # Find best configuration for each dataset
    print("Best configuration per dataset:")
    for _i, _name in enumerate(dist_names):
        _best_idx, _best_ks = find_best_config(result_3d.ks_dist[_i])
        _best_l1 = l1_3d[_best_idx[0]]
        _best_nt = num_terms_3d[_best_idx[1]]
        print(
            f"  {_name}: L1={float(_best_l1):.3f}, {_best_nt} terms (KS = {float(_best_ks):.4f})"
        )

    print()
    print("Full 3D grid for first dataset (Lognormal):")
    print("         ", end="")
    for _nt in num_terms_3d:
        print(f"{_nt:>8} terms", end="")
    print()
    for _j, _l1 in enumerate(l1_3d):
        print(f"L1={float(_l1):5.3f}", end="")
        for _k in range(len(num_terms_3d)):
            print(f"     {float(result_3d.ks_dist[0, _j, _k]):.4f}", end="")
        print()
    return l1_3d, num_terms_3d, params_3d, result_3d


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Working with Different Boundedness Types

    The `fit_grid` function works with all boundedness types.
    Here's an example with unbounded data.
    """
    )
    return


@app.cell
def _(
    DEFAULT_Y,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    fit_grid,
    norm,
):
    # Generate normally distributed data (unbounded)
    normal_samples = norm(loc=50, scale=10).rvs(size=200, random_state=42)
    data_unbounded = MetalogInputData.from_values(normal_samples, DEFAULT_Y, False)

    params_unbounded = MetalogParameters(
        boundedness=MetalogBoundedness.UNBOUNDED,
        lower_bound=0,  # ignored
        upper_bound=0,  # ignored
        method=MetalogFitMethod.OLS,
        num_terms=7,
    )

    result_unbounded = fit_grid(data_unbounded.x, data_unbounded.y, params_unbounded)
    print(f"Unbounded fit KS distance: {float(result_unbounded.ks_dist):.4f}")
    return data_unbounded, normal_samples, params_unbounded, result_unbounded


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Using Precomputed Quantiles

    If you already have precomputed quantiles (e.g., from expert elicitation),
    set `precomputed_quantiles=True`.
    """
    )
    return


@app.cell
def _(MetalogBoundedness, MetalogFitMethod, MetalogParameters, fit_grid, jnp):
    # Expert-elicited quantiles for a 0-100 bounded variable
    quantiles = jnp.array([10.0, 25.0, 40.0, 50.0, 60.0, 75.0, 90.0])
    probabilities = jnp.array([0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95])

    params_quantiles = MetalogParameters(
        boundedness=MetalogBoundedness.BOUNDED,
        lower_bound=0,
        upper_bound=100,
        method=MetalogFitMethod.OLS,
        num_terms=5,
    )

    result_quantiles = fit_grid(
        quantiles, probabilities, params_quantiles, precomputed_quantiles=True
    )

    print(
        f"Precomputed quantiles fit KS distance: {float(result_quantiles.ks_dist):.4f}"
    )
    print(f"Coefficients: {result_quantiles.metalog.a}")
    return params_quantiles, probabilities, quantiles, result_quantiles


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Summary

    The `fit_grid` function provides a unified interface for all grid search operations:

    | Inputs | Output Shape | Description |
    |--------|--------------|-------------|
    | x.ndim=1, no grids | () | Single fit |
    | x.ndim=1, num_terms | (n_terms,) | 1D num_terms search |
    | x.ndim=1, l1_penalties | (n_penalties,) | 1D L1 search |
    | x.ndim=1, both | (n_penalties, n_terms) | 2D search |
    | x.ndim=2, no grids | (n_datasets,) | Batch fit |
    | x.ndim=2, num_terms | (n_datasets, n_terms) | 2D batch + terms |
    | x.ndim=2, l1_penalties | (n_datasets, n_penalties) | 2D batch + L1 |
    | x.ndim=2, both | (n_datasets, n_penalties, n_terms) | Full 3D search |

    Use `find_best_config(ks_dist)` to find the best configuration in any grid.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
