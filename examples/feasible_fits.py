"""Batch feasible fitting demonstration (Baucells et al. 2025)."""

import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting Many Datasets with `MetalogFitMethod.Feasible`

    Ordinary least squares can return metalog coefficients whose density goes negative
    somewhere, especially with many terms. `MetalogFitMethod.Feasible` returns the best
    least-squares fit *among valid metalogs* (the coefficient vector a* of Baucells,
    Chrisman, Keelin and Xu, 2025), so every fit is a proper distribution.

    This notebook fits a batch of datasets with `Feasible`:

    1. **Fixed number of terms**: one call to `fit_grid_datasets`, which vmaps `fit`
       over the datasets.
    2. **Different numbers of terms**: one `fit_grid_datasets` call per term count, then
       pick the best term count for each dataset by KS distance.

    `fit_grid` cannot be used here: it varies the number of terms inside `vmap`, while the
    feasible solver needs it fixed at trace time, so it raises a `ValueError` for
    `Feasible`. `fit_grid_datasets` compiles once per term count instead.
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import gamma, lognorm, lomax, weibull_min

    from metalog_jax.base import (
        MetalogBoundedness,
        MetalogFitMethod,
        MetalogInputData,
        MetalogParameters,
    )
    from metalog_jax.feasibility import check_feasibility, get_engine
    from metalog_jax.grid_search import extract_metalog, fit_grid, fit_grid_datasets
    from metalog_jax.utils import DEFAULT_Y

    return (
        DEFAULT_Y,
        MetalogBoundedness,
        MetalogFitMethod,
        MetalogInputData,
        MetalogParameters,
        check_feasibility,
        extract_metalog,
        fit_grid,
        fit_grid_datasets,
        gamma,
        get_engine,
        jax,
        jnp,
        lognorm,
        lomax,
        np,
        weibull_min,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data

    Five positive datasets of 200 samples each, including a heavy tail (Lomax) and a
    two-peaked mixture. Each is summarized by its empirical quantiles at `DEFAULT_Y`
    and stacked into `(n_datasets, n_quantiles)` arrays. All fits use a lower bound of 0.
    """)
    return


@app.cell
def _(
    DEFAULT_Y,
    MetalogInputData,
    gamma,
    jnp,
    lognorm,
    lomax,
    np,
    weibull_min,
):
    _rng = np.random.default_rng(5)
    _bimodal = np.where(
        _rng.random(200) < 0.6, _rng.normal(3.0, 0.5, 200), _rng.normal(7.0, 1.0, 200)
    )
    _samples = {
        "Lognormal": lognorm(s=0.9).rvs(size=200, random_state=1),
        "Weibull": weibull_min(c=1.5, scale=2).rvs(size=200, random_state=2),
        "Gamma": gamma(a=2).rvs(size=200, random_state=3),
        "Lomax": lomax(c=2.5).rvs(size=200, random_state=4),
        "Bimodal": _bimodal,
    }
    dataset_names = list(_samples)

    _data = [
        MetalogInputData.from_values(jnp.asarray(v), DEFAULT_Y, False)
        for v in _samples.values()
    ]
    batched_x = jnp.stack([d.x for d in _data])
    batched_y = jnp.stack([d.y for d in _data])

    print(f"Batched x shape: {batched_x.shape}")
    print(f"Batched y shape: {batched_y.shape}")
    return batched_x, batched_y, dataset_names


@app.cell
def _(check_feasibility, get_engine, jax):
    def feasible_flags(coeffs, num_terms):
        """Exact feasibility test for each row of a (n_datasets, num_terms) array."""
        engine = get_engine(num_terms)
        return jax.vmap(lambda a: check_feasibility(engine, a).feasible)(coeffs)

    return (feasible_flags,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Case 1: Fixed Number of Terms

    First, fit all datasets with OLS and check each fit with `check_feasibility`. The
    check is exact: it finds every local minimum of the density and tests both tails,
    so it also catches negative density outside the probabilities that `fit` checks.
    """)
    return


@app.cell
def _(
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    batched_x,
    batched_y,
    feasible_flags,
    fit_grid,
):
    FIXED_TERMS = 10

    params_ols = MetalogParameters(
        boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
        lower_bound=0.0,
        upper_bound=0.0,
        method=MetalogFitMethod.OLS,
        num_terms=FIXED_TERMS,
    )

    # fit_grid returns OLS coefficients without validating them
    result_ols = fit_grid(batched_x, batched_y, params_ols)
    ols_feasible = feasible_flags(result_ols.metalog.a, FIXED_TERMS)
    print(
        f"OLS fits that are valid distributions: {int(ols_feasible.sum())} of {len(ols_feasible)}"
    )
    return FIXED_TERMS, ols_feasible, result_ols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now fit the same batch with `MetalogFitMethod.Feasible`. `fit_grid_datasets` vmaps
    `fit` over the datasets, so this is one compiled call for all of them. Where the OLS
    fit is already feasible, `Feasible` returns the same coefficients.
    """)
    return


@app.cell
def _(
    FIXED_TERMS,
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
    batched_x,
    batched_y,
    dataset_names,
    feasible_flags,
    fit_grid_datasets,
    ols_feasible,
    result_ols,
):
    params_feasible = MetalogParameters(
        boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
        lower_bound=0.0,
        upper_bound=0.0,
        method=MetalogFitMethod.Feasible,
        num_terms=FIXED_TERMS,
    )

    result_feasible = fit_grid_datasets(batched_x, batched_y, params_feasible)
    feasible_ok = feasible_flags(result_feasible.metalog.a, FIXED_TERMS)

    print(f"Coefficients shape: {result_feasible.metalog.a.shape}")
    print(f"KS Distances shape: {result_feasible.ks_dist.shape}")
    print()
    print(
        f"{'Dataset':<10} {'OLS valid':>10} {'OLS KS':>8} {'Feasible valid':>15} {'Feasible KS':>12}"
    )
    for _i, _name in enumerate(dataset_names):
        print(
            f"{_name:<10} {str(bool(ols_feasible[_i])):>10} {float(result_ols.ks_dist[_i]):>8.4f}"
            f" {str(bool(feasible_ok[_i])):>15} {float(result_feasible.ks_dist[_i]):>12.4f}"
        )
    return (params_feasible,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An infeasible OLS fit can have a *lower* KS distance than the feasible fit, because
    the KS distance only compares quantiles at the data points. It is still not a valid
    distribution: its density is negative somewhere, so its quantile function is not
    monotone.

    ---
    ## Case 2: Different Numbers of Terms

    The feasible solver needs the number of terms fixed when it compiles, so loop over
    the term counts in Python and batch the datasets inside each call. Each term count
    compiles once. Stacking the KS distances gives a `(n_datasets, n_terms)` grid.
    """)
    return


@app.cell
def _(
    batched_x,
    batched_y,
    dataset_names,
    fit_grid_datasets,
    jnp,
    params_feasible,
):
    num_terms_list = [4, 6, 8, 10, 12]

    # One compiled call per term count; each batches all datasets
    results_by_terms = {
        _k: fit_grid_datasets(
            batched_x, batched_y, params_feasible.replace(num_terms=_k)
        )
        for _k in num_terms_list
    }
    ks_grid = jnp.stack([results_by_terms[_k].ks_dist for _k in num_terms_list], axis=1)

    print(f"KS grid shape (n_datasets, n_terms): {ks_grid.shape}")
    print()
    print(f"{'Dataset':<10}" + "".join(f"{_k:>8} terms" for _k in num_terms_list))
    for _i, _name in enumerate(dataset_names):
        print(f"{_name:<10}" + "".join(f"{float(_v):>14.4f}" for _v in ks_grid[_i]))
    return ks_grid, num_terms_list, results_by_terms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pick the term count with the lowest KS distance for each dataset (ties go to the
    fewest terms), then use `extract_metalog` to pull out a ready-to-use `Metalog`.
    """)
    return


@app.cell
def _(
    dataset_names,
    extract_metalog,
    feasible_flags,
    jnp,
    ks_grid,
    num_terms_list,
    results_by_terms,
):
    best_term_idx = jnp.argmin(ks_grid, axis=1)
    best_metalogs = {}

    print(
        f"{'Dataset':<10} {'Terms':>6} {'KS':>8} {'Median':>8} {'P90':>8} {'Valid':>6}"
    )
    for _i, _name in enumerate(dataset_names):
        _k = num_terms_list[int(best_term_idx[_i])]
        _m = extract_metalog(results_by_terms[_k], _i)
        best_metalogs[_name] = _m
        _median, _p90 = _m.ppf(jnp.array([0.5, 0.9]))
        _valid = bool(feasible_flags(_m.a[None, :], _k)[0])
        print(
            f"{_name:<10} {_k:>6} {float(ks_grid[_i, int(best_term_idx[_i])]):>8.4f}"
            f" {float(_median):>8.3f} {float(_p90):>8.3f} {str(_valid):>6}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    - `MetalogFitMethod.Feasible` guarantees every fit is a valid distribution, and
      returns the OLS coefficients unchanged when OLS is already valid.
    - Use `fit_grid_datasets` to fit a batch of datasets with `Feasible` in one call.
    - To compare term counts, loop over them in Python with one `fit_grid_datasets` call
      each. `fit_grid` raises a `ValueError` for `Feasible`.
    - `check_feasibility` gives an exact validity check for any coefficient vector, for
      example to audit OLS fits.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
