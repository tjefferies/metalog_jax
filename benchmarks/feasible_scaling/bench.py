"""Benchmark batch fitting as in examples/feasible_fits.py.

Usage: uv run python benchmarks/feasible_scaling/bench.py MODE N K REPS
  MODE: feasible (fit_grid_datasets, Feasible), ols_datasets (fit_grid_datasets, OLS),
        ols_grid (fit_grid, OLS)
Prints one JSON line.
"""

import json
import resource
import sys
import time

import numpy as np

mode, N, K, REPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from data import samples  # noqa: E402

from metalog_jax.base import (  # noqa: E402
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
)
from metalog_jax.grid_search import fit_grid, fit_grid_datasets  # noqa: E402
from metalog_jax.utils import DEFAULT_Y  # noqa: E402

S = jnp.asarray(np.stack([samples(j) for j in range(N)]))
# Same as MetalogInputData.from_values(samples, DEFAULT_Y, False): x = jnp.quantile(samples, y)
bx = jax.vmap(lambda s: jnp.quantile(s, DEFAULT_Y))(S)
by = jnp.broadcast_to(DEFAULT_Y, bx.shape)
jax.block_until_ready(bx)
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS

method = MetalogFitMethod.Feasible if mode == "feasible" else MetalogFitMethod.OLS
params = MetalogParameters(
    boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
    lower_bound=0.0,
    upper_bound=0.0,
    method=method,
    num_terms=K,
)


def run():
    """Fit the whole batch once and block until the results are ready."""
    if mode == "ols_grid":
        r = fit_grid(bx, by, params)
    else:
        r = fit_grid_datasets(bx, by, params)
    return jax.block_until_ready((r.metalog.a, r.ks_dist))


t = time.perf_counter()
run()
first = time.perf_counter() - t
steady = []
for _ in range(REPS):
    t = time.perf_counter()
    run()
    steady.append(time.perf_counter() - t)
rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print(
    json.dumps(
        dict(
            mode=mode,
            N=N,
            K=K,
            first_s=round(first, 3),
            steady_s=round(min(steady), 4) if steady else None,
            per_dataset_ms=round(1e3 * min(steady) / N, 4) if steady else None,
            rss_before_mb=round(rss_before / 2**20),
            rss_peak_mb=round(rss_peak / 2**20),
        )
    )
)
