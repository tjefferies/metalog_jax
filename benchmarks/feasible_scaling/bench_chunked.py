"""Fit TOTAL datasets with Feasible in fixed-size chunks (one compiled shape).

Usage: uv run python benchmarks/feasible_scaling/bench_chunked.py TOTAL CHUNK K
Prints one JSON line.
"""

import json
import resource
import sys
import time

import numpy as np

TOTAL, CHUNK, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
assert TOTAL % CHUNK == 0

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from data import samples  # noqa: E402

from metalog_jax.base import (  # noqa: E402
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogParameters,
)
from metalog_jax.grid_search import fit_grid_datasets  # noqa: E402
from metalog_jax.utils import DEFAULT_Y  # noqa: E402

params = MetalogParameters(
    boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND,
    lower_bound=0.0,
    upper_bound=0.0,
    method=MetalogFitMethod.Feasible,
    num_terms=K,
)
quantiles = jax.jit(jax.vmap(lambda s: jnp.quantile(s, DEFAULT_Y)))

t_all = time.perf_counter()
t_fit = 0.0
first = None
for c in range(TOTAL // CHUNK):
    S = jnp.asarray(np.stack([samples(j) for j in range(c * CHUNK, (c + 1) * CHUNK)]))
    bx = quantiles(S)
    by = jnp.broadcast_to(DEFAULT_Y, bx.shape)
    t = time.perf_counter()
    r = fit_grid_datasets(bx, by, params)
    jax.block_until_ready((r.metalog.a, r.ks_dist))
    dt = time.perf_counter() - t
    first = dt if first is None else first
    t_fit += dt
wall = time.perf_counter() - t_all

print(
    json.dumps(
        dict(
            mode="feasible_chunked",
            N=TOTAL,
            chunk=CHUNK,
            K=K,
            first_chunk_s=round(first, 3),
            fit_s=round(t_fit, 3),
            fit_s_excl_compile=round(
                t_fit - first + (t_fit - first) / (TOTAL // CHUNK - 1), 3
            ),
            wall_s_incl_data_gen=round(wall, 3),
            rss_peak_mb=round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
        )
    )
)
