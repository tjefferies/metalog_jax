"""Benchmark datasets, drawn on the device with jax.random.

The five families of ``examples/feasible_fits.py`` and ``benchmarks/feasible_scaling``,
with the same parameters. Dataset ``j`` is family ``j % 5``, drawn with
``fold_in(key(seed), j)``, so a dataset doesn't depend on the chunk it's drawn in, and
every device draws the same ones (JAX's default PRNG is the same on every backend).
"""

from functools import partial

import jax
import jax.numpy as jnp

from metalog_jax.utils import DEFAULT_Y

FAMILIES = (
    "lognormal(sigma=0.9)",
    "weibull_min(concentration=1.5, scale=2)",
    "gamma(a=2)",
    "lomax(c=2.5)",
    "mixture: 0.6 normal(3, 0.5) + 0.4 normal(7, 1), absolute value",
)


def _draw(key: jax.Array, j: jax.Array, num_samples: int) -> jax.Array:
    """Draw dataset j's samples: family j % 5, all positive.

    Every family is drawn and one is kept, so that the draw vmaps without branches.
    The mixture's absolute value changes nothing in practice: a normal(3, 0.5) draw
    is negative with probability 1e-9. It keeps every sample inside the benchmark's
    lower bound of 0.

    Args:
        key: The dataset's PRNG key.
        j: The dataset's index.
        num_samples: Samples per dataset (static).

    Returns:
        jax.Array: The samples, shape (num_samples,).
    """
    keys = jax.random.split(key, 7)
    shape = (num_samples,)
    lognormal = jax.random.lognormal(keys[0], sigma=0.9, shape=shape)
    weibull = jax.random.weibull_min(keys[1], scale=2.0, concentration=1.5, shape=shape)
    gamma = jax.random.gamma(keys[2], 2.0, shape=shape)
    lomax = jax.random.pareto(keys[3], 2.5, shape=shape) - 1.0
    first = jax.random.bernoulli(keys[4], 0.6, shape=shape)
    mixture = jnp.abs(
        jnp.where(
            first,
            3.0 + 0.5 * jax.random.normal(keys[5], shape),
            7.0 + jax.random.normal(keys[6], shape),
        )
    )
    return jnp.stack([lognormal, weibull, gamma, lomax, mixture])[j % 5]


@partial(jax.jit, static_argnames=("count", "num_samples"))
def quantiles(
    seed: int, start: int, count: int, num_samples: int = 200, y: jax.Array = DEFAULT_Y
) -> jax.Array:
    """Datasets start to start + count, each summarized at its quantiles at y.

    Args:
        seed: The benchmark's seed.
        start: Index of the first dataset.
        count: Number of datasets (static).
        num_samples: Samples per dataset (static).
        y: Probability levels.

    Returns:
        jax.Array: Quantiles, shape (count, len(y)), as MetalogInputData.from_values
        computes them from samples.
    """
    base = jax.random.key(seed)
    index = start + jnp.arange(count)

    def one(j):
        samples = _draw(jax.random.fold_in(base, j), j, num_samples)
        return jnp.quantile(samples, y)

    return jax.vmap(one)(index)
