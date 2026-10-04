"""Dataset generator shared by the benchmarks (same families as the example)."""

import numpy as np
from scipy.stats import gamma, lognorm, lomax, weibull_min


def samples(j: int) -> np.ndarray:
    """200 samples from one of the example's five families, seeded by j."""
    fam = j % 5
    if fam == 0:
        return lognorm(s=0.9).rvs(size=200, random_state=j)
    if fam == 1:
        return weibull_min(c=1.5, scale=2).rvs(size=200, random_state=j)
    if fam == 2:
        return gamma(a=2).rvs(size=200, random_state=j)
    if fam == 3:
        return lomax(c=2.5).rvs(size=200, random_state=j)
    rng = np.random.default_rng(j)
    return np.where(
        rng.random(200) < 0.6, rng.normal(3.0, 0.5, 200), rng.normal(7.0, 1.0, 200)
    )
