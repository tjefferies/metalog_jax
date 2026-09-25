# Copyright: Travis Jefferies 2026
"""Metalog 2.0 feasibility tools (Baucells, Chrisman, Keelin and Xu).

JAX ports of the algorithms in "On the Properties of the Metalog Distribution" and
its reference implementation by Zixin (Stephen) Xu (CC BY 4.0; see NOTICE):

* :func:`best_feasible_fit` - the optimal feasible coefficients ``a*``.
* :func:`check_feasibility` - air-tight feasibility test (Algorithm 1 + Prop. 5).
* :func:`inflection_points` - all modes and anti-modes of the density.
* :func:`summary_stats` / :func:`raw_moment` - exact moments (Lemma 1, Prop. 3).

Every function is jit-compatible and can be vmapped over batches.

References:
    Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
    Properties of the Metalog Distribution. Darden Business School Working
    Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    Xu, Z. S. metalog_algorithm (reference implementation), licensed under
    CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); adapted and ported
    to JAX, see NOTICE. https://github.com/Stephenxuu/metalog_algorithm
"""

from metalog_jax.feasibility.a_star import (
    FeasibleFitResult,
    FeasibleModel,
    best_feasible_fit,
    design_matrix,
    fit_feasible,
    project_onto_polyhedron,
)
from metalog_jax.feasibility.analysis import (
    FeasibilityReport,
    check_feasibility,
    feasibility_function,
    is_feasible,
    mean_and_variance,
    raw_moment,
    summary_stats,
    tail_feasibility,
)
from metalog_jax.feasibility.engine import (
    Engine,
    TermOrder,
    get_engine,
    inflection_points,
    term_layout,
)

__all__ = [
    "Engine",
    "FeasibilityReport",
    "FeasibleFitResult",
    "FeasibleModel",
    "TermOrder",
    "best_feasible_fit",
    "check_feasibility",
    "design_matrix",
    "feasibility_function",
    "fit_feasible",
    "get_engine",
    "inflection_points",
    "is_feasible",
    "mean_and_variance",
    "project_onto_polyhedron",
    "raw_moment",
    "summary_stats",
    "tail_feasibility",
    "term_layout",
]
