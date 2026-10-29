# Copyright: Travis Jefferies 2026
"""Parameter configuration classes for the Metalog JAX library.

This module provides parameter configuration classes for metalog distributions:

Classes:
    MetalogRandomVariableParameters: Configuration for random variable sampling.
    MetalogParametersBase: Base class for metalog distribution parameters.
    MetalogParameters: Configuration for standard regression-based metalog fitting.
    SPTMetalogParameters: Configuration for Symmetric Percentile Triplet metalog fitting.

See Also:
    metalog_jax.base.enums: Enumeration types (MetalogBoundedness, MetalogFitMethod).
    metalog_jax.base.core: MetalogBase class that uses these parameters.
    metalog_jax.metalog: Fitting functions that accept these parameter configurations.
"""

from typing import ClassVar, Self

import chex
import numpy as np
from flax import struct

from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
from metalog_jax.feasibility.engine import DEFAULT_TERM_ORDER, TermOrder

# The order of a saved Metalog whose JSON has no term_order: every file saved before
# term_order existed is in the 2016 order, whatever the default is now
_UNSAVED_TERM_ORDER = TermOrder.KEELIN_2016
from metalog_jax.utils import (
    DEFAULT_MAX_NUMBER_RV_DRAWS,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
)


@struct.dataclass
class MetalogRandomVariableParameters:
    """Configuration parameters for drawing rvs from fitted metalog distribution.

    ``size`` and ``max_draws`` set array shapes, so they are static fields: they
    belong to the pytree structure rather than its leaves, and the parameters can be
    passed to ``jax.jit``. The PRNG parameters are leaves, so ``jax.vmap`` can run
    over seeds or HDR trials.

    Attributes:
        prng_params: Params for the psuedorandom number generator.
        size: The number of random variables to draw. Must be strictly positive.
            Static.
        max_draws: Maximum number of draws for rejection sampling. Static.
    """

    prng_params: JaxUniformDistributionParameters | HDRPRNGParameters
    size: int = struct.field(pytree_node=False)
    max_draws: int = struct.field(
        pytree_node=False, default=DEFAULT_MAX_NUMBER_RV_DRAWS
    )

    def __post_init__(self):
        """Post init data validation.

        Raises:
            AssertionError: If ``size`` is not a positive scalar.
        """  # noqa: DOC502
        chex.assert_scalar_positive(self.size)


# The boundedness types whose support each bound limits
_USES_LOWER_BOUND = (
    MetalogBoundedness.STRICTLY_LOWER_BOUND,
    MetalogBoundedness.BOUNDED,
)
_USES_UPPER_BOUND = (
    MetalogBoundedness.STRICTLY_UPPER_BOUND,
    MetalogBoundedness.BOUNDED,
)


def _concrete_bound(bound: object) -> np.ndarray | None:
    """Return a bound as a float array if it's a Python or NumPy number.

    Args:
        bound: A ``lower_bound`` or ``upper_bound`` value.

    Returns:
        np.ndarray | None: The bound as floats, or None for anything else: JAX
            arrays and tracers, ``None``, ``ShapeDtypeStruct`` and the other
            placeholders that pytree unflattening passes.
    """
    if isinstance(bound, (int, float, np.number)) or (
        isinstance(bound, np.ndarray) and np.issubdtype(bound.dtype, np.number)
    ):
        return np.asarray(bound, dtype=float)
    return None


@struct.dataclass
class MetalogParametersBase:
    """Base class for metalog distribution parameter configurations.

    This abstract base class defines the common parameters shared by all metalog
    distribution types, specifically the boundedness constraints and boundary values
    that determine the support of the distribution.

    Fields that select code paths or array shapes (``boundedness`` here, and
    ``method`` and ``num_terms`` in MetalogParameters) are static: they belong to
    the pytree structure rather than its leaves. So a model or parameters object
    can be passed to ``jax.jit``, which compiles once per combination of static
    values, and models stacked for ``jax.vmap`` must share them. The bounds are
    ordinary leaves and can be traced.

    The bounds are checked on construction when they're Python or NumPy numbers.
    JAX arrays aren't checked, because pytree transforms rebuild parameters with
    tracers, gradients, stacked batches or placeholders as the bounds.

    Attributes:
        boundedness: Type of domain constraint for the distribution. Static.
        lower_bound: Lower boundary value for the distribution's support. Must be
            finite for STRICTLY_LOWER_BOUND and BOUNDED, and below ``upper_bound``
            for BOUNDED; ignored otherwise.
        upper_bound: Upper boundary value for the distribution's support. Must be
            finite for STRICTLY_UPPER_BOUND and BOUNDED; ignored otherwise.
    """

    boundedness: MetalogBoundedness = struct.field(pytree_node=False)
    lower_bound: chex.Scalar
    upper_bound: chex.Scalar

    def __post_init__(self) -> None:
        """Check that the bounds the boundedness type uses describe a support.

        Raises:
            ValueError: If a bound that the boundedness type uses is infinite or
                NaN, or a BOUNDED distribution's ``lower_bound`` isn't below its
                ``upper_bound``.
        """
        lower = upper = None
        if self.boundedness in _USES_LOWER_BOUND:
            lower = _concrete_bound(self.lower_bound)
        if self.boundedness in _USES_UPPER_BOUND:
            upper = _concrete_bound(self.upper_bound)
        for name, bound in (("lower_bound", lower), ("upper_bound", upper)):
            if bound is not None and not np.all(np.isfinite(bound)):
                raise ValueError(
                    f"{self.boundedness.name} needs a finite {name}, got "
                    f"{getattr(self, name)!r}"
                )
        if lower is not None and upper is not None and not np.all(lower < upper):
            raise ValueError(
                "BOUNDED needs lower_bound < upper_bound, got "
                f"lower_bound={self.lower_bound!r} and upper_bound={self.upper_bound!r}"
            )


def _check_serialized_keys(
    cls: type,
    params: dict,
    keys: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> None:
    """Raise unless a serialized parameter dict has the keys ``to_dict`` writes.

    Args:
        cls: The parameter class being rebuilt.
        params: The ``metalog_params`` dict read from JSON.
        keys: The keys that ``cls.to_dict()`` always writes.
        optional: Keys that ``cls.to_dict()`` writes only for some values.

    Raises:
        ValueError: If a key is missing or unknown, for example SPT metalog JSON
            loaded as a Metalog. The message names the loader for each kind of JSON.
    """
    if not set(keys) <= set(params) <= set(keys) | set(optional):
        optional_keys = f", and optionally {list(optional)}," if optional else ""
        raise ValueError(
            f"{cls.__name__} expects the keys {list(keys)}{optional_keys} in "
            f"metalog_params, but got {sorted(params)}. Load a Metalog's JSON with "
            "Metalog.load or Metalog.loads, and an SPTMetalog's with SPTMetalog.load "
            "or SPTMetalog.loads."
        )


@struct.dataclass
class MetalogParameters(MetalogParametersBase):
    """Configuration parameters for standard regression-based metalog distributions.

    Extends MetalogParametersBase with regression-specific parameters for fitting
    metalog distributions via regression methods including OLS and Lasso.

    Attributes:
        boundedness: Inherited. Type of domain constraint for the distribution.
            Defined in metalog_jax.base.enums.MetalogBoundedness.
        lower_bound: Inherited. Lower boundary value.
        upper_bound: Inherited. Upper boundary value.
        method: Regression method used for fitting. Static. One of:
            - MetalogFitMethod.OLS: metalog_jax.regression.ols
            - MetalogFitMethod.Lasso: metalog_jax.regression.lasso
        num_terms: Number of terms in the metalog series expansion: an integer of
            at least 2. Static.
        term_order: The order of the basis terms, and so of the coefficients a.
            Static. TermOrder.METALOG_2, the default, is 1, L, xL, x, x^2, x^2 L,
            x^3 L, x^3, x^4, ..., with x = y - 0.5 and L = ln(y/(1-y)) (Baucells
            et al. 2025, Section 2): for probabilities symmetric about 0.5, its
            basis matrix has full rank for every k <= n (Proposition 2).
            TermOrder.KEELIN_2016 is 1, L, xL, x, x^2, x^2 L, x^3, x^3 L, ...
            (Keelin 2016, Definition 1), the only order in earlier releases, whose
            basis matrix is rank deficient for n = k in {7, 11, 15, ...}.
            The orders agree for up to 6 terms. For other k outside
            {7, 11, 15, ...}, they span the same functions in a different order,
            and Metalog.with_term_order converts between them.

    See Also:
        MetalogParametersBase: Parent class with boundedness and bound attributes.
        metalog_jax.base.enums.MetalogFitMethod: Enumeration of regression methods.
        metalog_jax.metalog.fit: Function that uses MetalogParameters for fitting.
        metalog_jax.regression: Module containing regression implementations.
    """

    method: MetalogFitMethod = struct.field(pytree_node=False)
    num_terms: int = struct.field(pytree_node=False)
    term_order: TermOrder = struct.field(pytree_node=False, default=DEFAULT_TERM_ORDER)

    # The keys to_dict writes, in order; Metalog.dumps output must not change
    _SERIALIZED_KEYS: ClassVar[tuple[str, ...]] = (
        "boundedness",
        "method",
        "lower_bound",
        "upper_bound",
        "num_terms",
    )
    # Written unless the order is _UNSAVED_TERM_ORDER, which a missing key means, so
    # that 2016-order JSON is unchanged and still loads in earlier versions
    _OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = ("term_order",)

    def __post_init__(self) -> None:
        """Check the bounds, as the base class does, and the number of terms.

        Raises:
            ValueError: If ``num_terms`` isn't an integer of at least 2 (a 1-term
                metalog is a constant, not a distribution), ``term_order`` isn't a
                TermOrder, or a bound is invalid.
        """
        super().__post_init__()
        if (
            isinstance(self.num_terms, bool)
            or not isinstance(self.num_terms, (int, np.integer))
            or self.num_terms < 2
        ):
            raise ValueError(
                f"num_terms must be an integer of at least 2, got {self.num_terms!r}"
            )
        if not isinstance(self.term_order, TermOrder):
            raise ValueError(f"term_order must be a TermOrder, got {self.term_order!r}")

    def to_dict(self) -> dict:
        """Return the parameters as the JSON-ready dict that Metalog.dumps writes.

        Returns:
            dict: ``boundedness`` and ``method`` as integers, the bounds as floats,
                and ``num_terms`` as an integer. ``term_order`` is added, as an
                integer, unless it's TermOrder.KEELIN_2016, the order of JSON
                without it, so that 2016-order JSON is the same as before.
        """
        params = {
            "boundedness": int(self.boundedness),
            "method": int(self.method),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "num_terms": int(self.num_terms),
        }
        if self.term_order != _UNSAVED_TERM_ORDER:
            params["term_order"] = int(self.term_order)
        return params

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        """Rebuild parameters from a dict written by to_dict.

        Args:
            params: The ``metalog_params`` dict of a saved Metalog.

        Returns:
            MetalogParameters: The parameters, with enum members for
                ``boundedness``, ``method`` and ``term_order``. JSON without
                ``term_order``, including every file saved before it existed, is
                in the 2016 order, not the default.

        Raises:
            ValueError: If the keys aren't those that to_dict writes (as for an
                SPTMetalog's parameters), or an enum value is unknown.
        """  # noqa: DOC502
        _check_serialized_keys(
            cls, params, cls._SERIALIZED_KEYS, optional=cls._OPTIONAL_KEYS
        )
        return cls(
            boundedness=MetalogBoundedness.from_value(params["boundedness"]),
            method=MetalogFitMethod.from_value(params["method"]),
            lower_bound=params["lower_bound"],
            upper_bound=params["upper_bound"],
            num_terms=params["num_terms"],
            term_order=TermOrder(params.get("term_order", _UNSAVED_TERM_ORDER)),
        )


@struct.dataclass
class SPTMetalogParameters(MetalogParametersBase):
    """Configuration parameters for Symmetric Percentile Triplet (SPT) metalog distributions.

    Extends MetalogParametersBase with the alpha parameter for SPT metalog fitting,
    which uses exactly 3 terms fitted from three symmetric quantiles.

    Attributes:
        boundedness: Inherited. Type of domain constraint for the distribution.
        lower_bound: Inherited. Lower boundary value.
        upper_bound: Inherited. Upper boundary value.
        alpha: The lower percentile parameter for SPT fitting. Must be in (0, 0.5).
    """

    alpha: float

    # The keys to_dict writes, in order. An SPT metalog has no fit method and
    # always 3 terms, so neither is saved.
    _SERIALIZED_KEYS: ClassVar[tuple[str, ...]] = (
        "boundedness",
        "lower_bound",
        "upper_bound",
        "alpha",
    )

    def to_dict(self) -> dict:
        """Return the parameters as the JSON-ready dict that SPTMetalog.dumps writes.

        Returns:
            dict: ``boundedness`` as an integer, and the bounds and ``alpha`` as
                floats.
        """
        return {
            "boundedness": int(self.boundedness),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "alpha": float(self.alpha),
        }

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        """Rebuild parameters from a dict written by to_dict.

        Args:
            params: The ``metalog_params`` dict of a saved SPTMetalog.

        Returns:
            SPTMetalogParameters: The parameters, with an enum member for
                ``boundedness``.

        Raises:
            ValueError: If the keys aren't exactly those that to_dict writes (as
                for a Metalog's parameters), or the boundedness value is unknown.
        """  # noqa: DOC502
        _check_serialized_keys(cls, params, cls._SERIALIZED_KEYS)
        return cls(
            boundedness=MetalogBoundedness.from_value(params["boundedness"]),
            lower_bound=params["lower_bound"],
            upper_bound=params["upper_bound"],
            alpha=params["alpha"],
        )
