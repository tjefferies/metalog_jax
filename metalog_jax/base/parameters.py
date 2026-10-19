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
from flax import struct

from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
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

    Attributes:
        boundedness: Type of domain constraint for the distribution. Static.
        lower_bound: Lower boundary value for the distribution's support.
        upper_bound: Upper boundary value for the distribution's support.
    """

    boundedness: MetalogBoundedness = struct.field(pytree_node=False)
    lower_bound: chex.Scalar
    upper_bound: chex.Scalar


def _check_serialized_keys(cls: type, params: dict, keys: tuple[str, ...]) -> None:
    """Raise unless a serialized parameter dict has exactly the given keys.

    Args:
        cls: The parameter class being rebuilt.
        params: The ``metalog_params`` dict read from JSON.
        keys: The keys that ``cls.to_dict()`` writes.

    Raises:
        ValueError: If the keys differ, for example SPT metalog JSON loaded as a
            Metalog. The message names the loader for each kind of JSON.
    """
    if set(params) != set(keys):
        raise ValueError(
            f"{cls.__name__} expects the keys {list(keys)} in metalog_params, but "
            f"got {sorted(params)}. Load a Metalog's JSON with Metalog.load or "
            "Metalog.loads, and an SPTMetalog's with SPTMetalog.load or "
            "SPTMetalog.loads."
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
        num_terms: Number of terms in the metalog series expansion. Static.

    See Also:
        MetalogParametersBase: Parent class with boundedness and bound attributes.
        metalog_jax.base.enums.MetalogFitMethod: Enumeration of regression methods.
        metalog_jax.metalog.fit: Function that uses MetalogParameters for fitting.
        metalog_jax.regression: Module containing regression implementations.
    """

    method: MetalogFitMethod = struct.field(pytree_node=False)
    num_terms: int = struct.field(pytree_node=False)

    # The keys to_dict writes, in order; Metalog.dumps output must not change
    _SERIALIZED_KEYS: ClassVar[tuple[str, ...]] = (
        "boundedness",
        "method",
        "lower_bound",
        "upper_bound",
        "num_terms",
    )

    def to_dict(self) -> dict:
        """Return the parameters as the JSON-ready dict that Metalog.dumps writes.

        Returns:
            dict: ``boundedness`` and ``method`` as integers, the bounds as floats,
                and ``num_terms`` as an integer.
        """
        return {
            "boundedness": int(self.boundedness),
            "method": int(self.method),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "num_terms": int(self.num_terms),
        }

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        """Rebuild parameters from a dict written by to_dict.

        Args:
            params: The ``metalog_params`` dict of a saved Metalog.

        Returns:
            MetalogParameters: The parameters, with enum members for ``boundedness``
                and ``method``.

        Raises:
            ValueError: If the keys aren't exactly those that to_dict writes (as
                for an SPTMetalog's parameters), or an enum value is unknown.
        """  # noqa: DOC502
        _check_serialized_keys(cls, params, cls._SERIALIZED_KEYS)
        return cls(
            boundedness=MetalogBoundedness.from_value(params["boundedness"]),
            method=MetalogFitMethod.from_value(params["method"]),
            lower_bound=params["lower_bound"],
            upper_bound=params["upper_bound"],
            num_terms=params["num_terms"],
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
