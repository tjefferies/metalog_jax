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
