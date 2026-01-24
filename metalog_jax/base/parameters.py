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

from typing import Union

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

    Attributes:
        prng_params: Params for the psuedorandom number generator.
        size: The number of random variables to draw. Must be strictly positive.
        max_draws: Maximum number of draws for rejection sampling.
    """

    prng_params: Union[JaxUniformDistributionParameters, HDRPRNGParameters]
    size: int
    max_draws: int = DEFAULT_MAX_NUMBER_RV_DRAWS

    def __post_init__(self):
        """Post init data validation.

        Raises:
            AssertionError if self.size is not a positive scalar.
        """
        chex.assert_scalar_positive(self.size)


@struct.dataclass
class MetalogParametersBase:
    """Base class for metalog distribution parameter configurations.

    This abstract base class defines the common parameters shared by all metalog
    distribution types, specifically the boundedness constraints and boundary values
    that determine the support of the distribution.

    Attributes:
        boundedness: Type of domain constraint for the distribution.
        lower_bound: Lower boundary value for the distribution's support.
        upper_bound: Upper boundary value for the distribution's support.
    """

    boundedness: MetalogBoundedness
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
        method: Regression method used for fitting. One of:
            - MetalogFitMethod.OLS: metalog_jax.regression.ols
            - MetalogFitMethod.Lasso: metalog_jax.regression.lasso
        num_terms: Number of terms in the metalog series expansion.

    See Also:
        MetalogParametersBase: Parent class with boundedness and bound attributes.
        metalog_jax.base.enums.MetalogFitMethod: Enumeration of regression methods.
        metalog_jax.metalog.fit: Function that uses MetalogParameters for fitting.
        metalog_jax.regression: Module containing regression implementations.
    """

    method: MetalogFitMethod
    num_terms: int


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
