# Copyright: Travis Jefferies 2026
"""Enumeration types for the Metalog JAX library.

This module provides enumeration types for configuring metalog distributions:

Classes:
    CustomIntEnum: Base class for integer enumerations with enhanced value lookup.
    MetalogBoundedness: Enumeration of boundedness types (UNBOUNDED, STRICTLY_LOWER_BOUND,
        STRICTLY_UPPER_BOUND, BOUNDED).
    MetalogFitMethod: Enumeration of regression methods (OLS, Lasso).
    MetalogPlotOptions: Enumeration of plot types (PDF, CDF, SF).

See Also:
    metalog_jax.base.parameters: Parameter configuration classes that use these enums.
    metalog_jax.regression: Regression implementations for each MetalogFitMethod.
"""

from enum import IntEnum, auto


class CustomIntEnum(IntEnum):
    """Base class for custom integer enumerations with enhanced value lookup.

    Extends Python's IntEnum to provide additional utility methods for working
    with enumeration values. This class serves as a base for all custom integer
    enumerations in the metalog JAX library, providing consistent behavior for
    value-to-member conversion and validation.

    The class maintains all IntEnum functionality (enum members are instances of int
    and can be used in arithmetic operations) while adding convenient class methods
    for value lookup and conversion.

    Note:
        This is an abstract base class. Use specific subclasses like
        MetalogBoundedness or MetalogFitMethod for actual enum definitions.
    """

    @classmethod
    def from_value(cls, value: int):
        """Convert an integer value to the corresponding enum member.

        Searches through all members of the enumeration and returns the member
        whose value matches the input. This provides a safe way to reconstruct
        enum members from integer values, such as when loading from JSON files
        or deserializing data.

        Args:
            value: Integer value to convert to an enum member. Must match the
                value of an existing enum member exactly.

        Returns:
            The enum member whose value matches the input value.

        Raises:
            ValueError: If no enum member with the specified value exists.

        Example:
            >>> boundedness = MetalogBoundedness.from_value(1)
            >>> assert boundedness == MetalogBoundedness.UNBOUNDED
            >>> MetalogBoundedness.from_value(99)  # Raises ValueError
        """
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No enum member with value {value}")


class MetalogBoundedness(CustomIntEnum):
    """Enumeration of boundedness types for metalog distributions.

    Defines the domain constraints for metalog probability distributions,
    specifying whether the distribution has lower bounds, upper bounds,
    both, or neither.

    Attributes:
        UNBOUNDED: Distribution with no bounds (support on entire real line).
        STRICTLY_LOWER_BOUND: Distribution with only a lower bound.
        STRICTLY_UPPER_BOUND: Distribution with only an upper bound.
        BOUNDED: Distribution with both lower and upper bounds.
    """

    UNBOUNDED = auto()
    STRICTLY_LOWER_BOUND = auto()
    STRICTLY_UPPER_BOUND = auto()
    BOUNDED = auto()


class MetalogFitMethod(CustomIntEnum):
    """Enumeration of regression methods for fitting metalog distributions.

    Defines the available regression techniques that can be used to estimate
    metalog distribution coefficients from data.

    Attributes:
        OLS: Ordinary Least Squares regression (no regularization, closed-form solution).
            Implementation: metalog_jax.regression.ols.fit_ordinary_least_squares
        Lasso: LASSO regression (L1 regularization via proximal gradient descent).
            Implementation: metalog_jax.regression.lasso.fit_lasso

    See Also:
        metalog_jax.regression: Module containing all regression implementations.
        metalog_jax.regression.ols: OLS regression implementation.
        metalog_jax.regression.lasso: LASSO regression implementation.
    """

    OLS = auto()
    Lasso = auto()


class MetalogPlotOptions(IntEnum):
    """Enumeration of available plot types for metalog distributions.

    Defines the types of statistical plots that can be generated for visualizing
    fitted metalog distributions. Each option corresponds to a different view of
    the probability distribution.

    Attributes:
        PDF: Probability Density Function plot showing the probability density
            across the distribution's support.
        CDF: Cumulative Distribution Function plot showing the cumulative
            probability from the lower bound to each point.
        SF: Survival Function plot showing the survival function
         from the lower bound to each point.
    """

    PDF = auto()
    CDF = auto()
    SF = auto()
