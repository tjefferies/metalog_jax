# Copyright: Travis Jefferies 2026
"""Data container classes for the Metalog JAX library.

This module provides data container classes for metalog input data:

Classes:
    MetalogBaseData: Base container for metalog input data without validation.
        Intended for internal use within JAX-traced functions.
    MetalogInputData: Validated container for metalog input data, created
        exclusively via the from_values() factory method.

See Also:
    metalog_jax.base.core: MetalogBase class.
    metalog_jax.base.parameters: Parameter configuration classes.
    metalog_jax.metalog.fit: Function that uses MetalogInputData for fitting.
"""

from dataclasses import fields
from typing import Type, TypeVar

import chex
import jax.numpy as jnp
from flax import struct

from metalog_jax.utils import (
    assert_numeric_array,
    assert_probability_range,
    assert_strictly_ascending,
)

T_MetalogInputData = TypeVar("T_MetalogInputData", bound="MetalogInputData")


@struct.dataclass
class MetalogBaseData:
    """Base container for metalog input data without validation.

    This is a minimal dataclass that holds the core data required for metalog
    fitting without performing any validation or transformation. It serves as
    the parent class for `MetalogInputData` and is primarily used for internal
    operations where validation has already been performed or needs to be bypassed,
    such as within JAX-traced functions like `vmap`.

    **IMPORTANT**: This class is intended for internal library use and advanced
    users only. Most users should use `MetalogInputData.from_values()` instead,
    which provides comprehensive validation and proper error handling.

    Use `MetalogBaseData` directly only when:
    - Working inside JAX-traced functions (vmap, jit, etc.) where validation
      functions would fail due to tracer boolean conversion issues
    - Data has already been validated externally
    - Building custom pipelines that require manual control over validation

    Attributes:
        x: Quantile values array corresponding to the probability levels in `y`.
            This should be a 1D JAX array of numeric values. When
            `precomputed_quantiles=False`, these are typically computed via
            `jnp.quantile(samples, y)`. When `precomputed_quantiles=True`,
            these are user-provided quantile values.

        y: Probability levels array in the open interval (0, 1), sorted in
            strictly ascending order. This should be a 1D JAX array with the
            same length as `x`, representing the cumulative probabilities at
            which the quantiles in `x` are evaluated.

        precomputed_quantiles: Boolean flag indicating whether `x` contains
            pre-computed quantiles (True) or was originally raw sample data (False).
            This is marked as `pytree_node=False` to exclude it from JAX
            transformations, as it's metadata rather than data to be transformed.

    Note:
        This class uses Flax's `struct.dataclass` decorator to make instances
        immutable and compatible with JAX transformations (jit, vmap, grad, etc.).
        The `precomputed_quantiles` field is excluded from the pytree structure
        via `pytree_node=False` since it's metadata.

    See Also:
        MetalogInputData: Subclass with validation via from_values() factory method.
        metalog_jax.base.parameters.MetalogParameters: Configuration for fitting.
        metalog_jax.metalog.fit: Function that processes input data for fitting.
    """

    x: chex.Numeric
    y: chex.Numeric
    precomputed_quantiles: bool = struct.field(pytree_node=False)


@struct.dataclass
class MetalogInputData(MetalogBaseData):
    """Container for input data used to fit metalog distributions.

    This dataclass encapsulates the input data for metalog fitting, supporting two modes:
    (1) fitting from pre-computed quantiles at specified probability levels, or
    (2) fitting from raw sample data where quantiles are computed automatically.

    The mode is controlled by the `precomputed_quantiles` flag, which determines how
    the `x` array should be interpreted and processed during fitting.

    **IMPORTANT**: Instances MUST be created using the `from_values()` classmethod.
    Direct instantiation via `__init__()` is prevented by a `__post_init__` validation
    that raises `TypeError` if the instance was not created through `from_values()`.

    Attributes:
        x: Input data array. The interpretation depends on `precomputed_quantiles`:
            - If `precomputed_quantiles=True`: Array of quantile values
            - If `precomputed_quantiles=False`: Array of raw sample data

        y: Array of probability levels in the open interval (0, 1), sorted in ascending
            order.

        precomputed_quantiles: Flag indicating the interpretation of `x`.

    See Also:
        MetalogBaseData: Parent class without validation (for advanced use).
        from_values: Required factory method for creating validated instances.
        metalog_jax.metalog.fit: Function that uses MetalogInputData for fitting.
    """

    @classmethod
    def from_values(
        cls: Type[T_MetalogInputData],
        x: chex.Numeric,
        y: chex.Numeric,
        precomputed_quantiles: bool,
    ) -> T_MetalogInputData:
        """Create a validated MetalogInputData instance with comprehensive input checks.

        This is the ONLY way to create valid MetalogInputData instances.

        Args:
            x: Input data array.
            y: Array of probability levels in the open interval (0, 1).
            precomputed_quantiles: Boolean flag indicating interpretation of `x`.

        Returns:
            A validated MetalogInputData instance ready for use in metalog fitting.

        Raises:
            AssertionError: If any validation check fails.
        """

        def _input_check(
            x: chex.Numeric, y: chex.Numeric, precomputed_quantiles: bool
        ) -> dict:
            """Validate inputs and return dictionary for dataclass construction."""
            # Validate boolean flag
            chex.assert_type(precomputed_quantiles, bool)

            # Validate x array: minimum size and numeric type
            chex.assert_axis_dimension_gteq(x, 0, 3)
            assert_numeric_array(x)

            # Validate y array: minimum size and numeric type
            chex.assert_axis_dimension_gteq(y, 0, 3)
            assert_numeric_array(y)

            # Validate y is strictly ascending (required for all cases)
            assert_strictly_ascending(y)
            assert_probability_range(y)

            # Validate x is strictly ascending (only for pre-computed quantiles)
            if precomputed_quantiles:
                assert_strictly_ascending(x)
            else:
                x = jnp.quantile(x, y)

            # Build and return dictionary for dataclass construction
            keys = [f.name for f in fields(cls) if f.name[0] != "_"]
            values = [x, y, precomputed_quantiles]
            return dict(zip(keys, values))

        return MetalogBaseData(**_input_check(x, y, precomputed_quantiles))

    _from_factory: bool = struct.field(pytree_node=False, default=False)

    def __post_init__(self):
        """Post-initialization validation enforcing factory pattern."""
        if not self._from_factory:
            raise TypeError(
                "Class can only be created using `from_values`! Direct init is not allowed."
            )
        object.__setattr__(self, "_from_factory", False)
