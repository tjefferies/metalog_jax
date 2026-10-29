# Copyright: Travis Jefferies 2026
"""Core MetalogBase class for metalog distributions.

This module contains the MetalogBase class, which is the abstract base class for all
metalog distribution implementations. It provides the common interface and core
functionality shared by Metalog and SPTMetalog classes defined in metalog_jax.metalog.

The MetalogBase class includes:
    - Quantile function (PPF) computation
    - Probability density function (PDF) computation
    - Cumulative distribution function (CDF) computation
    - Statistical moments (mean, median, variance, std, mode)
    - Random variable sampling (rvs)
    - Serialization/deserialization (save, load, dumps, loads)
    - Visualization (plot)
    - Display of the fitted equation (str, repr, _repr_latex_, print)

See Also:
    metalog_jax.base.data: Input data container classes.
    metalog_jax.base.display: Text and LaTeX formatting of the fitted equation.
    metalog_jax.base.enums: Enumeration types for boundedness and fit methods.
    metalog_jax.base.parameters: Parameter configuration classes.
    metalog_jax.metalog: Metalog and SPTMetalog distribution classes.
"""

import json
from enum import Enum, unique
from functools import partial
from pathlib import Path
from typing import ClassVar, TextIO, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from flax import struct
from jax.experimental import checkify
from jax.scipy.special import logit

from metalog_jax.base.data import MetalogInputData
from metalog_jax.base.display import (
    DEFAULT_PRECISION,
    format_block,
    format_latex,
    format_line,
)
from metalog_jax.base.enums import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogPlotOptions,
)
from metalog_jax.base.parameters import (
    MetalogParameters,
    MetalogRandomVariableParameters,
    SPTMetalogParameters,
)
from metalog_jax.feasibility.analysis import (
    highest_mode,
    mean_and_variance,
    summary_stats,
)
from metalog_jax.feasibility.engine import (
    Engine,
    TermOrder,
    _horner,
    get_engine,
    term_basis,
)
from metalog_jax.utils import (
    DEFAULT_Y,
    DEFAULT_Y_FULL,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
    NotFittedError,
    assert_float_array,
    assert_probability_range,
    hdrprng,
)

T_MetalogBase = TypeVar("T_MetalogBase", bound="MetalogBase")


# The smallest positive normal float64, the lower end of the uniform draws
_TINY = float(np.finfo(np.float64).tiny)


# The rvs uniform draws are module level: defined inside rvs, each would be a new
# function on every call, and jax.jit would recompile it every time.
@partial(jax.jit, static_argnames=["size"])
def _hdr_uniforms(prng_params: HDRPRNGParameters, size: int) -> chex.Array:
    """Draw size uniforms in (0, 1) with the HDR PRNG.

    Args:
        prng_params: HDR parameters; draw i uses trial ``prng_params.trial + i``.
        size: Number of draws.

    Returns:
        chex.Array: Uniforms in the open interval (0, 1), shape (size,).
    """
    trials = prng_params.trial + jnp.arange(size, dtype=int)
    return jax.vmap(lambda trial: hdrprng(prng_params.replace(trial=trial)))(trials)


@partial(jax.jit, static_argnames=["size"])
def _jax_uniforms(key: chex.PRNGKey, size: int) -> chex.Array:
    """Draw size uniforms in (0, 1) with jax.random.

    Args:
        key: PRNG key.
        size: Number of draws.

    Returns:
        chex.Array: float64 uniforms in the open interval (0, 1), shape (size,).
    """
    # minval excludes 0, where ppf is undefined; uniform already excludes maxval
    return jax.random.uniform(key, (size,), dtype=jnp.float64, minval=_TINY, maxval=1.0)


@unique
class _LogitScale(Enum):
    """Numerical settings for working on the logit scale, t = logit(y).

    cdf, the bounded moments and mode evaluate the quantile function at logits
    rather than probabilities, which keeps both tails exact. ``@unique`` stops two
    members with equal values from becoming aliases of each other.

    Attributes:
        BOUND: The logit range [-BOUND, BOUND] that cdf's bisection brackets and
            the quadrature nodes stay inside: y from about 1e-304 to 1 - 1e-304,
            above the smallest normal float64.
        BISECTION_STEPS: Halvings of cdf's and mode's brackets. After 80, a
            bracket as wide as [-BOUND, BOUND] is narrower than the float64
            spacing anywhere inside it.
        QUADRATURE_STEP: The trapezoid step of the tanh-sinh rule. 1/16 gives 195
            nodes, and an error far below rounding for integrands analytic inside
            (0, 1).
        MODE_GRID_BOUND: mode searches logits in [-MODE_GRID_BOUND,
            MODE_GRID_BOUND], y within about 4e-18 of 0 and 1, for the highest
            density.
        MODE_GRID_POINTS: Points of that grid: 2 * 40 / 0.01 + 1, so 0.01 apart.
        SCALE_PROBABILITIES: The quartiles. The bounded moments divide X - mean by
            their spread, the interquartile range, before taking powers, so
            (X - mean)^4 doesn't overflow in a heavy tail whatever the units of X.
        LOWER_TAIL_X: x = y - 0.5 as y approaches 0. The scale polynomial there
            sets the power-law lower tail of a STRICTLY_UPPER_BOUND metalog.
        UPPER_TAIL_X: x = y - 0.5 as y approaches 1. The scale polynomial there
            sets the power-law upper tail of a STRICTLY_LOWER_BOUND metalog.
    """

    BOUND = 700.0
    BISECTION_STEPS = 80
    QUADRATURE_STEP = 1 / 16
    MODE_GRID_BOUND = 40.0
    MODE_GRID_POINTS = 8001
    SCALE_PROBABILITIES = (0.25, 0.75)
    LOWER_TAIL_X = -0.5
    UPPER_TAIL_X = 0.5


def _tanh_sinh_rule(step: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for integrals over (0, 1) on the logit scale.

    The tanh-sinh (double exponential) substitution y = sigmoid(pi sinh(r)) makes
    logit(y) = pi sinh(r), so the trapezoid rule in r with nodes t = pi sinh(r)
    integrates g(y) dy = g(sigmoid(t)) sigmoid(t) sigmoid(-t) dt. It converges
    exponentially for integrands analytic inside (0, 1), and nodes reach the
    tails without rounding y. The outermost nodes are the last inside
    |t| <= _LogitScale.BOUND, at about 675.

    Args:
        step: The trapezoid step in r.

    Returns:
        tuple[np.ndarray, np.ndarray]: Logit nodes t and weights w, so that
        sum(w * g(sigmoid(t))) approximates the integral of g over (0, 1).
    """
    r_max = np.arcsinh(_LogitScale.BOUND.value / np.pi)
    r = step * np.arange(-np.floor(r_max / step), np.floor(r_max / step) + 1)
    t = np.pi * np.sinh(r)
    # sigmoid(t) sigmoid(-t), computed in logs so it doesn't underflow early
    density = np.exp(-np.logaddexp(0.0, t) - np.logaddexp(0.0, -t))
    return t, step * np.pi * np.cosh(r) * density


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = _tanh_sinh_rule(
    _LogitScale.QUADRATURE_STEP.value
)

# mode searches this logit grid for the highest density, then refines it
_MODE_GRID = np.linspace(
    -_LogitScale.MODE_GRID_BOUND.value,
    _LogitScale.MODE_GRID_BOUND.value,
    _LogitScale.MODE_GRID_POINTS.value,
)


def _unbounded_ppf_at_logit(
    mu: chex.Array, s: chex.Array, t: chex.Numeric
) -> chex.Numeric:
    """Evaluate M(y) = mu(y - 0.5) + s(y - 0.5) logit(y) at logit(y) = t.

    Using t rather than y keeps full precision in both tails, where y rounds to
    0 or 1; t = -inf and +inf give the limits of M.

    Args:
        mu: Ascending coefficients of the location polynomial (Engine.split).
        s: Ascending coefficients of the scale polynomial (Engine.split).
        t: Logits of the probabilities.

    Returns:
        chex.Numeric: The unbounded quantile function M at sigmoid(t).
    """
    x = 0.5 * jnp.tanh(0.5 * t)  # y - 0.5
    return _horner(mu, x) + _horner(s, x) * t


@struct.dataclass
class MetalogBase:
    """Base class for metalog distribution implementations.

    This abstract base class provides the common interface and core functionality
    shared by all metalog distribution types (Metalog and SPTMetalog). It implements
    the fundamental metalog distribution operations including quantile function (PPF),
    probability density function (PDF), cumulative distribution function (CDF),
    and statistical moments.

    Subclasses:
        Metalog: Standard metalog fitted via regression (OLS, Ridge, ElasticNet, Lasso).
            Defined in metalog_jax.metalog.
        SPTMetalog: Symmetric Percentile Triplet metalog with closed-form coefficients.
            Defined in metalog_jax.metalog.

    Attributes:
        metalog_params: Configuration parameters defining the distribution type.
            Can be MetalogParameters (for Metalog) or SPTMetalogParameters
            (for SPTMetalog). Defined in metalog_jax.base.parameters.
        a: Coefficient vector defining the metalog quantile function. Shape is
            (num_terms,) where num_terms depends on the configuration. A list or
            tuple is converted to a float JAX array on construction.

    Example:
        MetalogBase is not instantiated directly. Use Metalog or SPTMetalog:

            >>> import jax.numpy as jnp
            >>> from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
            >>> from metalog_jax.base import MetalogInputData, MetalogParameters
            >>> from metalog_jax.metalog import fit
            >>> from metalog_jax.utils import DEFAULT_Y
            >>>
            >>> samples = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1, 10.5])
            >>> data = MetalogInputData.from_values(
            ...     samples, DEFAULT_Y, precomputed_quantiles=False
            ... )
            >>> params = MetalogParameters(
            ...     boundedness=MetalogBoundedness.UNBOUNDED,
            ...     method=MetalogFitMethod.Feasible,
            ...     num_terms=5, lower_bound=0.0, upper_bound=0.0
            ... )
            >>> m = fit(data, params)  # Returns a Metalog
            >>> isinstance(m, MetalogBase)
            True

    See Also:
        metalog_jax.metalog.Metalog: Standard metalog distribution class.
        metalog_jax.metalog.SPTMetalog: SPT metalog distribution class.
        metalog_jax.base.parameters.MetalogParameters: Configuration for Metalog.
        metalog_jax.base.parameters.SPTMetalogParameters: Configuration for SPTMetalog.
    """

    metalog_params: SPTMetalogParameters | MetalogParameters
    a: chex.Numeric

    # The parameter class that load and loads rebuild; SPTMetalog overrides it
    _params_type: ClassVar[type[MetalogParameters | SPTMetalogParameters]] = (
        MetalogParameters
    )

    def __post_init__(self) -> None:
        """Coerce a list or tuple of coefficients to a float JAX array.

        Every method assumes ``a`` is an array (``jnp.result_type``, ``.shape``,
        ``.tolist()``, ``@``), so a plain sequence such as hand-copied coefficients
        would otherwise fail deep inside JAX (issue #9). Only lists and tuples are
        converted: pytree unflattening under ``jit``/``vmap``/``tree.map`` re-runs
        ``__init__`` with tracers, ``ShapeDtypeStruct`` or ``None`` as ``a``, and
        those must pass through untouched.
        """
        if isinstance(self.a, (list, tuple)):
            object.__setattr__(self, "a", jnp.asarray(self.a, dtype=float))

    def __eq__(self, other: object) -> bool:
        """Check equality between two metalog distribution instances.

        Compares the metalog parameters and coefficient arrays to determine
        equality. For JAX arrays, uses `jnp.allclose` for numerical comparison.

        Args:
            other: The object to compare against.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if not isinstance(other, MetalogBase):
            return False

        # Compare metalog_params attributes
        if self.metalog_params != other.metalog_params:
            return False

        # Compare coefficient arrays using allclose for numerical comparison
        # Check shape first, then values
        if self.a.shape != other.a.shape:
            return False

        return bool(jnp.allclose(self.a, other.a))

    def __str__(self) -> str:
        """Show the model and its quantile function on one line.

        The coefficients and bounds have 6 significant digits. A traced or batched
        model shows a one-line summary instead.

        Returns:
            str: The one-line equation, such as
                ``Metalog(3 terms, OLS, unbounded): Q(y) = 1.5 - 0.25 ln(y/(1-y)) ...``,
                or a one-line summary if the model is traced or batched.
        """
        return format_line(self)

    def __repr__(self) -> str:
        """Show the model and its quantile function on one line, at full precision.

        The same line as ``str``, with each number written as Python's shortest
        ``repr(float)``, so the coefficients parse back to ``a`` exactly. Each
        subclass defines ``__repr__`` to call this one, because
        ``flax.struct.dataclass`` would otherwise generate its own.

        Returns:
            str: The one-line equation, or a one-line summary if the model is traced
                or batched.
        """
        return format_line(self, precision=None)

    def _repr_latex_(self) -> str | None:
        """Show the quantile function as typeset math in Jupyter and the docs.

        Up to four terms stay on one line. Longer models wrap after every four
        terms, because MathJax doesn't break long equations.

        Returns:
            str | None: The LaTeX, or None if the model is traced or batched, which
                tells IPython to show ``repr`` instead.
        """
        return format_latex(self)

    def print(
        self,
        *,
        header: bool = True,
        q: bool = True,
        m: bool = True,
        verbose: bool = False,
        precision: int = DEFAULT_PRECISION,
        file: TextIO | None = None,
    ) -> None:
        """Print the quantile function, one term per line.

        For a bounded or semi-bounded model, Q(y) maps M(y) back to the support,
        and the options choose which lines appear. An unbounded model's Q(y) is
        M(y), so its terms print once, labelled Q(y) (or M(y) when ``q`` is False).
        A traced or batched model prints a one-line summary instead. The builtin
        ``print(m)`` still prints ``str(m)``.

        Args:
            header: Print a first line describing the model, such as ``Metalog: 5
                terms, OLS, bounded on (0, 15)``, and indent the equations under it.
                Without it, the equations start at the left margin.
            q: Print the quantile function Q(y).
            m: Print M(y), one term per line. For an unbounded model, ``m=False``
                keeps the terms on Q(y)'s line.
            verbose: Write Q(y) with M(y)'s terms in place of M(y), so it reads on
                its own. A bounded Q(y) uses M(y) twice, so the terms appear twice.
            precision: Significant digits of the coefficients and bounds.
            file: Where to write, such as an open file or ``io.StringIO``; standard
                output by default.

        Raises:
            ValueError: If ``q`` and ``m`` are both False, ``verbose`` is True while
                ``q`` is False, or ``precision`` isn't a positive integer.
        """  # noqa: DOC502
        text = format_block(
            self, header=header, q=q, m=m, verbose=verbose, precision=precision
        )
        print(text, file=file)

    @property
    def boundedness(self) -> MetalogBoundedness:
        """Get the boundedness type of the metalog distribution.

        Returns:
            MetalogBoundedness: The boundedness type (UNBOUNDED, STRICTLY_LOWER_BOUND,
                STRICTLY_UPPER_BOUND, or BOUNDED).
        """
        return self.metalog_params.boundedness

    @property
    def lower_bound(self) -> chex.Scalar:
        """Get the lower bound of the metalog distribution.

        Returns:
            chex.Scalar: The lower bound value. Only meaningful when boundedness
                is STRICTLY_LOWER_BOUND or BOUNDED.
        """
        return self.metalog_params.lower_bound

    @property
    def upper_bound(self) -> chex.Scalar:
        """Get the upper bound of the metalog distribution.

        Returns:
            chex.Scalar: The upper bound value. Only meaningful when boundedness
                is STRICTLY_UPPER_BOUND or BOUNDED.
        """
        return self.metalog_params.upper_bound

    @property
    def num_terms(self) -> int:
        """Get the number of terms in the metalog expansion.

        Returns:
            int: The number of terms (coefficients) in the metalog quantile function.
        """
        return self.metalog_params.num_terms

    @property
    def method(self) -> MetalogFitMethod:
        """Get the fit method used for the metalog distribution.

        Returns:
            MetalogFitMethod: The regression method (OLS or Lasso).
        """
        return self.metalog_params.method

    @property
    def term_order(self) -> TermOrder:
        """Get the order of the basis terms, and so of the coefficients ``a``.

        Returns:
            TermOrder: The parameters' ``term_order``. An SPT metalog has 3 terms,
                where the orders agree, so it reports TermOrder.KEELIN_2016.
        """
        return getattr(self.metalog_params, "term_order", TermOrder.KEELIN_2016)

    def _assert_has_coefficients(self) -> None:
        """Check that the distribution has been fitted with coefficients.

        This is a lightweight check used internally by methods like ``pdf`` to
        avoid recursion when ``assert_fitted`` calls ``pdf`` for validation.

        Raises:
            NotFittedError: If the distribution has not been fitted (no coefficients).
        """
        if len(self.a) < 1:
            raise NotFittedError(
                "Metalog has not been fit yet! Distribution must be fit "
                "before you can call methods on the distribution."
            )

    def assert_fitted(self) -> None:
        """Validate that the metalog distribution has been fitted and is feasible.

        Performs two validation checks:

        1. Verifies that coefficients exist (distribution has been fit).
        2. Computes the PDF at standard quantile points to verify feasibility,
           which internally calls ``assert_fit_valid`` to ensure all PDF values
           are strictly positive.

        This method is vmap-compatible through the use of ``jax.experimental.checkify``
        in the underlying PDF validation.

        Raises:
            NotFittedError: If the distribution has not been fitted (no coefficients).
            checkify.JaxRuntimeError: If the fitted distribution produces non-positive
                PDF values, indicating an infeasible fit.
        """  # noqa: DOC502
        self._assert_has_coefficients()
        self.pdf(DEFAULT_Y)

    @staticmethod
    def assert_fit_valid(m: chex.Numeric) -> None:
        """Validate that PDF values are strictly positive.

        Checks that all probability density values are positive, which is a
        necessary condition for a valid probability distribution. A metalog
        fit that produces non-positive PDF values is theoretically infeasible.

        This method is vmap-compatible by using ``jax.experimental.checkify``
        for runtime error checking inside JAX transforms.

        Args:
            m: Array of PDF values to validate. All elements must be > 0.

        Raises:
            checkify.JaxRuntimeError: If any PDF value is less than or equal
                to zero, indicating an infeasible metalog fit.
        """  # noqa: DOC502
        checkify.check(
            jnp.all(m > 0),
            "Metalog PDF returned a non-positive value! "
            "This is a theoretically invalid PDF.",
        )

    @staticmethod
    @jax.jit
    def strictly_lower_bound_quantile_transform(
        x: chex.Numeric, lower_bound: chex.Scalar
    ) -> chex.Numeric:
        """Apply logarithmic transformation for strictly lower-bounded quantiles.

        Transforms quantiles from (lower_bound, inf) to (-inf, inf) using log(x - L).

        Args:
            x: Quantile values to transform. Must be > lower_bound.
            lower_bound: The strict lower bound of the distribution.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        return jnp.log(x - lower_bound)

    @staticmethod
    @jax.jit
    def strictly_upper_bound_quantile_transform(
        x: chex.Numeric, upper_bound: chex.Scalar
    ) -> chex.Numeric:
        """Apply negative logarithmic transformation for strictly upper-bounded quantiles.

        Transforms quantiles from (-inf, upper_bound) to (-inf, inf) using -log(U - x).

        Args:
            x: Quantile values to transform. Must be < upper_bound.
            upper_bound: The strict upper bound of the distribution.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        return -jnp.log(upper_bound - x)

    @staticmethod
    @jax.jit
    def bounded_quantile_transform(
        x: chex.Numeric,
        lower_bound: chex.Scalar,
        upper_bound: chex.Scalar,
    ) -> chex.Numeric:
        """Apply logit-like transformation for fully bounded quantiles.

        Transforms quantiles from (lower_bound, upper_bound) to (-inf, inf)
        using log((x - L) / (U - x)).

        Args:
            x: Quantile values to transform. Must be in (lower_bound, upper_bound).
            lower_bound: The strict lower bound of the distribution.
            upper_bound: The strict upper bound of the distribution.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        return jnp.log((x - lower_bound) / (upper_bound - x))

    @staticmethod
    def quantile_transform(
        x: chex.Numeric,
        boundedness: MetalogBoundedness,
        lower_bound: chex.Scalar,
        upper_bound: chex.Scalar,
    ) -> chex.Numeric:
        """Transform quantiles to the scale on which the metalog is unbounded.

        Applies the transform that ``boundedness`` selects: none for UNBOUNDED,
        then strictly_lower_bound_quantile_transform,
        strictly_upper_bound_quantile_transform or bounded_quantile_transform.
        ``boundedness`` is a static parameter field, so this dispatches in Python
        and works inside ``jax.jit`` and ``jax.vmap``.

        Args:
            x: Quantile values inside the distribution's support.
            boundedness: Boundedness type of the distribution.
            lower_bound: Lower bound, used by STRICTLY_LOWER_BOUND and BOUNDED.
            upper_bound: Upper bound, used by STRICTLY_UPPER_BOUND and BOUNDED.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.
        """
        if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            return MetalogBase.strictly_lower_bound_quantile_transform(x, lower_bound)
        if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            return MetalogBase.strictly_upper_bound_quantile_transform(x, upper_bound)
        if boundedness == MetalogBoundedness.BOUNDED:
            return MetalogBase.bounded_quantile_transform(x, lower_bound, upper_bound)
        return x

    @staticmethod
    def inverse_quantile_transform(
        m: chex.Numeric,
        boundedness: MetalogBoundedness,
        lower_bound: chex.Scalar,
        upper_bound: chex.Scalar,
    ) -> chex.Numeric:
        """Map values of the unbounded metalog back to the distribution's support.

        The inverse of quantile_transform (Keelin 2016, Section 4): ``L + e^M``
        for STRICTLY_LOWER_BOUND, ``U - e^(-M)`` for STRICTLY_UPPER_BOUND and
        ``(L + U e^M) / (1 + e^M)`` for BOUNDED.

        Args:
            m: Values of the unbounded metalog quantile function.
            boundedness: Boundedness type of the distribution.
            lower_bound: Lower bound, used by STRICTLY_LOWER_BOUND and BOUNDED.
            upper_bound: Upper bound, used by STRICTLY_UPPER_BOUND and BOUNDED.

        Returns:
            chex.Numeric: Quantile values in the distribution's support.
        """
        if boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            return lower_bound + jnp.exp(m)
        if boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            return upper_bound - jnp.exp(-m)
        if boundedness == MetalogBoundedness.BOUNDED:
            return (lower_bound + upper_bound * jnp.exp(m)) / (1 + jnp.exp(m))
        return m

    @staticmethod
    @jax.jit
    def delta_term(x: chex.Numeric) -> chex.Numeric:
        """Compute the delta term (x - 0.5) used in metalog basis functions.

        Args:
            x: Probability values in (0, 1).

        Returns:
            chex.Numeric: The delta term values (x - 0.5).
        """
        return x - 0.5

    @staticmethod
    @jax.jit
    def log_term(x: chex.Numeric) -> chex.Numeric:
        """Compute the logit transformation log(x / (1 - x)).

        Args:
            x: Probability values in (0, 1).

        Returns:
            chex.Numeric: The logit-transformed values.
        """
        return jnp.log(x / (1 - x))

    @staticmethod
    @jax.jit
    def product_term(x: chex.Numeric) -> chex.Numeric:
        """Compute the product term x * (1 - x).

        This term appears in the derivative of the logit function and is used
        in PDF computation.

        Args:
            x: Probability values in (0, 1).

        Returns:
            chex.Numeric: The product term values x * (1 - x).
        """
        return x * (1 - x)

    @staticmethod
    def get_quantiles(
        data: MetalogInputData,
        metalog_params: MetalogParameters,
    ) -> chex.Numeric:
        """Extract and transform quantiles from validated input data.

        Applies the appropriate boundedness transformation to convert quantiles
        from their natural domain to the unbounded (-inf, inf) scale used
        internally by the metalog.

        Args:
            data: Validated input data containing quantile values (x) and
                probability levels (y).
            metalog_params: Configuration parameters specifying boundedness
                type and bound values.

        Returns:
            chex.Numeric: Transformed quantile values on the unbounded scale.

        Raises:
            chex.AssertionError: If the number of data points is less than num_terms.
        """  # noqa: DOC502
        x = data.x
        chex.assert_axis_dimension_gteq(x, 0, metalog_params.num_terms)
        return MetalogBase.quantile_transform(
            x,
            metalog_params.boundedness,
            metalog_params.lower_bound,
            metalog_params.upper_bound,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["num_terms", "term_order"])
    def _get_target(
        y: chex.Numeric,
        num_terms: int,
        term_order: TermOrder = TermOrder.KEELIN_2016,
    ) -> chex.Numeric:
        """Construct the design matrix of metalog basis functions from input data.

        Builds the matrix Y where each column corresponds to a metalog basis function
        evaluated at the probability levels y, in the order ``term_basis`` gives.
        Both orders start 1, L, xL, x, x^2, x^2 L, with x = y - 0.5 and
        L = log(y / (1-y)); later terms are powers of x, with and without L.

        Args:
            y: Probability levels in (0, 1), shape (m,).
            num_terms: Number of metalog terms (basis functions) to include.
            term_order: The order of the basis terms.

        Returns:
            chex.Numeric: Design matrix of shape (m, num_terms).
        """
        delta = y - 0.5
        log = jnp.log(y / (1 - y))
        columns = []
        for power, has_logit in term_basis(num_terms, term_order):
            if power == 0:
                column = log if has_logit else jnp.ones(y.shape[0])
            else:
                column = delta if power == 1 else jnp.power(delta, power)
                if has_logit:
                    column = log * column
            columns.append(column)
        return jnp.stack(columns, axis=1)

    @staticmethod
    def get_target(
        data: MetalogInputData, metalog_params: MetalogParameters
    ) -> chex.Numeric:
        """Construct the design matrix from input data and parameters.

        Convenience wrapper around _get_target that extracts the probability
        levels, num_terms and term_order from the input objects.

        Args:
            data: Validated input data containing probability levels (y).
            metalog_params: Configuration parameters specifying num_terms and
                term_order.

        Returns:
            chex.Numeric: Design matrix of shape (m, num_terms).
        """
        return MetalogBase._get_target(
            y=data.y,
            num_terms=metalog_params.num_terms,
            term_order=metalog_params.term_order,
        )

    @partial(jax.jit, static_argnames=["num_terms"])
    def _ppf(self, x: chex.Numeric, num_terms: int) -> chex.Numeric:
        """Compute the unbounded metalog quantile function (internal PPF computation).

        Evaluates the metalog quantile function on the unbounded scale before
        applying any boundedness transformations. This is the core computation
        M(y) = sum(a_k * basis_k(y)).

        Args:
            x: Probability values in (0, 1) at which to evaluate the quantile function.
            num_terms: Number of terms in the metalog expansion.

        Returns:
            chex.Numeric: Quantile values on the unbounded (-inf, inf) scale.
        """
        delta_term = self.delta_term(x)
        log_term = self.log_term(x)
        # Every order starts with 1 and L, so a[0] + a[1] L, then a_j x^p (times L)
        ppf = self.a[0] + (self.a[1] * log_term)
        for j, (power, has_logit) in enumerate(
            term_basis(num_terms, self.term_order)[2:], start=2
        ):
            term = self.a[j] * (
                delta_term if power == 1 else jnp.power(delta_term, power)
            )
            ppf += term * log_term if has_logit else term
        return ppf

    def ppf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the percent point function (quantile function) of the distribution.

        The PPF is the inverse of the CDF: for a probability p, ppf(p) returns
        the value x such that P(X <= x) = p.

        Args:
            x: Probability values in (0, 1) at which to evaluate the quantile function.
                Can be a scalar or array.

        Returns:
            chex.Numeric: Quantile values corresponding to the input probabilities.
                Same shape as input.

        Raises:
            ValueError: If x contains values outside (0, 1).
            NotFittedError: If the distribution has not been fitted.
        """  # noqa: DOC502
        assert_float_array(x)
        assert_probability_range(x)
        self.assert_fitted()

        # Evaluate the basis in the coefficients' precision, as pdf() does, so
        # float32 probabilities don't round the basis to float32
        x = jnp.asarray(x, dtype=jnp.result_type(x, self.a))
        ppf = self._ppf(x, self.num_terms)
        return self.inverse_quantile_transform(
            ppf, self.boundedness, self.lower_bound, self.upper_bound
        )

    @partial(jax.jit, static_argnames=["num_terms"])
    def _pdf(self, x: chex.Numeric, num_terms: int) -> chex.Numeric:
        """Compute the unbounded metalog density m(y) = 1 / M'(y) (internal PDF).

        A method compiled once per model structure, like ``_ppf``. Defined inside
        ``pdf``, it would be a new function on every call, so ``jax.jit`` would
        recompile it every time.

        Args:
            x: Probability values in (0, 1) at which to evaluate the density.
            num_terms: Number of terms in the metalog expansion.

        Returns:
            chex.Numeric: Density values on the unbounded scale.
        """
        product_term = self.product_term(x)
        delta_term = self.delta_term(x)
        log_term = self.log_term(x)

        if num_terms == 2:
            return product_term / self.a[1]

        # M'(y): d/dy L = 1 / (y(1-y)), d/dy x^p = p x^(p-1), and
        # d/dy x^p L = x^p / (y(1-y)) + p x^(p-1) L
        m_inv = self.a[1] / product_term
        for j, (power, has_logit) in enumerate(
            term_basis(num_terms, self.term_order)[2:], start=2
        ):
            if has_logit and power == 1:
                m_inv += self.a[j] * ((delta_term / product_term) + log_term)
            elif has_logit:
                m_inv += self.a[j] * (
                    (jnp.power(delta_term, power) / product_term)
                    + power * jnp.power(delta_term, power - 1) * log_term
                )
            elif power == 1:
                m_inv += self.a[j]
            else:
                m_inv += power * self.a[j] * jnp.power(delta_term, power - 1)
        return 1.0 / m_inv

    def pdf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the probability density function of the metalog distribution.

        The PDF is computed as the reciprocal of the derivative of the quantile
        function (PPF), with appropriate transformations for bounded distributions.

        Args:
            x: Probability values in (0, 1) at which to evaluate the PDF.
                Can be a scalar or array.

        Returns:
            chex.Numeric: Probability density values. Same shape as input.

        Raises:
            ValueError: If x contains values outside (0, 1).
            NotFittedError: If the distribution has not been fitted.
            checkify.JaxRuntimeError: If the PDF contains non-positive values,
                indicating an infeasible metalog fit.
        """  # noqa: DOC502
        assert_float_array(x)
        assert_probability_range(x)
        self._assert_has_coefficients()
        # Evaluate the basis in the coefficients' precision: float32 rounding of
        # log(y / (1 - y)) times large coefficients can flip the sign of a small but
        # positive density (e.g. float32 probabilities with a float64 fit).
        x = jnp.asarray(x, dtype=jnp.result_type(x, self.a))
        num_terms = self.num_terms
        m = self._pdf(x, num_terms)
        if self.boundedness != MetalogBoundedness.UNBOUNDED:
            M = self._ppf(x, num_terms)
            if self.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
                m = m * jnp.exp(-M)
            elif self.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
                m = m * jnp.exp(M)
            elif self.boundedness == MetalogBoundedness.BOUNDED:
                m = (m * (1 + jnp.exp(M)) ** 2) / (
                    (self.upper_bound - self.lower_bound) * jnp.exp(M)
                )
        self.assert_fit_valid(m=m)
        return m

    def _logit_parts(self) -> tuple[Engine, chex.Array, chex.Array]:
        """The engine and the location and scale polynomials of the coefficients.

        Returns:
            tuple[Engine, chex.Array, chex.Array]: ``(engine, mu, s)``, so that
            M(y) = mu(y - 0.5) + s(y - 0.5) logit(y).
        """
        engine = get_engine(int(self.num_terms), self.term_order)
        mu, s = engine.split(jnp.asarray(self.a, jnp.float64))
        return engine, mu, s

    def _from_unbounded(self, m: chex.Numeric) -> chex.Numeric:
        """Map the unbounded metalog M to the support, without overflowing.

        The same back-transform as inverse_quantile_transform, with BOUNDED written
        as L + (U - L) sigmoid(M) so that a large M gives U rather than inf / inf.

        Args:
            m: Values of the unbounded quantile function M.

        Returns:
            chex.Numeric: Quantile values in the distribution's support.
        """
        lower, upper = self.lower_bound, self.upper_bound
        if self.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            return lower + jnp.exp(m)
        if self.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            return upper - jnp.exp(-m)
        if self.boundedness == MetalogBoundedness.BOUNDED:
            return lower + (upper - lower) * jax.nn.sigmoid(m)
        return m

    def _log_dx_dm(self, m: chex.Numeric) -> chex.Numeric:
        """Log of the back-transform's slope dX/dM at M = m.

        Args:
            m: Values of the unbounded quantile function M.

        Returns:
            chex.Numeric: log(dX/dM), which is 0 for UNBOUNDED.
        """
        if self.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            return m
        if self.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            return -m
        if self.boundedness == MetalogBoundedness.BOUNDED:
            width = jnp.log(self.upper_bound - self.lower_bound)
            return width + jax.nn.log_sigmoid(m) + jax.nn.log_sigmoid(-m)
        return jnp.zeros_like(m)

    @jax.jit
    def _cdf_logit(self, q: chex.Numeric) -> chex.Numeric:
        """Return logit(cdf(q)), by inverting the quantile function.

        A feasible metalog's M is non-decreasing in t = logit(y), so bisection over
        t in [-_LogitScale.BOUND, _LogitScale.BOUND] converges to the t with M(t)
        equal to q on the unbounded scale. Working with t keeps both tails exact.

        Args:
            q: Quantile values, in the coefficients' float dtype.

        Returns:
            chex.Numeric: logit(P(X <= q)): -inf at or below the support, +inf at
            or above it, and NaN where q is NaN.
        """
        _, mu, s = self._logit_parts()
        z = self.quantile_transform(
            q, self.boundedness, self.lower_bound, self.upper_bound
        )

        def halve(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            below = _unbounded_ppf_at_logit(mu, s, mid) < z
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        bound = _LogitScale.BOUND.value
        start = (jnp.full_like(z, -bound), jnp.full_like(z, bound))
        lo, hi = jax.lax.fori_loop(0, _LogitScale.BISECTION_STEPS.value, halve, start)
        t = jnp.where(jnp.isnan(z), jnp.nan, 0.5 * (lo + hi))
        t = jnp.where(z == -jnp.inf, -jnp.inf, jnp.where(z == jnp.inf, jnp.inf, t))
        # Outside the support the transform gives NaN; the CDF is 0 or 1 there
        if self.boundedness in (
            MetalogBoundedness.STRICTLY_LOWER_BOUND,
            MetalogBoundedness.BOUNDED,
        ):
            t = jnp.where(q <= self.lower_bound, -jnp.inf, t)
        if self.boundedness in (
            MetalogBoundedness.STRICTLY_UPPER_BOUND,
            MetalogBoundedness.BOUNDED,
        ):
            t = jnp.where(q >= self.upper_bound, jnp.inf, t)
        return t

    def _checked_cdf_logit(self, q: chex.Numeric) -> chex.Numeric:
        """Check the fit, then return _cdf_logit of q as a float array.

        Args:
            q: Quantile values: a Python or NumPy number, or an array of any shape.

        Returns:
            chex.Numeric: logit(P(X <= q)), with the shape of q.

        Raises:
            NotFittedError: If the distribution has not been fitted.
            checkify.JaxRuntimeError: If the fit is infeasible.
        """  # noqa: DOC502
        self.assert_fitted()
        q = jnp.asarray(q, dtype=jnp.result_type(q, self.a, float))
        return self._cdf_logit(q)

    def cdf(self, q: chex.Numeric) -> chex.Numeric:
        """Compute the cumulative distribution function.

        Returns P(X <= q) for the fitted metalog distribution, by inverting the
        quantile function: bisection on the logit scale finds the y with
        ppf(y) = q, so cdf(ppf(y)) = y to rounding over all of (0, 1).

        Args:
            q: Quantile values at which to evaluate the CDF: a Python or NumPy
                number, or an array of any shape, including a 0-d array.

        Returns:
            chex.Numeric: Probabilities in [0, 1], with the shape of q: 0 at or
            below the support's lower bound, and 1 at or above its upper bound.

        Raises:
            NotFittedError: If the distribution has not been fitted.
            checkify.JaxRuntimeError: If the fit is infeasible.
        """  # noqa: DOC502
        return jax.nn.sigmoid(self._checked_cdf_logit(q))

    def rvs(self, rv_params: MetalogRandomVariableParameters) -> chex.Numeric:
        """Generate random variates from the fitted metalog distribution.

        Samples are generated by inverse transform sampling: uniform draws u in
        the open interval (0, 1), returned as ppf(u). With HDRPRNGParameters,
        draw i uses HDR trial ``trial + i``, so the default ``trial=1`` draws
        trials 1 to size, and ``trial=size + 1`` continues the same stream.

        Args:
            rv_params: Parameters controlling random variate generation, including
                PRNG configuration (HDR or JAX-based) and sample size.

        Returns:
            chex.Numeric: Random samples from the distribution, shape (size,).

        Raises:
            chex.AssertionError: If size is not positive or exceeds max_draws.
            checkify.JaxRuntimeError: If the PDF is infeasible (non-positive values).
        """  # noqa: DOC502
        prng_params = rv_params.prng_params
        size = rv_params.size
        chex.assert_scalar_positive(size)
        chex.assert_scalar_non_negative(rv_params.max_draws - size)
        if isinstance(prng_params, HDRPRNGParameters):
            u = _hdr_uniforms(prng_params, size)
        elif isinstance(prng_params, JaxUniformDistributionParameters):
            u = _jax_uniforms(jax.random.PRNGKey(prng_params.seed), size)

        self.pdf(DEFAULT_Y_FULL)  # asserts feasibility
        return self.ppf(u)

    def logpdf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the natural logarithm of the probability density function.

        Args:
            x: Probability values in (0, 1) at which to evaluate log(PDF).

        Returns:
            chex.Numeric: Log-density values. Same shape as input.
        """
        return jnp.log(self.pdf(x))

    def logppf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the natural logarithm of the percent point function.

        Args:
            x: Probability values in (0, 1) at which to evaluate log(PPF).

        Returns:
            chex.Numeric: Log-quantile values. Same shape as input.
        """
        return jnp.log(self.ppf(x))

    def sf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the survival function (complement of CDF).

        The survival function is S(x) = 1 - F(x) = P(X > x).

        Args:
            x: Quantile values at which to evaluate the survival function.

        Returns:
            chex.Numeric: Survival probabilities in [0, 1], with the shape of x.
            Computed from the complementary logit, so it stays accurate in the upper
            tail, where 1 - cdf(x) would round to 0.
        """
        return jax.nn.sigmoid(-self._checked_cdf_logit(x))

    def logsf(self, x: chex.Numeric) -> chex.Numeric:
        """Compute the natural logarithm of the survival function.

        Args:
            x: Quantile values at which to evaluate log(SF).

        Returns:
            chex.Numeric: Log-survival probabilities, with the shape of x.
        """
        return jax.nn.log_sigmoid(-self._checked_cdf_logit(x))

    def isf(self, q: chex.Numeric) -> chex.Numeric:
        """Compute the inverse survival function (inverse complementary CDF).

        Returns the value x such that P(X > x) = q, equivalent to ppf(1 - q).

        Args:
            q: Probability values in (0, 1).

        Returns:
            chex.Numeric: Quantile values corresponding to survival probabilities.
        """
        return self.ppf(1 - q)

    def _has_exact_moments(self) -> bool:
        """Whether closed-form moments apply (unbounded metalogs only)."""
        return self.boundedness == MetalogBoundedness.UNBOUNDED

    def _exact_mean_and_variance(self) -> tuple[chex.Scalar, chex.Scalar]:
        """Exact mean and variance of an unbounded metalog."""
        engine = get_engine(int(self.num_terms), self.term_order)
        return mean_and_variance(engine, jnp.asarray(self.a, jnp.float64))

    def _exact_shape(self) -> tuple[chex.Scalar, chex.Scalar]:
        """Exact skewness and (Pearson) kurtosis of an unbounded metalog."""
        engine = get_engine(int(self.num_terms), self.term_order)
        stats = summary_stats(engine, jnp.asarray(self.a, jnp.float64))
        return stats["skewness"], stats["kurtosis"]

    @jax.jit
    def _quadrature_moments(
        self,
    ) -> tuple[chex.Scalar, chex.Scalar, chex.Scalar, chex.Scalar]:
        """Mean, variance, skewness and kurtosis by quadrature on the logit scale.

        E[g(X)] is the integral of g(Q(y)) over (0, 1), here by the tanh-sinh rule
        of _tanh_sinh_rule. A semi-bounded metalog has a power-law tail: as y
        approaches the unbounded end, M(y) grows like s_end logit(y), where s_end is
        the scale polynomial at that end, so X behaves like a Pareto variable with
        index 1 / s_end, and E[|X|^k] is finite only when k s_end < 1. Moments
        that don't exist follow scipy.stats: the mean is -inf or inf, the variance
        inf, and the skewness and kurtosis NaN.

        Returns:
            tuple: ``(mean, variance, skewness, kurtosis)``, kurtosis as Pearson's.
        """
        _, mu, s = self._logit_parts()
        x = self._from_unbounded(_unbounded_ppf_at_logit(mu, s, _QUADRATURE_NODES))
        mean = jnp.sum(_QUADRATURE_WEIGHTS * x)
        # Centre and scale by the interquartile range before taking powers, so
        # (X - mean)^4 doesn't overflow in a heavy tail whatever the units of X
        quartile_logits = logit(jnp.asarray(_LogitScale.SCALE_PROBABILITIES.value))
        quartiles = self._from_unbounded(
            _unbounded_ppf_at_logit(mu, s, quartile_logits)
        )
        scale = quartiles[1] - quartiles[0]
        c = (x - mean) / scale
        m2, m3, m4 = (jnp.sum(_QUADRATURE_WEIGHTS * c**p) for p in (2, 3, 4))

        if self.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
            upper_end = jnp.asarray(_LogitScale.UPPER_TAIL_X.value)
            tail, infinite_mean = _horner(s, upper_end), jnp.inf
        elif self.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
            lower_end = jnp.asarray(_LogitScale.LOWER_TAIL_X.value)
            tail, infinite_mean = _horner(s, lower_end), -jnp.inf
        else:  # BOUNDED: every moment is finite
            tail, infinite_mean = jnp.asarray(0.0), jnp.nan

        def exists(order: int) -> chex.Array:
            return order * tail < 1

        return (
            jnp.where(exists(1), mean, infinite_mean),
            jnp.where(exists(2), m2 * scale**2, jnp.inf),
            jnp.where(exists(3), m3 / m2**1.5, jnp.nan),
            jnp.where(exists(4), m4 / m2**2, jnp.nan),
        )

    def _bounded_moments(
        self,
    ) -> tuple[chex.Scalar, chex.Scalar, chex.Scalar, chex.Scalar]:
        """Check the fit, then return _quadrature_moments.

        Returns:
            tuple: ``(mean, variance, skewness, kurtosis)``.

        Raises:
            NotFittedError: If the distribution has not been fitted.
            checkify.JaxRuntimeError: If the fit is infeasible.
        """  # noqa: DOC502
        self.assert_fitted()
        return self._quadrature_moments()

    @property
    def mean(self) -> chex.Scalar:
        """Compute the mean (expected value) of the metalog distribution.

        Exact for unbounded metalogs, via the closed-form integrals of Baucells,
        Chrisman, Keelin and Xu (2025) (Lemma 1, Proposition 3). Bounded and
        semi-bounded metalogs have no closed form after the log/logit
        back-transform, so their mean is the integral of the quantile function,
        by tanh-sinh quadrature on the logit scale. A semi-bounded metalog's mean
        is infinite when its tail is too heavy (see ``var``).

        Returns:
            chex.Scalar: The mean of the distribution.

        References:
            Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
            Properties of the Metalog Distribution. Darden Business School Working
            Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
        """
        if self._has_exact_moments():
            return self._exact_mean_and_variance()[0]
        return self._bounded_moments()[0]

    @property
    def median(self) -> chex.Scalar:
        """Compute the median of the metalog distribution.

        The median is the 50th percentile, computed exactly via ppf(0.5).

        Returns:
            chex.Scalar: The median of the distribution.
        """
        return self.ppf(0.5)

    @property
    def var(self) -> chex.Scalar:
        """Compute the variance of the metalog distribution.

        Exact for unbounded metalogs (see ``mean``), and by quadrature for bounded
        and semi-bounded metalogs. A semi-bounded metalog's open end has a
        power-law tail: its k-th moment is finite only when k times the scale
        polynomial at that end, s(0.5) for STRICTLY_LOWER_BOUND or s(-0.5) for
        STRICTLY_UPPER_BOUND, is below 1. As in scipy.stats, a mean that doesn't
        exist is -inf or inf, a variance inf, and a skewness or kurtosis NaN.

        Returns:
            chex.Scalar: The variance of the distribution.
        """
        if self._has_exact_moments():
            return self._exact_mean_and_variance()[1]
        return self._bounded_moments()[1]

    @property
    def std(self) -> chex.Scalar:
        """Compute the standard deviation of the metalog distribution.

        Exact for unbounded metalogs (see ``mean``), and by quadrature for bounded
        and semi-bounded metalogs (see ``var``).

        Returns:
            chex.Scalar: The standard deviation of the distribution.
        """
        if self._has_exact_moments():
            return jnp.sqrt(self._exact_mean_and_variance()[1])
        return jnp.sqrt(self._bounded_moments()[1])

    @property
    def skewness(self) -> chex.Scalar:
        """Compute the skewness of the metalog distribution.

        Skewness is the third standardized central moment, E[(X - mean)^3] / std^3.
        Exact for unbounded metalogs (see ``mean``), and by quadrature for bounded
        and semi-bounded metalogs (see ``var``).

        Returns:
            chex.Scalar: The skewness of the distribution.
        """
        if self._has_exact_moments():
            return self._exact_shape()[0]
        return self._bounded_moments()[2]

    @property
    def kurtosis(self) -> chex.Scalar:
        """Compute the kurtosis of the metalog distribution.

        Pearson kurtosis, E[(X - mean)^4] / var^2: 3 for a normal distribution and
        4.2 for the logistic. Subtract 3 for excess kurtosis (scipy's default
        convention). Exact for unbounded metalogs (see ``mean``), and by quadrature
        for bounded and semi-bounded metalogs (see ``var``).

        Returns:
            chex.Scalar: The kurtosis of the distribution.
        """
        if self._has_exact_moments():
            return self._exact_shape()[1]
        return self._bounded_moments()[3]

    @jax.jit
    def _density_mode(self) -> chex.Scalar:
        """The quantile with the highest density, found on the logit scale.

        The density of X at y = sigmoid(t) is y (1 - y) / (dM/dt dX/dM). Its log
        is maximized over a grid of t in [-40, 40], then refined to rounding by
        bisection on the sign of its derivative between the neighbouring grid
        points. A mode at an edge of the grid is where the density keeps rising
        toward a bound of the support, so it returns that bound, to rounding.

        Returns:
            chex.Scalar: The mode of the distribution.
        """
        engine, mu, s = self._logit_parts()

        def log_density(t: chex.Numeric) -> chex.Numeric:
            m = _unbounded_ppf_at_logit(mu, s, t)
            # dM/dt is 0 at a binding mode of a feasible fit: infinite density
            slope = jnp.maximum(engine.level(1, mu, s, t), 0.0)
            logw = jax.nn.log_sigmoid(t) + jax.nn.log_sigmoid(-t)
            return logw - jnp.log(slope) - self._log_dx_dm(m)

        grid = jnp.asarray(_MODE_GRID)
        i = jnp.argmax(jax.vmap(log_density)(grid))
        rising = jax.grad(log_density)

        def halve(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            up = rising(mid) > 0
            return jnp.where(up, mid, lo), jnp.where(up, hi, mid)

        start = (grid[jnp.maximum(i - 1, 0)], grid[jnp.minimum(i + 1, grid.size - 1)])
        lo, hi = jax.lax.fori_loop(0, _LogitScale.BISECTION_STEPS.value, halve, start)
        return self._from_unbounded(_unbounded_ppf_at_logit(mu, s, 0.5 * (lo + hi)))

    @property
    def mode(self) -> chex.Scalar:
        """Compute the mode (most likely value) of the fitted metalog distribution.

        For unbounded metalogs the mode is exact: the highest-density root of
        ``M''``, found by Algorithm 1 of Baucells, Chrisman, Keelin and Xu (2025).
        This also finds binding modes of a Feasible fit, which can sit at
        y < 0.001. Bounded and semi-bounded metalogs, whose back-transform moves
        the modes, and metalogs without an interior mode maximize the density on
        the logit scale, refined to rounding. Where the density rises all the way
        to a bound, the mode is that bound, to rounding.

        Returns:
            chex.Scalar: The mode of the distribution.

        Raises:
            NotFittedError: If the distribution has not been fitted.
            checkify.JaxRuntimeError: If the fit is infeasible.
        """  # noqa: DOC502
        self.assert_fitted()
        if not self._has_exact_moments():
            return self._density_mode()
        engine = get_engine(int(self.num_terms), self.term_order)
        value, _, found = highest_mode(engine, jnp.asarray(self.a, jnp.float64))
        return jnp.where(found, value, self._density_mode())

    def plot(self, plot_option: MetalogPlotOptions) -> go.Figure:
        """Visualize the fitted metalog distribution using interactive Plotly plots.

        Creates an interactive Plotly figure showing the selected distribution
        function (PDF, CDF, or survival function).

        Args:
            plot_option: Which function to plot. One of MetalogPlotOptions.PDF,
                MetalogPlotOptions.CDF, or MetalogPlotOptions.SF.

        Returns:
            go.Figure: The Plotly figure object.

        Raises:
            NotFittedError: If the distribution has not been fitted.
        """  # noqa: DOC502
        self.assert_fitted()
        x = DEFAULT_Y_FULL
        ppf = self.ppf(x)
        fig = go.Figure()
        if plot_option == MetalogPlotOptions.PDF:
            pdf = self.pdf(x)
            fig.add_trace(go.Scatter(x=ppf, y=pdf, mode="markers", name="Metalog Fit"))
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Density",
                title="Probability Density Function",
            )
        elif plot_option == MetalogPlotOptions.CDF:
            fig.add_trace(go.Scatter(x=ppf, y=x, mode="lines", name="Metalog Fit"))
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Probability",
                title="Cumulative Density Function",
            )
        elif plot_option == MetalogPlotOptions.SF:
            fig.add_trace(go.Scatter(x=ppf, y=1 - x, mode="lines", name="Metalog Fit"))
            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Probability",
                title="Survival Function",
            )
        fig.show()
        return fig

    def save(self, path: Path) -> None:
        """Save the fitted metalog distribution to a JSON file.

        Writes the JSON that ``dumps`` returns. Read it back with ``load`` on the
        same class: ``Metalog.load`` or ``SPTMetalog.load``.

        Args:
            path: File path where the JSON file will be written.
        """
        with open(path, "w") as f:
            f.write(self.dumps())

    def dumps(self) -> str:
        """Serialize the fitted metalog distribution to a JSON string.

        The JSON holds the coefficients ``a`` and the parameters' ``to_dict()``:
        boundedness, fit method, bounds and number of terms for a Metalog;
        boundedness, bounds and alpha for an SPTMetalog.

        Returns:
            str: JSON string representation of the distribution.
        """
        return json.dumps(
            {"a": self.a.tolist(), "metalog_params": self.metalog_params.to_dict()}
        )

    @classmethod
    def load(cls: type[T_MetalogBase], path: Path) -> T_MetalogBase:
        """Load a fitted metalog distribution from a JSON file.

        Args:
            path: File path to the JSON file containing the serialized distribution.

        Returns:
            T_MetalogBase: A new instance of the metalog class with loaded
                coefficients and parameters.

        Raises:
            KeyError: If the JSON has no ``a`` or ``metalog_params``.
            ValueError: If the parameters were saved by the other metalog class
                (for example, an SPTMetalog's JSON loaded with Metalog.load), or
                hold an unknown enum value.
        """  # noqa: DOC502
        with open(path) as f:
            return cls.loads(f.read())

    @classmethod
    def loads(cls: type[T_MetalogBase], json_string: str) -> T_MetalogBase:
        """Load a fitted metalog distribution from a JSON string.

        Args:
            json_string: JSON string containing the serialized distribution.

        Returns:
            T_MetalogBase: A new instance of the metalog class with loaded
                coefficients and parameters.

        Raises:
            KeyError: If the JSON has no ``a`` or ``metalog_params``.
            ValueError: If the parameters were saved by the other metalog class
                (for example, an SPTMetalog's JSON loaded with Metalog.loads), or
                hold an unknown enum value.
        """  # noqa: DOC502
        input_data = json.loads(json_string)
        metalog_params = cls._params_type.from_dict(input_data["metalog_params"])
        return cls(metalog_params=metalog_params, a=jnp.array(input_data["a"]))

    def q(self, x: chex.Numeric) -> chex.Numeric:
        """Quantile function alias for R metalog package compatibility.

        This method provides compatibility with the R metalog package's `qmetalog`
        function naming convention. Equivalent to ppf().

        Args:
            x: Probability values in (0, 1) at which to evaluate the quantile function.

        Returns:
            chex.Numeric: Quantile values corresponding to the input probabilities.

        Raises:
            ValueError: If x contains values outside (0, 1).
            NotFittedError: If the distribution has not been fitted.
        """  # noqa: DOC502
        return self.ppf(x)

    def p(self, q: chex.Numeric) -> chex.Numeric:
        """Probability function alias for R metalog package compatibility.

        This method provides compatibility with the R metalog package's `pmetalog`
        function naming convention. Equivalent to cdf().

        Args:
            q: Quantile values at which to evaluate the CDF.

        Returns:
            chex.Numeric: Probability values in (0, 1).
        """
        return self.cdf(q)

    def d(self, q: chex.Numeric) -> chex.Numeric:
        """Density function alias for R metalog package compatibility.

        This method provides compatibility with the R metalog package's `dmetalog`
        function naming convention. Computes pdf(cdf(q)).

        Args:
            q: Quantile values at which to evaluate the density.

        Returns:
            chex.Numeric: Probability density values; 0 outside the support,
            where the CDF is 0 or 1.
        """
        y = self.cdf(q)
        inside = (y > 0) & (y < 1)
        return jnp.where(inside, self.pdf(jnp.where(inside, y, 0.5)), 0.0)
