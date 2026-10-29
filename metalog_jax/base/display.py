# Copyright: Travis Jefferies 2026
"""Text and LaTeX display of fitted metalog distributions.

These functions write out a model's quantile function with its coefficients. The
model methods call them: ``str(m)`` and ``repr(m)`` use format_line, Jupyter's
``_repr_latex_`` uses format_latex, and ``m.print()`` uses format_block.

The terms follow the model's term order, the one ``MetalogBase._ppf`` evaluates:
1, ln(y/(1-y)), (y - 0.5) ln(y/(1-y)), (y - 0.5), then powers of (y - 0.5), with and
without the logit, in the Keelin (2016, Definition 1) or Metalog 2.0 order. Bounded and
semi-bounded models map M(y) back to their support (Keelin 2016, Section 4):
``L + exp(M(y))``, ``U - exp(-M(y))`` and ``(L + U exp(M(y))) / (1 + exp(M(y)))``.

Formatting never raises on a model, because JAX, flax and debuggers call ``repr`` on
traced and placeholder values. A model whose coefficients or bounds aren't concrete
(under ``jax.jit``, ``jax.vmap`` or ``jax.eval_shape``), or that holds a batch of
models, prints a one-line summary instead of its equation, and format_latex returns
None for it, which tells Jupyter to fall back to ``repr``.

See Also:
    metalog_jax.base.core.MetalogBase: The ``__str__``, ``__repr__``,
        ``_repr_latex_`` and ``print`` methods.
"""

import dataclasses
import numbers
from collections.abc import Callable, Sequence
from itertools import groupby
from operator import itemgetter
from typing import Any

import jax
import numpy as np

from metalog_jax.base.enums import MetalogBoundedness
from metalog_jax.feasibility.engine import TermOrder, term_basis

# Significant digits for str(m), _repr_latex_ and print's default
DEFAULT_PRECISION = 6

# _repr_latex_ keeps up to this many terms on one line, then wraps after every this many
_LATEX_TERMS_PER_LINE = 4

# The basis functions are powers of _DELTA, with and without the logit (see _basis)
_DELTA = "(y - 0.5)"

_NEEDS_LOWER = (MetalogBoundedness.STRICTLY_LOWER_BOUND, MetalogBoundedness.BOUNDED)
_NEEDS_UPPER = (MetalogBoundedness.STRICTLY_UPPER_BOUND, MetalogBoundedness.BOUNDED)

# The boundedness in a summary, which shows no bound values because they may be traced
_KIND = {
    MetalogBoundedness.UNBOUNDED: "unbounded",
    MetalogBoundedness.STRICTLY_LOWER_BOUND: "lower-bounded",
    MetalogBoundedness.STRICTLY_UPPER_BOUND: "upper-bounded",
    MetalogBoundedness.BOUNDED: "bounded",
}

# A term is (negative, body): its sign, and |a_j| times its basis function
_Term = tuple[bool, str]


def _number(x: float, precision: int | None) -> str:
    """Format x to ``precision`` significant digits, or exactly if it's None."""
    x = 0.0 if x == 0 else float(x)  # print -0.0 as 0
    return repr(x) if precision is None else f"{x:.{precision}g}"


def _latex_number(x: float, precision: int | None) -> str:
    """Format x as _number does, with LaTeX scientific notation and infinities."""
    text = _number(x, precision)
    if text in ("inf", "-inf", "nan"):
        return {"inf": r"\infty", "-inf": r"-\infty", "nan": r"\mathrm{NaN}"}[text]
    mantissa, _, exponent = text.partition("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}" if exponent else text


@dataclasses.dataclass(frozen=True)
class _Style:
    """How plain text or LaTeX writes the pieces of an equation."""

    number: Callable[[float, int | None], str]
    logit: str
    times: str
    alpha: str
    power: Callable[[int], str]
    exp: Callable[[str], str]
    fraction: Callable[[str, str], str]


_PLAIN = _Style(
    number=_number,
    logit="ln(y/(1-y))",
    times=" ",
    alpha="alpha",
    power=lambda k: f"^{k}",
    exp=lambda x: f"exp({x})",
    fraction=lambda top, bottom: f"({top}) / ({bottom})",
)
_LATEX = _Style(
    number=_latex_number,
    logit=r"\ell(y)",
    times=r"\,",
    alpha=r"\alpha",
    power=lambda k: f"^{{{k}}}",
    exp=lambda x: f"e^{{{x}}}",
    fraction=lambda top, bottom: rf"\frac{{{top}}}{{{bottom}}}",
)
_LATEX_LOGIT = r"\ell(y) = \ln\frac{y}{1-y}"


@dataclasses.dataclass(frozen=True)
class _Snapshot:
    """A model's values as Python numbers, or why it has none to show.

    Attributes:
        name: The model's class name.
        num_terms: The number of terms its parameters declare.
        method: The fit method's name, or None for an SPT metalog.
        boundedness: The boundedness type.
        term_order: The order of the basis terms.
        detail: Why the equation can't be shown (traced or batched values), or None.
        a: The coefficients, when ``detail`` is None.
        lower: The lower bound, when the boundedness uses it.
        upper: The upper bound, when the boundedness uses it.
        alpha: An SPT metalog's alpha, or None.
    """

    name: str
    num_terms: int
    method: str | None
    boundedness: MetalogBoundedness
    term_order: TermOrder = TermOrder.KEELIN_2016
    detail: str | None = None
    a: tuple[float, ...] = ()
    lower: float = 0.0
    upper: float = 0.0
    alpha: float | None = None


def _concrete(value: Any) -> np.ndarray | None:
    """Return value as a float NumPy array, or None if it's traced or not numeric."""
    try:
        array = np.asarray(value)
    except Exception:  # a tracer can't be converted, and neither can a deleted array
        return None
    # ShapeDtypeStruct, None and other placeholders become object arrays
    return array.astype(float) if array.dtype.kind in "iuf" else None


def _scalar(value: Any) -> float | None:
    """Return value as a float, or None if it isn't one concrete number."""
    array = _concrete(value)
    return None if array is None or array.size != 1 else float(array.reshape(()))


def _abstract_coefficients(a: Any) -> str:
    """Describe coefficients that can't be read, such as a tracer's shape and dtype."""
    shape, dtype = getattr(a, "shape", None), getattr(a, "dtype", None)
    if shape is None or dtype is None:
        return "no coefficients"
    kind = "abstract" if isinstance(a, jax.ShapeDtypeStruct) else "traced"
    return f"coefficients {kind}: {getattr(dtype, 'name', dtype)}[{','.join(map(str, shape))}]"


def _coefficients(a: Any) -> tuple[tuple[float, ...], str | None]:
    """Return the coefficients as floats, or why they can't be shown."""
    array = _concrete(a)
    if array is None:
        return (), _abstract_coefficients(a)
    if array.ndim > 1:
        return (
            (),
            f"batch of shape {array.shape[:-1]}, index one model to see its equation",
        )
    if array.ndim == 0 or array.size == 0:
        return (), "not fitted"
    return tuple(array.tolist()), None


def _parameters(params: Any, boundedness: MetalogBoundedness) -> dict:
    """Return the bounds and alpha the display needs, or why they can't be shown."""
    values = {
        "lower": _scalar(params.lower_bound) if boundedness in _NEEDS_LOWER else 0.0,
        "upper": _scalar(params.upper_bound) if boundedness in _NEEDS_UPPER else 0.0,
    }
    if None in values.values():
        return {"detail": "bounds traced or batched"}
    if hasattr(params, "alpha"):
        values["alpha"] = _scalar(params.alpha)
        if values["alpha"] is None:
            return {"detail": "alpha traced or batched"}
    return values


def _snapshot(model: Any) -> _Snapshot:
    """Read what the display needs from a Metalog or SPTMetalog."""
    params = model.metalog_params
    method = getattr(params, "method", None)
    static = {
        "name": type(model).__name__,
        "num_terms": int(model.num_terms),
        "method": None if method is None else getattr(method, "name", str(method)),
        "boundedness": params.boundedness,
        "term_order": getattr(params, "term_order", TermOrder.KEELIN_2016),
    }
    a, detail = _coefficients(model.a)
    if detail is not None:
        return _Snapshot(**static, detail=detail)
    return _Snapshot(**static, a=a, **_parameters(params, params.boundedness))


def _basis(power: int, has_logit: bool, style: _Style) -> str:
    """A basis function, (y - 0.5)^power times the logit if has_logit, as text."""
    delta = {0: "", 1: _DELTA}.get(power, _DELTA + style.power(power))
    if not has_logit:
        return delta
    return f"{delta}{style.times}{style.logit}" if delta else style.logit


def _terms(
    a: Sequence[float], order: TermOrder, style: _Style, precision: int | None
) -> list[_Term]:
    """Pair each coefficient's sign with its magnitude times its basis function."""
    terms = []
    for value, (power, has_logit) in zip(a, term_basis(len(a), order), strict=True):
        magnitude = style.number(abs(value), precision)
        basis = _basis(power, has_logit, style)
        terms.append(
            (value < 0, f"{magnitude}{style.times}{basis}" if basis else magnitude)
        )
    return terms


def _signed(term: _Term) -> str:
    """Write a term after another, as ``+ body`` or ``- body``."""
    negative, body = term
    return f"{'-' if negative else '+'} {body}"


def _join(terms: Sequence[_Term], continued: bool = False) -> str:
    """Write terms on one line; a continued line starts with its first term's sign."""
    pieces = [_signed(term) for term in terms]
    if not continued:
        negative, body = terms[0]
        pieces[0] = f"-{body}" if negative else body
    return " ".join(pieces)


def _bounded_parts(
    snap: _Snapshot, style: _Style, precision: int | None, x: str
) -> tuple[str, str]:
    """The numerator and denominator of a bounded Q(y): (L + U exp(x)) / (1 + exp(x))."""
    upper = (
        snap.upper < 0,
        style.number(abs(snap.upper), precision) + style.times + style.exp(x),
    )
    numerator = f"{style.number(snap.lower, precision)} {_signed(upper)}"
    return numerator, f"1 + {style.exp(x)}"


def _quantile(
    snap: _Snapshot, style: _Style, precision: int | None, x: str = "M(y)"
) -> str:
    """Q(y) of a bounded or semi-bounded model, with x for M(y)."""
    negated = "-M(y)" if x == "M(y)" else f"-({x})"
    if snap.boundedness == MetalogBoundedness.STRICTLY_LOWER_BOUND:
        return f"{style.number(snap.lower, precision)} + {style.exp(x)}"
    if snap.boundedness == MetalogBoundedness.STRICTLY_UPPER_BOUND:
        return f"{style.number(snap.upper, precision)} - {style.exp(negated)}"
    return style.fraction(*_bounded_parts(snap, style, precision, x))


def _count(num_terms: int) -> str:
    """Write the number of terms, such as ``5 terms``; there are at least 2."""
    return f"{num_terms} terms"


def _header_parts(
    snap: _Snapshot, style: _Style, precision: int | None
) -> list[tuple[bool, str]]:
    """The model's description as (is_math, text) pieces, such as ``5 terms, OLS``."""
    parts = [(False, _count(snap.num_terms))]
    if snap.method is not None:
        parts.append((False, f", {snap.method}"))
    if snap.alpha is not None:
        parts += [
            (False, ", "),
            (True, f"{style.alpha} = {style.number(snap.alpha, precision)}"),
        ]
    lower, upper = (
        style.number(bound, precision) for bound in (snap.lower, snap.upper)
    )
    support = {
        MetalogBoundedness.UNBOUNDED: [(False, ", unbounded")],
        MetalogBoundedness.STRICTLY_LOWER_BOUND: [
            (False, ", lower bound "),
            (True, lower),
        ],
        MetalogBoundedness.STRICTLY_UPPER_BOUND: [
            (False, ", upper bound "),
            (True, upper),
        ],
        MetalogBoundedness.BOUNDED: [
            (False, ", bounded on "),
            (True, f"({lower}, {upper})"),
        ],
    }
    return parts + support[snap.boundedness]


def _render(parts: list[tuple[bool, str]], latex: bool) -> str:
    r"""Join header pieces, putting each run of LaTeX text in one ``\text{}``."""
    if not latex:
        return "".join(text for _, text in parts)
    runs = (
        (is_math, "".join(map(itemgetter(1), run)))
        for is_math, run in groupby(parts, itemgetter(0))
    )
    return "".join(text if is_math else rf"\text{{{text}}}" for is_math, text in runs)


def _summary(snap: _Snapshot) -> str:
    """The one line shown instead of an equation, such as for a traced model."""
    method = "" if snap.method is None else f", {snap.method}"
    kind = _KIND[snap.boundedness]
    return f"{snap.name}({_count(snap.num_terms)}{method}, {kind}; {snap.detail})"


def _safely(
    format_model: Callable[..., str | None], fallback: Callable[[Any], str | None]
) -> Callable[..., str | None]:
    """Wrap a formatter so that a model it can't read gives ``fallback(model)``."""

    def wrapped(model: Any, *args: Any) -> str | None:
        try:
            return format_model(model, *args)
        except Exception:  # repr runs in tracebacks and debuggers, so it must not raise
            return fallback(model)

    return wrapped


def _unprintable(model: Any) -> str:
    """What format_line and format_block show for a model they can't read."""
    return f"{type(model).__name__}(<unprintable>)"


def _line(model: Any, precision: int | None) -> str:
    """format_line, without the guard."""
    snap = _snapshot(model)
    if snap.detail is not None:
        return _summary(snap)
    terms = _join(_terms(snap.a, snap.term_order, _PLAIN, precision))
    prefix = (
        f"{snap.name}({_render(_header_parts(snap, _PLAIN, precision), latex=False)}): "
    )
    if snap.boundedness == MetalogBoundedness.UNBOUNDED:
        return f"{prefix}Q(y) = {terms}"
    return f"{prefix}Q(y) = {_quantile(snap, _PLAIN, precision)}, M(y) = {terms}"


def _latex(model: Any, precision: int | None) -> str | None:
    """format_latex, without the guard."""
    snap = _snapshot(model)
    if snap.detail is not None:
        return None
    terms = _terms(snap.a, snap.term_order, _LATEX, precision)
    parts = _header_parts(snap, _LATEX, precision)
    unbounded = snap.boundedness == MetalogBoundedness.UNBOUNDED
    q = [] if unbounded else [f"Q(y) = {_quantile(snap, _LATEX, precision)}"]
    label = "Q(y)" if unbounded else "M(y)"
    if len(terms) <= _LATEX_TERMS_PER_LINE:
        header = _render([(False, f"{snap.name}("), *parts, (False, "):")], latex=True)
        equations = [*q, f"{label} = {_join(terms)}", _LATEX_LOGIT]
        return f"${header}\\quad " + r",\quad ".join(equations) + "$"
    step = _LATEX_TERMS_PER_LINE
    chunks = [terms[i : i + step] for i in range(0, len(terms), step)]
    lines = [
        "&" + _render([(False, f"{snap.name}: "), *parts], latex=True),
        *(line.replace(" = ", " &= ", 1) for line in q),
        f"{label} &= {_join(chunks[0])}",
        *(r"&\quad " + _join(chunk, continued=True) for chunk in chunks[1:]),
        _LATEX_LOGIT.replace(" = ", " &= ", 1),
    ]
    return "$$\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}$$"


def _term_lines(label: str, terms: Sequence[_Term]) -> list[str]:
    """Write ``label = `` and the terms one per line, aligned under the first."""
    lead = f"{label} = "
    return [lead + _join(terms[:1])] + [
        " " * len(lead) + _signed(term) for term in terms[1:]
    ]


def _quantile_lines(
    snap: _Snapshot, terms: Sequence[_Term], verbose: bool, precision: int
) -> list[str]:
    """The Q(y) line of a bounded or semi-bounded model; two lines if verbose and bounded."""
    if not verbose:
        return [f"Q(y) = {_quantile(snap, _PLAIN, precision)}"]
    x = _join(terms)
    if snap.boundedness != MetalogBoundedness.BOUNDED:
        return [f"Q(y) = {_quantile(snap, _PLAIN, precision, x)}"]
    numerator, denominator = _bounded_parts(snap, _PLAIN, precision, x)
    return [f"Q(y) = ({numerator})", f"{' ' * len('Q(y) = ')}/ ({denominator})"]


def _block(
    model: Any, header: bool, q: bool, m: bool, verbose: bool, precision: int
) -> str:
    """format_block, without the guard or the option checks."""
    snap = _snapshot(model)
    if snap.detail is not None:
        return _summary(snap)
    terms = _terms(snap.a, snap.term_order, _PLAIN, precision)
    if snap.boundedness == MetalogBoundedness.UNBOUNDED:
        # Q(y) is M(y): its terms print once, under whichever name was asked for
        label = "Q(y)" if q else "M(y)"
        lines = _term_lines(label, terms) if m else [f"{label} = {_join(terms)}"]
    else:
        lines = (_quantile_lines(snap, terms, verbose, precision) if q else []) + (
            _term_lines("M(y)", terms) if m else []
        )
    if not header:
        return "\n".join(lines)
    title = (
        f"{snap.name}: {_render(_header_parts(snap, _PLAIN, precision), latex=False)}"
    )
    return "\n".join([title] + ["  " + line for line in lines])


def _check_options(q: bool, m: bool, verbose: bool, precision: int) -> None:
    """Raise ValueError for print options that contradict each other.

    Args:
        q: Whether to print Q(y).
        m: Whether to print M(y).
        verbose: Whether to substitute M(y)'s terms into Q(y).
        precision: Significant digits.

    Raises:
        ValueError: If q and m are both False, verbose is True while q is False,
            or precision isn't a positive integer.
    """
    if not (q or m):
        raise ValueError(
            "q=False and m=False leave no equation to print; set q, m or both"
        )
    if verbose and not q:
        raise ValueError("verbose=True rewrites the Q(y) line, so it needs q=True")
    if (
        isinstance(precision, bool)
        or not isinstance(precision, numbers.Integral)
        or precision < 1
    ):
        raise ValueError(f"precision must be a positive integer, got {precision!r}")


_guarded_line = _safely(_line, _unprintable)
_guarded_latex = _safely(_latex, lambda model: None)
_guarded_block = _safely(_block, _unprintable)


def format_line(model: Any, precision: int | None = DEFAULT_PRECISION) -> str:
    """Write a model and its equation on one line.

    ``str(m)`` is ``format_line(m)``, and ``repr(m)`` is
    ``format_line(m, precision=None)``.

    Args:
        model: A Metalog or SPTMetalog.
        precision: Significant digits of the coefficients and bounds, or None for
            Python's shortest round-trip ``repr(float)``, which parses back to the
            exact values.

    Returns:
        str: The one-line equation, such as
            ``Metalog(3 terms, OLS, lower bound 0): Q(y) = 0 + exp(M(y)), M(y) = ...``,
            or a one-line summary if the model is traced or batched.
    """
    return _guarded_line(model, precision)


def format_latex(model: Any, precision: int = DEFAULT_PRECISION) -> str | None:
    r"""Write a model's equation in LaTeX, for Jupyter's ``_repr_latex_``.

    Up to four terms stay on one line, between ``$`` signs. A longer model is an
    ``aligned`` environment between ``$$`` signs, with Q(y) on its own line and
    M(y)'s terms wrapped after every four, because MathJax doesn't break long
    equations. ``\ell(y)`` stands for ``\ln(y / (1 - y))``.

    Args:
        model: A Metalog or SPTMetalog.
        precision: Significant digits of the coefficients and bounds.

    Returns:
        str | None: The LaTeX, or None if the model is traced or batched, which
            tells IPython to show ``repr`` instead.
    """
    return _guarded_latex(model, precision)


def format_block(
    model: Any,
    *,
    header: bool = True,
    q: bool = True,
    m: bool = True,
    verbose: bool = False,
    precision: int = DEFAULT_PRECISION,
) -> str:
    """Write a model's equation one term per line, as ``m.print()`` prints it.

    Args:
        model: A Metalog or SPTMetalog.
        header: Start with a line describing the model, such as ``Metalog: 5 terms,
            OLS, bounded on (0, 15)``, and indent the equations under it.
        q: Include the quantile function Q(y).
        m: Include M(y), one term per line. An unbounded model's Q(y) is M(y), so
            its terms appear once, labelled Q(y), or M(y) when ``q`` is False; with
            ``m=False`` they stay on Q(y)'s line.
        verbose: Write Q(y) with M(y)'s terms in place of M(y), so it can be read
            on its own.
        precision: Significant digits of the coefficients and bounds.

    Returns:
        str: The lines, without a trailing newline, or a one-line summary if the
            model is traced or batched.

    Raises:
        ValueError: If ``q`` and ``m`` are both False, ``verbose`` is True while
            ``q`` is False, or ``precision`` isn't a positive integer.
    """  # noqa: DOC502
    _check_options(q, m, verbose, precision)
    return _guarded_block(model, header, q, m, verbose, precision)
