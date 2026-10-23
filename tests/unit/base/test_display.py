# Copyright: Travis Jefferies 2026
"""Tests for issue #41: str, repr, _repr_latex_ and print show the fitted equation."""

import contextlib
import importlib.util
import io
import itertools
import math
import re
from typing import ClassVar

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    SPTMetalogParameters,
)
from metalog_jax.base.display import format_block, format_latex, format_line
from metalog_jax.grid_search import extract_metalog, fit_grid
from metalog_jax.metalog import Metalog, SPTMetalog
from metalog_jax.utils import DEFAULT_Y

UNBOUNDED = MetalogBoundedness.UNBOUNDED
LOWER = MetalogBoundedness.STRICTLY_LOWER_BOUND
UPPER = MetalogBoundedness.STRICTLY_UPPER_BOUND
BOUNDED = MetalogBoundedness.BOUNDED

# A coefficient is followed by a space and its basis function, or ends the equation;
# the 0.5 in "(y - 0.5)" is followed by ")", so it doesn't match
_COEFFICIENT = re.compile(r"(?:^|\s([+-])\s)(-?[\d.]+(?:e[+-]?\d+)?|nan|inf)(?=\s|$)")


def metalog(a, boundedness=UNBOUNDED, lower=0.0, upper=0.0, method=None):
    """Return a Metalog with the given coefficients and parameters."""
    params = MetalogParameters(
        boundedness=boundedness,
        method=MetalogFitMethod.OLS if method is None else method,
        lower_bound=lower,
        upper_bound=upper,
        num_terms=len(a),
    )
    return Metalog(metalog_params=params, a=jnp.asarray(a, dtype=float))


def spt_metalog(a, boundedness=LOWER, lower=0.0, upper=0.0, alpha=0.1):
    """Return an SPTMetalog with the given coefficients and parameters."""
    params = SPTMetalogParameters(
        boundedness=boundedness, alpha=alpha, lower_bound=lower, upper_bound=upper
    )
    return SPTMetalog(metalog_params=params, a=jnp.asarray(a, dtype=float))


def printed(model, **options):
    """Return what model.print(**options) writes."""
    buffer = io.StringIO()
    model.print(file=buffer, **options)
    return buffer.getvalue()


def coefficients_in(equation):
    """Parse the signed coefficients out of a one-line M(y) or unbounded Q(y)."""
    return [
        float(number) * (-1.0 if sign == "-" else 1.0)
        for sign, number in _COEFFICIENT.findall(equation)
    ]


class DisplayExamplesTest(absltest.TestCase):
    """The exact output of every form, for the models in the issue."""

    # A 5-term OLS fit bounded on (0, 15) to the sample in the Metalog docstring
    BOUNDED_A = (
        -0.7544638391916972,
        0.0719273440716332,
        -0.11075865713072672,
        2.4579362669066045,
        1.1187009327471362,
    )
    # The SPT fit of (2.08, 4.1, 8.06) with lower bound 0 and alpha 0.1
    SPT_A = (1.410986973710262, 0.30824014913566067, -0.001531760349245195)
    M_TERMS = (
        "-0.754464",
        "+ 0.0719273 ln(y/(1-y))",
        "- 0.110759 (y - 0.5) ln(y/(1-y))",
        "+ 2.45794 (y - 0.5)",
        "+ 1.1187 (y - 0.5)^2",
    )

    def setUp(self):
        """Build the bounded Metalog and the lower-bounded SPTMetalog."""
        super().setUp()
        self.m = metalog(self.BOUNDED_A, BOUNDED, 0.0, 15.0)
        self.spt = spt_metalog(self.SPT_A)

    def test_str(self):
        """The str form shows the header, Q(y) and M(y) on one line, to 6 digits."""
        self.assertEqual(
            str(self.m),
            "Metalog(5 terms, OLS, bounded on (0, 15)): "
            "Q(y) = (0 + 15 exp(M(y))) / (1 + exp(M(y))), M(y) = "
            + " ".join(self.M_TERMS),
        )

    def test_repr(self):
        """The repr form is the same line at full precision."""
        self.assertEqual(
            repr(self.m),
            "Metalog(5 terms, OLS, bounded on (0.0, 15.0)): "
            "Q(y) = (0.0 + 15.0 exp(M(y))) / (1 + exp(M(y))), "
            "M(y) = -0.7544638391916972 + 0.0719273440716332 ln(y/(1-y)) "
            "- 0.11075865713072672 (y - 0.5) ln(y/(1-y)) + 2.4579362669066045 (y - 0.5) "
            "+ 1.1187009327471362 (y - 0.5)^2",
        )

    def test_print(self):
        """The print method writes the header, Q(y), and M(y) one term per line."""
        self.assertEqual(
            printed(self.m),
            "Metalog: 5 terms, OLS, bounded on (0, 15)\n"
            "  Q(y) = (0 + 15 exp(M(y))) / (1 + exp(M(y)))\n"
            f"  M(y) = {self.M_TERMS[0]}\n"
            + "".join(f"         {term}\n" for term in self.M_TERMS[1:]),
        )

    def test_print_verbose_without_m(self):
        """verbose=True puts M(y)'s terms in both places a bounded Q(y) uses M(y)."""
        terms = " ".join(self.M_TERMS)
        self.assertEqual(
            printed(self.m, m=False, verbose=True),
            "Metalog: 5 terms, OLS, bounded on (0, 15)\n"
            f"  Q(y) = (0 + 15 exp({terms}))\n"
            f"         / (1 + exp({terms}))\n",
        )

    def test_print_m_only_at_3_digits(self):
        """q=False prints M(y) alone, and precision sets the significant digits."""
        self.assertEqual(
            printed(self.m, q=False, precision=3),
            "Metalog: 5 terms, OLS, bounded on (0, 15)\n"
            "  M(y) = -0.754\n"
            "         + 0.0719 ln(y/(1-y))\n"
            "         - 0.111 (y - 0.5) ln(y/(1-y))\n"
            "         + 2.46 (y - 0.5)\n"
            "         + 1.12 (y - 0.5)^2\n",
        )

    def test_latex_wraps_after_four_terms(self):
        """A 5-term model is an aligned environment with M(y) on two lines."""
        self.assertEqual(
            self.m._repr_latex_(),
            "$$\\begin{aligned}\n"
            r"&\text{Metalog: 5 terms, OLS, bounded on }(0, 15) \\"
            "\n"
            r"Q(y) &= \frac{0 + 15\,e^{M(y)}}{1 + e^{M(y)}} \\"
            "\n"
            r"M(y) &= -0.754464 + 0.0719273\,\ell(y) - 0.110759\,(y - 0.5)\,\ell(y)"
            r" + 2.45794\,(y - 0.5) \\"
            "\n"
            r"&\quad + 1.1187\,(y - 0.5)^{2} \\"
            "\n"
            r"\ell(y) &= \ln\frac{y}{1-y}"
            "\n"
            "\\end{aligned}$$",
        )

    def test_spt_print_verbose(self):
        """An SPT header names alpha; verbose substitutes M(y) into L + exp(M(y))."""
        self.assertEqual(
            printed(self.spt, m=False, verbose=True),
            "SPTMetalog: 3 terms, alpha = 0.1, lower bound 0\n"
            "  Q(y) = 0 + exp(1.41099 + 0.30824 ln(y/(1-y)) "
            "- 0.00153176 (y - 0.5) ln(y/(1-y)))\n",
        )

    def test_spt_print_without_header(self):
        """header=False prints just the equation, at the left margin."""
        self.assertEqual(
            printed(self.spt, header=False, m=False, verbose=True),
            "Q(y) = 0 + exp(1.41099 + 0.30824 ln(y/(1-y)) "
            "- 0.00153176 (y - 0.5) ln(y/(1-y)))\n",
        )

    def test_spt_str(self):
        """An SPT metalog's str has alpha in place of a fit method."""
        self.assertEqual(
            str(self.spt),
            "SPTMetalog(3 terms, alpha = 0.1, lower bound 0): Q(y) = 0 + exp(M(y)), "
            "M(y) = 1.41099 + 0.30824 ln(y/(1-y)) - 0.00153176 (y - 0.5) ln(y/(1-y))",
        )

    def test_spt_latex_stays_on_one_line(self):
        """A 3-term model's LaTeX is one inline equation."""
        self.assertEqual(
            self.spt._repr_latex_(),
            r"$\text{SPTMetalog(3 terms, }\alpha = 0.1\text{, lower bound }0\text{):}"
            r"\quad Q(y) = 0 + e^{M(y)},\quad M(y) = 1.41099 + 0.30824\,\ell(y)"
            r" - 0.00153176\,(y - 0.5)\,\ell(y),\quad \ell(y) = \ln\frac{y}{1-y}$",
        )

    def test_format_functions_match_the_methods(self):
        """The methods return what the display module's functions do."""
        self.assertEqual(str(self.m), format_line(self.m))
        self.assertEqual(repr(self.m), format_line(self.m, precision=None))
        self.assertEqual(self.m._repr_latex_(), format_latex(self.m))
        self.assertEqual(printed(self.m), format_block(self.m) + "\n")


class TermLayoutTest(parameterized.TestCase):
    """Every boundedness type at 3, 4, 5, 6 and 9 terms, against literal terms."""

    A = (1.5, -0.25, 0.125, 2.0, -0.5, 0.75, -1.0, 0.0625, 3.0)
    # The coefficients of A on their basis functions, Keelin (2016) Definition 1
    TERMS = (
        "1.5",
        "- 0.25 ln(y/(1-y))",
        "+ 0.125 (y - 0.5) ln(y/(1-y))",
        "+ 2 (y - 0.5)",
        "- 0.5 (y - 0.5)^2",
        "+ 0.75 (y - 0.5)^2 ln(y/(1-y))",
        "- 1 (y - 0.5)^3",
        "+ 0.0625 (y - 0.5)^3 ln(y/(1-y))",
        "+ 3 (y - 0.5)^4",
    )
    BOUNDS: ClassVar[dict] = {
        UNBOUNDED: (0.0, 0.0),
        LOWER: (1.0, 0.0),
        UPPER: (0.0, 20.0),
        BOUNDED: (1.0, 20.0),
    }
    SUPPORT: ClassVar[dict] = {
        UNBOUNDED: "unbounded",
        LOWER: "lower bound 1",
        UPPER: "upper bound 20",
        BOUNDED: "bounded on (1, 20)",
    }
    QUANTILE: ClassVar[dict] = {
        LOWER: "1 + exp(M(y))",
        UPPER: "20 - exp(-M(y))",
        BOUNDED: "(1 + 20 exp(M(y))) / (1 + exp(M(y)))",
    }

    def model(self, boundedness, k):
        """Return a k-term Metalog with the first k coefficients of A."""
        return metalog(self.A[:k], boundedness, *self.BOUNDS[boundedness])

    @parameterized.product(boundedness=tuple(MetalogBoundedness), k=(3, 4, 5, 6, 9))
    def test_str(self, boundedness, k):
        """The str form has every term in order, and Q(y) for the boundedness."""
        terms = " ".join(self.TERMS[:k])
        prefix = f"Metalog({k} terms, OLS, {self.SUPPORT[boundedness]}): Q(y) = "
        expected = (
            prefix + terms
            if boundedness == UNBOUNDED
            else f"{prefix}{self.QUANTILE[boundedness]}, M(y) = {terms}"
        )
        self.assertEqual(str(self.model(boundedness, k)), expected)

    @parameterized.product(boundedness=tuple(MetalogBoundedness), k=(3, 4, 5, 6, 9))
    def test_print(self, boundedness, k):
        """Each term gets a line; an unbounded model's terms are its Q(y)."""
        label = "Q(y)" if boundedness == UNBOUNDED else "M(y)"
        lines = [f"Metalog: {k} terms, OLS, {self.SUPPORT[boundedness]}"]
        if boundedness != UNBOUNDED:
            lines.append(f"  Q(y) = {self.QUANTILE[boundedness]}")
        lines.append(f"  {label} = {self.TERMS[0]}")
        lines += [f"         {term}" for term in self.TERMS[1:k]]
        self.assertEqual(printed(self.model(boundedness, k)), "\n".join(lines) + "\n")

    @parameterized.product(boundedness=tuple(MetalogBoundedness), k=(3, 4, 5, 6, 9))
    def test_repr_coefficients_parse_back_exactly(self, boundedness, k):
        """The coefficients in repr are a, exactly."""
        model = self.model(boundedness, k)
        equation = repr(model).rpartition(" = ")[2]
        self.assertEqual(coefficients_in(equation), list(self.A[:k]))
        self.assertEqual(coefficients_in(equation), model.a.tolist())


class PrintOptionsTest(parameterized.TestCase):
    """Every combination of header, q, m and verbose, for each boundedness type."""

    A = (1.5, -0.25, 0.125, 2.0, -0.5)
    BOUNDS: ClassVar[dict] = {
        UNBOUNDED: (0.0, 0.0),
        LOWER: (1.0, 0.0),
        UPPER: (0.0, 20.0),
        BOUNDED: (1.0, 20.0),
    }

    @parameterized.product(
        boundedness=tuple(MetalogBoundedness),
        options=tuple(itertools.product((True, False), repeat=4)),
    )
    def test_options(self, boundedness, options):
        """Each option adds or removes exactly the lines it names."""
        header, q, m, verbose = options
        model = metalog(self.A, boundedness, *self.BOUNDS[boundedness])
        kwargs = {"header": header, "q": q, "m": m, "verbose": verbose}
        if not (q or m) or (verbose and not q):
            with self.assertRaises(ValueError):
                model.print(**kwargs)
            return
        lines = printed(model, **kwargs).splitlines()
        self.assertEqual(lines[0].startswith("Metalog: 5 terms"), header)
        body = lines[1:] if header else lines
        indent = "  " if header else ""
        labels = [
            line.lstrip()[:4] for line in body if line.lstrip()[:4] in ("Q(y)", "M(y)")
        ]
        for line in body:
            if line.lstrip()[:4] in (
                "Q(y)",
                "M(y)",
            ):  # an equation starts at the indent
                self.assertTrue(line.startswith(indent + line.lstrip()[:4]))
            else:  # its continuation lines go further
                self.assertTrue(line.startswith(indent + " "))
        if boundedness == UNBOUNDED:
            self.assertEqual(labels, ["Q(y)" if q else "M(y)"])
            self.assertEqual(len(body), 5 if m else 1)
            return
        self.assertEqual(labels, ["Q(y)"] * q + ["M(y)"] * m)
        if q:
            q_line = body[0]
            self.assertEqual("M(y)" in q_line, not verbose)
            self.assertEqual("ln(y/(1-y))" in q_line, verbose)
        two_line_q = q and verbose and boundedness == BOUNDED
        self.assertLen(body, (q + two_line_q) + 5 * m)

    def test_unbounded_without_m_keeps_the_terms_on_one_line(self):
        """An unbounded model with m=False prints Q(y) = its terms, on one line."""
        self.assertEqual(
            printed(metalog(self.A), header=False, m=False),
            "Q(y) = 1.5 - 0.25 ln(y/(1-y)) + 0.125 (y - 0.5) ln(y/(1-y)) + 2 (y - 0.5) "
            "- 0.5 (y - 0.5)^2\n",
        )

    def test_unbounded_verbose_changes_nothing(self):
        """An unbounded Q(y) has no M(y) to substitute."""
        model = metalog(self.A)
        self.assertEqual(printed(model, verbose=True), printed(model))

    def test_upper_bound_verbose_negates_the_terms(self):
        """With an upper bound, verbose writes U - exp(-(terms))."""
        model = metalog(self.A[:3], UPPER, upper=20.0)
        self.assertEqual(
            printed(model, header=False, m=False, verbose=True),
            "Q(y) = 20 - exp(-(1.5 - 0.25 ln(y/(1-y)) + 0.125 (y - 0.5) ln(y/(1-y))))\n",
        )

    @parameterized.parameters(0, -1, 2.5, True, None)
    def test_invalid_precision_raises(self, precision):
        """The precision must be a positive integer."""
        with self.assertRaisesRegex(ValueError, "precision"):
            metalog(self.A).print(precision=precision)

    def test_error_messages(self):
        """The two contradictory option sets explain themselves."""
        model = metalog(self.A)
        with self.assertRaisesRegex(ValueError, "no equation to print"):
            model.print(q=False, m=False)
        with self.assertRaisesRegex(ValueError, "needs q=True"):
            model.print(q=False, verbose=True)

    def test_writes_to_standard_output_by_default(self):
        """The method writes to sys.stdout as it is when called."""
        model = metalog(self.A)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            model.print()
        self.assertEqual(stdout.getvalue(), printed(model))

    def test_builtin_print_shows_str(self):
        """The builtin print(m) still prints str(m)."""
        model = metalog(self.A)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            print(model)
        self.assertEqual(stdout.getvalue(), str(model) + "\n")


class LatexTest(parameterized.TestCase):
    """_repr_latex_ wraps long equations, since MathJax doesn't."""

    @parameterized.product(boundedness=(UNBOUNDED, BOUNDED), k=(3, 4))
    def test_up_to_four_terms_is_one_inline_equation(self, boundedness, k):
        """Up to four terms, the LaTeX is one line between single dollar signs."""
        latex = metalog([1.0] * k, boundedness, 0.0, 1.0)._repr_latex_()
        self.assertRegex(latex, r"^\$[^$\n]+\$$")
        label = "Q(y) = " if boundedness == UNBOUNDED else "M(y) = "
        parts = re.split(r",?\\quad ", latex)  # the equations follow the header
        (terms,) = [part for part in parts if part.startswith(label)]
        self.assertEqual(terms.count(" + "), k - 1)

    @parameterized.product(boundedness=(UNBOUNDED, BOUNDED), k=(5, 9, 16))
    def test_longer_models_wrap_every_four_terms(self, boundedness, k):
        """M(y)'s terms take ceil(k / 4) lines of an aligned environment."""
        latex = metalog([1.0] * k, boundedness, 0.0, 1.0)._repr_latex_()
        self.assertTrue(latex.startswith("$$\\begin{aligned}\n"))
        self.assertTrue(latex.endswith("\n\\end{aligned}$$"))
        lines = latex.split("\n")[1:-1]
        label = "Q(y) &= " if boundedness == UNBOUNDED else "M(y) &= "
        start = next(i for i, line in enumerate(lines) if line.startswith(label))
        continued = itertools.takewhile(
            lambda line: line.startswith(r"&\quad"), lines[start + 1 :]
        )
        term_lines = [lines[start], *continued]
        self.assertLen(term_lines, math.ceil(k / 4))
        self.assertEqual(sum(line.count(" + ") for line in term_lines), k - 1)
        self.assertEqual(
            any(line.startswith("Q(y) &= \\frac") for line in lines),
            boundedness == BOUNDED,
        )
        self.assertEqual(latex.count("{"), latex.count("}"))

    def test_sixteen_term_powers(self):
        r"""The 16th term is (y - 0.5)^7 \ell(y)."""
        latex = metalog([1.0] * 16)._repr_latex_()
        self.assertIn(r"+ 1\,(y - 0.5)^{7}\,\ell(y) \\", latex)


class NumberFormatTest(absltest.TestCase):
    """Signs, zeros, scientific notation and non-finite values."""

    A = (-1.0, -0.0, 4.49171e-06, -2.5e-07, 1e16)

    def test_plain_text(self):
        """Negative terms subtract, -0.0 prints as 0, and small values use %g."""
        self.assertEqual(
            str(metalog(self.A)),
            "Metalog(5 terms, OLS, unbounded): Q(y) = -1 + 0 ln(y/(1-y)) "
            "+ 4.49171e-06 (y - 0.5) ln(y/(1-y)) - 2.5e-07 (y - 0.5) + 1e+16 (y - 0.5)^2",
        )

    def test_latex_scientific_notation(self):
        """LaTeX writes scientific notation as a power of 10."""
        latex = metalog(self.A)._repr_latex_()
        self.assertIn(r"+ 4.49171 \times 10^{-6}\,(y - 0.5)\,\ell(y)", latex)
        self.assertIn(r"- 2.5 \times 10^{-7}\,(y - 0.5)", latex)
        self.assertIn(r"+ 1 \times 10^{16}\,(y - 0.5)^{2}", latex)

    def test_non_finite_coefficients(self):
        """NaN and infinite coefficients print instead of raising."""
        model = metalog([float("nan"), float("inf"), -float("inf")])
        self.assertEqual(
            str(model),
            "Metalog(3 terms, OLS, unbounded): Q(y) = nan + inf ln(y/(1-y)) "
            "- inf (y - 0.5) ln(y/(1-y))",
        )
        self.assertIn(
            r"\mathrm{NaN} + \infty\,\ell(y) - \infty\,", model._repr_latex_()
        )
        self.assertEqual(
            coefficients_in(repr(model).rpartition(" = ")[2])[1:], [math.inf, -math.inf]
        )

    def test_negative_bounds(self):
        """A negative upper bound subtracts in a bounded Q(y)."""
        model = metalog([1.0, 2.0], BOUNDED, -10.0, -5.0)
        self.assertIn("Q(y) = (-10 - 5 exp(M(y))) / (1 + exp(M(y)))", str(model))
        self.assertIn(r"\frac{-10 - 5\,e^{M(y)}}{1 + e^{M(y)}}", model._repr_latex_())

    def test_small_bounds_in_latex(self):
        """A bound in scientific notation stays math in the LaTeX header."""
        latex = metalog([1.0, 2.0], LOWER, lower=1e-7)._repr_latex_()
        self.assertIn(r"lower bound }1 \times 10^{-7}\text{):}", latex)


class NeverRaisesTest(absltest.TestCase):
    """Traced, abstract, batched, unfitted and infeasible models still print."""

    A = (-0.75, 0.07, -0.11, 2.46, 1.12)
    TRACED = "Metalog(5 terms, OLS, bounded; coefficients traced: float64[5])"

    def setUp(self):
        """Build a bounded 5-term Metalog."""
        super().setUp()
        self.m = metalog(self.A, BOUNDED, 0.0, 15.0)

    def shown(self, model):
        """Return str, repr, _repr_latex_ and print's output."""
        return [str(model), repr(model), model._repr_latex_(), printed(model)]

    def test_jit(self):
        """Under jax.jit, each form shows a summary without concretizing."""
        seen = []

        @jax.jit
        def f(model):
            """Record the forms of a traced model."""
            seen.extend(self.shown(model))
            return model.a

        f(self.m)
        self.assertEqual(seen, [self.TRACED, self.TRACED, None, self.TRACED + "\n"])

    def test_vmap(self):
        """Under jax.vmap, each model in the batch is traced."""
        seen = []

        def f(model):
            """Record the forms of a batched, traced model."""
            seen.extend(self.shown(model))
            return model.a

        batch = jax.tree.map(lambda *xs: jnp.stack(xs), self.m, self.m)
        jax.vmap(f)(batch)
        self.assertEqual(seen, [self.TRACED, self.TRACED, None, self.TRACED + "\n"])

    def test_eval_shape(self):
        """A model of ShapeDtypeStructs from eval_shape is abstract."""
        model = jax.eval_shape(lambda: self.m)
        abstract = "Metalog(5 terms, OLS, bounded; coefficients abstract: float64[5])"
        self.assertEqual(self.shown(model), [abstract, abstract, None, abstract + "\n"])

    def test_placeholder_leaves(self):
        """A model whose leaves are None, as tree utilities build, has no coefficients."""
        model = jax.tree.map(lambda _: None, self.m)
        self.assertEqual(str(model), "Metalog(5 terms, OLS, bounded; no coefficients)")

    def test_batched_model(self):
        """A stacked model says to index one, and has no LaTeX."""
        batch = jax.tree.map(lambda *xs: jnp.stack(xs), self.m, self.m, self.m)
        summary = (
            "Metalog(5 terms, OLS, bounded; batch of shape (3,), "
            "index one model to see its equation)"
        )
        self.assertEqual(self.shown(batch), [summary, summary, None, summary + "\n"])

    def test_batched_bounds(self):
        """Concrete coefficients with batched bounds give a summary."""
        params = self.m.metalog_params.replace(lower_bound=jnp.zeros(2))
        model = self.m.replace(metalog_params=params)
        self.assertEqual(
            str(model), "Metalog(5 terms, OLS, bounded; bounds traced or batched)"
        )

    def test_batched_alpha(self):
        """An SPT metalog with a batch of alphas gives a summary."""
        params = SPTMetalogParameters(
            boundedness=UNBOUNDED,
            alpha=jnp.array([0.1, 0.2]),
            lower_bound=0.0,
            upper_bound=0.0,
        )
        model = SPTMetalog(metalog_params=params, a=jnp.ones(3))
        self.assertEqual(
            str(model), "SPTMetalog(3 terms, unbounded; alpha traced or batched)"
        )
        self.assertIsNone(model._repr_latex_())

    def test_grid_result(self):
        """A GridResult's metalog is a batch; an extracted one prints its equation."""
        samples = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1, 10.5])
        data = MetalogInputData.from_values(
            samples, DEFAULT_Y, precomputed_quantiles=False
        )
        params = MetalogParameters(
            boundedness=UNBOUNDED,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
            lower_bound=0.0,
            upper_bound=0.0,
        )
        result = fit_grid(
            data.x, data.y, params, l1_penalties=jnp.array([0.0, 0.01, 0.1])
        )
        self.assertIn(
            "metalog=Metalog(5 terms, Lasso, unbounded; batch of shape (3,)",
            repr(result),
        )
        self.assertTrue(
            str(extract_metalog(result, 0)).startswith(
                "Metalog(5 terms, Lasso, unbounded): Q(y) = "
            )
        )

    def test_not_fitted(self):
        """A model without coefficients says so."""
        params = self.m.metalog_params
        model = Metalog(metalog_params=params, a=jnp.array([]))
        self.assertEqual(str(model), "Metalog(5 terms, OLS, bounded; not fitted)")

    def test_infeasible_fit(self):
        """An infeasible model prints, because printing evaluates nothing."""
        model = metalog([0.0, -1.0])
        self.assertEqual(
            str(model), "Metalog(2 terms, OLS, unbounded): Q(y) = 0 - 1 ln(y/(1-y))"
        )
        self.assertEqual(
            printed(model, header=False), "Q(y) = 0\n       - 1 ln(y/(1-y))\n"
        )

    def test_unreadable_model(self):
        """A model the formatter can't read falls back instead of raising."""
        model = Metalog(metalog_params=None, a=[1.0, 2.0])
        self.assertEqual(str(model), "Metalog(<unprintable>)")
        self.assertEqual(repr(model), "Metalog(<unprintable>)")
        self.assertIsNone(model._repr_latex_())
        self.assertEqual(printed(model), "Metalog(<unprintable>)\n")

    def test_options_are_checked_even_when_unreadable(self):
        """Contradictory options raise whatever the model holds."""
        model = jax.eval_shape(lambda: self.m)
        with self.assertRaises(ValueError):
            model.print(q=False, m=False)


class DunderPlacementTest(absltest.TestCase):
    """flax.struct.dataclass generates __repr__ per class, so each defines its own."""

    def test_subclasses_define_repr(self):
        """Metalog and SPTMetalog define __repr__ in their own class bodies."""
        self.assertIn("__repr__", Metalog.__dict__)
        self.assertIn("__repr__", SPTMetalog.__dict__)

    def test_str_is_not_repr(self):
        """The str form rounds to 6 digits; repr doesn't."""
        model = metalog([1 / 3, 2 / 3])
        self.assertIn("0.333333 ", str(model))
        self.assertIn("0.3333333333333333 ", repr(model))

    @absltest.skipUnless(importlib.util.find_spec("IPython"), "IPython isn't installed")
    def test_ipython_display(self):
        """IPython shows LaTeX for a concrete model and falls back to repr otherwise."""
        from IPython.core.formatters import DisplayFormatter

        model = metalog([1.0, 2.0, 3.0])
        data, _ = DisplayFormatter().format(model)
        self.assertEqual(data["text/latex"], model._repr_latex_())
        self.assertEqual(data["text/plain"], repr(model))
        data, _ = DisplayFormatter().format(jax.eval_shape(lambda: model))
        self.assertNotIn("text/latex", data)


if __name__ == "__main__":
    absltest.main()
