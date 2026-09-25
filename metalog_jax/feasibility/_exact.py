# Copyright: Travis Jefferies 2026
"""Exact rational tables used by the Metalog 2.0 feasibility and moment routines.

Everything in this module runs in pure Python with ``fractions.Fraction`` at trace
time, so the numbers baked into jitted code carry no accumulated rounding error.

References:
    Baucells, M., Chrisman, L., Keelin, T. W., & Xu, Z. S. (2025). On the
    Properties of the Metalog Distribution. Darden Business School Working
    Paper No. 5279416. https://doi.org/10.2139/ssrn.5279416
    (Lemma 1, Lemma 3 and Lemma 4).
    Xu, Z. S. metalog_algorithm (reference implementation), licensed under
    CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); adapted and ported
    to JAX, see NOTICE. https://github.com/Stephenxuu/metalog_algorithm
"""

from fractions import Fraction
from functools import lru_cache
from math import comb, factorial, pi

Poly = list[Fraction]

_Y: Poly = [Fraction(0), Fraction(1)]
_ONE_MINUS_Y: Poly = [Fraction(1), Fraction(-1)]
_SHIFT: Poly = [Fraction(-1, 2), Fraction(1)]  # y - 1/2


def _padd(p: Poly, q: Poly) -> Poly:
    n = max(len(p), len(q))
    return [
        (p[i] if i < len(p) else Fraction(0)) + (q[i] if i < len(q) else Fraction(0))
        for i in range(n)
    ]


def _pmul(p: Poly, q: Poly) -> Poly:
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _pscale(p: Poly, c: Fraction) -> Poly:
    return [c * a for a in p]


def _ppow(p: Poly, n: int) -> Poly:
    out: Poly = [Fraction(1)]
    for _ in range(n):
        out = _pmul(out, p)
    return out


def _pder(p: Poly, n: int = 1) -> Poly:
    for _ in range(n):
        p = [i * p[i] for i in range(1, len(p))] or [Fraction(0)]
    return p


def _trim(p: Poly) -> Poly:
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _to_one_minus_y_basis(p: Poly) -> Poly:
    """Re-express ascending coefficients in y as ascending coefficients in z = 1 - y."""
    out: Poly = [Fraction(0)]
    for i, c in enumerate(p):
        out = _padd(out, _pscale(_ppow(_ONE_MINUS_Y, i), c))
    return out


@lru_cache(maxsize=None)
def rational_part_row(i: int, t: int) -> tuple[Fraction, ...]:
    """Polynomial (in y) multiplying ``s_t`` in the rational part of ``(y(1-y))^i M^(i)``.

    For ``s_t(y) = (y - 1/2)^t`` this is
    ``y^i (1-y)^i * sum_{j=1}^{i} C(i, j) s_t^(i-j)(y) logit^(j)(y)`` with
    ``logit^(j)(y) = (j-1)! [y^j - (-1)^j (1-y)^j] / [y^j (1-y)^j]``. By Lemma 4 its
    degree is at most ``i - 1`` whenever ``t < i``.

    Args:
        i: Derivative order (>= 1).
        t: Power of (y - 1/2) in the s-polynomial term.

    Returns:
        Ascending coefficients in y.
    """
    s = _ppow(_SHIFT, t)
    acc: Poly = [Fraction(0)]
    for j in range(1, i + 1):
        num = _padd(
            _ppow(_Y, j), _pscale(_ppow(_ONE_MINUS_Y, j), Fraction(-((-1) ** j)))
        )
        term = _pmul(
            _pmul(_pder(s, i - j), num),
            _pmul(_ppow(_Y, i - j), _ppow(_ONE_MINUS_Y, i - j)),
        )
        acc = _padd(acc, _pscale(term, Fraction(comb(i, j) * factorial(j - 1))))
    return tuple(_trim(acc))


def rational_part_tables(
    i: int, num_s: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Rational-part rows for ``t = 0..num_s-1`` in the y- and (1-y)-bases, zero padded.

    Evaluating near y = 1 in the ``z = 1 - y`` basis avoids cancellation.

    Args:
        i: Derivative order.
        num_s: Number of s-polynomial coefficients.

    Returns:
        ``(rows_y, rows_z)`` as nested float lists of shape (num_s, L).
    """
    rows_y = [list(rational_part_row(i, t)) for t in range(num_s)]
    rows_z = [_trim(_to_one_minus_y_basis(r)) for r in rows_y]
    width = max([len(r) for r in rows_y] + [len(r) for r in rows_z] + [1])

    def pad(r: Poly) -> list[float]:
        return [float(c) for c in r] + [0.0] * (width - len(r))

    return [pad(r) for r in rows_y], [pad(r) for r in rows_z]


# ----------------------------------------------------------------------------------
# Lemma 1: I(m, u) = int_0^1 (y - 1/2)^m logit(y)^u dy, exactly as polynomials in pi^2
# ----------------------------------------------------------------------------------
PiPoly = dict[int, Fraction]  # {e: q} represents sum_e q * pi^(2e)


def _vadd(a: PiPoly, b: PiPoly, c: Fraction = Fraction(1)) -> PiPoly:
    out = dict(a)
    for e, q in b.items():
        out[e] = out.get(e, Fraction(0)) + c * q
    return {e: q for e, q in out.items() if q != 0}


def _eta(jmax: int) -> dict[int, Fraction]:
    eta = {1: Fraction(1, 6)}
    for j in range(2, jmax + 1):
        v = Fraction((-1) ** (j + 1) * j, factorial(2 * j + 1))
        for i in range(1, j):
            v += Fraction((-1) ** (i - 1), factorial(2 * i + 1)) * eta[j - i]
        eta[j] = v
    return eta


def _s_entry(S: dict[tuple[int, int], PiPoly], n: int, u: int) -> PiPoly:
    """Recurrence for ``S(n, u)`` from entries with smaller ``n`` or ``u``."""
    acc: PiPoly = {0: Fraction(1, n) - Fraction(1, (n + 1) ** 2)}
    for k in range(1, u + 1):
        acc = _vadd(
            acc,
            _vadd(S[(n - 1, k)], S[(n, k - 1)], Fraction(-1)),
            Fraction(n, n + 1),
        )
    for k in range(1, u):
        acc = _vadd(
            acc,
            _vadd(S[(n - 1, k)], S[(n, k)], Fraction(-1)),
            Fraction(1, n + 1),
        )
    return acc


def _s_table(mmax: int, umax: int) -> dict[tuple[int, int], PiPoly]:
    """``S(n, u)`` (moments about y = 0, scaled by 1/u!) for ``n <= mmax, u <= umax``."""
    eta = _eta(max(1, umax // 2))
    S: dict[tuple[int, int], PiPoly] = {}
    for n in range(mmax + 1):
        S[(n, 0)] = {0: Fraction(1, n + 1)}
    for u in range(1, umax + 1):
        S[(0, u)] = (
            {} if u % 2 else {u // 2: 2 * (1 - Fraction(2) ** (1 - u)) * eta[u // 2]}
        )
    for n in range(1, mmax + 1):
        for u in range(1, umax + 1):
            S[(n, u)] = _s_entry(S, n, u)
    return S


def _centered_entry(S: dict[tuple[int, int], PiPoly], m: int, u: int) -> PiPoly:
    """``I(m, u)``: shift ``S(., u)`` from y^n to (y - 1/2)^m by the binomial theorem."""
    if (m + u) % 2:
        return {}  # odd integrand about y = 1/2
    acc: PiPoly = {}
    for n in range(m + 1):
        acc = _vadd(acc, S[(n, u)], Fraction(comb(m, n)) * Fraction(-1, 2) ** (m - n))
    return {e: q * factorial(u) for e, q in acc.items()}


@lru_cache(maxsize=None)
def _i_table(mmax: int, umax: int) -> dict[tuple[int, int], PiPoly]:
    S = _s_table(mmax, umax)
    return {
        (m, u): _centered_entry(S, m, u)
        for m in range(mmax + 1)
        for u in range(umax + 1)
    }


def i_matrix(mmax: int, umax: int) -> list[list[float]]:
    """``I(m, u)`` for ``0 <= m <= mmax``, ``0 <= u <= umax`` as floats (Lemma 1).

    Args:
        mmax: Largest power of (y - 1/2).
        umax: Largest power of logit(y).

    Returns:
        Nested list of shape (mmax + 1, umax + 1).
    """
    table = _i_table(mmax, umax)
    return [
        [
            sum(float(q) * pi ** (2 * e) for e, q in table[(m, u)].items())
            for u in range(umax + 1)
        ]
        for m in range(mmax + 1)
    ]
