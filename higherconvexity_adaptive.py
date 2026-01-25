
"""
Certified integration strategies based on refined Gauss--Lobatto bounds.

This module implements two practical strategies for producing certified
approximations of ∫ f on an interval when f is known to be (2n-1)-convex or
(2n-1)-concave:

1) Greedy bisection (adaptive): repeatedly bisect the subinterval with the
   largest local certified bound.

2) Uniform partition (non-adaptive): use N equal subintervals and the composite
   estimator. Optionally search for the minimal N that reaches a prescribed
   tolerance.

The code is intended as a companion to the paper

    "Refined Gauss--Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

Version
-------
0.99.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import heapq
import numpy as np

from higherconvexity_rules import QuadratureRule, ScalarFunc, gauss_legendre_rule, gauss_lobatto_rule

__all__ = [
    "IntegrationResult",
    "FunctionCache",
    "CertifiedAdaptiveIntegrator",
    "integrate_uniform",
    "find_min_intervals_uniform",
]

__version__ = "0.99.1"


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """
    Result of a certified integration run.

    Parameters
    ----------
    approx:
        The computed approximation (composite estimator Q_n).
    bound:
        Certified a posteriori bound on the absolute error.
    n_intervals:
        Number of subintervals in the final partition.
    num_evals:
        Number of *distinct* function evaluations used (with caching).
    strategy:
        Strategy identifier.
    """

    approx: float
    bound: float
    n_intervals: int
    num_evals: int
    strategy: Literal["greedy_bisection", "uniform_fixed", "uniform_minimal"]


class FunctionCache:
    """
    Cache for scalar function evaluations f(x) with an evaluation counter.

    Notes
    -----
    We cache values keyed by the floating-point value of x. This is safe for
    reusing shared endpoints in composite quadrature, because those endpoints
    are generated deterministically (exact bitwise repeats inside the code).

    If you generate nodes by different arithmetic paths and want aggressive
    caching, consider quantising keys yourself before calling the cache.
    """

    def __init__(self, f: ScalarFunc) -> None:
        self._f: ScalarFunc = f
        self._cache: dict[float, float] = {}
        self._num_evals: int = 0

    @property
    def num_evals(self) -> int:
        """Number of distinct points at which f was evaluated."""
        return self._num_evals

    def __call__(self, x: float) -> float:
        """
        Evaluate f(x) with caching.

        Parameters
        ----------
        x:
            Point at which to evaluate the integrand.

        Returns
        -------
        float
            The value f(x).
        """
        key = float(x)
        if key in self._cache:
            return self._cache[key]
        val = float(self._f(key))
        self._cache[key] = val
        self._num_evals += 1
        return val


@dataclass(slots=True)
class _IntervalEstimate:
    a: float
    b: float
    q: float
    bound: float


def _local_estimate_Qn(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    rule_g: QuadratureRule,
    rule_l: QuadratureRule,
    eval_fn: ScalarFunc | None = None,
) -> _IntervalEstimate:
    """
    Compute the local Q_n estimate and its certified bound on an interval.

    Parameters
    ----------
    f:
        Integrand.
    interval:
        (a, b) with a < b.
    rule_g:
        Gauss--Legendre rule G_n on [-1, 1].
    rule_l:
        Gauss--Lobatto rule L_{n+1} on [-1, 1].
    eval_fn:
        Optional cached evaluator.

    Returns
    -------
    _IntervalEstimate
        Local data for the interval.
    """
    a, b = interval
    G = rule_g.apply(f, a=a, b=b, eval_fn=eval_fn)
    L = rule_l.apply(f, a=a, b=b, eval_fn=eval_fn)
    q = 0.75 * G + 0.25 * L
    bound = 0.25 * abs(L - G)
    return _IntervalEstimate(a=a, b=b, q=float(q), bound=float(bound))


def integrate_uniform(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
    n_intervals: int,
) -> IntegrationResult:
    """
    Composite certified estimator on a uniform partition.

    Parameters
    ----------
    f:
        Integrand.
    interval:
        (a, b) with a < b.
    n:
        Parameter n from the paper (odd-order convexity 2n-1).
    n_intervals:
        Number of equal subintervals.

    Returns
    -------
    IntegrationResult
        Approximation, certified bound, and evaluation statistics.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    a, b = interval
    if not (a < b):
        raise ValueError("Require a < b")

    rule_g = gauss_legendre_rule(n)
    rule_l = gauss_lobatto_rule(n)

    cache = FunctionCache(f)

    # Use boundaries computed once to guarantee exact repeats at shared endpoints.
    boundaries = np.linspace(a, b, n_intervals + 1, dtype=float)

    approx = 0.0
    bound = 0.0
    for j in range(n_intervals):
        est = _local_estimate_Qn(
            f=f,
            interval=(float(boundaries[j]), float(boundaries[j + 1])),
            rule_g=rule_g,
            rule_l=rule_l,
            eval_fn=cache,
        )
        approx += est.q
        bound += est.bound

    return IntegrationResult(
        approx=float(approx),
        bound=float(bound),
        n_intervals=int(n_intervals),
        num_evals=int(cache.num_evals),
        strategy="uniform_fixed",
    )


def find_min_intervals_uniform(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
    tol: float,
    n_start: int = 1,
    max_intervals: int = 200_000,
) -> IntegrationResult:
    """
    Find the minimal number of equal subintervals N such that the certified bound <= tol.

    This routine assumes (and empirically observes for our benchmark problems)
    that the uniform composite bound is non-increasing in N.

    Parameters
    ----------
    f:
        Integrand.
    interval:
        (a, b) with a < b.
    n:
        Parameter n from the paper (odd-order convexity 2n-1).
    tol:
        Target tolerance for the certified bound.
    n_start:
        Starting value for the search (useful if you run a sequence of tolerances).
    max_intervals:
        Hard cap on N to prevent runaway searches.

    Returns
    -------
    IntegrationResult
        Result for the minimal N. The `strategy` is set to "uniform_minimal".
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    if n_start < 1:
        raise ValueError("n_start must be >= 1")
    if max_intervals < 1:
        raise ValueError("max_intervals must be >= 1")

    # Cache results of expensive evaluations for specific N.
    computed: dict[int, IntegrationResult] = {}

    def _get(N: int) -> IntegrationResult:
        if N not in computed:
            computed[N] = integrate_uniform(f=f, interval=interval, n=n, n_intervals=N)
        return computed[N]

    # If already good, return immediately.
    res0 = _get(n_start)
    if res0.bound <= tol:
        return IntegrationResult(
            approx=res0.approx,
            bound=res0.bound,
            n_intervals=res0.n_intervals,
            num_evals=res0.num_evals,
            strategy="uniform_minimal",
        )

    lo = n_start
    hi = max(2 * n_start, n_start + 1)

    while True:
        if hi > max_intervals:
            raise RuntimeError(
                f"Could not reach tol={tol:g} with N up to max_intervals={max_intervals}."
            )
        res_hi = _get(hi)
        if res_hi.bound <= tol:
            break
        lo = hi
        hi *= 2

    # Binary search for minimal N in (lo, hi].
    left = lo + 1
    right = hi
    while left < right:
        mid = (left + right) // 2
        if _get(mid).bound <= tol:
            right = mid
        else:
            left = mid + 1

    res = _get(left)
    return IntegrationResult(
        approx=res.approx,
        bound=res.bound,
        n_intervals=res.n_intervals,
        num_evals=res.num_evals,
        strategy="uniform_minimal",
    )


class CertifiedAdaptiveIntegrator:
    """
    Greedy certified adaptive integrator based on local Q_n bounds.

    Parameters
    ----------
    n:
        Parameter n from the paper. The estimator is Q_n = 3/4 G_n + 1/4 L_{n+1}.
    tol:
        Target tolerance for the *global* certified bound.
    max_intervals:
        Maximal number of subintervals allowed.
    """

    def __init__(self, *, n: int, tol: float, max_intervals: int = 200_000) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_intervals < 1:
            raise ValueError("max_intervals must be >= 1")
        self.n: int = int(n)
        self.tol: float = float(tol)
        self.max_intervals: int = int(max_intervals)

        self._rule_g: QuadratureRule = gauss_legendre_rule(self.n)
        self._rule_l: QuadratureRule = gauss_lobatto_rule(self.n)

    def integrate(self, *, f: ScalarFunc, interval: tuple[float, float]) -> IntegrationResult:
        """
        Run greedy bisection until the certified global bound <= tol.

        Parameters
        ----------
        f:
            Integrand.
        interval:
            (a, b) with a < b.

        Returns
        -------
        IntegrationResult
            Approximation, certified bound, and evaluation statistics.
        """
        a, b = interval
        if not (a < b):
            raise ValueError("Require a < b")

        cache = FunctionCache(f)

        # Initial interval.
        est0 = _local_estimate_Qn(
            f=f, interval=(a, b), rule_g=self._rule_g, rule_l=self._rule_l, eval_fn=cache
        )

        approx = est0.q
        bound = est0.bound

        # Max-heap keyed by local bound.
        heap: list[tuple[float, int, _IntervalEstimate]] = []
        counter = 0
        heapq.heappush(heap, (-est0.bound, counter, est0))

        while bound > self.tol:
            if len(heap) >= self.max_intervals:
                raise RuntimeError(
                    f"Reached max_intervals={self.max_intervals} before meeting tol={self.tol:g}."
                )

            neg_bnd, _, est = heapq.heappop(heap)
            # Remove old interval contribution.
            approx -= est.q
            bound -= est.bound

            mid = 0.5 * (est.a + est.b)
            left = _local_estimate_Qn(
                f=f,
                interval=(est.a, mid),
                rule_g=self._rule_g,
                rule_l=self._rule_l,
                eval_fn=cache,
            )
            right = _local_estimate_Qn(
                f=f,
                interval=(mid, est.b),
                rule_g=self._rule_g,
                rule_l=self._rule_l,
                eval_fn=cache,
            )

            # Add new contributions.
            approx += left.q + right.q
            bound += left.bound + right.bound

            counter += 1
            heapq.heappush(heap, (-left.bound, counter, left))
            counter += 1
            heapq.heappush(heap, (-right.bound, counter, right))

        return IntegrationResult(
            approx=float(approx),
            bound=float(bound),
            n_intervals=int(len(heap)),
            num_evals=int(cache.num_evals),
            strategy="greedy_bisection",
        )
