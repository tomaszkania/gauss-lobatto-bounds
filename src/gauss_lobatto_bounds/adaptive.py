"""
Certified integration strategies based on refined Gauss–Lobatto bounds.

This module implements practical strategies for producing certified
approximations of ∫ f on an interval when f is known to be (2n-1)-convex or
(2n-1)-concave:

1. **Greedy bisection** (adaptive): repeatedly bisect the subinterval with the
   largest local certified bound.

2. **Uniform partition** (non-adaptive): use N equal subintervals and the
   composite estimator. Optionally search for the minimal N that reaches a
   prescribed tolerance.

3. **Convergence analysis**: tools for estimating and verifying convergence
   rates.

The code is intended as a companion to the paper

    "Refined Gauss–Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

Notes
-----
British English is used throughout the documentation.

Version
-------
1.1.0
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .rules import (
    QuadratureRule,
    ScalarFunc,
    gauss_legendre_rule,
    gauss_lobatto_rule,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "IntegrationResult",
    "FunctionCache",
    "CertifiedAdaptiveIntegrator",
    "integrate_uniform",
    "find_min_intervals_uniform",
    "CompositeRule",
    "estimate_convergence_rate",
]

__version__ = "1.1.0"


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """
    Result of a certified integration run.

    This dataclass encapsulates the outcome of a certified quadrature
    computation, including the approximation, error bound, and diagnostics.

    Parameters
    ----------
    approx : float
        The computed approximation (composite estimator Q_n).
    bound : float
        Certified a posteriori bound on the absolute error.
    n_intervals : int
        Number of subintervals in the final partition.
    num_evals : int
        Number of *distinct* function evaluations used (with caching).
    strategy : {'greedy_bisection', 'uniform_fixed', 'uniform_minimal'}
        Strategy identifier.

    Notes
    -----
    The certified bound satisfies |I[f] - approx| ≤ bound for any (2n-1)-convex
    or (2n-1)-concave integrand.
    """

    approx: float
    bound: float
    n_intervals: int
    num_evals: int
    strategy: Literal["greedy_bisection", "uniform_fixed", "uniform_minimal"]

    @property
    def is_certified(self) -> bool:
        """Return True if a finite certified bound is available."""
        return bool(np.isfinite(self.bound))

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"IntegrationResult(approx={self.approx:.15e}, "
            f"bound={self.bound:.6e}, "
            f"n_intervals={self.n_intervals}, "
            f"num_evals={self.num_evals}, "
            f"strategy='{self.strategy}')"
        )


class FunctionCache:
    """
    Cache for scalar function evaluations f(x) with an evaluation counter.

    This class wraps a scalar function and caches its values, avoiding
    redundant computations when the same point is evaluated multiple times
    (e.g., at shared endpoints in composite quadrature).

    Parameters
    ----------
    f : callable
        The scalar function to wrap.

    Notes
    -----
    Values are cached using the floating-point value of x as the key. This is
    safe for reusing shared endpoints in composite quadrature, because those
    endpoints are generated deterministically (exact bitwise repeats).

    If you generate nodes by different arithmetic paths and want aggressive
    caching, consider quantising keys yourself before calling the cache.

    Examples
    --------
    >>> import math
    >>> cache = FunctionCache(math.sin)
    >>> cache(0.5)
    0.479425538604203
    >>> cache(0.5)  # Cached, no recomputation
    0.479425538604203
    >>> cache.num_evals
    1
    """

    def __init__(self, f: ScalarFunc) -> None:
        self._f: ScalarFunc = f
        self._cache: dict[float, float] = {}
        self._num_evals: int = 0

    @property
    def num_evals(self) -> int:
        """Return the number of distinct points at which f was evaluated."""
        return self._num_evals

    @property
    def cache_size(self) -> int:
        """Return the current size of the cache."""
        return len(self._cache)

    def __call__(self, x: float) -> float:
        """
        Evaluate f(x) with caching.

        Parameters
        ----------
        x : float
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

    def clear(self) -> None:
        """Clear the cache and reset the evaluation counter."""
        self._cache.clear()
        self._num_evals = 0

    def get_cached_points(self) -> list[float]:
        """Return a sorted list of all cached evaluation points."""
        return sorted(self._cache.keys())


@dataclass(slots=True)
class _IntervalEstimate:
    """Internal representation of a local quadrature estimate on a subinterval."""

    a: float
    b: float
    q: float
    bound: float
    gauss_val: float = 0.0
    lobatto_val: float = 0.0


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
    f : callable
        Integrand.
    interval : tuple of float
        (a, b) with a < b.
    rule_g : QuadratureRule
        Gauss–Legendre rule G_n on [-1, 1].
    rule_l : QuadratureRule
        Gauss–Lobatto rule L_{n+1} on [-1, 1].
    eval_fn : callable, optional
        Optional cached evaluator.

    Returns
    -------
    _IntervalEstimate
        Local data for the interval including Q_n value and certified bound.
    """
    a, b = interval
    gauss_val = rule_g.apply(f, a=a, b=b, eval_fn=eval_fn)
    lobatto_val = rule_l.apply(f, a=a, b=b, eval_fn=eval_fn)

    # Q_n = (3/4) G_n + (1/4) L_{n+1}
    q_val = 0.75 * gauss_val + 0.25 * lobatto_val

    # Certified bound: (1/4) |L_{n+1} - G_n|
    certified_bound = 0.25 * abs(lobatto_val - gauss_val)

    return _IntervalEstimate(
        a=a,
        b=b,
        q=float(q_val),
        bound=float(certified_bound),
        gauss_val=float(gauss_val),
        lobatto_val=float(lobatto_val),
    )


def integrate_uniform(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
    n_intervals: int,
) -> IntegrationResult:
    """
    Compute the composite certified estimator on a uniform partition.

    Parameters
    ----------
    f : callable
        Integrand.
    interval : tuple of float
        (a, b) with a < b.
    n : int
        Parameter n from the paper (odd-order convexity 2n - 1).
    n_intervals : int
        Number of equal subintervals.

    Returns
    -------
    IntegrationResult
        Approximation, certified bound, and evaluation statistics.

    Raises
    ------
    ValueError
        If n_intervals < 1 or a >= b.

    Examples
    --------
    >>> import math
    >>> result = integrate_uniform(f=math.exp, interval=(0.0, 1.0), n=3, n_intervals=10)
    >>> abs(result.approx - (math.e - 1.0)) < result.bound
    True
    """
    if n_intervals < 1:
        raise ValueError(
            f"Number of intervals must be at least 1, but received {n_intervals}."
        )

    a, b = interval
    if not (a < b):
        raise ValueError(f"Require a < b, but received a={a}, b={b}.")

    rule_g = gauss_legendre_rule(n)
    rule_l = gauss_lobatto_rule(n)

    cache = FunctionCache(f)

    # Compute boundaries once to guarantee exact repeats at shared endpoints.
    boundaries = np.linspace(a, b, n_intervals + 1, dtype=np.float64)

    total_approx = 0.0
    total_bound = 0.0

    for j in range(n_intervals):
        est = _local_estimate_Qn(
            f=f,
            interval=(float(boundaries[j]), float(boundaries[j + 1])),
            rule_g=rule_g,
            rule_l=rule_l,
            eval_fn=cache,
        )
        total_approx += est.q
        total_bound += est.bound

    return IntegrationResult(
        approx=float(total_approx),
        bound=float(total_bound),
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
    Find the minimal number of equal subintervals N such that bound ≤ tol.

    This routine uses exponential search followed by binary search to find
    the smallest N. It assumes (and empirically observes for benchmark problems)
    that the uniform composite bound is non-increasing in N.

    Parameters
    ----------
    f : callable
        Integrand.
    interval : tuple of float
        (a, b) with a < b.
    n : int
        Parameter n from the paper (odd-order convexity 2n - 1).
    tol : float
        Target tolerance for the certified bound.
    n_start : int, default=1
        Starting value for the search (useful for sequences of tolerances).
    max_intervals : int, default=200_000
        Hard cap on N to prevent runaway searches.

    Returns
    -------
    IntegrationResult
        Result for the minimal N. The `strategy` is set to "uniform_minimal".

    Raises
    ------
    ValueError
        If tol <= 0, n_start < 1, or max_intervals < 1.
    RuntimeError
        If the tolerance cannot be reached within max_intervals.

    Examples
    --------
    >>> import math
    >>> result = find_min_intervals_uniform(
    ...     f=math.exp, interval=(0.0, 1.0), n=3, tol=1e-8
    ... )
    >>> result.bound <= 1e-8
    True
    """
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, but received tol={tol}.")
    if n_start < 1:
        raise ValueError(f"Starting value must be at least 1, but received n_start={n_start}.")
    if max_intervals < 1:
        raise ValueError(
            f"Maximum intervals must be at least 1, but received max_intervals={max_intervals}."
        )

    # Cache results of expensive evaluations for specific N.
    computed: dict[int, IntegrationResult] = {}

    def _get(num_intervals: int) -> IntegrationResult:
        if num_intervals not in computed:
            computed[num_intervals] = integrate_uniform(
                f=f, interval=interval, n=n, n_intervals=num_intervals
            )
        return computed[num_intervals]

    # Check if already good.
    res0 = _get(n_start)
    if res0.bound <= tol:
        return IntegrationResult(
            approx=res0.approx,
            bound=res0.bound,
            n_intervals=res0.n_intervals,
            num_evals=res0.num_evals,
            strategy="uniform_minimal",
        )

    # Exponential search for upper bound.
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

    result = _get(left)
    return IntegrationResult(
        approx=result.approx,
        bound=result.bound,
        n_intervals=result.n_intervals,
        num_evals=result.num_evals,
        strategy="uniform_minimal",
    )


class CertifiedAdaptiveIntegrator:
    """
    Greedy certified adaptive integrator based on local Q_n bounds.

    This integrator uses a greedy bisection strategy: it repeatedly bisects
    the subinterval with the largest local certified bound until the global
    bound reaches the target tolerance.

    Parameters
    ----------
    n : int
        Parameter n from the paper. The estimator is Q_n = (3/4) G_n + (1/4) L_{n+1}.
    tol : float
        Target tolerance for the *global* certified bound.
    max_intervals : int, default=200_000
        Maximal number of subintervals allowed.

    Raises
    ------
    ValueError
        If n < 1, tol <= 0, or max_intervals < 1.

    Examples
    --------
    >>> import math
    >>> integrator = CertifiedAdaptiveIntegrator(n=3, tol=1e-10)
    >>> result = integrator.integrate(f=math.exp, interval=(0.0, 1.0))
    >>> result.bound <= 1e-10
    True
    """

    def __init__(
        self,
        *,
        n: int,
        tol: float,
        max_intervals: int = 200_000,
    ) -> None:
        if n < 1:
            raise ValueError(f"Parameter n must be at least 1, but received n={n}.")
        if tol <= 0:
            raise ValueError(f"Tolerance must be positive, but received tol={tol}.")
        if max_intervals < 1:
            raise ValueError(
                f"Maximum intervals must be at least 1, but received max_intervals={max_intervals}."
            )

        self.n: int = int(n)
        self.tol: float = float(tol)
        self.max_intervals: int = int(max_intervals)

        self._rule_g: QuadratureRule = gauss_legendre_rule(self.n)
        self._rule_l: QuadratureRule = gauss_lobatto_rule(self.n)

    def integrate(
        self,
        *,
        f: ScalarFunc,
        interval: tuple[float, float],
    ) -> IntegrationResult:
        """
        Run greedy bisection until the certified global bound ≤ tol.

        Parameters
        ----------
        f : callable
            Integrand.
        interval : tuple of float
            (a, b) with a < b.

        Returns
        -------
        IntegrationResult
            Approximation, certified bound, and evaluation statistics.

        Raises
        ------
        ValueError
            If a >= b.
        RuntimeError
            If max_intervals is reached before meeting the tolerance.
        """
        a, b = interval
        if not (a < b):
            raise ValueError(f"Require a < b, but received a={a}, b={b}.")

        cache = FunctionCache(f)

        # Initial interval.
        est0 = _local_estimate_Qn(
            f=f,
            interval=(a, b),
            rule_g=self._rule_g,
            rule_l=self._rule_l,
            eval_fn=cache,
        )

        total_approx = est0.q
        total_bound = est0.bound

        # Max-heap keyed by local bound (negated for max-heap behaviour).
        heap: list[tuple[float, int, _IntervalEstimate]] = []
        counter = 0
        heapq.heappush(heap, (-est0.bound, counter, est0))

        while total_bound > self.tol:
            if len(heap) >= self.max_intervals:
                raise RuntimeError(
                    f"Reached max_intervals={self.max_intervals} before meeting "
                    f"tol={self.tol:g}. Current bound: {total_bound:g}."
                )

            _, _, est = heapq.heappop(heap)

            # Remove old interval contribution.
            total_approx -= est.q
            total_bound -= est.bound

            # Bisect the interval.
            mid = 0.5 * (est.a + est.b)

            left_est = _local_estimate_Qn(
                f=f,
                interval=(est.a, mid),
                rule_g=self._rule_g,
                rule_l=self._rule_l,
                eval_fn=cache,
            )
            right_est = _local_estimate_Qn(
                f=f,
                interval=(mid, est.b),
                rule_g=self._rule_g,
                rule_l=self._rule_l,
                eval_fn=cache,
            )

            # Add new contributions.
            total_approx += left_est.q + right_est.q
            total_bound += left_est.bound + right_est.bound

            counter += 1
            heapq.heappush(heap, (-left_est.bound, counter, left_est))
            counter += 1
            heapq.heappush(heap, (-right_est.bound, counter, right_est))

        return IntegrationResult(
            approx=float(total_approx),
            bound=float(total_bound),
            n_intervals=int(len(heap)),
            num_evals=int(cache.num_evals),
            strategy="greedy_bisection",
        )


@dataclass(frozen=True, slots=True)
class CompositeRule:
    """
    Composite quadrature rule on a partitioned interval.

    This class represents a composite application of a base quadrature rule
    over multiple subintervals, enabling higher accuracy through refinement.

    Parameters
    ----------
    base_rule_gauss : QuadratureRule
        The base Gauss–Legendre rule.
    base_rule_lobatto : QuadratureRule
        The base Gauss–Lobatto rule.
    interval : tuple of float
        The overall integration interval (a, b).
    n_subintervals : int
        Number of subintervals.

    Examples
    --------
    >>> rule = CompositeRule.create(n=3, interval=(0.0, 1.0), n_subintervals=10)
    >>> import math
    >>> rule.apply_certified(math.exp)
    (1.7182818284590453, 7.12e-12)  # (approximation, bound)
    """

    base_rule_gauss: QuadratureRule
    base_rule_lobatto: QuadratureRule
    interval: tuple[float, float]
    n_subintervals: int
    boundaries: np.ndarray = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        n: int,
        interval: tuple[float, float],
        n_subintervals: int,
    ) -> CompositeRule:
        """
        Create a composite rule.

        Parameters
        ----------
        n : int
            Parameter n (odd-order convexity 2n - 1).
        interval : tuple of float
            Integration interval (a, b).
        n_subintervals : int
            Number of subintervals.

        Returns
        -------
        CompositeRule
            A composite rule ready for evaluation.
        """
        a, b = interval
        if not (a < b):
            raise ValueError(f"Require a < b, but received a={a}, b={b}.")
        if n_subintervals < 1:
            raise ValueError(
                f"Number of subintervals must be at least 1, but received {n_subintervals}."
            )

        boundaries = np.linspace(a, b, n_subintervals + 1, dtype=np.float64)

        return cls(
            base_rule_gauss=gauss_legendre_rule(n),
            base_rule_lobatto=gauss_lobatto_rule(n),
            interval=interval,
            n_subintervals=n_subintervals,
            boundaries=boundaries,
        )

    def apply_certified(
        self,
        f: ScalarFunc,
        *,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Apply the composite rule and return (approximation, certified_bound).

        Parameters
        ----------
        f : callable
            Integrand.
        use_cache : bool, default=True
            Whether to cache function evaluations.

        Returns
        -------
        tuple of float
            (Q_n approximation, certified error bound).
        """
        cache = FunctionCache(f) if use_cache else None
        eval_fn = cache if cache is not None else None

        total_approx = 0.0
        total_bound = 0.0

        for j in range(self.n_subintervals):
            a_j, b_j = float(self.boundaries[j]), float(self.boundaries[j + 1])

            gauss_val = self.base_rule_gauss.apply(f, a=a_j, b=b_j, eval_fn=eval_fn)
            lobatto_val = self.base_rule_lobatto.apply(f, a=a_j, b=b_j, eval_fn=eval_fn)

            total_approx += 0.75 * gauss_val + 0.25 * lobatto_val
            total_bound += 0.25 * abs(lobatto_val - gauss_val)

        return float(total_approx), float(total_bound)

    def subinterval_bounds(self, f: ScalarFunc) -> np.ndarray:
        """
        Return an array of certified bounds for each subinterval.

        Parameters
        ----------
        f : callable
            Integrand.

        Returns
        -------
        ndarray
            Array of shape (n_subintervals,) with local certified bounds.
        """
        bounds = np.empty(self.n_subintervals, dtype=np.float64)

        for j in range(self.n_subintervals):
            a_j, b_j = float(self.boundaries[j]), float(self.boundaries[j + 1])

            gauss_val = self.base_rule_gauss.apply(f, a=a_j, b=b_j)
            lobatto_val = self.base_rule_lobatto.apply(f, a=a_j, b=b_j)

            bounds[j] = 0.25 * abs(lobatto_val - gauss_val)

        return bounds


def estimate_convergence_rate(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
    interval_counts: list[int],
    exact: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Estimate the convergence rate of the certified estimator.

    Computes the approximation error (if exact value is known) and certified
    bound for various numbers of subintervals, enabling convergence analysis.

    Parameters
    ----------
    f : callable
        Integrand.
    interval : tuple of float
        Integration interval (a, b).
    n : int
        Parameter n (odd-order convexity 2n - 1).
    interval_counts : list of int
        List of subinterval counts to test.
    exact : float, optional
        Exact value of the integral, if known.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'n_intervals': array of interval counts
        - 'bounds': array of certified bounds
        - 'errors': array of true errors (if exact is provided)
        - 'bound_rate': estimated convergence rate for bounds
        - 'error_rate': estimated convergence rate for errors (if exact is provided)

    Examples
    --------
    >>> import math
    >>> results = estimate_convergence_rate(
    ...     f=math.exp,
    ...     interval=(0.0, 1.0),
    ...     n=3,
    ...     interval_counts=[10, 20, 40, 80],
    ...     exact=math.e - 1.0,
    ... )
    >>> results['bound_rate']  # Should be approximately 2n
    5.98...
    """
    interval_counts = sorted(interval_counts)
    bounds = []
    errors = []

    for num_int in interval_counts:
        result = integrate_uniform(f=f, interval=interval, n=n, n_intervals=num_int)
        bounds.append(result.bound)
        if exact is not None:
            errors.append(abs(result.approx - exact))

    n_arr = np.array(interval_counts, dtype=np.float64)
    bounds_arr = np.array(bounds, dtype=np.float64)

    # Estimate convergence rate from log-log slope.
    log_n = np.log(n_arr)
    log_bounds = np.log(bounds_arr + 1e-50)  # Avoid log(0).

    # Linear regression for rate: bound ~ C / N^rate.
    bound_rate = -float(np.polyfit(log_n, log_bounds, 1)[0])

    result_dict: dict[str, np.ndarray] = {
        "n_intervals": n_arr,
        "bounds": bounds_arr,
        "bound_rate": np.array([bound_rate]),
    }

    if exact is not None and errors:
        errors_arr = np.array(errors, dtype=np.float64)
        log_errors = np.log(errors_arr + 1e-50)
        error_rate = -float(np.polyfit(log_n, log_errors, 1)[0])
        result_dict["errors"] = errors_arr
        result_dict["error_rate"] = np.array([error_rate])

    return result_dict
