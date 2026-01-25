"""
Experiment helpers and benchmark suite.

This module provides tools for numerical experiments with the certified
Gauss–Lobatto quadrature estimator:

- A benchmark suite with functions of known exact integrals.
- Runners that produce tidy pandas DataFrames.
- Comparisons between estimator variants.
- Convergence rate analysis utilities.

The code is a computational companion to the paper

    "Refined Gauss–Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

Notes
-----
British English is used throughout the documentation.

Version
-------
1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import numpy as np

from .adaptive import (
    CertifiedAdaptiveIntegrator,
    IntegrationResult,
    find_min_intervals_uniform,
    integrate_uniform,
)
from .rules import ScalarFunc, gauss_legendre_rule, gauss_lobatto_rule

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BenchmarkProblem",
    "default_benchmark_suite",
    "extended_benchmark_suite",
    "run_suite",
    "run_suite_single_tol",
    "run_table_min_intervals_by_tol",
    "compare_Qn_vs_Pn",
    "run_experiment",
    "convergence_table",
]

__version__ = "1.0.0"


@dataclass(frozen=True, slots=True)
class BenchmarkProblem:
    """
    A benchmark integration problem.

    This dataclass encapsulates a test problem for quadrature, including
    the integrand, interval, exact value, and metadata.

    Parameters
    ----------
    name : str
        Short identifier (used in tables).
    f : callable
        Integrand.
    interval : tuple of float
        Integration interval (a, b).
    exact : float
        Exact value of the integral.
    convexity_order : int, optional
        The smallest m such that f is m-convex. Use -1 if unknown.
    notes : str, default=""
        Optional free-form notes (e.g., convexity/concavity information).

    Examples
    --------
    >>> import math
    >>> prob = BenchmarkProblem(
    ...     name="exp",
    ...     f=math.exp,
    ...     interval=(0.0, 1.0),
    ...     exact=math.e - 1.0,
    ...     notes="All derivatives positive.",
    ... )
    """

    name: str
    f: ScalarFunc
    interval: tuple[float, float]
    exact: float
    convexity_order: int = -1
    notes: str = ""


def default_benchmark_suite() -> list[BenchmarkProblem]:
    """
    Return the default benchmark suite used in the paper.

    Returns
    -------
    list of BenchmarkProblem
        A small list of problems with closed-form exact integrals.

    Notes
    -----
    The suite includes:
    - 1/x on [1, 2]: f^{(2n)} > 0 for all n, hence (2n-1)-convex.
    - exp on [0, 1]: all derivatives positive, hence (2n-1)-convex.
    - log(1+x) on [0, 1]: even derivatives negative, hence (2n-1)-concave.
    """

    def inv(x: float) -> float:
        """Reciprocal function."""
        return 1.0 / x

    def expf(x: float) -> float:
        """Exponential function."""
        return math.exp(x)

    def log1p_f(x: float) -> float:
        """Natural logarithm of (1 + x)."""
        return math.log1p(x)

    return [
        BenchmarkProblem(
            name="1/x on [1, 2]",
            f=inv,
            interval=(1.0, 2.0),
            exact=math.log(2.0),
            notes="(2n-1)-convex for all n (since f^{(2n)} > 0).",
        ),
        BenchmarkProblem(
            name="exp on [0, 1]",
            f=expf,
            interval=(0.0, 1.0),
            exact=math.e - 1.0,
            notes="(2n-1)-convex for all n (all derivatives positive).",
        ),
        BenchmarkProblem(
            name="log(1+x) on [0, 1]",
            f=log1p_f,
            interval=(0.0, 1.0),
            exact=2.0 * math.log(2.0) - 1.0,
            notes="(2n-1)-concave for all n (even derivatives negative).",
        ),
    ]


def extended_benchmark_suite() -> list[BenchmarkProblem]:
    """
    Return an extended benchmark suite with additional test functions.

    Returns
    -------
    list of BenchmarkProblem
        An extended list including oscillatory and polynomial test cases.

    Notes
    -----
    The extended suite adds:
    - sin(x) on [0, π]: Complete sinusoidal period.
    - cos(x) on [0, π/2]: Quarter period.
    - x^6 on [-1, 1]: Monomial (exactly integrable).
    - sqrt(x) on [0, 1]: Fractional power with endpoint singularity in derivative.
    - (1 + x)^{-2} on [0, 1]: Rapidly decaying function.
    """
    suite = default_benchmark_suite()

    def sin_f(x: float) -> float:
        """Sine function."""
        return math.sin(x)

    def cos_f(x: float) -> float:
        """Cosine function."""
        return math.cos(x)

    def x6(x: float) -> float:
        """Monomial x^6."""
        return x**6

    def sqrt_f(x: float) -> float:
        """Square root function."""
        return math.sqrt(x)

    def inv_sq(x: float) -> float:
        """Function 1/(1+x)²."""
        return 1.0 / (1.0 + x) ** 2

    def atan_f(x: float) -> float:
        """Arctangent function."""
        return math.atan(x)

    suite.extend(
        [
            BenchmarkProblem(
                name="sin on [0, π]",
                f=sin_f,
                interval=(0.0, math.pi),
                exact=2.0,
                notes="Complete period; sign changes in higher derivatives.",
            ),
            BenchmarkProblem(
                name="cos on [0, π/2]",
                f=cos_f,
                interval=(0.0, math.pi / 2),
                exact=1.0,
                notes="Quarter period.",
            ),
            BenchmarkProblem(
                name="x^6 on [-1, 1]",
                f=x6,
                interval=(-1.0, 1.0),
                exact=2.0 / 7.0,
                notes="Monomial; 6-convex (f^{(6)} = 720 > 0).",
            ),
            BenchmarkProblem(
                name="sqrt on [0, 1]",
                f=sqrt_f,
                interval=(0.0, 1.0),
                exact=2.0 / 3.0,
                notes="Endpoint singularity in derivative at x = 0.",
            ),
            BenchmarkProblem(
                name="1/(1+x)² on [0, 1]",
                f=inv_sq,
                interval=(0.0, 1.0),
                exact=0.5,
                notes="(2n-1)-convex for all n.",
            ),
            BenchmarkProblem(
                name="arctan on [0, 1]",
                f=atan_f,
                interval=(0.0, 1.0),
                exact=math.pi / 4 - 0.5 * math.log(2.0),
                notes="Higher derivatives change sign.",
            ),
        ]
    )

    return suite


def _ensure_pandas() -> "type[pd]":
    """Import and return the pandas module, or raise with a helpful message."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for this function. "
            "Install it via `pip install pandas`."
        ) from exc
    return pd


def run_suite(
    *,
    n: int,
    problems: Sequence[BenchmarkProblem],
    tol: float,
    method: Literal["uniform_minimal", "uniform_fixed", "greedy_bisection"] = "uniform_minimal",
    n_intervals: int = 1,
    max_intervals: int = 200_000,
) -> "pd.DataFrame":
    """
    Run a batch of certified integrations and return a pandas DataFrame.

    Parameters
    ----------
    n : int
        Parameter from the paper (odd order 2n - 1).
    problems : sequence of BenchmarkProblem
        Sequence of benchmark problems.
    tol : float
        Target tolerance for the certified bound.
    method : {'uniform_minimal', 'uniform_fixed', 'greedy_bisection'}, default='uniform_minimal'
        Integration strategy:
        - "uniform_minimal": search for the minimal N of equal subintervals such that bound ≤ tol;
        - "uniform_fixed": use exactly `n_intervals` equal subintervals;
        - "greedy_bisection": adaptive greedy bisection.
    n_intervals : int, default=1
        Used only when method="uniform_fixed".
    max_intervals : int, default=200_000
        Cap for the number of subintervals (used in searches / bisection).

    Returns
    -------
    pandas.DataFrame
        Table with approximation, certified bound, true error, and evaluation counts.

    Raises
    ------
    ValueError
        If tol <= 0 or an unknown method is specified.
    RuntimeError
        If pandas is not installed.

    Examples
    --------
    >>> problems = default_benchmark_suite()
    >>> df = run_suite(n=4, problems=problems, tol=1e-8)
    >>> df.columns.tolist()
    ['problem', 'interval', 'n', 'tol', 'method', 'approx', 'bound', ...]
    """
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, but received tol={tol}.")

    pd = _ensure_pandas()

    rows: list[dict[str, object]] = []

    for prob in problems:
        if method == "uniform_fixed":
            res = integrate_uniform(
                f=prob.f, interval=prob.interval, n=n, n_intervals=n_intervals
            )
        elif method == "uniform_minimal":
            res = find_min_intervals_uniform(
                f=prob.f,
                interval=prob.interval,
                n=n,
                tol=tol,
                n_start=1,
                max_intervals=max_intervals,
            )
        elif method == "greedy_bisection":
            integrator = CertifiedAdaptiveIntegrator(
                n=n, tol=tol, max_intervals=max_intervals
            )
            res = integrator.integrate(f=prob.f, interval=prob.interval)
        else:
            raise ValueError(f"Unknown method: {method}")

        error = float(res.approx - prob.exact)
        abs_error = abs(error)

        rows.append(
            {
                "problem": prob.name,
                "interval": prob.interval,
                "n": n,
                "tol": tol,
                "method": method,
                "approx": res.approx,
                "bound": res.bound,
                "abs_error": abs_error,
                "bound/abs_error": (res.bound / abs_error) if abs_error > 1e-50 else np.inf,
                "n_intervals": res.n_intervals,
                "evaluations": res.num_evals,
                "notes": prob.notes,
            }
        )

    return pd.DataFrame(rows)


def run_suite_single_tol(
    *,
    n: int,
    problems: Sequence[BenchmarkProblem],
    tol: float,
    method: Literal["uniform_minimal", "uniform_fixed", "greedy_bisection"] = "uniform_minimal",
    n_intervals: int = 1,
    max_intervals: int = 200_000,
) -> "pd.DataFrame":
    """
    Backwards-compatible wrapper mirroring interactive notebook usage.

    This function mirrors the signature used in earlier drafts:

        df = run_suite_single_tol(n=4, problems=problems, tol=1e-8)

    Parameters
    ----------
    n, problems, tol, method, n_intervals, max_intervals
        See `run_suite`.

    Returns
    -------
    pandas.DataFrame
        Same output as `run_suite`.
    """
    return run_suite(
        n=n,
        problems=problems,
        tol=tol,
        method=method,
        n_intervals=n_intervals,
        max_intervals=max_intervals,
    )


def run_experiment(
    *,
    n: int,
    problems: Sequence[BenchmarkProblem],
    tol: float,
    method: Literal["uniform_minimal", "uniform_fixed", "greedy_bisection"] = "uniform_minimal",
    n_intervals: int = 1,
    max_intervals: int = 200_000,
) -> dict[str, dict[str, object]]:
    """
    Backwards-compatible runner returning a plain dictionary.

    Earlier drafts used a dictionary output rather than a DataFrame. This helper
    keeps that interface available whilst `run_suite` remains the recommended
    entry point for tabular output.

    Parameters
    ----------
    n, problems, tol, method, n_intervals, max_intervals
        See `run_suite`.

    Returns
    -------
    dict
        Mapping problem name → a small results dictionary.
    """
    out: dict[str, dict[str, object]] = {}

    for prob in problems:
        if method == "uniform_fixed":
            res = integrate_uniform(
                f=prob.f, interval=prob.interval, n=n, n_intervals=n_intervals
            )
        elif method == "uniform_minimal":
            res = find_min_intervals_uniform(
                f=prob.f,
                interval=prob.interval,
                n=n,
                tol=tol,
                n_start=1,
                max_intervals=max_intervals,
            )
        elif method == "greedy_bisection":
            integrator = CertifiedAdaptiveIntegrator(
                n=n, tol=tol, max_intervals=max_intervals
            )
            res = integrator.integrate(f=prob.f, interval=prob.interval)
        else:
            raise ValueError(f"Unknown method: {method}")

        out[prob.name] = {
            "approx": res.approx,
            "bound": res.bound,
            "n_intervals": res.n_intervals,
            "evaluations": res.num_evals,
            "exact": prob.exact,
            "abs_error": abs(res.approx - prob.exact),
            "method": method,
        }

    return out


def run_table_min_intervals_by_tol(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n_list: Sequence[int],
    tol_list: Sequence[float],
    max_intervals: int = 200_000,
) -> "pd.DataFrame":
    """
    Produce a table of minimal N (uniform partitions) required to reach each tolerance.

    This function is intended to reproduce the style of tables shared during
    development (columns labelled by convexity order 2n - 1).

    Parameters
    ----------
    f : callable
        Integrand.
    interval : tuple of float
        (a, b) integration interval.
    n_list : sequence of int
        List of n values (e.g., [2, 3, 4, 5] corresponds to 3-, 5-, 7-, 9-convex).
    tol_list : sequence of float
        Tolerances to test.
    max_intervals : int, default=200_000
        Maximal N allowed in the search.

    Returns
    -------
    pandas.DataFrame
        A table with rows indexed by tolerance and columns "N_(2n-1)".

    Examples
    --------
    >>> import math
    >>> df = run_table_min_intervals_by_tol(
    ...     f=math.exp,
    ...     interval=(0.0, 1.0),
    ...     n_list=[2, 3, 4],
    ...     tol_list=[1e-4, 1e-6, 1e-8],
    ... )
    """
    pd = _ensure_pandas()

    # IMPORTANT: warm-starting n_start relies on the monotonicity
    #   minimal N(tol) is non-decreasing as tol decreases.
    # Therefore we iterate tolerances from coarse to tight (largest to smallest).
    tol_sorted = sorted((float(t) for t in tol_list), reverse=True)
    data: dict[str, list[int | float]] = {"tol": tol_sorted}

    for n in n_list:
        n_values: list[int] = []
        # Warm start: minimal N is non-decreasing as tolerance decreases.
        n_start = 1

        for tol in tol_sorted:
            res = find_min_intervals_uniform(
                f=f,
                interval=interval,
                n=int(n),
                tol=float(tol),
                n_start=n_start,
                max_intervals=max_intervals,
            )
            n_values.append(int(res.n_intervals))
            n_start = int(res.n_intervals)

        data[f"N_{2 * n - 1}"] = n_values

    return pd.DataFrame(data)


def _Qn_single_interval(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
) -> tuple[float, float, float]:
    """
    Compute Q_n, G_n, and L_{n+1} on a single interval.

    Returns (Q_n, G_n, L_{n+1}).
    """
    a, b = interval
    gauss_val = gauss_legendre_rule(n).apply(f, a=a, b=b)
    lobatto_val = gauss_lobatto_rule(n).apply(f, a=a, b=b)
    q_val = 0.75 * gauss_val + 0.25 * lobatto_val
    return float(q_val), float(gauss_val), float(lobatto_val)


def _Pn_single_interval(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
) -> float:
    """
    Compute the degree-raising combination P_n on a single interval.

    P_n = (n+1)/(2n+1) · G_n + n/(2n+1) · L_{n+1}.

    This combination has degree of exactness 2n (one higher than the components).
    """
    a, b = interval
    gauss_val = gauss_legendre_rule(n).apply(f, a=a, b=b)
    lobatto_val = gauss_lobatto_rule(n).apply(f, a=a, b=b)
    return float(((n + 1) / (2 * n + 1)) * gauss_val + (n / (2 * n + 1)) * lobatto_val)


def compare_Qn_vs_Pn(
    *,
    n: int,
    problems: Sequence[BenchmarkProblem],
) -> "pd.DataFrame":
    """
    Compare Q_n (certified) against P_n (degree-raising) on single-interval problems.

    Q_n = (3/4) G_n + (1/4) L_{n+1} is the certified estimator with θ = 1/4.
    P_n = ((n+1)/(2n+1)) G_n + (n/(2n+1)) L_{n+1} is the degree-raising combination.

    Parameters
    ----------
    n : int
        Parameter n from the paper.
    problems : sequence of BenchmarkProblem
        Problems with known exact integrals.

    Returns
    -------
    pandas.DataFrame
        Table with absolute errors of Q_n, P_n, and their component rules.

    Notes
    -----
    P_n achieves degree of exactness 2n (vs. 2n-1 for both G_n and L_{n+1}),
    but Q_n has a certified error bound for (2n-1)-convex functions.
    """
    pd = _ensure_pandas()

    rows: list[dict[str, object]] = []

    for prob in problems:
        q_val, gauss_val, lobatto_val = _Qn_single_interval(
            f=prob.f, interval=prob.interval, n=n
        )
        p_val = _Pn_single_interval(f=prob.f, interval=prob.interval, n=n)

        certified_bound = 0.25 * abs(lobatto_val - gauss_val)

        rows.append(
            {
                "problem": prob.name,
                "interval": prob.interval,
                "n": n,
                "|I - G_n|": abs(gauss_val - prob.exact),
                "|I - L_{n+1}|": abs(lobatto_val - prob.exact),
                "|I - Q_n|": abs(q_val - prob.exact),
                "|I - P_n|": abs(p_val - prob.exact),
                "certified_bound": certified_bound,
                "Q_n": q_val,
                "P_n": p_val,
                "G_n": gauss_val,
                "L_{n+1}": lobatto_val,
                "exact": prob.exact,
            }
        )

    return pd.DataFrame(rows)


def convergence_table(
    *,
    problem: BenchmarkProblem,
    n_list: Sequence[int],
    interval_counts: Sequence[int],
) -> "pd.DataFrame":
    """
    Create a convergence table showing errors across n values and subinterval counts.

    Parameters
    ----------
    problem : BenchmarkProblem
        The integration problem to analyse.
    n_list : sequence of int
        List of n values to test.
    interval_counts : sequence of int
        List of subinterval counts.

    Returns
    -------
    pandas.DataFrame
        Table with true errors and certified bounds for each (n, N) combination.

    Examples
    --------
    >>> import math
    >>> prob = BenchmarkProblem(
    ...     name="exp", f=math.exp, interval=(0.0, 1.0), exact=math.e - 1.0
    ... )
    >>> df = convergence_table(
    ...     problem=prob, n_list=[2, 3, 4], interval_counts=[1, 2, 4, 8]
    ... )
    """
    pd = _ensure_pandas()

    rows: list[dict[str, object]] = []

    for n in n_list:
        for num_intervals in interval_counts:
            res = integrate_uniform(
                f=problem.f,
                interval=problem.interval,
                n=n,
                n_intervals=num_intervals,
            )

            abs_error = abs(res.approx - problem.exact)

            rows.append(
                {
                    "n": n,
                    "n_intervals": num_intervals,
                    "convexity_order": 2 * n - 1,
                    "approx": res.approx,
                    "abs_error": abs_error,
                    "bound": res.bound,
                    "bound/error": res.bound / abs_error if abs_error > 1e-50 else np.inf,
                    "evaluations": res.num_evals,
                }
            )

    return pd.DataFrame(rows)
