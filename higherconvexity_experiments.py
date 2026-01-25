
"""
Experiment helpers for the companion notebook.

This module is meant to be imported from the Jupyter notebook
`numerical_experiments_frontend.ipynb`.  It provides:

- A small benchmark suite with exact integrals.
- Runners that produce tidy pandas DataFrames.
- A few convenience comparisons (Q_n versus the degree-raising P_n).

The code is a computational companion to the paper

    "Refined Gauss--Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

Version
-------
0.99.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

import math
import numpy as np

from higherconvexity_rules import ScalarFunc, gauss_legendre_rule, gauss_lobatto_rule
from higherconvexity_adaptive import (
    CertifiedAdaptiveIntegrator,
    IntegrationResult,
    find_min_intervals_uniform,
    integrate_uniform,
)

__all__ = [
    "BenchmarkProblem",
    "default_benchmark_suite",
    "run_suite",
    "run_suite_single_tol",
    "run_table_min_intervals_by_tol",
    "compare_Qn_vs_Pn",
    "run_experiment",
]

__version__ = "0.99.1"


@dataclass(frozen=True, slots=True)
class BenchmarkProblem:
    """
    A benchmark integration problem.

    Parameters
    ----------
    name:
        Short identifier (used in tables).
    f:
        Integrand.
    interval:
        Integration interval (a, b).
    exact:
        Exact value of the integral, if known.
    notes:
        Optional free-form notes (e.g. convexity/concavity information).
    """

    name: str
    f: ScalarFunc
    interval: tuple[float, float]
    exact: float
    notes: str = ""


def default_benchmark_suite() -> list[BenchmarkProblem]:
    """
    Default benchmark suite used in the notebook and paper drafts.

    Returns
    -------
    list of BenchmarkProblem
        A small list of problems with closed-form exact integrals.
    """

    def inv(x: float) -> float:
        return 1.0 / x

    def expf(x: float) -> float:
        return math.exp(x)

    def log1p(x: float) -> float:
        return math.log1p(x)

    problems: list[BenchmarkProblem] = [
        BenchmarkProblem(
            name="1/x on [1,2]",
            f=inv,
            interval=(1.0, 2.0),
            exact=math.log(2.0),
            notes="(2n-1)-convex for all n (since f^{(2n)} > 0).",
        ),
        BenchmarkProblem(
            name="exp on [0,1]",
            f=expf,
            interval=(0.0, 1.0),
            exact=math.e - 1.0,
            notes="(2n-1)-convex for all n (all derivatives positive).",
        ),
        BenchmarkProblem(
            name="log(1+x) on [0,1]",
            f=log1p,
            interval=(0.0, 1.0),
            exact=2.0 * math.log(2.0) - 1.0,
            notes="(2n-1)-concave for all n (even derivatives negative).",
        ),
    ]
    return problems


def _ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required for run_suite/compare_Qn_vs_Pn. "
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
) -> "object":
    """
    Run a batch of certified integrations and return a pandas DataFrame.

    Parameters
    ----------
    n:
        Parameter from the paper (odd order 2n-1).
    problems:
        Sequence of benchmark problems.
    tol:
        Target tolerance for the certified bound.
    method:
        Integration strategy:
        - "uniform_minimal": search for the minimal N of equal subintervals such that bound <= tol;
        - "uniform_fixed": use exactly `n_intervals` equal subintervals;
        - "greedy_bisection": adaptive greedy bisection.
    n_intervals:
        Used only when method="uniform_fixed".
    max_intervals:
        Cap for the number of subintervals (used in searches / bisection).

    Returns
    -------
    pandas.DataFrame
        Table with approximation, certified bound, true error, and evaluation counts.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")

    pd = _ensure_pandas()

    rows: list[dict[str, object]] = []
    for p in problems:
        if method == "uniform_fixed":
            res = integrate_uniform(f=p.f, interval=p.interval, n=n, n_intervals=n_intervals)
        elif method == "uniform_minimal":
            res = find_min_intervals_uniform(
                f=p.f, interval=p.interval, n=n, tol=tol, n_start=1, max_intervals=max_intervals
            )
        elif method == "greedy_bisection":
            integrator = CertifiedAdaptiveIntegrator(n=n, tol=tol, max_intervals=max_intervals)
            res = integrator.integrate(f=p.f, interval=p.interval)
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {method}")

        err = float(res.approx - p.exact)
        rows.append(
            {
                "problem": p.name,
                "interval": p.interval,
                "n": n,
                "tol": tol,
                "method": method,
                "approx": res.approx,
                "bound": res.bound,
                "abs_error": abs(err),
                "bound/abs_error": (res.bound / abs(err)) if err != 0 else np.inf,
                "n_intervals": res.n_intervals,
                "evals": res.num_evals,
                "notes": p.notes,
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
) -> "object":
    """
    Backwards-compatible wrapper mirroring interactive notebook usage.

    This mirrors the signature used in earlier drafts:

        df = run_suite_single_tol(n=4, problems=problems, tol=1e-8)

    Parameters
    ----------
    n, problems, tol, method, n_intervals, max_intervals:
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

    Earlier drafts used a dictionary output rather than a DataFrame.  This helper
    keeps that interface available while `run_suite` remains the recommended
    entry point for tabular output.

    Parameters
    ----------
    n, problems, tol, method, n_intervals, max_intervals:
        See `run_suite`.

    Returns
    -------
    dict
        Mapping problem name -> a small results dictionary.
    """
    out: dict[str, dict[str, object]] = {}

    for p in problems:
        if method == "uniform_fixed":
            res = integrate_uniform(f=p.f, interval=p.interval, n=n, n_intervals=n_intervals)
        elif method == "uniform_minimal":
            res = find_min_intervals_uniform(
                f=p.f, interval=p.interval, n=n, tol=tol, n_start=1, max_intervals=max_intervals
            )
        elif method == "greedy_bisection":
            integrator = CertifiedAdaptiveIntegrator(n=n, tol=tol, max_intervals=max_intervals)
            res = integrator.integrate(f=p.f, interval=p.interval)
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {method}")

        out[p.name] = {
            "approx": res.approx,
            "bound": res.bound,
            "n_intervals": res.n_intervals,
            "evals": res.num_evals,
            "exact": p.exact,
            "abs_error": abs(res.approx - p.exact),
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
) -> "object":
    """
    Produce a table of minimal N (uniform partitions) required to reach each tolerance.

    This is intended to reproduce the style of tables shared during development
    (columns labelled by convexity order 2n-1).

    Parameters
    ----------
    f:
        Integrand.
    interval:
        (a, b) integration interval.
    n_list:
        List of n values (e.g. [2, 3, 4, 5] corresponds to 3-, 5-, 7-, 9-convex).
    tol_list:
        Tolerances to test.
    max_intervals:
        Maximal N allowed in the search.

    Returns
    -------
    pandas.DataFrame
        A table with rows indexed by tol and columns "N_(2n-1)".
    """
    pd = _ensure_pandas()

    # IMPORTANT: warm-starting n_start relies on the monotonicity
    #   minimal N(tol) is non-decreasing as tol decreases.
    # Therefore we iterate tolerances from coarse to tight (largest to smallest).
    tol_sorted = sorted((float(t) for t in tol_list), reverse=True)
    data: dict[str, list[int]] = {"tol": tol_sorted}

    for n in n_list:
        Ns: list[int] = []
        # Warm start: minimal N is non-decreasing as tol decreases.
        n_start = 1
        for tol in tol_sorted:
            res = find_min_intervals_uniform(
                f=f, interval=interval, n=int(n), tol=float(tol), n_start=n_start, max_intervals=max_intervals
            )
            Ns.append(int(res.n_intervals))
            n_start = int(res.n_intervals)
        data[f"N_{2*n-1}"] = Ns

    return pd.DataFrame(data)


def _Qn_single_interval(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
) -> float:
    """
    Compute Q_n on a single interval.

    Q_n = 3/4 G_n + 1/4 L_{n+1}.
    """
    a, b = interval
    G = gauss_legendre_rule(n).apply(f, a=a, b=b)
    L = gauss_lobatto_rule(n).apply(f, a=a, b=b)
    return float(0.75 * G + 0.25 * L)


def _Pn_single_interval(
    *,
    f: ScalarFunc,
    interval: tuple[float, float],
    n: int,
) -> float:
    """
    Compute the degree-raising combination P_n on a single interval.

    P_n = (n+1)/(2n+1) G_n + n/(2n+1) L_{n+1}.
    """
    a, b = interval
    G = gauss_legendre_rule(n).apply(f, a=a, b=b)
    L = gauss_lobatto_rule(n).apply(f, a=a, b=b)
    return float(((n + 1) / (2 * n + 1)) * G + (n / (2 * n + 1)) * L)


def compare_Qn_vs_Pn(
    *,
    n: int,
    problems: Sequence[BenchmarkProblem],
) -> "object":
    """
    Compare Q_n (certified) against P_n (degree-raising) on single-interval problems.

    Parameters
    ----------
    n:
        Parameter n from the paper.
    problems:
        Problems with known exact integrals.

    Returns
    -------
    pandas.DataFrame
        Table with absolute errors of Q_n and P_n.
    """
    pd = _ensure_pandas()

    rows: list[dict[str, object]] = []
    for p in problems:
        q = _Qn_single_interval(f=p.f, interval=p.interval, n=n)
        pn = _Pn_single_interval(f=p.f, interval=p.interval, n=n)
        rows.append(
            {
                "problem": p.name,
                "interval": p.interval,
                "n": n,
                "|I - Q_n|": abs(q - p.exact),
                "|I - P_n|": abs(pn - p.exact),
                "Q_n approx": q,
                "P_n approx": pn,
                "exact": p.exact,
            }
        )

    return pd.DataFrame(rows)