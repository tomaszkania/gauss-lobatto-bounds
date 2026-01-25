"""
Refined Gauss–Lobatto bounds for odd-order convexity.

This package provides quadrature rules and certified integration strategies
based on the refined Gauss–Lobatto bounds described in the paper

    "Refined Gauss–Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

The key result is that for (2n-1)-convex or (2n-1)-concave integrands f,
the estimator

    Q_n = (3/4) G_n + (1/4) L_{n+1}

satisfies the certified error bound

    |I[f] - Q_n[f]| ≤ (1/4) |L_{n+1}[f] - G_n[f]|,

where G_n is the n-point Gauss–Legendre rule and L_{n+1} is the (n+1)-point
Gauss–Lobatto rule, both exact on polynomials of degree ≤ 2n - 1.

Modules
-------
rules
    Quadrature rules and Peano kernels.
adaptive
    Certified integration strategies (uniform and adaptive).
experiments
    Benchmark suite and experiment runners.

Examples
--------
Basic usage with the certified estimator:

>>> import math
>>> from gauss_lobatto_bounds import integrate_uniform
>>> result = integrate_uniform(f=math.exp, interval=(0.0, 1.0), n=3, n_intervals=10)
>>> result.bound  # Certified error bound
2.83e-10
>>> abs(result.approx - (math.e - 1.0)) < result.bound  # Bound is valid
True

Adaptive integration:

>>> from gauss_lobatto_bounds import CertifiedAdaptiveIntegrator
>>> integrator = CertifiedAdaptiveIntegrator(n=4, tol=1e-12)
>>> result = integrator.integrate(f=math.exp, interval=(0.0, 1.0))
>>> result.bound <= 1e-12
True

Notes
-----
British English is used throughout the documentation.

Version
-------
1.0.0
"""

from .adaptive import (
    CertifiedAdaptiveIntegrator,
    CompositeRule,
    FunctionCache,
    IntegrationResult,
    estimate_convergence_rate,
    find_min_intervals_uniform,
    integrate_uniform,
)
from .experiments import (
    BenchmarkProblem,
    compare_Qn_vs_Pn,
    convergence_table,
    default_benchmark_suite,
    extended_benchmark_suite,
    run_experiment,
    run_suite,
    run_suite_single_tol,
    run_table_min_intervals_by_tol,
)
from .rules import (
    QuadratureRule,
    ScalarFunc,
    apply_rule_on_interval,
    certified_estimator_weights,
    gauss_legendre_rule,
    gauss_lobatto_rule,
    gauss_radau_left_rule,
    gauss_radau_right_rule,
    kernel_dominance_ratio,
    peano_kernel_gauss_legendre,
    peano_kernel_gauss_lobatto,
    radau_curvature_kernel_midpoint,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Type aliases
    "ScalarFunc",
    # Core classes
    "QuadratureRule",
    "IntegrationResult",
    "FunctionCache",
    "CertifiedAdaptiveIntegrator",
    "CompositeRule",
    "BenchmarkProblem",
    # Quadrature rules
    "gauss_legendre_rule",
    "gauss_lobatto_rule",
    "gauss_radau_left_rule",
    "gauss_radau_right_rule",
    "apply_rule_on_interval",
    # Peano kernels
    "peano_kernel_gauss_legendre",
    "peano_kernel_gauss_lobatto",
    "radau_curvature_kernel_midpoint",
    "kernel_dominance_ratio",
    # Integration functions
    "integrate_uniform",
    "find_min_intervals_uniform",
    "certified_estimator_weights",
    "estimate_convergence_rate",
    # Experiment functions
    "default_benchmark_suite",
    "extended_benchmark_suite",
    "run_suite",
    "run_suite_single_tol",
    "run_experiment",
    "run_table_min_intervals_by_tol",
    "compare_Qn_vs_Pn",
    "convergence_table",
]
