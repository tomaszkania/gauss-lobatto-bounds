"""
Certified Gauss-Lobatto bounds for odd-order convexity.

The package provides quadrature rules, Peano-kernel diagnostics, certified
uniform and adaptive integration routines, and reproducible numerical
experiments accompanying the paper

    "Refined Gauss-Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

For an integrand known to be (2n-1)-convex or (2n-1)-concave, the estimator

    Q_n = (3/4) G_n + (1/4) L_{n+1}

satisfies the a posteriori certificate

    |I[f] - Q_n[f]| <= (1/4) |L_{n+1}[f] - G_n[f]|,

where G_n is the n-point Gauss-Legendre rule and L_{n+1} is the
(n+1)-point Gauss-Lobatto rule.
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
    application_benchmark_suite,
    compare_Qn_vs_Pn,
    compare_with_global_gauss_legendre,
    convergence_table,
    default_benchmark_suite,
    extended_benchmark_suite,
    global_gauss_legendre_integral,
    run_experiment,
    run_suite,
    run_suite_single_tol,
    run_table_min_intervals_by_tol,
    stieltjes_kernel_problem,
    truncated_power_problem,
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

__version__ = "1.1.0"

__all__ = [
    "__version__",
    "ScalarFunc",
    "QuadratureRule",
    "IntegrationResult",
    "FunctionCache",
    "CertifiedAdaptiveIntegrator",
    "CompositeRule",
    "BenchmarkProblem",
    "gauss_legendre_rule",
    "gauss_lobatto_rule",
    "gauss_radau_left_rule",
    "gauss_radau_right_rule",
    "apply_rule_on_interval",
    "certified_estimator_weights",
    "peano_kernel_gauss_legendre",
    "peano_kernel_gauss_lobatto",
    "radau_curvature_kernel_midpoint",
    "kernel_dominance_ratio",
    "integrate_uniform",
    "find_min_intervals_uniform",
    "estimate_convergence_rate",
    "default_benchmark_suite",
    "extended_benchmark_suite",
    "application_benchmark_suite",
    "truncated_power_problem",
    "stieltjes_kernel_problem",
    "global_gauss_legendre_integral",
    "compare_with_global_gauss_legendre",
    "run_suite",
    "run_suite_single_tol",
    "run_table_min_intervals_by_tol",
    "run_experiment",
    "compare_Qn_vs_Pn",
    "convergence_table",
]
