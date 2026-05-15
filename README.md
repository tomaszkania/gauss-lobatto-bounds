# Gauss-Lobatto Bounds

Companion code for the manuscript

> **Refined Gauss-Lobatto bounds for odd-order convexity**  
> Tomasz Kania and Szymon Wąsowicz

The package implements the quadrature rules, certified estimators, Peano-kernel diagnostics, and reproducible experiments used in the paper.  The focus is certified integration under a higher-order convexity or concavity assumption, not merely high-order approximation of analytic test functions.

## What is included

- Gauss-Legendre, Gauss-Lobatto, and Gauss-Radau rules on `[-1, 1]`, with affine rescaling to arbitrary intervals.
- The certified estimator
  
  ```text
  Q_n = (3/4) G_n + (1/4) L_{n+1}
  ```
  with the a posteriori certificate
  
  ```text
  |I[f] - Q_n[f]| <= (1/4) |L_{n+1}[f] - G_n[f]|,
  ```
  valid for `(2n-1)`-convex or `(2n-1)`-concave integrands.
- Uniform and greedy-bisection composite strategies.
- Peano-kernel evaluators for the Gauss/Lobatto dominance check and the Radau midpoint obstruction.
- Application-focused benchmarks, including truncated powers from spline models and near-pole Stieltjes kernels.
- Pytest coverage for rule exactness, kernel properties, certified error bounds, and experiment helpers.

## Installation

```bash
python -m pip install -e .
```

Optional experiment and development dependencies are available with extras:

```bash
python -m pip install -e ".[experiments]"
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer and NumPy.

## Quick start

```python
from __future__ import annotations

import math

from gauss_lobatto_bounds import CertifiedAdaptiveIntegrator, integrate_uniform

# Certified composite integration on a fixed uniform partition.
result = integrate_uniform(
    f=math.exp,
    interval=(0.0, 1.0),
    n=4,
    n_intervals=5,
)
print(result.approx)
print(result.bound)

# Greedy bisection until the certified global bound is below tolerance.
integrator = CertifiedAdaptiveIntegrator(n=4, tol=1e-8)
result = integrator.integrate(f=lambda x: 1.0 / (x + 1e-6), interval=(0.0, 1.0))
print(result)
```

A truncated-power example, matching the revised numerical section:

```python
from gauss_lobatto_bounds import CertifiedAdaptiveIntegrator

f = lambda x: max(x - 0.37, 0.0) ** 7
exact = 0.63 ** 8 / 8.0

result = CertifiedAdaptiveIntegrator(n=4, tol=1e-8).integrate(
    f=f,
    interval=(0.0, 1.0),
)

assert abs(result.approx - exact) <= result.bound + 1e-15
print(result.num_evals, result.bound)
```

## Repository layout

```text
src/gauss_lobatto_bounds/
├── __init__.py       # Public exports
├── rules.py          # Quadrature rules and Peano kernels
├── adaptive.py       # Certified integration strategies
└── experiments.py    # Benchmark suites and experiment runners

tests/
├── test_rules.py
├── test_adaptive.py
└── test_experiments.py

notebooks/
├── numerical_experiments_frontend.ipynb
└── classical_error_exercise.ipynb
```

## Core API

### Rules and kernels

- `gauss_legendre_rule(n)` returns the `n`-point Gauss-Legendre rule.
- `gauss_lobatto_rule(n)` returns the `(n+1)`-point Gauss-Lobatto rule in the paper's indexing.
- `gauss_radau_left_rule(n)` and `gauss_radau_right_rule(n)` return the `(n+1)`-point Radau endpoint rules.
- `peano_kernel_gauss_legendre(n, t)` and `peano_kernel_gauss_lobatto(n, t)` evaluate the Peano kernels used in the refined bracket.
- `radau_curvature_kernel_midpoint(n, t)` evaluates the sign-changing Radau midpoint kernel.

### Certified integration

- `integrate_uniform(f, interval, n, n_intervals)` evaluates a fixed uniform composite rule.
- `find_min_intervals_uniform(f, interval, n, tol)` searches for the smallest uniform partition size reaching the certificate.
- `CertifiedAdaptiveIntegrator(n, tol).integrate(f, interval)` performs greedy bisection using local certified bounds.
- `CompositeRule.create(...)` constructs a reusable composite rule object.

### Experiments

- `default_benchmark_suite()` contains smooth sanity checks.
- `application_benchmark_suite()` contains the truncated-power and Stieltjes-kernel examples used to address the revised numerical discussion.
- `compare_with_global_gauss_legendre(...)` compares the certified estimator with a global Gauss-Legendre rule when an exact integral is available.
- `run_suite(...)` and `convergence_table(...)` produce pandas DataFrames for notebooks and manuscript tables.

## Running checks

```bash
pytest
ruff check src tests
mypy src
```

The main mathematical checks covered by the tests are:

- polynomial exactness of Gauss, Lobatto, and Radau rules;
- positivity and dominance of Peano kernels on stable grids;
- oddness/sign change of the Radau midpoint kernel;
- validity of the certified bound on convex and concave examples;
- exact integral metadata for the application benchmark suite.

## Numerical precision

The implementation uses double precision (`float64`) and NumPy's polynomial routines.  This is appropriate for the small and moderate orders used in the paper.  For very high-order global rules or tolerances far below `1e-14`, arbitrary precision or a specialised node generator may be preferable.

## Citation

If you use this code, cite the associated paper and the metadata in `CITATION.cff`.
