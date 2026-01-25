# Gauss-Lobatto Bounds

Companion code for the paper

> **Refined Gauss–Lobatto bounds for odd-order convexity**  
> Tomasz Kania and Szymon Wąsowicz  
> (arXiv identifier to be added)

This repository contains a self-contained Python implementation of:

- Gauss–Legendre, Gauss–Lobatto, and Gauss–Radau quadrature rules on `[-1, 1]`,
- the certified estimator  
  $$Q_n = \tfrac{3}{4}G_n + \tfrac{1}{4}L_{n+1},$$
  together with the a posteriori bound  
  $$\lvert I[f] - Q_n[f]\rvert \le \tfrac{1}{4}\lvert L_{n+1}[f] - G_n[f]\rvert,$$
  valid for $(2n-1)$-convex or $(2n-1)$-concave integrands,
- basic certified integration strategies (uniform partitions and greedy bisection),
- reproducible numerical experiments and kernel plots.

Tomasz Kania's webpage: https://users.math.cas.cz/~kania/

## Installation

```bash
# Basic installation
pip install -e .

# With experiment dependencies (pandas, matplotlib)
pip install -e ".[experiments]"

# With development dependencies (pytest, mypy, ruff)
pip install -e ".[dev]"
```

## Repository Layout

```
src/gauss_lobatto_bounds/
├── __init__.py       # Clean exports
├── rules.py          # Quadrature rules and Peano kernels
├── adaptive.py       # Certified integration strategies
└── experiments.py    # Benchmark suite and runners
tests/
├── test_rules.py     # Comprehensive rule tests
└── test_adaptive.py  # Integration strategy tests
notebooks/
├── numerical_experiments_frontend.ipynb
└── classical_error_exercise.ipynb
```

## Notebooks

The repository contains two Jupyter notebooks used to generate figures and tables in the paper:

- `notebooks/numerical_experiments_frontend.ipynb` — the main numerical experiments (kernel plots, certified stopping criteria,
  and 3/5/7/9-convex benchmarks).
- `notebooks/classical_error_exercise.ipynb` — a short companion note on **classical a priori error bounds** (max-norm bounds
  involving $\|f^{(2n)}\|_\infty$), common implementation pitfalls (e.g. tolerance sweep ordering), and how these relate to the
  certified a posteriori bound used in the paper.


## Quick Start

```python
from gauss_lobatto_bounds import (
    gauss_legendre_rule,
    gauss_lobatto_rule,
    integrate_uniform,
    CertifiedAdaptiveIntegrator,
    benchmark_suite,
)
import math

# Create quadrature rules
G3 = gauss_legendre_rule(3)
L4 = gauss_lobatto_rule(4)

# Integrate with certified bounds
result = integrate_uniform(f=math.exp, interval=(0.0, 1.0), n=3, n_intervals=10)
print(f"Approximation: {result.approx}")
print(f"Certified bound: {result.bound}")

# Adaptive integration to tolerance
integrator = CertifiedAdaptiveIntegrator(n=3, tol=1e-10)
result = integrator.integrate(f=math.exp, interval=(0.0, 1.0))
print(f"Reached tolerance with {result.n_intervals} intervals")

# Run benchmark suite
for problem in benchmark_suite():
    print(f"{problem.name}: {problem.exact:.10f}")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=gauss_lobatto_bounds

# Type checking
mypy src/

# Linting
ruff check src/
```

## Key Features

### Quadrature Rules

- `gauss_legendre_rule(n)` — n-point Gauss–Legendre rule (exact for polynomials up to degree 2n−1)
- `gauss_lobatto_rule(n)` — n-point Gauss–Lobatto rule (includes endpoints ±1)
- `gauss_radau_left_rule(n)` — (n+1)-point left Radau rule (includes −1)
- `gauss_radau_right_rule(n)` — (n+1)-point right Radau rule (includes +1)

### Peano Kernels

- `gauss_peano_kernel(n, x)` — Peano kernel K_G for Gauss–Legendre
- `lobatto_peano_kernel(n, x)` — Peano kernel K_L for Gauss–Lobatto
- `radau_curvature_kernel(n, x)` — Curvature kernel for Radau rule
- `kernel_dominance_ratio(n, num_points)` — Verify K_L ≥ K_G inequality

### Certified Integration

- `integrate_uniform(f, interval, n, m)` — Fixed uniform partitioning
- `find_min_intervals_uniform(f, interval, n, tol)` — Find minimal partition count
- `CertifiedAdaptiveIntegrator` — Greedy bisection to tolerance

### Experiments

- `benchmark_suite()` — Standard test problems
- `extended_benchmark_suite()` — Additional test functions
- `compare_Qn_vs_Pn(f, interval, n, ...)` — Compare certified vs standard estimators
- `convergence_table(f, interval, n, ...)` — Convergence analysis

## Notes on Numerical Precision

The rules are computed in double precision (`float64`) using `numpy.polynomial`.
For the experiment sizes in the paper (small n, moderate refinement), this is
typically sufficient. If you need higher precision arithmetic (e.g. to push
tolerances well below `1e-14` on ill-conditioned problems), consider replacing
the backend for node computation with an arbitrary-precision library.

## Citation

If you use this code, please cite the paper (arXiv information forthcoming).
You can also use the metadata in `CITATION.cff`.
