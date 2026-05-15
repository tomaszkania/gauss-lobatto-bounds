"""Tests for benchmark and experiment helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gauss_lobatto_bounds import (
    CertifiedAdaptiveIntegrator,
    application_benchmark_suite,
    compare_with_global_gauss_legendre,
    global_gauss_legendre_integral,
    stieltjes_kernel_problem,
    truncated_power_problem,
)


def test_truncated_power_problem_exact_integral() -> None:
    """Check the closed-form integral for a truncated-power basis function."""
    problem = truncated_power_problem(knot=0.37, power=7.0)

    assert problem.name == "(x-0.37)_+^7 on [0, 1]"
    assert problem.convexity_order == 7
    assert problem.exact == pytest.approx(0.63**8 / 8.0)
    assert problem.f(0.2) == pytest.approx(0.0)
    assert problem.f(0.9) == pytest.approx(0.53**7)


def test_truncated_power_problem_handles_shifted_interval() -> None:
    """Check the exact integral formula on a non-default interval."""
    problem = truncated_power_problem(knot=-0.5, power=3.0, interval=(-1.0, 2.0))
    exact = ((2.0 + 0.5) ** 4 - 0.0**4) / 4.0

    assert problem.exact == pytest.approx(exact)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"interval": (1.0, 0.0)}, "Require a < b"),
        ({"power": -1.0}, "power > -1"),
    ],
)
def test_truncated_power_problem_validation(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Check validation for invalid truncated-power benchmark parameters."""
    with pytest.raises(ValueError, match=match):
        truncated_power_problem(**kwargs)


def test_stieltjes_kernel_problem_exact_integral() -> None:
    """Check the closed-form integral for a near-pole Stieltjes kernel."""
    problem = stieltjes_kernel_problem(delta=1e-4)

    assert problem.name == "1/(x+0.0001) on [0, 1]"
    assert problem.exact == pytest.approx(math.log(1.0001 / 0.0001))
    assert problem.f(0.0) == pytest.approx(10_000.0)
    assert problem.f(1.0) == pytest.approx(1.0 / 1.0001)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"delta": 0.0}, "delta > 0"),
        ({"interval": (1.0, 0.0)}, "Require a < b"),
        ({"delta": 0.25, "interval": (-1.0, 1.0)}, r"a \+ delta > 0"),
    ],
)
def test_stieltjes_kernel_problem_validation(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Check validation for invalid Stieltjes-kernel benchmark parameters."""
    with pytest.raises(ValueError, match=match):
        stieltjes_kernel_problem(**kwargs)


def test_application_benchmark_suite_contains_revision_examples() -> None:
    """Check that the application suite contains the revised paper examples."""
    suite = application_benchmark_suite()
    names = [problem.name for problem in suite]

    assert names == [
        "(x-0.37)_+^7 on [0, 1]",
        "1/(x+0.0001) on [0, 1]",
        "1/(x+1e-06) on [0, 1]",
    ]


def test_global_gauss_legendre_integral_exact_for_polynomial() -> None:
    """Check exactness of the global Gauss-Legendre helper on a polynomial."""
    approx = global_gauss_legendre_integral(
        f=lambda x: x**5 - 2.0 * x**2 + 1.0,
        interval=(-1.0, 1.0),
        n_nodes=3,
    )
    exact = 0.0 - 2.0 * (2.0 / 3.0) + 2.0

    assert approx == pytest.approx(exact)


def test_global_gauss_legendre_integral_validation() -> None:
    """Check validation in the global Gauss-Legendre helper."""
    with pytest.raises(ValueError, match="n_nodes >= 1"):
        global_gauss_legendre_integral(f=math.exp, interval=(0.0, 1.0), n_nodes=0)

    with pytest.raises(ValueError, match="Require a < b"):
        global_gauss_legendre_integral(f=math.exp, interval=(1.0, 0.0), n_nodes=2)


def test_compare_with_global_gauss_legendre_table() -> None:
    """Check the comparison table format and certification semantics."""
    problem = truncated_power_problem(knot=0.37, power=7.0)
    table = compare_with_global_gauss_legendre(
        problem=problem,
        certified_n=4,
        tol=1e-8,
        gauss_nodes=[8, 16],
    )

    assert list(table["method"]) == ["certified Q_4", "global G_8", "global G_16"]
    assert table.loc[0, "certified_bound"] <= 1e-8
    assert table.loc[0, "abs_error"] <= table.loc[0, "certified_bound"] + 1e-15
    assert np.isnan(table.loc[1, "certified_bound"])
    assert np.isnan(table.loc[2, "certified_bound"])


def test_compare_with_global_gauss_legendre_validation() -> None:
    """Check validation in the certified-versus-global comparison helper."""
    with pytest.raises(ValueError, match="tol > 0"):
        compare_with_global_gauss_legendre(
            problem=truncated_power_problem(),
            certified_n=4,
            tol=0.0,
            gauss_nodes=[8],
        )


def test_paper_truncated_power_certificate_is_reproducible() -> None:
    """Reproduce the truncated-power certificate used in the revised paper."""
    problem = truncated_power_problem(knot=0.37, power=7.0)
    result = CertifiedAdaptiveIntegrator(n=4, tol=1e-8).integrate(
        f=problem.f,
        interval=problem.interval,
    )

    assert result.n_intervals == 2
    assert result.num_evals == 23
    assert result.bound == pytest.approx(2.381761365277867e-09)
    assert abs(result.approx - problem.exact) == pytest.approx(1.6136058869284375e-09)


def test_paper_stieltjes_certificate_is_reproducible() -> None:
    """Reproduce a near-pole certificate used in the revised paper."""
    problem = stieltjes_kernel_problem(delta=1e-4)
    result = CertifiedAdaptiveIntegrator(n=4, tol=1e-8).integrate(
        f=problem.f,
        interval=problem.interval,
    )

    assert result.n_intervals == 37
    assert result.num_evals == 513
    assert result.bound == pytest.approx(7.890367630103423e-09)
    assert abs(result.approx - problem.exact) == pytest.approx(6.10904393738565e-09)
