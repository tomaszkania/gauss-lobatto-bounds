"""
Tests for the adaptive integration module.

This test suite verifies the correctness of the certified integration
strategies: uniform partitioning, minimal interval search, and greedy
bisection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gauss_lobatto_bounds import (
    CertifiedAdaptiveIntegrator,
    CompositeRule,
    FunctionCache,
    IntegrationResult,
    estimate_convergence_rate,
    find_min_intervals_uniform,
    integrate_uniform,
)


class TestFunctionCache:
    """Tests for the FunctionCache class."""

    def test_basic_caching(self) -> None:
        """Test that repeated calls return cached values."""
        call_count = [0]

        def f(x: float) -> float:
            call_count[0] += 1
            return x**2

        cache = FunctionCache(f)

        # First call should evaluate.
        result1 = cache(0.5)
        assert call_count[0] == 1
        assert result1 == pytest.approx(0.25)

        # Second call should use cache.
        result2 = cache(0.5)
        assert call_count[0] == 1  # No new evaluation.
        assert result2 == pytest.approx(0.25)

    def test_num_evals_counter(self) -> None:
        """Test that the evaluation counter is correct."""
        cache = FunctionCache(lambda x: x)

        cache(1.0)
        cache(2.0)
        cache(1.0)  # Cached.
        cache(3.0)

        assert cache.num_evals == 3

    def test_cache_size(self) -> None:
        """Test that cache_size returns the correct count."""
        cache = FunctionCache(lambda x: x)

        cache(1.0)
        cache(2.0)
        cache(1.0)

        assert cache.cache_size == 2

    def test_clear(self) -> None:
        """Test that clear() resets the cache."""
        cache = FunctionCache(lambda x: x)

        cache(1.0)
        cache(2.0)
        assert cache.num_evals == 2

        cache.clear()
        assert cache.num_evals == 0
        assert cache.cache_size == 0

    def test_get_cached_points(self) -> None:
        """Test that get_cached_points returns sorted points."""
        cache = FunctionCache(lambda x: x)

        cache(3.0)
        cache(1.0)
        cache(2.0)

        points = cache.get_cached_points()
        assert points == [1.0, 2.0, 3.0]


class TestIntegrationResult:
    """Tests for the IntegrationResult dataclass."""

    def test_is_certified(self) -> None:
        """Test the is_certified property."""
        result1 = IntegrationResult(
            approx=1.0, bound=0.01, n_intervals=5, num_evals=10, strategy="uniform_fixed"
        )
        assert result1.is_certified  # finite bound means certified

        result2 = IntegrationResult(
            approx=1.0, bound=np.inf, n_intervals=5, num_evals=10, strategy="uniform_fixed"
        )
        assert not result2.is_certified  # infinite bound means not certified

    def test_repr(self) -> None:
        """Test the string representation."""
        result = IntegrationResult(
            approx=1.718281828, bound=1e-10, n_intervals=5, num_evals=10, strategy="uniform_fixed"
        )
        repr_str = repr(result)
        assert "approx" in repr_str
        assert "bound" in repr_str
        assert "uniform_fixed" in repr_str


class TestIntegrateUniform:
    """Tests for the integrate_uniform function."""

    def test_exp_on_0_1(self) -> None:
        """Test integration of exp on [0, 1]."""
        exact = math.e - 1.0
        result = integrate_uniform(
            f=math.exp, interval=(0.0, 1.0), n=4, n_intervals=5
        )

        error = abs(result.approx - exact)
        # Certified bound should hold.
        assert error <= result.bound + 1e-15

    def test_inv_on_1_2(self) -> None:
        """Test integration of 1/x on [1, 2]."""
        exact = math.log(2.0)
        result = integrate_uniform(
            f=lambda x: 1.0 / x, interval=(1.0, 2.0), n=3, n_intervals=10
        )

        error = abs(result.approx - exact)
        assert error <= result.bound + 1e-15

    def test_single_interval(self) -> None:
        """Test with a single interval."""
        result = integrate_uniform(
            f=lambda x: x**2, interval=(0.0, 1.0), n=4, n_intervals=1
        )
        exact = 1.0 / 3.0
        assert result.approx == pytest.approx(exact, rel=1e-10)

    def test_strategy_is_uniform_fixed(self) -> None:
        """Test that the strategy is correctly set."""
        result = integrate_uniform(
            f=math.exp, interval=(0.0, 1.0), n=2, n_intervals=3
        )
        assert result.strategy == "uniform_fixed"

    def test_invalid_n_intervals_raises(self) -> None:
        """Test that n_intervals < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            integrate_uniform(f=math.exp, interval=(0.0, 1.0), n=2, n_intervals=0)

    def test_invalid_interval_raises(self) -> None:
        """Test that a >= b raises ValueError."""
        with pytest.raises(ValueError, match="Require a < b"):
            integrate_uniform(f=math.exp, interval=(1.0, 0.0), n=2, n_intervals=5)


class TestFindMinIntervalsUniform:
    """Tests for the find_min_intervals_uniform function."""

    def test_finds_minimal_n(self) -> None:
        """Test that the function finds the minimal N."""
        tol = 1e-8
        result = find_min_intervals_uniform(
            f=math.exp, interval=(0.0, 1.0), n=3, tol=tol, max_intervals=10_000
        )

        # Should meet tolerance.
        assert result.bound <= tol

        # Previous N should fail (if N > 1).
        if result.n_intervals > 1:
            res_prev = integrate_uniform(
                f=math.exp, interval=(0.0, 1.0), n=3, n_intervals=result.n_intervals - 1
            )
            # Previous N should NOT meet tolerance.
            assert res_prev.bound > tol * (1 - 1e-10)

    def test_strategy_is_uniform_minimal(self) -> None:
        """Test that the strategy is correctly set."""
        result = find_min_intervals_uniform(
            f=math.exp, interval=(0.0, 1.0), n=2, tol=1e-4
        )
        assert result.strategy == "uniform_minimal"

    def test_already_good(self) -> None:
        """Test when the initial estimate already meets tolerance."""
        # With a very loose tolerance, even n_start=1 should be good.
        result = find_min_intervals_uniform(
            f=lambda x: 1.0, interval=(0.0, 1.0), n=2, tol=1.0
        )
        assert result.n_intervals == 1
        assert result.bound <= 1.0

    def test_warm_start(self) -> None:
        """Test that n_start accelerates the search."""
        # First, find the minimal N for a tolerance.
        res1 = find_min_intervals_uniform(
            f=math.exp, interval=(0.0, 1.0), n=3, tol=1e-6
        )

        # Now use that as a warm start for a tighter tolerance.
        res2 = find_min_intervals_uniform(
            f=math.exp, interval=(0.0, 1.0), n=3, tol=1e-8, n_start=res1.n_intervals
        )

        assert res2.n_intervals >= res1.n_intervals

    def test_invalid_tol_raises(self) -> None:
        """Test that tol <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            find_min_intervals_uniform(
                f=math.exp, interval=(0.0, 1.0), n=2, tol=0.0
            )

    def test_invalid_n_start_raises(self) -> None:
        """Test that n_start < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            find_min_intervals_uniform(
                f=math.exp, interval=(0.0, 1.0), n=2, tol=1e-4, n_start=0
            )

    def test_max_intervals_exceeded_raises(self) -> None:
        """Test that exceeding max_intervals raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Could not reach"):
            find_min_intervals_uniform(
                f=math.exp, interval=(0.0, 1.0), n=2, tol=1e-20, max_intervals=10
            )


class TestCertifiedAdaptiveIntegrator:
    """Tests for the CertifiedAdaptiveIntegrator class."""

    def test_meets_tolerance(self) -> None:
        """Test that the integrator meets the specified tolerance."""
        integrator = CertifiedAdaptiveIntegrator(n=3, tol=1e-10, max_intervals=50_000)
        result = integrator.integrate(f=lambda x: 1.0 / x, interval=(1.0, 2.0))

        assert result.bound <= integrator.tol

    def test_certified_bound_valid(self) -> None:
        """Test that the certified bound is valid."""
        exact = math.log(2.0)
        integrator = CertifiedAdaptiveIntegrator(n=4, tol=1e-8)
        result = integrator.integrate(f=lambda x: 1.0 / x, interval=(1.0, 2.0))

        error = abs(result.approx - exact)
        assert error <= result.bound + 1e-15

    def test_strategy_is_greedy_bisection(self) -> None:
        """Test that the strategy is correctly set."""
        integrator = CertifiedAdaptiveIntegrator(n=2, tol=1e-4)
        result = integrator.integrate(f=math.exp, interval=(0.0, 1.0))

        assert result.strategy == "greedy_bisection"

    def test_exp_on_0_1(self) -> None:
        """Test integration of exp on [0, 1]."""
        exact = math.e - 1.0
        integrator = CertifiedAdaptiveIntegrator(n=3, tol=1e-12)
        result = integrator.integrate(f=math.exp, interval=(0.0, 1.0))

        assert result.bound <= 1e-12
        assert abs(result.approx - exact) <= result.bound + 1e-15

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            CertifiedAdaptiveIntegrator(n=0, tol=1e-4)

    def test_invalid_tol_raises(self) -> None:
        """Test that tol <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            CertifiedAdaptiveIntegrator(n=2, tol=0.0)

    def test_invalid_interval_raises(self) -> None:
        """Test that a >= b raises ValueError."""
        integrator = CertifiedAdaptiveIntegrator(n=2, tol=1e-4)
        with pytest.raises(ValueError, match="Require a < b"):
            integrator.integrate(f=math.exp, interval=(1.0, 0.0))

    def test_max_intervals_exceeded_raises(self) -> None:
        """Test that exceeding max_intervals raises RuntimeError."""
        integrator = CertifiedAdaptiveIntegrator(n=2, tol=1e-20, max_intervals=5)
        with pytest.raises(RuntimeError, match="max_intervals"):
            integrator.integrate(f=math.exp, interval=(0.0, 1.0))


class TestCompositeRule:
    """Tests for the CompositeRule class."""

    def test_create_and_apply(self) -> None:
        """Test creating and applying a composite rule."""
        rule = CompositeRule.create(n=3, interval=(0.0, 1.0), n_subintervals=5)
        approx, bound = rule.apply_certified(math.exp)

        exact = math.e - 1.0
        error = abs(approx - exact)
        assert error <= bound + 1e-15

    def test_subinterval_bounds(self) -> None:
        """Test computing bounds for each subinterval."""
        rule = CompositeRule.create(n=3, interval=(0.0, 1.0), n_subintervals=4)
        bounds = rule.subinterval_bounds(math.exp)

        assert len(bounds) == 4
        assert all(b >= 0 for b in bounds)

    def test_invalid_interval_raises(self) -> None:
        """Test that a >= b raises ValueError."""
        with pytest.raises(ValueError, match="Require a < b"):
            CompositeRule.create(n=2, interval=(1.0, 0.0), n_subintervals=3)

    def test_invalid_n_subintervals_raises(self) -> None:
        """Test that n_subintervals < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            CompositeRule.create(n=2, interval=(0.0, 1.0), n_subintervals=0)


class TestEstimateConvergenceRate:
    """Tests for the estimate_convergence_rate function."""

    def test_basic_usage(self) -> None:
        """Test basic usage with exp on [0, 1]."""
        results = estimate_convergence_rate(
            f=math.exp,
            interval=(0.0, 1.0),
            n=3,
            interval_counts=[10, 20, 40, 80],
            exact=math.e - 1.0,
        )

        assert "n_intervals" in results
        assert "bounds" in results
        assert "errors" in results
        assert "bound_rate" in results
        assert "error_rate" in results

    def test_without_exact(self) -> None:
        """Test usage without an exact value."""
        results = estimate_convergence_rate(
            f=math.exp,
            interval=(0.0, 1.0),
            n=3,
            interval_counts=[10, 20, 40],
        )

        assert "errors" not in results
        assert "error_rate" not in results

    def test_convergence_rate_reasonable(self) -> None:
        """Test that the estimated rate is reasonable."""
        # For n=3, convexity order is 5, so expect rate ~ 2n = 6.
        results = estimate_convergence_rate(
            f=math.exp,
            interval=(0.0, 1.0),
            n=3,
            interval_counts=[8, 16, 32, 64, 128],
            exact=math.e - 1.0,
        )

        bound_rate = float(results["bound_rate"][0])
        # Should be roughly 2n = 6, allow tolerance for numerical variation.
        assert 3.5 < bound_rate < 8.5


class TestCertifiedBoundValidity:
    """Integration tests verifying that certified bounds are always valid."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_exp_uniform(self, n: int) -> None:
        """Test certified bounds for exp with uniform partitioning."""
        exact = math.e - 1.0

        for n_intervals in [1, 5, 10, 20]:
            result = integrate_uniform(
                f=math.exp, interval=(0.0, 1.0), n=n, n_intervals=n_intervals
            )
            error = abs(result.approx - exact)
            assert error <= result.bound + 1e-14

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_inv_uniform(self, n: int) -> None:
        """Test certified bounds for 1/x with uniform partitioning."""
        exact = math.log(2.0)

        for n_intervals in [1, 5, 10]:
            result = integrate_uniform(
                f=lambda x: 1.0 / x, interval=(1.0, 2.0), n=n, n_intervals=n_intervals
            )
            error = abs(result.approx - exact)
            assert error <= result.bound + 1e-14

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_log1p_uniform(self, n: int) -> None:
        """Test certified bounds for log(1+x) with uniform partitioning."""
        exact = 2.0 * math.log(2.0) - 1.0

        for n_intervals in [1, 5, 10]:
            result = integrate_uniform(
                f=math.log1p, interval=(0.0, 1.0), n=n, n_intervals=n_intervals
            )
            error = abs(result.approx - exact)
            assert error <= result.bound + 1e-14

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_exp_adaptive(self, n: int) -> None:
        """Test certified bounds for exp with adaptive integration."""
        exact = math.e - 1.0

        for tol in [1e-6, 1e-8, 1e-10]:
            integrator = CertifiedAdaptiveIntegrator(n=n, tol=tol)
            result = integrator.integrate(f=math.exp, interval=(0.0, 1.0))

            assert result.bound <= tol
            error = abs(result.approx - exact)
            assert error <= result.bound + 1e-15
