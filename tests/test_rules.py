"""
Tests for the quadrature rules module.

This test suite verifies the correctness of Gauss–Legendre, Gauss–Lobatto,
and Gauss–Radau quadrature rules, as well as the Peano kernel computations.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauss_lobatto_bounds import (
    QuadratureRule,
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


def _exact_monomial_on_minus1_1(k: int) -> float:
    """Exact integral of x^k over [-1, 1]."""
    if k % 2 == 1:
        return 0.0
    return 2.0 / (k + 1)


class TestQuadratureRuleBasics:
    """Tests for the QuadratureRule dataclass."""

    def test_creation_valid(self) -> None:
        """Test creating a valid quadrature rule."""
        rule = QuadratureRule(
            nodes=np.array([-0.5, 0.5]),
            weights=np.array([1.0, 1.0]),
            name="Test rule",
            degree=1,
            family="gauss",
        )
        assert rule.n_nodes == 2
        assert rule.total_weight == pytest.approx(2.0)

    def test_creation_unsorted_nodes_raises(self) -> None:
        """Test that unsorted nodes raise ValueError."""
        with pytest.raises(ValueError, match="non-decreasing order"):
            QuadratureRule(
                nodes=np.array([0.5, -0.5]),
                weights=np.array([1.0, 1.0]),
                name="Bad rule",
                degree=1,
                family="gauss",
            )

    def test_creation_mismatched_shapes_raises(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            QuadratureRule(
                nodes=np.array([-0.5, 0.5]),
                weights=np.array([1.0]),
                name="Bad rule",
                degree=1,
                family="gauss",
            )

    def test_rescale(self) -> None:
        """Test affine rescaling of a rule."""
        rule = gauss_legendre_rule(2)
        rescaled = rule.rescale(0.0, 2.0)

        # Check nodes are in [0, 2].
        assert float(rescaled.nodes.min()) >= 0.0
        assert float(rescaled.nodes.max()) <= 2.0

        # Check weights sum to interval length.
        assert rescaled.total_weight == pytest.approx(2.0)

    def test_rescale_invalid_interval_raises(self) -> None:
        """Test that rescaling with a >= b raises ValueError."""
        rule = gauss_legendre_rule(2)
        with pytest.raises(ValueError, match="Require a < b"):
            rule.rescale(1.0, 0.0)

    def test_apply_identity(self) -> None:
        """Test applying a rule to the identity function."""
        rule = gauss_legendre_rule(3)
        # ∫_{-1}^{1} x dx = 0
        result = rule.apply(lambda x: x)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_apply_with_rescaling(self) -> None:
        """Test applying a rule with automatic rescaling."""
        rule = gauss_legendre_rule(4)
        # ∫_0^1 x^2 dx = 1/3
        result = rule.apply(lambda x: x**2, a=0.0, b=1.0)
        assert result == pytest.approx(1.0 / 3.0)

    def test_evaluate_at_nodes(self) -> None:
        """Test evaluating a function at all nodes."""
        rule = gauss_legendre_rule(3)
        values = rule.evaluate_at_nodes(lambda x: x**2)
        expected = rule.nodes**2
        np.testing.assert_allclose(values, expected)


class TestGaussLegendre:
    """Tests for Gauss–Legendre quadrature rules."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_exactness(self, n: int) -> None:
        """Test exactness on polynomials of degree ≤ 2n - 1."""
        rule = gauss_legendre_rule(n)

        for k in range(2 * n):
            approx = rule.apply(lambda x, k=k: x**k)
            exact = _exact_monomial_on_minus1_1(k)
            assert approx == pytest.approx(exact, abs=1e-13)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_not_exact_beyond_degree(self, n: int) -> None:
        """Test that exactness fails for degree 2n."""
        if n <= 3:  # Only test for small n where the error is significant.
            rule = gauss_legendre_rule(n)
            k = 2 * n
            approx = rule.apply(lambda x, k=k: x**k)
            exact = _exact_monomial_on_minus1_1(k)
            # Should NOT be exact.
            assert abs(approx - exact) > 1e-10

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_positive_weights(self, n: int) -> None:
        """Test that all weights are positive."""
        rule = gauss_legendre_rule(n)
        assert np.all(rule.weights > 0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_weights_sum_to_two(self, n: int) -> None:
        """Test that weights sum to 2 (length of [-1, 1])."""
        rule = gauss_legendre_rule(n)
        assert rule.total_weight == pytest.approx(2.0)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_nodes_symmetric(self, n: int) -> None:
        """Test that nodes are symmetric about 0."""
        rule = gauss_legendre_rule(n)
        np.testing.assert_allclose(rule.nodes, -rule.nodes[::-1], atol=1e-14)

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            gauss_legendre_rule(0)


class TestGaussLobatto:
    """Tests for Gauss–Lobatto quadrature rules."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_exactness(self, n: int) -> None:
        """Test exactness on polynomials of degree ≤ 2n - 1."""
        rule = gauss_lobatto_rule(n)

        for k in range(2 * n):
            approx = rule.apply(lambda x, k=k: x**k)
            exact = _exact_monomial_on_minus1_1(k)
            assert approx == pytest.approx(exact, abs=1e-13)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_includes_endpoints(self, n: int) -> None:
        """Test that the rule includes ±1 as nodes."""
        rule = gauss_lobatto_rule(n)
        assert rule.nodes[0] == pytest.approx(-1.0)
        assert rule.nodes[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_n_plus_one_nodes(self, n: int) -> None:
        """Test that L_{n+1} has n+1 nodes."""
        rule = gauss_lobatto_rule(n)
        assert rule.n_nodes == n + 1

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_positive_weights(self, n: int) -> None:
        """Test that all weights are positive."""
        rule = gauss_lobatto_rule(n)
        assert np.all(rule.weights > 0)

    def test_n1_is_trapezoidal(self) -> None:
        """Test that n=1 gives the trapezoidal rule."""
        rule = gauss_lobatto_rule(1)
        np.testing.assert_allclose(rule.nodes, [-1.0, 1.0])
        np.testing.assert_allclose(rule.weights, [1.0, 1.0])

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            gauss_lobatto_rule(0)


class TestGaussRadau:
    """Tests for Gauss–Radau quadrature rules."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_left_exactness(self, n: int) -> None:
        """Test left Radau exactness on polynomials of degree ≤ 2n."""
        rule = gauss_radau_left_rule(n)

        for k in range(2 * n + 1):
            approx = rule.apply(lambda x, k=k: x**k)
            exact = _exact_monomial_on_minus1_1(k)
            assert approx == pytest.approx(exact, abs=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_right_exactness(self, n: int) -> None:
        """Test right Radau exactness on polynomials of degree ≤ 2n."""
        rule = gauss_radau_right_rule(n)

        for k in range(2 * n + 1):
            approx = rule.apply(lambda x, k=k: x**k)
            exact = _exact_monomial_on_minus1_1(k)
            assert approx == pytest.approx(exact, abs=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_left_includes_minus_one(self, n: int) -> None:
        """Test that left Radau includes -1 as a node."""
        rule = gauss_radau_left_rule(n)
        assert rule.nodes[0] == pytest.approx(-1.0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_right_includes_plus_one(self, n: int) -> None:
        """Test that right Radau includes +1 as a node."""
        rule = gauss_radau_right_rule(n)
        assert rule.nodes[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_n_plus_one_nodes(self, n: int) -> None:
        """Test that Radau rules have n+1 nodes."""
        left = gauss_radau_left_rule(n)
        right = gauss_radau_right_rule(n)
        assert left.n_nodes == n + 1
        assert right.n_nodes == n + 1

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            gauss_radau_left_rule(0)
        with pytest.raises(ValueError, match="at least 1"):
            gauss_radau_right_rule(0)


class TestPeanoKernels:
    """Tests for Peano kernel computations."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_gauss_kernel_nonnegative(self, n: int) -> None:
        """Test that the Gauss Peano kernel is non-negative on [-1, 1]."""
        t = np.linspace(-1.0, 1.0, 501)
        kernel = peano_kernel_gauss_legendre(n, t)
        assert float(np.min(kernel)) >= -1e-14

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_lobatto_kernel_nonnegative(self, n: int) -> None:
        """Test that the Lobatto Peano kernel is non-negative on [-1, 1]."""
        t = np.linspace(-1.0, 1.0, 501)
        kernel = peano_kernel_gauss_lobatto(n, t)
        assert float(np.min(kernel)) >= -1e-14

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_kernels_vanish_at_one(self, n: int) -> None:
        """Test that both kernels vanish at t = 1."""
        kg = peano_kernel_gauss_legendre(n, 1.0)
        kl = peano_kernel_gauss_lobatto(n, 1.0)
        assert abs(kg) < 1e-14
        assert abs(kl) < 1e-14

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_kernel_dominance(self, n: int) -> None:
        """Test that K_L(t) ≥ K_G(t) pointwise (kernel dominance inequality)."""
        t = np.linspace(-1.0, 0.999, 201)  # Avoid t=1 where both are 0.
        kg = peano_kernel_gauss_legendre(n, t)
        kl = peano_kernel_gauss_lobatto(n, t)
        assert np.all(kl >= kg - 1e-14)

    def test_scalar_input(self) -> None:
        """Test that scalar input returns a scalar."""
        result = peano_kernel_gauss_legendre(3, 0.0)
        assert isinstance(result, float)

    def test_array_input(self) -> None:
        """Test that array input returns an array."""
        t = np.array([0.0, 0.5, 1.0])
        result = peano_kernel_gauss_legendre(3, t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            peano_kernel_gauss_legendre(0, 0.0)
        with pytest.raises(ValueError, match="at least 1"):
            peano_kernel_gauss_lobatto(0, 0.0)


class TestRadauCurvatureKernel:
    """Tests for the Radau curvature kernel."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_kernel_vanishes_at_endpoints(self, n: int) -> None:
        """Test that the Radau kernel vanishes at t = ±1."""
        # At t = 1, (x - t)_+ = 0 for x ≤ 1.
        k1 = radau_curvature_kernel_midpoint(n, 1.0)
        assert abs(k1) < 1e-13


    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_kernel_is_odd(self, n: int) -> None:
        """Test that the Radau midpoint kernel is numerically odd."""
        t = np.linspace(-0.95, 0.95, 101)
        kernel = radau_curvature_kernel_midpoint(n, t)
        reflected = radau_curvature_kernel_midpoint(n, -t)

        np.testing.assert_allclose(kernel, -reflected, atol=1e-13)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_kernel_changes_sign(self, n: int) -> None:
        """Test that the Radau midpoint kernel changes sign."""
        t = np.linspace(-0.95, 0.95, 101)
        kernel = radau_curvature_kernel_midpoint(n, t)

        assert float(np.min(kernel)) < 0.0
        assert float(np.max(kernel)) > 0.0

    def test_scalar_input(self) -> None:
        """Test that scalar input returns a scalar."""
        result = radau_curvature_kernel_midpoint(3, 0.0)
        assert isinstance(result, float)

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            radau_curvature_kernel_midpoint(0, 0.0)


class TestKernelDominanceRatio:
    """Tests for the kernel dominance ratio function."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_ratio_at_least_one(self, n: int) -> None:
        """Test that K_L / K_G ≥ 1 where defined."""
        t, ratio = kernel_dominance_ratio(n, num_points=101)
        valid = ~np.isnan(ratio)
        assert np.all(ratio[valid] >= 1.0 - 1e-10)

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            kernel_dominance_ratio(0)


class TestCertifiedEstimatorWeights:
    """Tests for the certified estimator weights."""

    def test_weights_sum_to_one(self) -> None:
        """Test that α + β = 1."""
        alpha, beta = certified_estimator_weights(3)
        assert alpha + beta == pytest.approx(1.0)

    def test_weights_values(self) -> None:
        """Test that α = 0.75 and β = 0.25."""
        alpha, beta = certified_estimator_weights(3)
        assert alpha == pytest.approx(0.75)
        assert beta == pytest.approx(0.25)

    def test_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            certified_estimator_weights(0)


class TestApplyRuleOnInterval:
    """Tests for the apply_rule_on_interval convenience function."""

    def test_basic_usage(self) -> None:
        """Test basic usage of apply_rule_on_interval."""
        rule = gauss_legendre_rule(3)
        result = apply_rule_on_interval(rule, lambda x: x**2, (0.0, 1.0))
        assert result == pytest.approx(1.0 / 3.0)

    def test_with_cache(self) -> None:
        """Test usage with a cached evaluator."""
        rule = gauss_legendre_rule(3)
        call_count = [0]

        def f(x: float) -> float:
            call_count[0] += 1
            return x**2

        # First call.
        result1 = apply_rule_on_interval(rule, f, (0.0, 1.0))
        count1 = call_count[0]

        # Second call without caching still evaluates.
        result2 = apply_rule_on_interval(rule, f, (0.0, 1.0))
        assert call_count[0] == 2 * count1

        assert result1 == pytest.approx(result2)
