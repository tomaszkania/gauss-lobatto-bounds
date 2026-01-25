"""
Higher-convexity quadrature rules and Peano kernels.

This module provides Gauss–Legendre, Gauss–Lobatto, and Gauss–Radau rules on
the reference interval [-1, 1], together with Peano-kernel evaluators used in
the paper

    "Refined Gauss–Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

The implementation is intentionally lightweight and depends only on NumPy.

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
from typing import TYPE_CHECKING, Callable, Literal, Union, overload

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre, leggauss

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "ScalarFunc",
    "QuadratureRule",
    "gauss_legendre_rule",
    "gauss_lobatto_rule",
    "gauss_radau_left_rule",
    "gauss_radau_right_rule",
    "apply_rule_on_interval",
    "peano_kernel_gauss_legendre",
    "peano_kernel_gauss_lobatto",
    "radau_curvature_kernel_midpoint",
    "kernel_dominance_ratio",
    "certified_estimator_weights",
]

__version__ = "1.0.0"

ScalarFunc = Callable[[float], float]
ArrayLike = Union[float, "NDArray[np.floating]"]


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    """
    Interpolatory quadrature rule on the reference interval [-1, 1].

    This class represents a quadrature rule with nodes and weights, supporting
    affine rescaling to arbitrary intervals and function evaluation.

    Parameters
    ----------
    nodes : array_like
        Nodes in [-1, 1], sorted in increasing order.
    weights : array_like
        Weights corresponding to `nodes`.
    name : str
        Human-readable name for the rule.
    degree : int
        Degree of exactness (largest d such that the rule is exact on Π_d).
    family : {'gauss', 'lobatto', 'radau_left', 'radau_right'}
        Family identifier for the quadrature rule.

    Raises
    ------
    ValueError
        If nodes and weights have mismatched shapes or nodes are not sorted.

    Examples
    --------
    >>> rule = gauss_legendre_rule(3)
    >>> rule.apply(lambda x: x**2)  # Integrates x² over [-1, 1]
    0.6666666666666666
    """

    nodes: np.ndarray
    weights: np.ndarray
    name: str
    degree: int
    family: Literal["gauss", "lobatto", "radau_left", "radau_right"]

    def __post_init__(self) -> None:
        """Validate and normalise input arrays."""
        nodes = np.asarray(self.nodes, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)

        if nodes.ndim != 1 or weights.ndim != 1:
            raise ValueError("Nodes and weights must be one-dimensional arrays.")
        if nodes.shape != weights.shape:
            raise ValueError("Nodes and weights must have the same shape.")
        if nodes.size > 1 and not np.all(np.diff(nodes) >= 0):
            raise ValueError("Nodes must be sorted in non-decreasing order.")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def n_nodes(self) -> int:
        """Return the number of quadrature nodes."""
        return int(self.nodes.size)

    @property
    def total_weight(self) -> float:
        """Return the sum of all weights (equals 2 for rules on [-1, 1])."""
        return float(np.sum(self.weights))

    def rescale(self, a: float, b: float) -> QuadratureRule:
        """
        Affinely rescale the rule from [-1, 1] to [a, b].

        The affine transformation is x = (a + b)/2 + (b - a)/2 · t, where t is
        the original node in [-1, 1].

        Parameters
        ----------
        a : float
            Left endpoint of the target interval.
        b : float
            Right endpoint of the target interval.

        Returns
        -------
        QuadratureRule
            A new rule with nodes in [a, b] and weights scaled by (b - a)/2.

        Raises
        ------
        ValueError
            If a >= b.

        Examples
        --------
        >>> rule = gauss_legendre_rule(2)
        >>> rescaled = rule.rescale(0.0, 1.0)
        >>> rescaled.nodes  # Nodes now in [0, 1]
        array([0.21132487, 0.78867513])
        """
        if not (a < b):
            raise ValueError(f"Require a < b, but received a={a}, b={b}.")

        midpoint = 0.5 * (a + b)
        half_width = 0.5 * (b - a)

        return QuadratureRule(
            nodes=midpoint + half_width * self.nodes,
            weights=half_width * self.weights,
            name=f"{self.name} on [{a}, {b}]",
            degree=self.degree,
            family=self.family,
        )

    def apply(
        self,
        f: ScalarFunc,
        *,
        a: float = -1.0,
        b: float = 1.0,
        eval_fn: ScalarFunc | None = None,
    ) -> float:
        """
        Apply the quadrature rule to a scalar function on [a, b].

        Parameters
        ----------
        f : callable
            Scalar integrand accepting a float and returning a float.
        a : float, default=-1.0
            Left endpoint of the integration interval.
        b : float, default=1.0
            Right endpoint of the integration interval.
        eval_fn : callable, optional
            Optional evaluation function. If provided, it is called instead
            of `f` and can be used to implement caching of function values.

        Returns
        -------
        float
            Quadrature approximation of ∫_a^b f(x) dx.

        Examples
        --------
        >>> rule = gauss_legendre_rule(4)
        >>> rule.apply(lambda x: x**4, a=0.0, b=1.0)
        0.2
        """
        rule = self if (a == -1.0 and b == 1.0) else self.rescale(a, b)
        evaluator = f if eval_fn is None else eval_fn

        total = 0.0
        for node, weight in zip(rule.nodes, rule.weights, strict=True):
            total += float(weight) * float(evaluator(float(node)))

        return float(total)

    def evaluate_at_nodes(self, f: ScalarFunc) -> np.ndarray:
        """
        Evaluate a function at all quadrature nodes.

        Parameters
        ----------
        f : callable
            Scalar function to evaluate.

        Returns
        -------
        ndarray
            Array of function values at each node.
        """
        return np.array([f(float(x)) for x in self.nodes], dtype=np.float64)


def apply_rule_on_interval(
    rule: QuadratureRule,
    f: ScalarFunc,
    interval: tuple[float, float],
    *,
    eval_fn: ScalarFunc | None = None,
) -> float:
    """
    Convenience wrapper: apply a quadrature rule on a given interval.

    Parameters
    ----------
    rule : QuadratureRule
        Quadrature rule on [-1, 1] (will be rescaled internally).
    f : callable
        Integrand.
    interval : tuple of float
        Pair (a, b) with a < b.
    eval_fn : callable, optional
        Optional cached evaluator.

    Returns
    -------
    float
        Approximation of ∫_a^b f(x) dx.

    Examples
    --------
    >>> rule = gauss_legendre_rule(3)
    >>> apply_rule_on_interval(rule, lambda x: x**2, (0.0, 1.0))
    0.3333333333333333
    """
    a, b = interval
    return rule.apply(f, a=a, b=b, eval_fn=eval_fn)


def gauss_legendre_rule(n: int) -> QuadratureRule:
    """
    Construct the n-point Gauss–Legendre rule on [-1, 1].

    The Gauss–Legendre rule with n nodes achieves the maximal degree of
    exactness 2n - 1 among all interpolatory quadrature rules with n nodes.

    Parameters
    ----------
    n : int
        Number of nodes (n ≥ 1).

    Returns
    -------
    QuadratureRule
        Gauss–Legendre rule, exact on Π_{2n-1}.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    The nodes are the roots of the Legendre polynomial P_n, and the weights
    are computed using the standard formula involving P'_n.

    Examples
    --------
    >>> rule = gauss_legendre_rule(2)
    >>> rule.degree
    3
    >>> rule.n_nodes
    2
    """
    if n < 1:
        raise ValueError(f"Number of nodes n must be at least 1, but received n={n}.")

    nodes, weights = leggauss(n)
    # leggauss returns nodes in increasing order and positive weights.
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"G_{n} (Gauss–Legendre)",
        degree=2 * n - 1,
        family="gauss",
    )


def gauss_lobatto_rule(n: int) -> QuadratureRule:
    """
    Construct the (n+1)-point Gauss–Lobatto rule on [-1, 1].

    The indexing matches the paper's convention: G_n has n nodes and L_{n+1}
    has n + 1 nodes, with both rules having degree of exactness 2n - 1.

    Parameters
    ----------
    n : int
        Parameter n ≥ 1 (total number of nodes = n + 1).

    Returns
    -------
    QuadratureRule
        Gauss–Lobatto rule, exact on Π_{2n-1}.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    The Gauss–Lobatto rule includes both endpoints ±1 as nodes. The interior
    nodes are the roots of P'_n (the derivative of the Legendre polynomial).

    Examples
    --------
    >>> rule = gauss_lobatto_rule(2)
    >>> rule.n_nodes
    3
    >>> rule.nodes[0], rule.nodes[-1]  # Endpoints
    (-1.0, 1.0)
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    if n == 1:
        # Special case: trapezoidal rule on [-1, 1].
        nodes = np.array([-1.0, 1.0], dtype=np.float64)
        weights = np.array([1.0, 1.0], dtype=np.float64)
        return QuadratureRule(
            nodes=nodes,
            weights=weights,
            name="L_2 (Gauss–Lobatto / trapezoidal)",
            degree=1,
            family="lobatto",
        )

    # Compute interior nodes as roots of P'_n.
    legendre_n = Legendre.basis(n)
    deriv_legendre_n = legendre_n.deriv()
    interior_nodes = np.asarray(deriv_legendre_n.roots(), dtype=np.float64)
    interior_nodes.sort()

    nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))
    weights = np.empty_like(nodes)

    # Endpoint weights: 2 / (n(n+1)).
    endpoint_weight = 2.0 / (n * (n + 1))
    weights[0] = endpoint_weight
    weights[-1] = endpoint_weight

    # Interior weights: 2 / (n(n+1) [P_n(x_i)]²).
    legendre_at_interior = legendre_n(interior_nodes)
    weights[1:-1] = 2.0 / (n * (n + 1) * legendre_at_interior**2)

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"L_{n+1} (Gauss–Lobatto)",
        degree=2 * n - 1,
        family="lobatto",
    )


def _poly_divide_by_linear(p: Polynomial, root: float) -> Polynomial:
    """
    Divide a polynomial by (x - root), assuming exact divisibility.

    Parameters
    ----------
    p : Polynomial
        Polynomial in power basis.
    root : float
        Root to factor out.

    Returns
    -------
    Polynomial
        Quotient q such that p(x) = (x - root) q(x).

    Raises
    ------
    RuntimeError
        If the remainder is not negligible (indicating numerical issues).
    """
    divisor = Polynomial([-root, 1.0])  # x - root
    quotient, remainder = divmod(p, divisor)

    if remainder.coef.size != 1 or abs(float(remainder.coef[0])) > 1e-10:
        raise RuntimeError(
            "Polynomial division did not yield a negligible remainder; "
            "check the formula or numerical precision."
        )

    return quotient


def gauss_radau_left_rule(n: int) -> QuadratureRule:
    """
    Construct the (n+1)-point Gauss–Radau rule with a fixed node at -1.

    Parameters
    ----------
    n : int
        Parameter n ≥ 1 (total number of nodes = n + 1, degree of exactness = 2n).

    Returns
    -------
    QuadratureRule
        Left Gauss–Radau rule, exact on Π_{2n}.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    The nodes are -1 together with the roots of (P_n + P_{n+1})/(1 + x).
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    legendre_n = Legendre.basis(n)
    legendre_np1 = Legendre.basis(n + 1)
    sum_poly = (legendre_n + legendre_np1).convert(kind=Polynomial)

    # Factor out (x + 1) since sum_poly(-1) = 0.
    quotient = _poly_divide_by_linear(sum_poly, root=-1.0)
    roots = np.asarray(quotient.roots(), dtype=complex)
    real_roots = roots[np.abs(roots.imag) < 1e-12].real
    real_roots.sort()

    nodes = np.concatenate(([-1.0], real_roots))
    weights = np.empty_like(nodes)

    # Endpoint weight (classical closed form).
    weights[0] = 2.0 / ((n + 1) ** 2)

    # Interior weights: (1 - x_i) / ((n+1)² [P_n(x_i)]²).
    legendre_at_roots = legendre_n(real_roots)
    weights[1:] = (1.0 - real_roots) / (((n + 1) ** 2) * legendre_at_roots**2)

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"R^ℓ_{n+1} (Gauss–Radau left)",
        degree=2 * n,
        family="radau_left",
    )


def gauss_radau_right_rule(n: int) -> QuadratureRule:
    """
    Construct the (n+1)-point Gauss–Radau rule with a fixed node at +1.

    Parameters
    ----------
    n : int
        Parameter n ≥ 1 (total number of nodes = n + 1, degree of exactness = 2n).

    Returns
    -------
    QuadratureRule
        Right Gauss–Radau rule, exact on Π_{2n}.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    The nodes are the roots of (P_{n+1} - P_n)/(x - 1) together with +1.
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    legendre_n = Legendre.basis(n)
    legendre_np1 = Legendre.basis(n + 1)
    diff_poly = (legendre_np1 - legendre_n).convert(kind=Polynomial)

    # Factor out (x - 1) since diff_poly(1) = 0.
    quotient = _poly_divide_by_linear(diff_poly, root=1.0)
    roots = np.asarray(quotient.roots(), dtype=complex)
    real_roots = roots[np.abs(roots.imag) < 1e-12].real
    real_roots.sort()

    nodes = np.concatenate((real_roots, [1.0]))
    weights = np.empty_like(nodes)

    # Endpoint weight.
    weights[-1] = 2.0 / ((n + 1) ** 2)

    # Interior weights: (1 + x_i) / ((n+1)² [P_n(x_i)]²).
    legendre_at_roots = legendre_n(real_roots)
    weights[:-1] = (1.0 + real_roots) / (((n + 1) ** 2) * legendre_at_roots**2)

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"R^r_{n+1} (Gauss–Radau right)",
        degree=2 * n,
        family="radau_right",
    )


@overload
def peano_kernel_gauss_legendre(n: int, t: float) -> float: ...


@overload
def peano_kernel_gauss_legendre(n: int, t: np.ndarray) -> np.ndarray: ...


def peano_kernel_gauss_legendre(
    n: int,
    t: ArrayLike,
) -> ArrayLike:
    """
    Compute the Peano kernel K_G for the Gauss remainder of order 2n.

    The kernel is defined by
        K_G(t) = 1/(2n-1)! · ( I[(·-t)_+^{2n-1}] - G_n[(·-t)_+^{2n-1}] ).

    Parameters
    ----------
    n : int
        Gauss parameter (number of nodes).
    t : float or array_like
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    float or ndarray
        Values of the Peano kernel K_G at `t`.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    For (2n-1)-convex functions, this kernel is non-negative, which is the
    foundation for the refined Gauss–Lobatto bounds.
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    t_arr = np.asarray(t, dtype=np.float64)
    scalar_input = t_arr.ndim == 0
    t_arr = np.atleast_1d(t_arr)

    rule = gauss_legendre_rule(n)
    nodes = rule.nodes[np.newaxis, :]
    weights = rule.weights[np.newaxis, :]

    power = 2 * n - 1
    phi = np.maximum(nodes - t_arr[:, np.newaxis], 0.0) ** power
    gauss_term = np.sum(weights * phi, axis=-1)

    # Exact integral: ∫_{-1}^1 (x-t)_+^{2n-1} dx = (1-t)_+^{2n} / (2n).
    integral_term = np.maximum(1.0 - t_arr, 0.0) ** (2 * n) / (2 * n)
    kernel = (integral_term - gauss_term) / float(math.factorial(2 * n - 1))

    return float(kernel[0]) if scalar_input else kernel


@overload
def peano_kernel_gauss_lobatto(n: int, t: float) -> float: ...


@overload
def peano_kernel_gauss_lobatto(n: int, t: np.ndarray) -> np.ndarray: ...


def peano_kernel_gauss_lobatto(
    n: int,
    t: ArrayLike,
) -> ArrayLike:
    """
    Compute the Peano kernel K_L for the Lobatto remainder of order 2n.

    The kernel is defined by
        K_L(t) = 1/(2n-1)! · ( L_{n+1}[(·-t)_+^{2n-1}] - I[(·-t)_+^{2n-1}] ).

    Parameters
    ----------
    n : int
        Parameter matching the paper: Lobatto has n + 1 nodes.
    t : float or array_like
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    float or ndarray
        Values of the Peano kernel K_L at `t`.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    For (2n-1)-convex functions, this kernel is non-negative and dominates
    the Gauss kernel K_G pointwise, yielding the refined bracket inequality.
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    t_arr = np.asarray(t, dtype=np.float64)
    scalar_input = t_arr.ndim == 0
    t_arr = np.atleast_1d(t_arr)

    rule = gauss_lobatto_rule(n)
    nodes = rule.nodes[np.newaxis, :]
    weights = rule.weights[np.newaxis, :]

    power = 2 * n - 1
    phi = np.maximum(nodes - t_arr[:, np.newaxis], 0.0) ** power
    lobatto_term = np.sum(weights * phi, axis=-1)

    integral_term = np.maximum(1.0 - t_arr, 0.0) ** (2 * n) / (2 * n)
    kernel = (lobatto_term - integral_term) / float(math.factorial(2 * n - 1))

    return float(kernel[0]) if scalar_input else kernel


@overload
def radau_curvature_kernel_midpoint(n: int, t: float) -> float: ...


@overload
def radau_curvature_kernel_midpoint(n: int, t: np.ndarray) -> np.ndarray: ...


def radau_curvature_kernel_midpoint(
    n: int,
    t: ArrayLike,
) -> ArrayLike:
    """
    Compute the Radau curvature kernel K^R_n for the Radau midpoint functional.

    The midpoint functional is defined as M^R_n = (R^ℓ_{n+1} + R^r_{n+1}) / 2,
    and the kernel is
        K^R_n(t) = 1/(2n)! · ( M^R_n[(·-t)_+^{2n}] - I[(·-t)_+^{2n}] ).

    Parameters
    ----------
    n : int
        Parameter n ≥ 1 (Radau rules have n + 1 nodes, exact on Π_{2n}).
    t : float or array_like
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    float or ndarray
        Values of K^R_n at `t`.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    This kernel is relevant for analysing even-order (2n)-convex functions
    under the Radau midpoint combination.
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    t_arr = np.asarray(t, dtype=np.float64)
    scalar_input = t_arr.ndim == 0
    t_arr = np.atleast_1d(t_arr)

    left_rule = gauss_radau_left_rule(n)
    right_rule = gauss_radau_right_rule(n)

    power = 2 * n
    phi_left = np.maximum(left_rule.nodes[np.newaxis, :] - t_arr[:, np.newaxis], 0.0) ** power
    phi_right = np.maximum(right_rule.nodes[np.newaxis, :] - t_arr[:, np.newaxis], 0.0) ** power

    radau_left = np.sum(left_rule.weights[np.newaxis, :] * phi_left, axis=-1)
    radau_right = np.sum(right_rule.weights[np.newaxis, :] * phi_right, axis=-1)
    midpoint_term = 0.5 * (radau_left + radau_right)

    integral_term = np.maximum(1.0 - t_arr, 0.0) ** (2 * n + 1) / (2 * n + 1)
    kernel = (midpoint_term - integral_term) / float(math.factorial(2 * n))

    return float(kernel[0]) if scalar_input else kernel


def kernel_dominance_ratio(
    n: int,
    num_points: int = 1001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pointwise ratio K_L(t) / K_G(t) over [-1, 1].

    This ratio demonstrates the kernel dominance inequality K_G ≤ K_L used
    in the refined bounds.

    Parameters
    ----------
    n : int
        Parameter n ≥ 1.
    num_points : int, default=1001
        Number of evaluation points.

    Returns
    -------
    t : ndarray
        Array of points in [-1, 1].
    ratio : ndarray
        Values of K_L(t) / K_G(t) at each point.

    Notes
    -----
    At points where K_G(t) ≈ 0, the ratio may be numerically unstable.
    The ratio is set to NaN where K_G < 1e-15.
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    t = np.linspace(-1.0, 1.0, num_points, dtype=np.float64)
    k_gauss = peano_kernel_gauss_legendre(n, t)
    k_lobatto = peano_kernel_gauss_lobatto(n, t)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(k_gauss > 1e-15, k_lobatto / k_gauss, np.nan)

    return t, ratio


def certified_estimator_weights(n: int) -> tuple[float, float]:
    """
    Return the weights (α, β) for the certified estimator Q_n = α G_n + β L_{n+1}.

    For the certified bounds with θ = 1/4 (minimax optimal), we have
    α = 3/4 and β = 1/4.

    Parameters
    ----------
    n : int
        Parameter n ≥ 1 (included for API consistency, though weights are fixed).

    Returns
    -------
    alpha : float
        Weight for the Gauss–Legendre rule (= 0.75).
    beta : float
        Weight for the Gauss–Lobatto rule (= 0.25).

    Notes
    -----
    These weights arise from the minimax optimality condition:
    choosing θ = 1/4 minimises the worst-case error bound
    |I[f] - Q_n[f]| ≤ θ |L_{n+1}[f] - G_n[f]|.
    """
    if n < 1:
        raise ValueError(f"Parameter n must be at least 1, but received n={n}.")

    return 0.75, 0.25
