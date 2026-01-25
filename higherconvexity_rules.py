
"""
Higher-convexity quadrature rules and Peano kernels.

This module provides Gauss--Legendre, Gauss--Lobatto, and Gauss--Radau rules on
the reference interval [-1, 1], together with Peano-kernel evaluators used in the
paper

    "Refined Gauss--Lobatto bounds for odd-order convexity"
    Tomasz Kania and Szymon Wąsowicz.

The implementation is intentionally lightweight and depends only on NumPy.

Version
-------
0.99.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, overload

import math
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre, leggauss

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
]

__version__ = "0.99.1"

ScalarFunc = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    """
    Interpolatory quadrature rule on the reference interval [-1, 1].

    Parameters
    ----------
    nodes:
        Nodes in [-1, 1], sorted increasingly.
    weights:
        Weights corresponding to `nodes`.
    name:
        Human-readable name.
    degree:
        Degree of exactness (largest d such that the rule is exact on Π_d).
    family:
        Family identifier ("gauss", "lobatto", "radau_left", "radau_right").
    """

    nodes: np.ndarray
    weights: np.ndarray
    name: str
    degree: int
    family: Literal["gauss", "lobatto", "radau_left", "radau_right"]

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if nodes.ndim != 1 or weights.ndim != 1:
            raise ValueError("nodes and weights must be 1D arrays")
        if nodes.shape != weights.shape:
            raise ValueError("nodes and weights must have the same shape")
        if not np.all(np.diff(nodes) >= 0):
            raise ValueError("nodes must be sorted increasingly")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.nodes.size)

    def rescale(self, a: float, b: float) -> "QuadratureRule":
        """
        Affinely rescale the rule from [-1, 1] to [a, b].

        The affine map is x = (a+b)/2 + (b-a)/2 * t.

        Parameters
        ----------
        a, b:
            Interval endpoints with a < b.

        Returns
        -------
        QuadratureRule
            A new rule with nodes in [a, b] and weights scaled by (b-a)/2.
        """
        if not (a < b):
            raise ValueError("Require a < b")
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        return QuadratureRule(
            nodes=mid + half * self.nodes,
            weights=half * self.weights,
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
        f:
            Scalar integrand.
        a, b:
            Interval endpoints with a < b.
        eval_fn:
            Optional evaluation function. If provided, it is called instead of `f`
            and can be used to implement caching of function values.

        Returns
        -------
        float
            Quadrature approximation.
        """
        rule = self if (a == -1.0 and b == 1.0) else self.rescale(a, b)
        g = f if eval_fn is None else eval_fn
        s = 0.0
        for x, w in zip(rule.nodes, rule.weights, strict=True):
            s += float(w) * float(g(float(x)))
        return float(s)


def apply_rule_on_interval(
    rule: QuadratureRule,
    f: ScalarFunc,
    interval: tuple[float, float],
    *,
    eval_fn: ScalarFunc | None = None,
) -> float:
    """
    Convenience wrapper: apply `rule` on a given interval.

    Parameters
    ----------
    rule:
        Quadrature rule on [-1, 1] (will be rescaled internally).
    f:
        Integrand.
    interval:
        Pair (a, b) with a < b.
    eval_fn:
        Optional cached evaluator.

    Returns
    -------
    float
        Approximation of ∫_a^b f(x) dx.
    """
    a, b = interval
    return rule.apply(f, a=a, b=b, eval_fn=eval_fn)


def gauss_legendre_rule(n: int) -> QuadratureRule:
    """
    n-point Gauss--Legendre rule on [-1, 1].

    Parameters
    ----------
    n:
        Number of nodes (n >= 1).

    Returns
    -------
    QuadratureRule
        Gauss--Legendre rule, exact on Π_{2n-1}.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes, weights = leggauss(n)
    # leggauss returns nodes in increasing order and positive weights.
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"G_{n} (Gauss--Legendre)",
        degree=2 * n - 1,
        family="gauss",
    )


def gauss_lobatto_rule(n: int) -> QuadratureRule:
    """
    (n+1)-point Gauss--Lobatto rule on [-1, 1].

    Here `n` matches the paper's indexing: G_n has n nodes and L_{n+1} has n+1 nodes.

    Parameters
    ----------
    n:
        Parameter n >= 1 (total nodes = n+1).

    Returns
    -------
    QuadratureRule
        Gauss--Lobatto rule, exact on Π_{2n-1}.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    if n == 1:
        # Trapezoidal rule on [-1, 1].
        nodes = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
        return QuadratureRule(
            nodes=nodes,
            weights=weights,
            name="L_2 (Gauss--Lobatto / trapezoidal)",
            degree=1,
            family="lobatto",
        )

    Pn = Legendre.basis(n)
    dPn = Pn.deriv()
    interior = np.asarray(dPn.roots(), dtype=float)
    interior.sort()

    nodes = np.concatenate(([-1.0], interior, [1.0]))
    weights = np.empty_like(nodes)

    # Endpoint weights.
    w_end = 2.0 / (n * (n + 1))
    weights[0] = w_end
    weights[-1] = w_end

    # Interior weights: 2 / (n(n+1) [P_n(x_i)]^2), where x_i are roots of P_n'.
    Pn_vals = Pn(interior)
    weights[1:-1] = 2.0 / (n * (n + 1) * (Pn_vals**2))

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"L_{n+1} (Gauss--Lobatto)",
        degree=2 * n - 1,
        family="lobatto",
    )


def _poly_divide_by_linear(p: Polynomial, root: float) -> Polynomial:
    """
    Divide a power-basis polynomial by (x - root), assuming exact divisibility.

    Parameters
    ----------
    p:
        Polynomial in power basis.
    root:
        Root to factor out.

    Returns
    -------
    Polynomial
        Quotient q such that p(x) = (x-root) q(x).
    """
    divisor = Polynomial([-root, 1.0])  # x - root
    q, r = divmod(p, divisor)
    if r.coef.size != 1 or abs(float(r.coef[0])) > 1e-10:
        raise RuntimeError("Polynomial division did not have negligible remainder; check formula.")
    return q


def gauss_radau_left_rule(n: int) -> QuadratureRule:
    """
    (n+1)-point Gauss--Radau rule with the fixed node at -1.

    Parameters
    ----------
    n:
        Parameter n >= 1 (total nodes = n+1, degree of exactness 2n).

    Returns
    -------
    QuadratureRule
        Left Gauss--Radau rule, exact on Π_{2n}.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    Pn = Legendre.basis(n)
    Pnp1 = Legendre.basis(n + 1)
    S = (Pn + Pnp1).convert(kind=Polynomial)
    # S(-1)=0, so factor (x+1).
    Q = _poly_divide_by_linear(S, root=-1.0)
    roots = np.asarray(Q.roots(), dtype=complex)
    roots = roots[np.isreal(roots)].real
    roots.sort()

    nodes = np.concatenate(([-1.0], roots))
    weights = np.empty_like(nodes)

    # Endpoint weight (classical closed form).
    weights[0] = 2.0 / ((n + 1) ** 2)

    # Interior weights:
    # w_i = (1 - x_i) / ((n+1)^2 [P_n(x_i)]^2), see e.g. Davis--Rabinowitz.
    Pn_vals = Pn(roots)
    weights[1:] = (1.0 - roots) / (((n + 1) ** 2) * (Pn_vals**2))

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"R^ℓ_{n+1} (Gauss--Radau left)",
        degree=2 * n,
        family="radau_left",
    )


def gauss_radau_right_rule(n: int) -> QuadratureRule:
    """
    (n+1)-point Gauss--Radau rule with the fixed node at +1.

    Parameters
    ----------
    n:
        Parameter n >= 1 (total nodes = n+1, degree of exactness 2n).

    Returns
    -------
    QuadratureRule
        Right Gauss--Radau rule, exact on Π_{2n}.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    Pn = Legendre.basis(n)
    Pnp1 = Legendre.basis(n + 1)
    S = (Pnp1 - Pn).convert(kind=Polynomial)
    # S(1)=0, so factor (x-1).
    Q = _poly_divide_by_linear(S, root=1.0)
    roots = np.asarray(Q.roots(), dtype=complex)
    roots = roots[np.isreal(roots)].real
    roots.sort()

    nodes = np.concatenate((roots, [1.0]))
    weights = np.empty_like(nodes)

    weights[-1] = 2.0 / ((n + 1) ** 2)

    Pn_vals = Pn(roots)
    weights[:-1] = (1.0 + roots) / (((n + 1) ** 2) * (Pn_vals**2))

    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        name=f"R^r_{n+1} (Gauss--Radau right)",
        degree=2 * n,
        family="radau_right",
    )


@overload
def peano_kernel_gauss_legendre(n: int, t: float) -> float: ...


@overload
def peano_kernel_gauss_legendre(n: int, t: np.ndarray) -> np.ndarray: ...


def peano_kernel_gauss_legendre(n: int, t: float | np.ndarray) -> float | np.ndarray:
    """
    Peano kernel K_G for the Gauss remainder of order 2n.

    The kernel is defined by
        K_G(t) = 1/(2n-1)! * ( I[ (·-t)_+^{2n-1} ] - G_n[ (·-t)_+^{2n-1} ] ).

    Parameters
    ----------
    n:
        Gauss parameter (number of nodes).
    t:
        Evaluation point(s) in [-1, 1]. Can be a float or a NumPy array.

    Returns
    -------
    float or numpy.ndarray
        Values of the Peano kernel K_G at `t`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    t_arr = np.asarray(t, dtype=float)
    rule = gauss_legendre_rule(n)
    nodes = rule.nodes[None, :]
    weights = rule.weights[None, :]

    power = 2 * n - 1
    phi = np.maximum(nodes - t_arr[..., None], 0.0) ** power
    G = np.sum(weights * phi, axis=-1)

    I = np.maximum(1.0 - t_arr, 0.0) ** (2 * n) / (2 * n)
    # NOTE: use stdlib factorial for NumPy 1.x / 2.x compatibility.
    K = (I - G) / float(math.factorial(int(2 * n - 1)))

    return float(K) if np.isscalar(t) else K


@overload
def peano_kernel_gauss_lobatto(n: int, t: float) -> float: ...


@overload
def peano_kernel_gauss_lobatto(n: int, t: np.ndarray) -> np.ndarray: ...


def peano_kernel_gauss_lobatto(n: int, t: float | np.ndarray) -> float | np.ndarray:
    """
    Peano kernel K_L for the Lobatto remainder of order 2n.

    The kernel is defined by
        K_L(t) = 1/(2n-1)! * ( L_{n+1}[ (·-t)_+^{2n-1} ] - I[ (·-t)_+^{2n-1} ] ).

    Parameters
    ----------
    n:
        Parameter matching the paper: Lobatto has n+1 nodes.
    t:
        Evaluation point(s) in [-1, 1]. Can be a float or a NumPy array.

    Returns
    -------
    float or numpy.ndarray
        Values of the Peano kernel K_L at `t`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    t_arr = np.asarray(t, dtype=float)
    rule = gauss_lobatto_rule(n)
    nodes = rule.nodes[None, :]
    weights = rule.weights[None, :]

    power = 2 * n - 1
    phi = np.maximum(nodes - t_arr[..., None], 0.0) ** power
    L = np.sum(weights * phi, axis=-1)

    I = np.maximum(1.0 - t_arr, 0.0) ** (2 * n) / (2 * n)
    # NOTE: use stdlib factorial for NumPy 1.x / 2.x compatibility.
    K = (L - I) / float(math.factorial(int(2 * n - 1)))

    return float(K) if np.isscalar(t) else K


@overload
def radau_curvature_kernel_midpoint(n: int, t: float) -> float: ...


@overload
def radau_curvature_kernel_midpoint(n: int, t: np.ndarray) -> np.ndarray: ...


def radau_curvature_kernel_midpoint(n: int, t: float | np.ndarray) -> float | np.ndarray:
    """
    Radau curvature kernel K^R_n for the Radau midpoint functional.

    We define M^R_n = (R^ℓ_{n+1} + R^r_{n+1}) / 2 (exact on Π_{2n}), and then
        K^R_n(t) = 1/(2n)! * ( M^R_n[ (·-t)_+^{2n} ] - I[ (·-t)_+^{2n} ] ).

    Parameters
    ----------
    n:
        Parameter n >= 1 (Radau rules have n+1 nodes, exact on Π_{2n}).
    t:
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    float or numpy.ndarray
        Values of K^R_n at `t`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    t_arr = np.asarray(t, dtype=float)

    left = gauss_radau_left_rule(n)
    right = gauss_radau_right_rule(n)

    power = 2 * n
    # Evaluate (x - t)_+^{2n} at the nodes via broadcasting.
    phi_left = np.maximum(left.nodes[None, :] - t_arr[..., None], 0.0) ** power
    phi_right = np.maximum(right.nodes[None, :] - t_arr[..., None], 0.0) ** power

    Rl = np.sum(left.weights[None, :] * phi_left, axis=-1)
    Rr = np.sum(right.weights[None, :] * phi_right, axis=-1)
    M = 0.5 * (Rl + Rr)

    I = np.maximum(1.0 - t_arr, 0.0) ** (2 * n + 1) / (2 * n + 1)

    # NOTE: use stdlib factorial for NumPy 1.x / 2.x compatibility.
    K = (M - I) / float(math.factorial(int(2 * n)))

    return float(K) if np.isscalar(t) else K
