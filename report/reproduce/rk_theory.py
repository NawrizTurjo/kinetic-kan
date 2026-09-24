"""
Runge-Kutta theory used by the order-condition, work-precision and stability scripts.

  * tableau(name): the Butcher tableau (A, b) exactly as implemented in
    implementation/ode/solvers.py
  * rooted trees up to a given order, with order, density gamma(t) and symmetry sigma(t)
  * elementary weights Phi(t) = b . (product of A-powers over the tree)
  * the linear stability function R(z) = 1 + z b^T (I - zA)^{-1} 1
"""

import math
from collections import Counter

import numpy as np

from common import IMPL  # noqa: F401  (puts implementation/ on sys.path)
from ode.solvers import DOPRI5_A, DOPRI5_B, TSIT5_A, TSIT5_B  # noqa: E402


def tableau(name):
    if name == "euler":
        return np.zeros((1, 1)), np.array([1.0])
    if name == "midpoint":
        return np.array([[0, 0], [0.5, 0]]), np.array([0, 1.0])
    if name == "heun":
        return np.array([[0, 0], [1.0, 0]]), np.array([0.5, 0.5])
    if name == "rk4":
        A = np.zeros((4, 4)); A[1, 0] = A[2, 1] = 0.5; A[3, 2] = 1.0
        return A, np.array([1, 2, 2, 1]) / 6.0
    rows, b = (TSIT5_A, TSIT5_B) if name == "tsit5" else (DOPRI5_A, DOPRI5_B)
    s = 6  # stage 7 (FSAL) has zero solution weight and is not evaluated
    A = np.zeros((s, s))
    for i in range(s):
        A[i, :len(rows[i])] = rows[i]
    return A, np.array(b[:s], dtype=float)


def rooted_trees(max_order):
    """All rooted trees up to max_order; a tree is a sorted tuple of its child subtrees."""
    by_order = {1: [()]}
    for n in range(2, max_order + 1):
        found = set()

        # children multiset with total order n-1
        def parts(rem, min_key):
            if rem == 0:
                yield ()
                return
            for k in range(1, rem + 1):
                for t in by_order[k]:
                    key = (k, t)
                    if min_key is not None and key < min_key:
                        continue
                    for rest in parts(rem - k, key):
                        yield (t,) + rest
        for ch in parts(n - 1, None):
            found.add(tuple(sorted(ch)))
        by_order[n] = sorted(found)
    return by_order


def order_of(t):
    return 1 + sum(order_of(c) for c in t)


def gamma_of(t):
    return order_of(t) * math.prod(gamma_of(c) for c in t)


def sigma_of(t):
    s = 1
    for child, mult in Counter(t).items():
        s *= math.factorial(mult) * sigma_of(child) ** mult
    return s


def elem_weight(t, A):
    g = np.ones(A.shape[0])
    for child in t:
        g = g * (A @ elem_weight(child, A))
    return g


def stability_function(name, z):
    A, b = tableau(name)
    s = len(b)
    one = np.ones(s)
    R = np.empty_like(z, dtype=complex)
    for idx, zz in np.ndenumerate(z):
        R[idx] = 1 + zz * b @ np.linalg.solve(np.eye(s) - zz * A, one)
    return R
