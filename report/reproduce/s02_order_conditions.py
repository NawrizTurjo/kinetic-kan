"""
Runge-Kutta order conditions, checked on the tableaux the code actually uses.

For every rooted tree t with |t| <= 6 (37 trees) the residual
b . Phi(t) - 1/gamma(t) is evaluated. The order p is the largest q with all
residuals of order <= q below 1e-10; the principal error norm is
A^(p+1) = sqrt(sum over |t| = p+1 of (residual / sigma(t))^2).

Report: Table "order-conditions" (Section 3.1) and the order-condition listing
in Appendix A.
Output: numbers.json -> order_conditions, n_trees
"""

import numpy as np

from common import SOLVERS, save_numbers
from rk_theory import elem_weight, gamma_of, rooted_trees, sigma_of, tableau


def main():
    trees = rooted_trees(6)
    res = {}
    for name in SOLVERS:
        A, b = tableau(name)
        worst_by_order, err_norm = {}, {}
        for q in range(1, 7):
            devs = [b @ elem_weight(t, A) - 1.0 / gamma_of(t) for t in trees[q]]
            worst_by_order[q] = float(np.max(np.abs(devs)))
            err_norm[q] = float(np.sqrt(sum((d / sigma_of(t)) ** 2 for d, t in zip(devs, trees[q]))))
        p = max(q for q in range(1, 7) if all(worst_by_order[r] < 1e-10 for r in range(1, q + 1)))
        res[name] = dict(stages=int(len(b)), order=p, worst_residual=worst_by_order,
                         principal_error_norm=err_norm[p + 1] if p < 6 else None)
    for k, v in res.items():
        print(f"  {k:8s} s={v['stages']} p={v['order']} A^(p+1)={v['principal_error_norm']:.3e}")
    save_numbers(dict(order_conditions=res, n_trees={q: len(trees[q]) for q in trees}))


if __name__ == "__main__":
    main()
