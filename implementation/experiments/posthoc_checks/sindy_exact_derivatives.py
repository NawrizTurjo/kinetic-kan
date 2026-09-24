"""How much of SINDy's noise-free error is the finite-difference derivative estimate?"""
import sys

import numpy as np
import pysindy as ps

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
sys.path.insert(0, IMPL)
from data import generate_lotka_volterra_data

d = generate_lotka_volterra_data(t_end=14.0, dt=0.1, t_train_end=3.5, noise_std=0.0)
t_tr, y_tr = d.t_train.numpy().astype(float), d.y_train.numpy().astype(float)
t_full, y_full = d.t_full.numpy().astype(float), d.y_full.numpy().astype(float)
n = len(t_tr)
true_dot = np.stack([1.5 * y_tr[:, 0] - y_tr[:, 0] * y_tr[:, 1],
                     y_tr[:, 0] * y_tr[:, 1] - 3.0 * y_tr[:, 1]], axis=1)


def fit(x_dot=None, diff=None):
    kw = dict(feature_library=ps.PolynomialLibrary(degree=2), optimizer=ps.STLSQ(threshold=0.05))
    if diff is not None:
        kw["differentiation_method"] = diff
    m = ps.SINDy(**kw)
    m.fit(y_tr, t=t_tr, x_dot=x_dot)
    sim = m.simulate(y_tr[0], t_full)
    return m, float(np.mean((sim[n:] - y_full[n:]) ** 2))


m_fd, e_fd = fit(diff=ps.FiniteDifference())
fd_dot = ps.FiniteDifference()._differentiate(y_tr, t_tr)
m_ex, e_ex = fit(x_dot=true_dot)
print("derivative RMS error of FD on the 36 points:", float(np.sqrt(np.mean((fd_dot - true_dot) ** 2))),
      " (true derivative RMS", float(np.sqrt(np.mean(true_dot ** 2))), ")")
print(f"SINDy, finite differences : extrap MSE = {e_fd:.3e}")
m_fd.print()
print(f"SINDy, exact derivatives  : extrap MSE = {e_ex:.3e}")
m_ex.print()
