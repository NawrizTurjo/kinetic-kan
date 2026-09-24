"""Root-find the learned field's equilibria and measure its Jacobian spectrum (stiffness study)."""
import sys

import numpy as np
import torch
from torch.autograd.functional import jacobian

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
sys.path.insert(0, IMPL)
from kan import KAN
from data import generate_damped_pendulum_data

H = 0.05 / 2  # dt=0.05 with two substeps


def load(solver, mu):
    p = IMPL + rf"\results\phase3\stiffness_map\probe\{solver}_mu{mu}_dt0.05\checkpoint.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = KAN(layers_hidden=ck["layers_hidden"], grid_len=ck["grid_len"], grid_lims=tuple(ck["grid_lims"]),
            basis_func=ck["basis_func"], normalizer=ck["normalizer"], base_act=ck["base_act"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return lambda u: m(u.view(1, 2)).view(2)


def newton(f, u0, iters=50):
    u = torch.tensor(u0, dtype=torch.float64).float()
    for _ in range(iters):
        J = jacobian(f, u)
        step = torch.linalg.solve(J, f(u))
        u = u - step
        if step.norm() < 1e-9:
            break
    return u


print("True linearization at (0,0):")
for mu in (0.1, 0.5, 1.0, 2.0, 5.0, 8.0):
    lam = np.linalg.eigvals(np.array([[0, 1], [-9.81, -mu]]))
    print(f"  mu={mu}: eig={np.round(lam, 3)}  max h|lam|={H * np.abs(lam).max():.3f}  "
          f"stiffness ratio={np.abs(lam).max() / np.abs(lam).min():.2f}")

print("\nLearned field (seed 42 probe checkpoints):")
overall = 0.0
for mu in (0.1, 0.5, 1.0, 2.0, 5.0, 8.0):
    data = generate_damped_pendulum_data(mu=mu, t_end=10.0, dt=0.05, t_train_end=5.0)
    states = data.y_full
    for solver in ("euler", "midpoint", "rk4", "tsit5"):
        f = load(solver, mu)
        with torch.no_grad():
            f00 = f(torch.zeros(2)).norm().item()
        # spectral radius of the learned Jacobian along the true trajectory
        rad = max(np.abs(torch.linalg.eigvals(jacobian(f, s)).numpy()).max() for s in states[::5])
        overall = max(overall, H * rad)
        line = (f"  mu={mu} {solver:8s} |f(0,0)|={f00:.3f}  "
                f"max|eig| on trajectory={rad:.2f} -> h|lam|={H * rad:.3f}")
        if mu in (1.0, 2.0):  # the equilibrium search is reported for the trapped cases
            root = newton(f, [-0.05, -0.02])
            eig = torch.linalg.eigvals(jacobian(f, root)).numpy()
            with torch.no_grad():
                res = f(root).norm().item()
            line += (f"  root=({root[0]:+.4f},{root[1]:+.4f}) |f(root)|={res:.1e}  "
                     f"eig(J)={np.round(eig, 3)}")
        print(line)
print(f"\nLargest h|lambda| over all 24 learned fields: {overall:.3f}")
