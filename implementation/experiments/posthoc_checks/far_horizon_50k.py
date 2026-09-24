"""Score checkpoints on near (3.5,14] and far (14,28] Lotka-Volterra windows."""
import sys
import torch
import numpy as np

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
ROOT = IMPL  # this script calls the implementation directory ROOT
sys.path.insert(0, ROOT)
from kan import KAN, MLP_ODE
from ode import NeuralODE
from data import generate_lotka_volterra_data

RUNS = {
    "KAN 10k": r"\results\benchmarks\kanode_flagship\best_model.pt",
    "MLP-SiLU 10k": r"\results\benchmarks\mlpode_baseline_silu\best_model.pt",
    "MLP-tanh 10k": r"\results\benchmarks\mlpode_baseline\best_model.pt",
    "KAN 50k": r"\results\phase4\epoch_budget_check\tsit5_rbf_50k\best_model.pt",
    "MLP-SiLU 50k": r"\results\phase4\epoch_budget_check\mlp_silu_50k\best_model.pt",
    "MLP-tanh 50k": r"\results\phase4\epoch_budget_check\mlp_paperspec_50k\best_model.pt",
    "Euler 50k": r"\results\phase4\epoch_budget_check\euler_50k\best_model.pt",
    "B-spline 25k": r"\results\phase4\epoch_budget_check\bspline_25k\best_model.pt",
}
data = generate_lotka_volterra_data(t_end=28.0, dt=0.1, t_train_end=3.5)
t = data.t_full
y = data.y_full.numpy()
i14 = int(round(14.0 / 0.1)) + 1
n_tr = len(data.t_train)
for name, rel in RUNS.items():
    ck = torch.load(ROOT + rel, map_location="cpu", weights_only=False)
    c = ck["config"]
    if c["model_type"] == "mlp":
        m = MLP_ODE(layers_hidden=c["layers_hidden"], activation=c["mlp_act"])
    else:
        m = KAN(layers_hidden=c["layers_hidden"], grid_len=c["grid_len"],
                grid_lims=tuple(c.get("grid_lims", [-1, 1])), basis_func=c["basis_func"],
                normalizer=c["normalizer"], base_act=c["base_act"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    node = NeuralODE(func=m, method=c["solver"], substeps=c["substeps"])
    with torch.no_grad():
        p = node(y0=data.y0, t=t).numpy()
    mse = lambda a, b: float(np.mean((a - b) ** 2))
    print(f"{name:14s} epoch={ck['epoch']:6d} train={mse(y[:n_tr], p[:n_tr]):.3e} "
          f"near={mse(y[n_tr:i14], p[n_tr:i14]):.3e} far={mse(y[i14:], p[i14:]):.3e}")
