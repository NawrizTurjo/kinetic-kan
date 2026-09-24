"""Post-hoc analyses on saved checkpoints (report Sections 6.3, 9 and 12).

1. Pendulum models rescored on a common test window (5, 10].
2. Hybrid-basis models: actual RMS contribution of each basis.
3. RBF vs B-spline training loss at a matched 25,000-epoch budget.

The original failing pendulum checkpoint (t_train=3) was removed from the
working tree in a later commit; it is extracted from git history on first run.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
ORIG_PENDULUM_REV = "6f37a66"
ORIG_PENDULUM_PATH = "implementation/results/benchmarks/pendulum/best_model.pt"
_orig = os.path.join(SCR, "pend_orig", "best_model.pt")
if not os.path.exists(_orig):
    os.makedirs(os.path.dirname(_orig), exist_ok=True)
    blob = subprocess.run(["git", "-C", ROOT, "show", f"{ORIG_PENDULUM_REV}:{ORIG_PENDULUM_PATH}"],
                          check=True, capture_output=True).stdout
    open(_orig, "wb").write(blob)
sys.path.insert(0, IMPL)
sys.path.insert(0, IMPL + r"\experiments\hybrid_basis")
from kan import KAN
from kan.basis import bspline_basis, rbf
from ode import NeuralODE
from evaluate import _rebuild_data
from utils import compute_r2_score
from load_hybrid import load as load_hybrid


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def build_kan(c, sd):
    m = KAN(layers_hidden=c["layers_hidden"], grid_len=c["grid_len"],
            grid_lims=tuple(c.get("grid_lims", [-1, 1])), basis_func=c["basis_func"],
            normalizer=c["normalizer"], base_act=c["base_act"])
    m.load_state_dict(sd)
    m.eval()
    return m


# ---------------------------------------------------------------- 1. pendulum
print("=== 1. Pendulum: both models scored on the common window (5, 10] ===")
runs = {
    "original (train [0,3], SiLU)": SCR + r"\pend_orig\best_model.pt",
    "adopted (train [0,5], SiLU)": IMPL + r"\results\_fixed\pendulum_control_win5\best_model.pt",
    "identity (train [0,5])": IMPL + r"\results\_fixed\pendulum_fixed\best_model.pt",
}
for name, path in runs.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    node = NeuralODE(func=build_kan(c, ck["model_state_dict"]), method=c["solver"], substeps=c["substeps"])
    data, _ = _rebuild_data(c)
    with torch.no_grad():
        p = node(y0=data.y0, t=data.t_full).numpy()
    t = data.t_full.numpy()
    y = data.y_full.numpy()
    own = t > c["t_train_end"] + 1e-9
    common = t > 5.0 + 1e-9
    print(f"{name:30s} t_train={c['t_train_end']} own-window R2={compute_r2_score(y[own], p[own]):+.4f} "
          f"common (5,10] R2={compute_r2_score(y[common], p[common]):+.4f} "
          f"MSE(5,10]={mse(y[common], p[common]):.3e} full MSE={mse(y, p):.3e}")

# ------------------------------------------------------------------ 2. hybrid
print("\n=== 2. Hybrid basis: actual contribution of each basis ===")


def contributions(model, x):
    """RMS over batch/outputs of alpha*C*Bspline and beta*C*RBF per layer."""
    out = []
    hb = model.layers[0].basis_func
    a, b = hb.blend_weights()
    h = x
    for li, layer in enumerate(model.layers):
        u = layer.normalizer(h)
        C = layer.C.view(layer.out_features, layer.in_features, layer.grid_len)
        Bs = bspline_basis(u, layer.grid, layer.denominator)
        Br = rbf(u, layer.grid, layer.denominator)
        ys = torch.einsum("big,oig->bo", a * Bs, C)
        yr = torch.einsum("big,oig->bo", b * Br, C)
        yb = torch.nn.functional.linear(layer.base_act(h), layer.W)
        rms = lambda z: float(z.pow(2).mean().sqrt())
        out.append((li, rms(ys), rms(yr), rms(yb)))
        h = layer(h)
    return a, b, out


for name in ["lv_full", "pendulum_3k", "pendulum_full"]:
    path = IMPL + rf"\results\phase3\hybrid_basis\{name}\best_model.pt"
    model, cfg = load_hybrid(path)
    dcfg = dict(cfg)
    if cfg["dataset"] == "damped_pendulum":
        dcfg.update(t_end=10.0, dt=0.05, t_train_end=5.0)
    data, _ = _rebuild_data(dcfg)
    with torch.no_grad():
        node = NeuralODE(func=model, method="tsit5", substeps=2)
        traj = node(y0=data.y0, t=data.t_full)
        yfull = data.y_full.numpy()
        ntr = len(data.t_train)
        pr = traj.numpy()
        a, b, rows = contributions(model, traj)
    print(f"{name}: alpha={a:.4f} beta={b:.4f}  check extrap R2={compute_r2_score(yfull[ntr:], pr[ntr:]):+.4f}")
    for li, s, r, w in rows:
        share = s / (s + r)
        print(f"   layer {li}: RMS alpha*Spline={s:.4f}  beta*RBF={r:.4f}  residual={w:.4f}  "
              f"spline share of basis output={share:.1%}")
hist = json.load(open(IMPL + r"\results\phase3\hybrid_basis\pendulum_full\training_history.json"))
beta = np.array(hist["beta"])
print("pendulum_full beta at epochs 250/1000/3000/5000/10000:",
      [round(float(beta[i - 1]), 4) for i in (250, 1000, 3000, 5000, 10000)],
      " max", round(float(beta.max()), 4), "at", int(beta.argmax()) + 1)

# ------------------------------------------------------ 3. matched-budget ranks
print("\n=== 3. RBF vs B-spline at matched budget (from training histories) ===")
N, NTR = 141, 36
NEX = N - NTR


def at_budget(run, budget):
    h = json.load(open(IMPL + rf"\results\phase4\epoch_budget_check\{run}\training_history.json"))
    tr = np.array(h["train_losses"][:budget])
    te = np.array(h["test_losses"][:budget])
    # monitor (full-horizon MSE) is exactly evaluated on multiples of 10
    idx = np.arange(9, budget, 10)
    best = idx[np.argmin(tr[idx])]
    ext = (N * te[best] - NTR * tr[best]) / NEX
    return best + 1, tr.min(), tr[best], ext


for run, budget in [("tsit5_rbf_50k", 50000), ("tsit5_rbf_50k", 25000), ("bspline_25k", 25000)]:
    ep, trmin, trb, ext = at_budget(run, budget)
    print(f"{run} @ {budget}: min train={trmin:.3e}  (monitored epoch {ep}: train={trb:.3e}, "
          f"approx extrap MSE={ext:.3e})")
for run in ["tsit5_rbf_50k", "bspline_25k"]:
    m = json.load(open(IMPL + rf"\results\phase4\epoch_budget_check\{run}\metrics.json"))
    print(f"   {run} saved best checkpoint: epoch={m.get('best_epoch')} train={m['best']['train_mse']:.3e} "
          f"extrap={m['best']['extrap_mse']:.3e}")
