"""
Shared helpers for every reproduction script in this folder.

  * paths (repository root, implementation/, results/, output folder)
  * the registry of trained runs the report analyses (RUNS)
  * loaders for a run's metrics.json, training_history.json and best_model.pt
  * the Lotka-Volterra ground truth (vector field, Jacobian, first integral, data)
  * rollouts of a saved checkpoint, cached per process
  * the matplotlib style of the analysis figures
  * numbers.json: every derived number quoted in the report text

Nothing here trains a model. Every quantity is either read from a run's own
files or recomputed from its saved checkpoint.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # report/reproduce
REPORT = HERE.parent                            # report
ROOT = REPORT.parent                            # repository root
IMPL = ROOT / "implementation"
RES = IMPL / "results"
OUT = REPORT / "figures" / "analysis"           # PDFs, numbers.json, tables.md
OUT.mkdir(parents=True, exist_ok=True)
NUMBERS = OUT / "numbers.json"
sys.path.insert(0, str(IMPL))

from kan import KAN, MLP_ODE  # noqa: E402
from ode import NeuralODE  # noqa: E402
from data.lotka_volterra import generate_lotka_volterra_data  # noqa: E402

torch.set_grad_enabled(False)

# ------------------------------------------------------------------------------
# Runs analysed in the report (all produced by implementation/train.py)
# ------------------------------------------------------------------------------
B = RES / "benchmarks"
P4 = RES / "phase4" / "epoch_budget_check"
SOLVERS = ["euler", "heun", "midpoint", "rk4", "dopri5", "tsit5"]
SOLVER_LABEL = dict(euler="Euler", heun="Heun", midpoint="Midpoint", rk4="RK4",
                    dopri5="DOPRI5", tsit5="Tsit5")
BASES = ["bspline", "rbf", "chebyshev", "lagrange", "iqf", "rswaf", "newton"]
BASIS_LABEL = dict(bspline="B-spline", rbf="Gaussian RBF", chebyshev="Chebyshev",
                   lagrange="Lagrange", iqf="IQF", rswaf="RSWAF", newton="Newton")
RUNS = {f"solver_{s}": B / "ablation_solvers" / f"solver_{s}" for s in SOLVERS}
RUNS.update({f"basis_{b}": B / "ablation_activations" / f"basis_{b}" for b in BASES})
RUNS.update({
    "mlp_silu": B / "mlpode_baseline_silu", "mlp_tanh": B / "mlpode_baseline",
    "kan_50k": P4 / "tsit5_rbf_50k", "euler_50k": P4 / "euler_50k",
    "bspline_25k": P4 / "bspline_25k", "mlp_silu_50k": P4 / "mlp_silu_50k",
    "mlp_tanh_50k": P4 / "mlp_paperspec_50k",
    # paper's own learning rate (1e-2) and near-zero init scale (Glorot / 1e5)
    # for the identical [2,50,2]+tanh architecture; see mlp_tanh / mlp_tanh_50k
    # above for the same architecture under this project's DEFAULT lr/init.
    "mlp_tanh_exact": RES / "mlp_paperspec_exact_10k_result",
    "mlp_tanh_exact_50k": RES / "mlp_paperspec_exact_50k_result",
})
for s in ["0", "0.01", "0.05", "0.1"]:
    RUNS[f"noise_{s}"] = B / "noise" / f"sigma{s}"
for d in ["0.05", "0.1", "0.2"]:
    RUNS[f"dt_{d}"] = B / "stepsize" / f"dt{d}"


def metrics(key):
    return json.loads((RUNS[key] / "metrics.json").read_text())


@lru_cache(maxsize=None)
def history(key):
    h = json.loads((RUNS[key] / "training_history.json").read_text())
    return {k: np.asarray(v, dtype=float) for k, v in h.items() if isinstance(v, list)}


@lru_cache(maxsize=None)
def load_model(key, double=False):
    ck = torch.load(RUNS[key] / "best_model.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    if c["model_type"] == "mlp":
        m = MLP_ODE(layers_hidden=c["layers_hidden"], activation=c.get("mlp_act", "tanh"))
    else:
        m = KAN(layers_hidden=c["layers_hidden"], grid_len=c["grid_len"],
                grid_lims=tuple(c.get("grid_lims", [-1.0, 1.0])), basis_func=c["basis_func"],
                normalizer=c["normalizer"], base_act=c["base_act"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    if double:
        m = m.double()
    return m, c


def learned_jacobian(model, u):
    J = torch.autograd.functional.jacobian(lambda v: model(v[None])[0], torch.tensor(u))
    return J.numpy()


# ------------------------------------------------------------------------------
# Lotka-Volterra ground truth (same parameters and data as training)
# ------------------------------------------------------------------------------
LV = dict(alpha=1.5, beta=1.0, gamma=3.0, delta=1.0)
EQ_TRUE = np.array([LV["gamma"] / LV["delta"], LV["alpha"] / LV["beta"]])
DATA28 = generate_lotka_volterra_data(t_end=28.0, dt=0.1, seed=42, dtype=torch.float64)
T28 = DATA28.t_full.numpy()
Y28 = DATA28.y_full.numpy()
N_TRAIN, N_14 = 36, 141   # samples in [0, 3.5] and in [0, 14]


def f_true_np(u):
    x, y = u[..., 0], u[..., 1]
    return np.stack([LV["alpha"] * x - LV["beta"] * x * y, LV["delta"] * x * y - LV["gamma"] * y], -1)


def f_true_torch(t, u):
    x, y = u[..., 0], u[..., 1]
    return torch.stack([LV["alpha"] * x - LV["beta"] * x * y, LV["delta"] * x * y - LV["gamma"] * y], -1)


def jac_true(u):
    x, y = u
    return np.array([[LV["alpha"] - LV["beta"] * y, -LV["beta"] * x],
                     [LV["delta"] * y, LV["delta"] * x - LV["gamma"]]])


def first_integral(u):
    """Lotka-Volterra conserved quantity H(x, y) = delta x - gamma ln x + beta y - alpha ln y."""
    x, y = u[..., 0], u[..., 1]
    return LV["delta"] * x - LV["gamma"] * np.log(x) + LV["beta"] * y - LV["alpha"] * np.log(y)


@lru_cache(maxsize=None)
def rollout(key, solver=None, t_end=28.0):
    """Integrate a run's checkpoint from u(0) on the 0.1 grid (training solver unless overridden)."""
    m, c = load_model(key, double=True)
    node = NeuralODE(m, method=solver or c["solver"], substeps=c["substeps"])
    n = int(round(t_end / 0.1)) + 1
    return node(DATA28.y0, DATA28.t_full[:n]).numpy()


# ------------------------------------------------------------------------------
# numbers.json (merged, so each script can be run on its own)
# ------------------------------------------------------------------------------
def load_numbers():
    return json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}


def save_numbers(new):
    num = load_numbers()
    num.update(json.loads(json.dumps(new, default=float)))
    NUMBERS.write_text(json.dumps(num, indent=1, default=float))
    print(f"  updated {NUMBERS.relative_to(ROOT)}: {', '.join(new)}")


# ------------------------------------------------------------------------------
# Figure style: report palette (validated for colour-blind separation)
# ------------------------------------------------------------------------------
C = dict(crimson="#9B1A20", blue="#1E5AA8", amber="#D0901C", teal="#0E9484",
         purple="#7A4FA8", ink="#1C2026", gray="#8A939E", rule="#D1D5DB",
         soft="#F7E4E5")
SOLVER_COLOR = dict(euler=C["crimson"], heun=C["amber"], midpoint=C["blue"], rk4=C["teal"],
                    dopri5=C["gray"], tsit5=C["purple"])
SEQ = LinearSegmentedColormap.from_list("crimson_seq", ["#FFFFFF", "#F7E4E5", "#E08A8E", "#9B1A20", "#4A0A0E"])
TW = 6.69  # \textwidth in inches (A4, 2 cm margins)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": C["ink"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["rule"], "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "lines.linewidth": 1.3, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {(OUT / name).relative_to(ROOT)}.pdf")


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom", color=C["ink"])
