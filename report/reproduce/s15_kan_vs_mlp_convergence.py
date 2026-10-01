"""
KAN-ODE against the paper's parameter-matched tanh MLP-ODE: training-loss decay (Figure "kan-vs-mlp").

Replaces implementation/results/figures/03_kan_vs_mlp_convergence.png, whose tanh
curve is the run at this project's default learning rate and initialization (the
third column of the KAN-vs-MLP table). Here the tanh curve is the paper's literal
network trained with the paper's own learning rate and initialization, 10^4 epochs
(the fourth column). The SiLU MLP is left out of this figure. Both curves are the
per-epoch training MSE, the same series the table's "epochs to 1e-2 / 1e-3 / 1e-4"
row is counted from.

Style comes from implementation/utils/plotting.py, as for the original figure.

Output: report/figures/ablation/03_kan_vs_mlp_convergence.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
IMPL = REPORT.parent / "implementation"
RES = IMPL / "results"
sys.path.insert(0, str(IMPL))
import utils.plotting  # noqa: E402,F401  (applies the project's rcParams)

RUNS = {
    "KAN-ODE (RBF, 240p)": RES / "benchmarks" / "kanode_flagship",
    "MLP-ODE (tanh, paper lr/init, 252p)": RES / "mlp_paperspec_exact_10k_result",
}
COLORS = ["#1f77b4", "#2ca02c"]
STYLES = ["-", "-."]
OUT = REPORT / "figures" / "ablation" / "03_kan_vs_mlp_convergence.png"


def main():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for (name, run), c, ls in zip(RUNS.items(), COLORS, STYLES):
        tr = json.load(open(run / "training_history.json"))["train_losses"]
        ax.semilogy(np.arange(1, len(tr) + 1), tr, label=name, color=c, linestyle=ls, linewidth=2.0)
    ax.axhline(1e-4, color="black", linestyle=":", linewidth=1.8, label=r"Target $\mathrm{MSE} = 10^{-4}$")
    ax.set_title("KAN-ODE vs. Paper-Spec MLP-ODE: Loss Decay", pad=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE, Log Scale)")
    ax.grid(True, which="both")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"  wrote {OUT.relative_to(REPORT.parent).as_posix()}")


if __name__ == "__main__":
    main()
