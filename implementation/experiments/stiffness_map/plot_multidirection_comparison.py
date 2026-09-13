"""
Renders the 3 random direction-pair loss-landscape grids for one (solver, mu)
cell side by side, so the seed-to-seed variation in "rise" (docs/16 section 8)
is visible directly, not just as three numbers in a table.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "results", "phase3", "stiffness_map", "loss_landscape")


def plot_comparison(solver, mu, out_path, seeds=(0, 1, 2)):
    fig = plt.figure(figsize=(7 * len(seeds), 6))
    for i, seed in enumerate(seeds):
        path = os.path.join(ROOT, f"grid_{solver}_mu{mu}_seed{seed}.npz")
        d = np.load(path)
        alphas, betas, loss, center = d["alphas"], d["betas"], d["loss"], float(d["center_loss"])
        log_loss = np.log10(np.clip(loss, 1e-12, None))
        rise = log_loss.max() - np.log10(max(center, 1e-12))

        ax = fig.add_subplot(1, len(seeds), i + 1, projection="3d")
        ax.computed_zorder = False
        A, B = np.meshgrid(alphas, betas, indexing="ij")
        surf = ax.plot_surface(A, B, log_loss, cmap="viridis", alpha=0.9, linewidth=0.15,
                                edgecolor="#00000022", antialiased=True, zorder=0)
        ax.scatter(0, 0, np.log10(max(center, 1e-12)), color="red", s=90, marker="*",
                   edgecolor="black", linewidth=0.8, zorder=10)
        ax.set_xlabel("alpha"); ax.set_ylabel("beta"); ax.set_zlabel("log10(loss)")
        ax.set_title(f"direction-pair seed={seed}\nrise={rise:.2f}", fontsize=11, fontweight="bold")

    fig.suptitle(f"{solver}, mu={mu} -- same trained weights, 3 different random direction-pairs",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    solver = sys.argv[1] if len(sys.argv) > 1 else "euler"
    mu = sys.argv[2] if len(sys.argv) > 2 else "2.0"
    out_path = os.path.join(ROOT, f"multidirection_comparison_{solver}_mu{mu}.png")
    plot_comparison(solver, mu, out_path)
