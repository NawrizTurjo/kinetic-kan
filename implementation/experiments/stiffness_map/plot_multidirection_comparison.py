"""
Renders the 3 random direction-pair loss-landscape grids for one (solver, mu)
cell side by side, so the seed-to-seed variation in "rise" (docs/16 section 8)
is visible directly, not just as three numbers in a table. Also builds
collages -- one per mu (all 4 solvers stacked) and one full grid (all 6 mu) --
so every cell's 3-seed comparison can be browsed and compared at once, not
just the 3 flagship cells docs/16 singles out in prose.

Reads/writes the mu/solver hierarchy under loss_landscape/ (grid_seed<N>.npz,
landscape.png, direction_comparison.png all live at
loss_landscape/mu<mu>/<solver>/, collage.png at loss_landscape/mu<mu>/) --
NOT flat filenames, which is how this used to be laid out before a full
reorganization made 152 flat files navigable.
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
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]


def _cell_dir(mu, solver):
    return os.path.join(ROOT, f"mu{mu}", solver)


def plot_comparison(solver, mu, out_path, seeds=(0, 1, 2)):
    fig = plt.figure(figsize=(7 * len(seeds), 6))
    for i, seed in enumerate(seeds):
        path = os.path.join(_cell_dir(mu, solver), f"grid_seed{seed}.npz")
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


def make_collage_for_mu(mu, solvers, out_path):
    """Stack all 4 solvers' 3-panel comparison images for one mu into a single
    4-row collage, so every solver's seed-to-seed spread at that mu is
    browsable in one image instead of 4 separate files."""
    paths = [os.path.join(_cell_dir(mu, s), "direction_comparison.png") for s in solvers]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print(f"mu={mu}: skipping collage, missing {missing}")
        return None

    fig, axes = plt.subplots(len(solvers), 1, figsize=(16, 4.2 * len(solvers)))
    if len(solvers) == 1:
        axes = [axes]
    for ax, solver, p in zip(axes, solvers, paths):
        ax.imshow(plt.imread(p))
        ax.axis("off")
        ax.set_title(solver, fontsize=12, fontweight="bold", loc="left")
    fig.suptitle(f"μ = {mu} -- all 4 solvers, 3 direction-pairs each", fontsize=15, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def make_full_grid(collage_paths_by_mu, out_path, mus):
    """Stack every mu's collage into one browsable image -- the complete
    24-cell x 3-seed picture in a single file."""
    valid = [(mu, p) for mu, p in collage_paths_by_mu.items() if p and os.path.isfile(p)]
    if not valid:
        print("no collages to combine into a full grid")
        return
    fig, axes = plt.subplots(len(valid), 1, figsize=(16, 16 * len(valid)))
    if len(valid) == 1:
        axes = [axes]
    for ax, (mu, p) in zip(axes, valid):
        ax.imshow(plt.imread(p))
        ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    collage_paths = {}
    for mu in MUS:
        for solver in SOLVERS:
            out_path = os.path.join(_cell_dir(mu, solver), "direction_comparison.png")
            plot_comparison(solver, mu, out_path)
        collage_path = os.path.join(ROOT, f"mu{mu}", "collage.png")
        collage_paths[mu] = make_collage_for_mu(mu, SOLVERS, collage_path)

    full_grid_path = os.path.join(ROOT, "multidirection_full_grid.png")
    make_full_grid(collage_paths, full_grid_path, MUS)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        solver = sys.argv[1]
        mu = sys.argv[2] if len(sys.argv) > 2 else "2.0"
        out_path = os.path.join(_cell_dir(mu, solver), "direction_comparison.png")
        plot_comparison(solver, mu, out_path)
    else:
        main()
