"""
Kaggle-notebook script: the actual TRAINING loss landscape (weight-space),
Li et al. ("Visualizing the Loss Landscape of Neural Nets") style -- distinct
from plot_landscape.py's phase-space (theta, omega) vector-field terrain,
which is about the pendulum's physics, not about how training went.

WHAT THIS PLOTS: for a trained model (a saved checkpoint.pt), pick the actual
final weights theta* as the center, pick two random "filter-normalized"
directions d1, d2 in weight-space (normalized per Li et al. so no single
layer's scale dominates), and for a grid of (alpha, beta) evaluate the REAL
training loss (MSE against the actual training data, recomputed through the
full NeuralODE forward pass) at theta* + alpha*d1 + beta*d2. The result is a
3D surface: height = training loss, with theta* itself at (alpha=0, beta=0).

WHY IT MATTERS: a wide, gentle bowl around theta* means training found a
robust, stable minimum -- small weight perturbations barely change the loss.
A narrow spike means the opposite -- the found minimum is fragile. This is a
genuinely different question from "does the model's LEARNED FIELD have the
right equilibrium" (plot_landscape.py) -- a model can sit in a wide, stable
loss basin and STILL have learned a spurious phase-space equilibrium (the
mu=2.0 finding in docs/16), because the loss landscape only sees the training
window's trajectory-matching error, not the equilibrium behavior specifically.

COST: this is NOT cheap -- each grid point requires a full forward pass
through the ODE solver over the whole training trajectory, so an NxN grid
costs N^2 forward passes (no backward pass needed, so still much cheaper than
training, but not free). Defaults to a modest 21x21 grid and a handful of
cells, not all 24/72 -- see CELLS below.

SETUP:
1. Upload the kinetic-kan repo (implementation/ and experiments/stiffness_map/)
   as a Kaggle Dataset, attach it, set REPO_ROOT below.
2. Point CHECKPOINT_ROOT at wherever your retrained checkpoints live (e.g. the
   extracted full_retrain_results/probe/ from kaggle_full_retrain.py, uploaded
   as a second Dataset, OR just re-run this after kaggle_full_retrain.py in the
   same session so the files are already on disk at OUT_ROOT).
3. CELLS defaults to ALL 4 solvers x ALL 6 mu (24 cells) -- trim it if that's
   too much compute for one session. At GRID_N=21 (441 forward-pass
   evaluations per cell), this is roughly 10,584 evaluations total. Using
   this project's own measured PER-EPOCH costs (README: euler 0.73s,
   midpoint 0.91s, rk4 1.82s, tsit5 3.02s -- an upper-bound proxy, since a
   forward-only eval skips the backward pass + optimizer step those numbers
   include) as a stand-in per-evaluation cost: roughly 4.8h run sequentially,
   or roughly 2.2h if you run 4 processes in parallel (one per solver, each
   handling its own 6 mu values) -- see kaggle_full_retrain.py /
   kaggle_seed_test.py for the ProcessPoolExecutor pattern to parallelize
   this the same way, if 24 cells run one-at-a-time is too slow for your
   session budget.

OUTPUT: one PNG per cell (loss_landscape_<solver>_mu<mu>.png) plus a
grid.npz per cell (raw alpha/beta/loss arrays, for later re-plotting without
recomputing), zipped to /kaggle/working/loss_landscape_results.zip.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
CHECKPOINT_ROOT = "/kaggle/working/results/phase3/stiffness_map"  # where probe/<solver>_mu<mu>_dt<dt>/checkpoint.pt lives
OUT_DIR = "/kaggle/working/loss_landscape_results"
DT = 0.05
MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
CELLS = [(s, m) for s in SOLVERS for m in MUS]  # all 4 solvers x all 6 mu = 24 cells
MAX_WORKERS = 4      # one process per solver is the natural split (see main())
GRID_N = 21          # NxN grid -- N^2 forward passes per cell, keep modest
SPAN = 1.0           # alpha, beta each range over [-SPAN, SPAN]
SEED_FOR_DIRECTIONS = 0  # RNG seed for the two random directions (reproducible)
SKIP_EXISTING = True     # resumable across Kaggle session timeouts

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "stiffness_map"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from kan import KAN
from ode import NeuralODE
from data import generate_damped_pendulum_data
import run_sweep as rs


def _cell_dir(solver, mu):
    return os.path.join(CHECKPOINT_ROOT, "probe", f"{solver}_mu{mu}_dt{DT}")


def load_model_and_data(solver, mu):
    ckpt_path = os.path.join(_cell_dir(solver, mu), "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = KAN(
        layers_hidden=ckpt["layers_hidden"], grid_len=ckpt["grid_len"], grid_lims=ckpt["grid_lims"],
        basis_func=ckpt["basis_func"], normalizer=ckpt["normalizer"], base_act=ckpt["base_act"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Reconstruct the EXACT training data this checkpoint was trained on --
    # same seed, same mu, same window -- so the loss surface is evaluated
    # against the real training target, not an approximation of it.
    data = generate_damped_pendulum_data(
        mu=mu, t_start=rs.BASELINE["t_start"], t_end=rs.BASELINE["t_end"], dt=DT,
        t_train_end=rs.BASELINE["t_train_end"], seed=rs.BASELINE["seed"],
    )
    node = NeuralODE(func=model, method=solver, substeps=rs.BASELINE["substeps"])
    return model, node, data


def filter_normalized_direction(model, generator):
    """One random direction per parameter tensor, rescaled so each tensor's
    direction has the SAME norm as that tensor's own weights (Li et al.'s
    'filter normalization') -- without this, a direction dominated by one
    layer's scale would make the plot mostly show that one layer's
    sensitivity, not a balanced picture of the whole model."""
    direction = []
    for p in model.parameters():
        d = torch.randn(p.shape, generator=generator)
        d = d * (p.norm() / (d.norm() + 1e-10))
        direction.append(d)
    return direction


def loss_at(model, node, data, base_params, d1, d2, alpha, beta):
    with torch.no_grad():
        for p, base, a1, a2 in zip(model.parameters(), base_params, d1, d2):
            p.copy_(base + alpha * a1 + beta * a2)
        pred_train = node(y0=data.y0, t=data.t_train)
        loss = F.mse_loss(pred_train, data.y_train).item()
    return loss


def _paths_for(solver, mu):
    return (os.path.join(OUT_DIR, f"grid_{solver}_mu{mu}.npz"),
            os.path.join(OUT_DIR, f"loss_landscape_{solver}_mu{mu}.png"))


def plot_loss_landscape(solver, mu):
    """Runs in a WORKER PROCESS -- all heavy imports (torch, matplotlib, KAN,
    NeuralODE, run_sweep) happen at module level so each spawned process gets
    its own clean init, same pattern as kaggle_full_retrain.py /
    kaggle_seed_test.py."""
    import time
    t0 = time.time()
    npz_path, png_path = _paths_for(solver, mu)
    if SKIP_EXISTING and os.path.isfile(npz_path) and os.path.isfile(png_path):
        return solver, mu, "skipped", None, 0.0

    model, node, data = load_model_and_data(solver, mu)
    base_params = [p.detach().clone() for p in model.parameters()]

    gen = torch.Generator().manual_seed(SEED_FOR_DIRECTIONS)
    d1 = filter_normalized_direction(model, gen)
    d2 = filter_normalized_direction(model, gen)

    alphas = np.linspace(-SPAN, SPAN, GRID_N)
    betas = np.linspace(-SPAN, SPAN, GRID_N)
    loss_grid = np.zeros((GRID_N, GRID_N))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            loss_grid[i, j] = loss_at(model, node, data, base_params, d1, d2, a, b)

    # Restore the model to its real trained weights before computing the
    # center point's own loss -- loss_at already leaves alpha=beta=0's value
    # in the grid at its own index, but recomputing explicitly documents it.
    with torch.no_grad():
        for p, base in zip(model.parameters(), base_params):
            p.copy_(base)
    center_loss = loss_at(model, node, data, base_params, d1, d2, 0.0, 0.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(npz_path, alphas=alphas, betas=betas, loss=loss_grid, center_loss=center_loss)

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    log_loss = np.log10(np.clip(loss_grid, 1e-12, None))

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    surf = ax.plot_surface(A, B, log_loss, cmap="viridis", alpha=0.9, linewidth=0.15,
                            edgecolor="#00000022", antialiased=True, zorder=0)
    ax.scatter(0, 0, np.log10(max(center_loss, 1e-12)), color="red", s=90, marker="*",
               edgecolor="black", linewidth=0.8, zorder=10, label="actual trained weights")
    ax.set_xlabel("alpha (direction 1)"); ax.set_ylabel("beta (direction 2)")
    ax.set_zlabel("log10(training MSE)")
    ax.set_title(f"Training loss landscape -- {solver}, mu={mu}\n"
                 f"(wide bowl = stable minimum, spike = fragile)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="log10(training MSE)")

    fig.patch.set_facecolor("white")
    fig.savefig(png_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return solver, mu, "done", center_loss, time.time() - t0


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_cores = os.cpu_count() or 1
    max_workers = min(MAX_WORKERS, n_cores, len(CELLS))
    print(f"Detected {n_cores} CPU core(s); using {max_workers} concurrent worker(s) "
          f"for {len(CELLS)} cell(s).")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(plot_loss_landscape, s, m): (s, m) for s, m in CELLS}
        for future in as_completed(futures):
            solver, mu = futures[future]
            completed += 1
            try:
                solver, mu, status, center_loss, elapsed = future.result()
                if status == "skipped":
                    print(f"[{completed}/{len(CELLS)}] {solver} mu={mu} -> skipped (already done)")
                else:
                    print(f"[{completed}/{len(CELLS)}] {solver} mu={mu} -> done "
                          f"(center_loss={center_loss:.4e}, {elapsed:.1f}s)")
            except Exception as e:
                print(f"[{completed}/{len(CELLS)}] {solver} mu={mu} -> FAILED: {e}")

    import shutil
    zip_base = "/kaggle/working/loss_landscape_results"
    shutil.make_archive(zip_base, "zip", OUT_DIR)
    print(f"\nZipped to {zip_base}.zip -- download from the notebook's Output tab.")


if __name__ == "__main__":
    main()
