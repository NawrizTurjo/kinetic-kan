"""
Kaggle-notebook script: robustness check for kaggle_loss_landscape.py's
"rise" measurement (docs/16 section 8) -- computes the SAME quantity from 3
independent random direction-pairs per (solver, mu) cell instead of 1, so the
reported rise is a mean over seeds with a range, not a single arbitrary
slice through weight-space. Directly answers that section's own caveat
("a different pair of directions could show a different rise").

Feasible to do this at all because the model is small: 360 total parameters
(2 KAN layers, [2, 10, 2], grid_len=8 -- confirmed by direct inspection of a
checkpoint), so a 21x21-grid evaluation is cheap per direction-pair (measured
locally: ~55ms/eval for euler, ~431ms/eval for tsit5).

SELF-CONTAINED: computes direction-pair seed=0 fresh here too (not reusing
kaggle_loss_landscape.py's earlier grid_*.npz files), so this needs only the
SAME checkpoint dataset kaggle_loss_landscape.py used -- nothing else to
upload. Because torch.Generator().manual_seed(0) is deterministic, seed=0's
result here is identical to kaggle_loss_landscape.py's original run; seeds 1
and 2 are the two new ones.

SETUP: same as kaggle_loss_landscape.py -- set REPO_ROOT and CHECKPOINT_ROOT
below (CHECKPOINT_ROOT must contain probe/<solver>_mu<mu>_dt<dt>/checkpoint.pt
for every cell, i.e. wherever kaggle_full_retrain.py's output landed).

OUTPUT: rise_multiseed.json ({solver: {mu: {"rises": [...], "mean", "min",
"max"}}}) PLUS the full loss grid for every (solver, mu, seed) combination
(grid_<solver>_mu<mu>_seed<seed>.npz -- same format kaggle_loss_landscape.py
already writes for seed 0), so all 3 direction-pairs' terrains can be
plotted and visually compared side by side later, not just their rise
numbers. 72 grid files total (24 cells x 3 seeds) -- each is small (441
floats), so this adds negligible size to the download. Zipped to
/kaggle/working/multi_direction_results.zip.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
CHECKPOINT_ROOT = "/kaggle/working/implementation/results/phase3/stiffness_map"  # where probe/<solver>_mu<mu>_dt<dt>/checkpoint.pt lives
OUT_DIR = "/kaggle/working/multi_direction_results"
DT = 0.05
MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
DIRECTION_SEEDS = [0, 1, 2]  # seed 0 reproduces kaggle_loss_landscape.py's original rise exactly
GRID_N = 21
SPAN = 1.0
MAX_WORKERS = 4
SKIP_EXISTING = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
sys.path.insert(0, os.path.join(REPO_ROOT, "implementation", "experiments", "stiffness_map"))


def _cell_dir(solver, mu):
    return os.path.join(CHECKPOINT_ROOT, "probe", f"{solver}_mu{mu}_dt{DT}")


def _load(solver, mu):
    from kan import KAN
    from ode import NeuralODE
    from data import generate_damped_pendulum_data
    import run_sweep as rs

    ckpt = torch.load(os.path.join(_cell_dir(solver, mu), "checkpoint.pt"), map_location="cpu", weights_only=False)
    model = KAN(
        layers_hidden=ckpt["layers_hidden"], grid_len=ckpt["grid_len"], grid_lims=ckpt["grid_lims"],
        basis_func=ckpt["basis_func"], normalizer=ckpt["normalizer"], base_act=ckpt["base_act"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    data = generate_damped_pendulum_data(
        mu=mu, t_start=rs.BASELINE["t_start"], t_end=rs.BASELINE["t_end"], dt=DT,
        t_train_end=rs.BASELINE["t_train_end"], seed=rs.BASELINE["seed"],
    )
    node = NeuralODE(func=model, method=solver, substeps=rs.BASELINE["substeps"])
    return model, node, data


def _direction(model, generator):
    direction = []
    for p in model.parameters():
        d = torch.randn(p.shape, generator=generator)
        d = d * (p.norm() / (d.norm() + 1e-10))
        direction.append(d)
    return direction


def _loss_at(model, node, data, base_params, d1, d2, alpha, beta):
    with torch.no_grad():
        for p, base, a1, a2 in zip(model.parameters(), base_params, d1, d2):
            p.copy_(base + alpha * a1 + beta * a2)
        pred = node(y0=data.y0, t=data.t_train)
        return F.mse_loss(pred, data.y_train).item()


def _rise_for_seed(solver, mu, direction_seed):
    """Also saves the full loss grid (not just the scalar rise) -- every point
    on it is already being evaluated to find the max for `rise`, so keeping
    the array costs no extra compute, only a small amount of disk space. This
    lets a later plotting pass render this direction-pair's terrain the same
    way kaggle_loss_landscape.py's original (seed=0) plots do, so all 3
    seeds' terrains can be compared side by side, not just their rise numbers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    grid_path = os.path.join(OUT_DIR, f"grid_{solver}_mu{mu}_seed{direction_seed}.npz")
    if SKIP_EXISTING and os.path.isfile(grid_path):
        d = np.load(grid_path)
        loss, center = d["loss"], float(d["center_loss"])
        rise = float(np.log10(max(loss.max(), center, 1e-12)) - np.log10(max(center, 1e-12)))
        return solver, mu, direction_seed, rise, 0.0

    t0 = time.time()
    model, node, data = _load(solver, mu)
    base_params = [p.detach().clone() for p in model.parameters()]
    gen = torch.Generator().manual_seed(direction_seed)
    d1 = _direction(model, gen)
    d2 = _direction(model, gen)

    center = _loss_at(model, node, data, base_params, d1, d2, 0.0, 0.0)
    alphas = np.linspace(-SPAN, SPAN, GRID_N)
    betas = np.linspace(-SPAN, SPAN, GRID_N)
    loss_grid = np.zeros((GRID_N, GRID_N))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            loss_grid[i, j] = _loss_at(model, node, data, base_params, d1, d2, a, b)

    max_loss = max(loss_grid.max(), center)
    rise = float(np.log10(max(max_loss, 1e-12)) - np.log10(max(center, 1e-12)))

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(grid_path, alphas=alphas, betas=betas, loss=loss_grid, center_loss=center)

    return solver, mu, direction_seed, rise, time.time() - t0


def main():
    n_cores = os.cpu_count() or 1
    max_workers = min(MAX_WORKERS, n_cores)
    jobs = [(s, m, seed) for s in SOLVERS for m in MUS for seed in DIRECTION_SEEDS]
    print(f"Detected {n_cores} CPU core(s); using {max_workers} concurrent worker(s) for {len(jobs)} job(s).")

    results = {}
    completed = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_rise_for_seed, s, m, sd): (s, m, sd) for s, m, sd in jobs}
        for future in as_completed(futures):
            solver, mu, direction_seed = futures[future]
            completed += 1
            try:
                solver, mu, direction_seed, rise, elapsed = future.result()
                results.setdefault(solver, {}).setdefault(mu, {})[direction_seed] = rise
                print(f"[{completed}/{len(jobs)}] {solver} mu={mu} seed={direction_seed} "
                      f"-> rise={rise:.2f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"[{completed}/{len(jobs)}] {solver} mu={mu} seed={direction_seed} -> FAILED: {e}")

    print(f"\nAll done in {(time.time()-t_start)/60:.1f} min.")

    out = {}
    for solver in SOLVERS:
        out[solver] = {}
        for mu in MUS:
            rises = [results[solver][mu][seed] for seed in DIRECTION_SEEDS if seed in results.get(solver, {}).get(mu, {})]
            out[solver][str(mu)] = {"rises": rises, "mean": float(np.mean(rises)),
                                     "min": float(np.min(rises)), "max": float(np.max(rises))}

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rise_multiseed.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")

    import shutil
    zip_base = "/kaggle/working/multi_direction_results"
    shutil.make_archive(zip_base, "zip", OUT_DIR)
    print(f"Zipped to {zip_base}.zip -- download from the notebook's Output tab.")


if __name__ == "__main__":
    main()
