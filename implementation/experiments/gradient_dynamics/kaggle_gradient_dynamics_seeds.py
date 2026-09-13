"""
Kaggle-notebook script: Phase 4 multi-seed extension of Phase 2's 6-solver
Lotka-Volterra ablation (the data Track B's own analyze_gradient_dynamics.py
reads). Track B's own scope note says "read-only access to
results/benchmarks/ablation_solvers/" -- so this deliberately does NOT write
into that folder. New seeds land under results/phase4/ablation_solvers_seeds/
instead, a brand-new tree kept separate from every Phase 2/3 track's own
protected results folder (docs/18_phase4_roadmap.md §2.5).

WHAT IT DOES: retrains all 6 solvers (euler, heun, midpoint, rk4, dopri5,
tsit5) on Lotka-Volterra, RBF basis, 10,000 epochs -- the exact Phase 2 recipe
confirmed against results/benchmarks/ablation_solvers/solver_tsit5/metrics.json's
own config block -- at whichever SEEDS are listed below, via `train_kan_ode`
(implementation/train.py) called directly, not through a subprocess.

SKIP-EXISTING: a (solver, seed) cell already having metrics.json is skipped --
safe to re-run this same script across multiple Kaggle sessions.

SETUP:
1. Upload the kinetic-kan repo (at least implementation/) as a Kaggle Dataset,
   attach it to this notebook, and set REPO_ROOT below to its mount path.
2. Set SEEDS to whichever new seeds this notebook should cover (see the
   Kaggle execution guide, docs/19_kaggle_execution_guide.md, for how the team
   splits seeds across notebooks -- typically ONE seed per notebook so 2
   notebooks running at once cover both of Phase 4's extra seeds).
3. Run this as a single notebook cell. No internet access needed.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
OUT_ROOT = "/kaggle/working/implementation/results/phase4/ablation_solvers_seeds"
SOLVERS = ["euler", "heun", "midpoint", "rk4", "dopri5", "tsit5"]
SEEDS = [1337]  # <-- one seed per notebook is the recommended split (see docstring)
NUM_EPOCHS = 10000
LR = 2e-3
MAX_WORKERS = 4  # further capped to os.cpu_count() below -- matches a Kaggle CPU notebook's 4 cores
SKIP_EXISTING = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))


def _cell_dir(solver, seed):
    return os.path.join(OUT_ROOT, f"seed{seed}", f"solver_{solver}")


def _cell_complete(cell_dir):
    return os.path.isfile(os.path.join(cell_dir, "metrics.json"))


def _run_one_cell(solver, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from train import train_kan_ode  # imported inside the worker, same reasoning as Track D's kaggle scripts

    cell_dir = _cell_dir(solver, seed)
    t0 = time.time()
    metrics = train_kan_ode(
        model_type="kan", dataset="lotka_volterra", basis_func="rbf",
        solver=solver, lr=LR, num_epochs=NUM_EPOCHS, seed=seed,
        save_dir=cell_dir, print_freq=0, device="cpu",
    )
    elapsed = time.time() - t0
    best = metrics.get("best", metrics)
    return solver, seed, best.get("train_mse"), elapsed


def main():
    n_cores = os.cpu_count() or 1
    max_workers = min(MAX_WORKERS, n_cores)
    print(f"Detected {n_cores} CPU core(s); using {max_workers} concurrent worker(s).")

    jobs = []
    skipped = 0
    for seed in SEEDS:
        for solver in SOLVERS:
            cell_dir = _cell_dir(solver, seed)
            if SKIP_EXISTING and _cell_complete(cell_dir):
                skipped += 1
                continue
            jobs.append((solver, seed))

    print(f"{len(jobs)} job(s) to run, {skipped} already complete (skipped).")
    if not jobs:
        print("Nothing to do.")
        return

    t_start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_cell, s, sd): (s, sd) for (s, sd) in jobs}
        for future in as_completed(futures):
            solver, seed = futures[future]
            try:
                solver, seed, mse, elapsed = future.result()
                completed += 1
                print(f"[{completed}/{len(jobs)}] solver={solver} seed={seed} "
                      f"-> train_mse={mse:.4e} ({elapsed/60:.1f} min)")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] solver={solver} seed={seed} -> FAILED: {e}")

    total_elapsed = time.time() - t_start
    print(f"\nAll jobs finished in {total_elapsed/60:.1f} min.")

    zip_base = "/kaggle/working/ablation_solvers_seeds_results"
    shutil.make_archive(zip_base, "zip", OUT_ROOT)
    print(f"Zipped to {zip_base}.zip -- download it from the notebook's Output tab, then extract "
          f"straight into implementation/results/phase4/ablation_solvers_seeds/ locally.")


if __name__ == "__main__":
    main()
