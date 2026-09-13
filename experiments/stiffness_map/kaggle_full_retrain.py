"""
Kaggle-notebook script: retrain EVERY (solver, mu, seed) cell in Track D's
multi-seed sweep -- 4 solvers x 6 mu x 3 seeds = 72 cells -- uniformly saving
checkpoint.pt + trajectory.npz + full metrics.json for each one.

WHY THIS EXISTS (supersedes kaggle_landscape_retrain.py and kaggle_seed_test.py):
the multi-seed sweep this project already ran (seed_test.py / kaggle_seed_test.py,
seeds 1 and 7 across all 6 mu) only ever saved metrics.json -- no checkpoint or
trajectory -- so verdicts/MSE numbers exist for all 72 cells, but there is no
way to PLOT any of them (trajectory curves, or the vector-field "landscape"
terrain from plot_landscape.py) beyond the couple of cells that happened to get
--save-checkpoints by hand during the mu=2.0 investigation. This script closes
that gap for the WHOLE sweep at once, not just mu=2.0, so a landscape or
trajectory plot can be built for ANY (solver, mu, seed) cell without needing
yet another special-case retrain later.

Per cell, saves:
  - checkpoint.pt   -- the trained model (lets plot_landscape.py evaluate that
                        cell's own learned vector field at any (theta, omega))
  - trajectory.npz  -- t_full, y_true (truth), y_pred (this cell's own
                        predicted path), t_train_end, n_train, mu, solver
  - metrics.json    -- the FULL dict train_cell returns (verdict, best_train_mse,
                        final_train_mse, full_mse, extrap_mse, extrap_r2,
                        epochs_run, aborted_at_epoch, nonfinite_grad_steps,
                        first_nonfinite_epoch, best_epoch, grad_norm_max,
                        seconds) -- everything docs/16's tables or a future
                        report section could need, not just verdict+mse.

LAYOUT (so results drop into the existing local directory structure with no
renaming):
  - seed 42 (the run_sweep.py default) -> OUT_ROOT/probe/<solver>_mu<mu>_dt<dt>/
    (matches probe_traj/'s existing layout; OVERWRITES euler's older
    hand-run mu=2.0 checkpoint there with this run's uniformly-produced one --
    intentional, so every seed-42 cell comes from the same script/settings)
  - seed 1, 7          -> OUT_ROOT/_seed_test/seed<seed>_mu<mu>_<solver>/
    (matches _seed_test/'s existing naming; note this uses the WITH-suffix
    name for every solver including euler, unlike the older seed_test.py's
    legacy no-suffix euler folders -- plot_seed_heatmap.py already checks the
    suffixed name first, so this is additive, not a naming conflict)

SKIP-EXISTING: a cell already having all three files (checkpoint.pt,
trajectory.npz, metrics.json) is skipped -- makes this resumable across
multiple Kaggle sessions if a 9-12 hour session limit is hit partway through.

REALISTIC COST WARNING: 72 cells at NUM_EPOCHS=2000 is a LOT of compute --
based on this project's own local timing, a single tsit5/rk4 cell can take on
the order of an hour, so 72 cells even at 4-wide parallelism could be in the
range of many hours to well over a day of wall-clock time, and may not finish
in one Kaggle session. Reduce NUM_EPOCHS, or trim MUS/SEEDS/SOLVERS below, if
that budget doesn't work -- SKIP_EXISTING makes re-running the unmodified
script later (after trimming, or across sessions) safe either way.

SETUP:
1. Upload the kinetic-kan repo (at least implementation/ and
   experiments/stiffness_map/) as a Kaggle Dataset, attach it to this
   notebook, and set REPO_ROOT below to its mount path.
2. Run this as a single notebook cell. No internet access needed.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
OUT_ROOT = "/kaggle/working/results/phase3/stiffness_map"
MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
SEEDS = [42, 1, 7]
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
DT = 0.05
NUM_EPOCHS = 2000
MAX_WORKERS = 8  # further capped to os.cpu_count() below
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
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "stiffness_map"))


def _fmt_mu(mu):
    return f"{int(mu)}.0" if mu == int(mu) else str(mu)


def _cell_dir(solver, mu, seed):
    mu_str = _fmt_mu(mu)
    if seed == 42:
        return os.path.join(OUT_ROOT, "probe", f"{solver}_mu{mu_str}_dt{DT}")
    return os.path.join(OUT_ROOT, "_seed_test", f"seed{seed}_mu{mu_str}_{solver}")


def _cell_complete(cell_dir):
    return all(os.path.isfile(os.path.join(cell_dir, name))
               for name in ("checkpoint.pt", "trajectory.npz", "metrics.json"))


def _run_one_cell(solver, mu, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import run_sweep as rs

    cell_dir = _cell_dir(solver, mu, seed)
    os.makedirs(cell_dir, exist_ok=True)

    t0 = time.time()
    metrics = rs.train_cell(
        solver=solver, mu=mu, dt=DT, num_epochs=NUM_EPOCHS, seed=seed, print_freq=0,
        save_trajectory_to=os.path.join(cell_dir, "trajectory.npz"),
        save_checkpoint_to=os.path.join(cell_dir, "checkpoint.pt"),
    )
    metrics["verdict"] = rs.classify_cell(metrics)
    elapsed = time.time() - t0

    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return solver, mu, seed, metrics["verdict"], metrics["best_train_mse"], elapsed


def main():
    n_cores = os.cpu_count() or 1
    max_workers = min(MAX_WORKERS, n_cores)
    print(f"Detected {n_cores} CPU core(s); using {max_workers} concurrent worker(s).")

    jobs = []
    skipped = 0
    for solver in SOLVERS:
        for mu in MUS:
            for seed in SEEDS:
                cell_dir = _cell_dir(solver, mu, seed)
                if SKIP_EXISTING and _cell_complete(cell_dir):
                    skipped += 1
                    continue
                jobs.append((solver, mu, seed))

    print(f"{len(jobs)} job(s) to run, {skipped} already complete (skipped). "
          f"At NUM_EPOCHS={NUM_EPOCHS}, this can take many hours -- see the "
          f"REALISTIC COST WARNING in this file's docstring.")
    if not jobs:
        print("Nothing to do.")
        return

    t_start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_cell, s, m, sd): (s, m, sd) for (s, m, sd) in jobs}
        for future in as_completed(futures):
            solver, mu, seed = futures[future]
            try:
                solver, mu, seed, verdict, mse, elapsed = future.result()
                completed += 1
                print(f"[{completed}/{len(jobs)}] solver={solver} mu={mu} seed={seed} "
                      f"-> {verdict} (mse={mse:.4e}, {elapsed/60:.1f} min)")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] solver={solver} mu={mu} seed={seed} -> FAILED: {e}")

    total_elapsed = time.time() - t_start
    print(f"\nAll jobs finished in {total_elapsed/60:.1f} min.")

    zip_base = "/kaggle/working/full_retrain_results"
    shutil.make_archive(zip_base, "zip", OUT_ROOT)
    print(f"Zipped to {zip_base}.zip -- download it from the notebook's Output tab, then extract "
          f"straight into results/phase3/stiffness_map/ locally (it merges into probe/ and "
          f"_seed_test/ alongside what's already there).")


if __name__ == "__main__":
    main()
