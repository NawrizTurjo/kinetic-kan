"""
Kaggle-notebook version of the multi-seed sensitivity check (run_seed_test.ps1's
logic, rewired for a single-process notebook environment instead of spawning
separate PowerShell windows).

SETUP (do this before running this cell):
1. Upload the kinetic-kan repo -- at minimum `implementation/` and
   `experiments/stiffness_map/` -- as a Kaggle Dataset, and attach it to this
   notebook.
2. Set REPO_ROOT below to wherever Kaggle mounts it (typically
   "/kaggle/input/<your-dataset-name>/kinetic-kan").
3. Run this as a single notebook cell (or `%run` it, or paste its contents in).
   It downloads nothing and needs no internet access -- PyTorch is preinstalled
   on Kaggle's standard images.

WHAT IT DOES: runs the same 36 (solver, mu, seed) cells run_seed_test.ps1 would
run locally -- midpoint/rk4/tsit5 x {0.1,0.5,1.0,2.0,5.0,8.0} x seeds {1,7} --
skipping euler (already completed locally) and skipping any cell whose
metrics.json already exists in OUT_ROOT (so re-running this cell after a
Kaggle session timeout resumes rather than restarting).

PARALLELISM: capped at 5 concurrent worker processes (matching the "no more
than 5 at a time" constraint from the local PowerShell version), each pinned
to 1 BLAS thread -- for a model this tiny, multi-threaded BLAS doesn't help an
individual job, and 5 processes x >1 thread each would oversubscribe Kaggle's
CPU allocation (typically 4 cores on a CPU notebook) for no benefit. Actually
effective parallelism is further capped to the notebook's real core count, so
on a 4-core Kaggle CPU instance this runs 4-wide, not 5.

OUTPUT: results/phase3/stiffness_map/_seed_test/seed<seed>_mu<mu>_<solver>/
metrics.json, trajectory.npz (ground-truth + predicted trajectories, for
plot_trajectories.py-style plots) and checkpoint.pt under OUT_ROOT, then zipped
to /kaggle/working/seed_test_results.zip for download via the notebook's Output
tab. Trajectories/checkpoints noticeably grow the zip vs. the metrics-only first
version of this script -- still fine for 36 cells, but if this is ever scaled up
much further, consider dropping save_checkpoint_to (trajectory.npz alone is
enough for plots; checkpoint.pt is only needed for further mechanistic digging
like the mu=2.0 investigation in docs/16).
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
OUT_ROOT = "/kaggle/working/implementation/results/phase3/stiffness_map"
MAX_WORKERS = 5  # further capped to os.cpu_count() below

MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
SEEDS = [1, 7]
SOLVERS = ["tsit5", "rk4", "midpoint"]  # slowest first; euler already done locally
DT = 0.05
NUM_EPOCHS = 2000

# ==============================================================================
# Set thread-limiting env vars BEFORE importing torch -- these only take effect
# if set prior to the BLAS backend's own initialization.
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
sys.path.insert(0, os.path.join(REPO_ROOT, "implementation", "experiments", "stiffness_map"))


def _cell_dir(solver, mu, seed):
    return os.path.join(OUT_ROOT, "_seed_test", f"seed{seed}_mu{mu}_{solver}")


def _run_one_cell(solver, mu, seed):
    """
    Runs in a WORKER PROCESS -- imports happen here, not at module level, so
    each process gets its own clean torch/BLAS initialization honoring the
    thread-limiting env vars set above (they must be in the environment before
    the worker process's own torch import, which multiprocessing's spawn/fork
    handles correctly since env vars are inherited at process creation).
    """
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
                if os.path.isfile(os.path.join(cell_dir, "metrics.json")):
                    skipped += 1
                    continue
                jobs.append((solver, mu, seed))

    print(f"{len(jobs)} job(s) to run, {skipped} already done (skipped).")
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
                      f"-> {verdict} (mse={mse:.4e}, {elapsed:.1f}s)")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] solver={solver} mu={mu} seed={seed} -> FAILED: {e}")

    total_elapsed = time.time() - t_start
    print(f"\nAll jobs finished in {total_elapsed/60:.1f} min.")

    # Zip the results for easy download from Kaggle's Output tab.
    zip_base = "/kaggle/working/seed_test_results"
    shutil.make_archive(zip_base, "zip", OUT_ROOT)
    print(f"Zipped results to {zip_base}.zip -- download it from the notebook's Output tab.")


if __name__ == "__main__":
    main()
