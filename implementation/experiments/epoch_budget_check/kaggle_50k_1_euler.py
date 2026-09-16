"""
Kaggle notebook 1 of 5 -- Phase 4 epoch-budget sensitivity check (docs/18 SS2).

CONFIG UNDER TEST: Euler solver, RBF basis, Lotka-Volterra -- the one solver
that does NOT reach 1e-4 training loss within the original 10,000-epoch
budget (results/benchmarks/ablation_solvers/solver_euler). Retrained here at
50,000 epochs (5x) to see whether it's still climbing, i.e. whether Table 1's
"Euler is worst" ranking is an artifact of stopping too early.

This is a SINGLE training run -- unlike the earlier seed-extension scripts,
there is only one job in this notebook, so we deliberately do NOT pin
OMP_NUM_THREADS to 1. We want this one run to use all 4 of the notebook's
cores via PyTorch's own multi-threaded BLAS, not restrict itself to 1 thread
the way a 4-way-parallel multi-job script would need to.

Every setting below except num_epochs is identical to the original 10k run's
own recorded config (results/benchmarks/ablation_solvers/solver_euler/metrics.json)
-- only the epoch budget changes, so the comparison is apples-to-apples.

SETUP:
1. Upload the kinetic-kan repo (at least implementation/) as a Kaggle Dataset,
   attach it to this notebook, and set REPO_ROOT below to its mount path.
2. Run this as a single notebook cell, then "Save & Run All (Commit)" for the
   real 50k-epoch run. No internet access needed.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
SAVE_DIR = "/kaggle/working/implementation/results/phase4/epoch_budget_check/euler_50k"
ORIGINAL_10K_METRICS = os.path.join(
    REPO_ROOT, "implementation", "results", "benchmarks",
    "ablation_solvers", "solver_euler", "metrics.json",
)
NUM_EPOCHS = 50000

import sys
import json
import shutil
import time

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
from train import train_kan_ode


def main():
    print(f"Detected {os.cpu_count()} CPU core(s) -- this single run will use all of them "
          f"via PyTorch's own multi-threaded BLAS (no thread pinning here, unlike a "
          f"multi-job-per-notebook script).")

    t0 = time.time()
    result = train_kan_ode(
        model_type="kan", dataset="lotka_volterra", basis_func="rbf", solver="euler",
        lr=2e-3, num_epochs=NUM_EPOCHS, seed=42,
        save_dir=SAVE_DIR, print_freq=500, device="cpu",
    )
    elapsed = time.time() - t0

    best = result["best"]
    print(f"\n50k-epoch run finished in {elapsed/60:.1f} min.")
    print(f"  train_mse={best['train_mse']:.4e}  extrap_mse={best['extrap_mse']:.4e}  "
          f"extrap_r2={best['extrap_r2']:.4f}")

    if os.path.isfile(ORIGINAL_10K_METRICS):
        orig = json.load(open(ORIGINAL_10K_METRICS))["best"]
        print(f"\nOriginal 10k-epoch run, for comparison:")
        print(f"  train_mse={orig['train_mse']:.4e}  extrap_mse={orig['extrap_mse']:.4e}  "
              f"extrap_r2={orig['extrap_r2']:.4f}")
        ratio = orig["train_mse"] / best["train_mse"] if best["train_mse"] > 0 else float("inf")
        print(f"\n50k vs. 10k train_mse improvement: {ratio:.2f}x")
    else:
        print(f"\n(Could not find original 10k metrics at {ORIGINAL_10K_METRICS} to compare "
              f"against -- compare manually once downloaded.)")

    zip_base = "/kaggle/working/euler_50k_result"
    shutil.make_archive(zip_base, "zip", SAVE_DIR)
    print(f"\nZipped to {zip_base}.zip -- download from the notebook's Output tab, then extract "
          f"into implementation/results/phase4/epoch_budget_check/euler_50k/ locally.")


if __name__ == "__main__":
    main()
