"""
Kaggle notebook 3 of 5 -- Phase 4 epoch-budget sensitivity check (docs/18 SS2).

CONFIG UNDER TEST: Tsit5 solver, B-spline basis, Lotka-Volterra
(`ablation_activations/basis_bspline`) -- Phase 2 called this "statistically
tied" with RBF at 10k epochs. This checks whether that tie holds at 50k, or
one basis pulls ahead. Deliberately the SAME solver (tsit5) as notebook 2's
RBF run, not `kanode_bspline`'s rk4 variant -- keeping the solver fixed is
what makes this an apples-to-apples basis-only comparison.

SINGLE training run -- no thread pinning (see notebook 1's docstring). This
is also the SLOWEST of the five (B-spline is the most expensive basis to
evaluate) -- expect roughly 9 hours for this one, the long pole of the batch.

Every setting below except num_epochs matches the original 10k run's own
recorded config (results/benchmarks/ablation_activations/basis_bspline/metrics.json).

SETUP: same as notebook 1.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
SAVE_DIR = "/kaggle/working/implementation/results/phase4/epoch_budget_check/bspline_50k"
ORIGINAL_10K_METRICS = os.path.join(
    REPO_ROOT, "implementation", "results", "benchmarks",
    "ablation_activations", "basis_bspline", "metrics.json",
)
NUM_EPOCHS = 50000

import sys
import json
import shutil
import time

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
from train import train_kan_ode


def main():
    print(f"Detected {os.cpu_count()} CPU core(s) -- this single run will use all of them.")
    print("This is the slowest of the 5 notebooks (B-spline basis) -- expect ~9 hours.")

    t0 = time.time()
    result = train_kan_ode(
        model_type="kan", dataset="lotka_volterra", basis_func="bspline", solver="tsit5",
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

    zip_base = "/kaggle/working/bspline_50k_result"
    shutil.make_archive(zip_base, "zip", SAVE_DIR)
    print(f"\nZipped to {zip_base}.zip -- download from the notebook's Output tab, then extract "
          f"into implementation/results/phase4/epoch_budget_check/bspline_50k/ locally.")


if __name__ == "__main__":
    main()
