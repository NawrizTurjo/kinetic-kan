"""
Kaggle-notebook version of run_sweep.py's `--stage sanity` check -- run as its
own separate Kaggle job/notebook so it doesn't compete with kaggle_full_retrain.py
for the same session's compute.

WHAT IT DOES: trains exactly one cell -- (tsit5, mu=0.5, dt=0.05) at the real
FULL_EPOCHS (10000) budget -- and compares its best_train_mse / full_mse /
extrap_r2 against the known pendulum_control_win5 reference values. This is
Track D's Definition-of-Done built-in correctness check (docs/12_phase3_roadmap.md):
confirms the sweep script's own training loop reproduces a result this project
already trusts, independent of anything mu-sweep-specific.

Reference values (results/_fixed/pendulum_control_win5/metrics.json):
  best_train_mse = 9.2797e-05 | full_mse = 4.4792e-02 | extrap_r2 = 0.6472

SETUP:
1. Upload the kinetic-kan repo (at least implementation/ and
   experiments/stiffness_map/) as a Kaggle Dataset, attach it, and set
   REPO_ROOT below to its mount path.
2. Run this as a single notebook cell.

OUTPUT: prints the comparison directly, and saves metrics.json to
OUT_ROOT/sanity/tsit5_mu0.5_dt0.05/metrics.json, zipped to
/kaggle/working/sanity_check_result.zip for download.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
OUT_ROOT = "/kaggle/working/results/phase3/stiffness_map"
NUM_EPOCHS = 10000  # FULL_EPOCHS -- the real budget this check requires, not a shortcut

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import shutil

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "stiffness_map"))

import run_sweep as rs

REFERENCE = {"best_train_mse": 9.2797e-05, "full_mse": 4.4792e-02, "extrap_r2": 0.6472}


def main():
    print("Sanity check: (tsit5, mu=0.5, dt=0.05) should reproduce pendulum_control_win5.")
    print(f"Reference: best_train_mse={REFERENCE['best_train_mse']:.4e} | "
          f"full_mse={REFERENCE['full_mse']:.4e} | extrap_r2={REFERENCE['extrap_r2']:.4f}")

    metrics = rs.train_cell(solver="tsit5", mu=0.5, dt=0.05, num_epochs=NUM_EPOCHS, print_freq=200)

    print("\nThis run:")
    print(f"  best_train_mse={metrics['best_train_mse']:.4e} | "
          f"full_mse={metrics['full_mse']:.4e} | extrap_r2={metrics['extrap_r2']:.4f}")

    cell_dir = os.path.join(OUT_ROOT, "sanity", "tsit5_mu0.5_dt0.05")
    os.makedirs(cell_dir, exist_ok=True)
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    zip_base = "/kaggle/working/sanity_check_result"
    shutil.make_archive(zip_base, "zip", OUT_ROOT)
    print(f"\nZipped to {zip_base}.zip -- download from the notebook's Output tab, then extract "
          f"straight into results/phase3/stiffness_map/ locally.")


if __name__ == "__main__":
    main()
