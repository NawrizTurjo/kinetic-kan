"""
Kaggle notebook 5 of 5 -- Phase 4 epoch-budget sensitivity check (docs/18 SS2).

CONFIG UNDER TEST: the ORIGINAL PAPER'S literal MLP architecture, [2,50,2] +
tanh (`mlpode_baseline`) -- this is the single most important config in the
whole batch for docs/18 SS3 (the baseline-vs-paper audit), because it
currently FAILS TO TRAIN AT ALL at 10k epochs (best train MSE 1.08 -- worse
than predicting the dataset's own mean). This run answers directly: does it
eventually converge given 5x the budget (an epoch-budget story), or does it
stay stuck (an architecture/implementation-mismatch story)?

SINGLE training run -- no thread pinning (see notebook 1's docstring).

Every setting below except num_epochs matches the original 10k run's own
recorded config (results/benchmarks/mlpode_baseline/metrics.json) --
deliberately including the ORIGINAL failing architecture, not the fixed one
(that's notebook 4).

SETUP: same as notebook 1.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
SAVE_DIR = "/kaggle/working/implementation/results/phase4/epoch_budget_check/mlp_paperspec_50k"
ORIGINAL_10K_METRICS = os.path.join(
    REPO_ROOT, "implementation", "results", "benchmarks", "mlpode_baseline", "metrics.json",
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
    print("Testing the ORIGINAL PAPER-SPEC MLP ([2,50,2]+tanh), which currently fails "
          "to train at 10k epochs -- watch for whether train_mse actually descends.")

    t0 = time.time()
    result = train_kan_ode(
        model_type="mlp", dataset="lotka_volterra",
        mlp_layers=[2, 50, 2], mlp_act="tanh",
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
        print(f"\nOriginal 10k-epoch run, for comparison (this is the one that failed to train):")
        print(f"  train_mse={orig['train_mse']:.4e}  extrap_mse={orig['extrap_mse']:.4e}  "
              f"extrap_r2={orig['extrap_r2']:.4f}")
        if best["train_mse"] < 0.01:
            print("\n  -> CONVERGED at 50k where it failed at 10k: this looks like an "
                  "epoch-budget story for this architecture, not an implementation bug.")
        else:
            print("\n  -> STILL NOT CONVERGED at 50k (5x the budget): this points toward "
                  "an architecture/implementation mismatch vs. the paper, not just a slow "
                  "training story -- flag for docs/18 SS3's audit.")
    else:
        print(f"\n(Could not find original 10k metrics at {ORIGINAL_10K_METRICS} to compare "
              f"against -- compare manually once downloaded.)")

    zip_base = "/kaggle/working/mlp_paperspec_50k_result"
    shutil.make_archive(zip_base, "zip", SAVE_DIR)
    print(f"\nZipped to {zip_base}.zip -- download from the notebook's Output tab, then extract "
          f"into implementation/results/phase4/epoch_budget_check/mlp_paperspec_50k/ locally.")


if __name__ == "__main__":
    main()
