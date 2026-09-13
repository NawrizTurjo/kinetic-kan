"""
Kaggle-notebook script: Phase 4 multi-seed extension of Track E, both halves.

E1 (SINDy vs. KAN under noise) needs NEW noise-sweep KAN checkpoints at a new
seed -- run_sindy_noise.py's own run_sweep() only ever RE-INTEGRATES existing
checkpoints under results/benchmarks/noise/ (read-only per Track E's own
scope note), it never retrains. This script does the retraining half; the
SINDy fit itself is fast (seconds, closed-form) and is left for a local
follow-up once the new-seed KAN checkpoints exist (point run_sindy_noise.py's
NOISE_SWEEP_DIR at this script's output and re-run its sweep locally).

E2 (real epidemic fit) needs the 3 headline arms (full_plain, full_vanish,
ts24 -- docs/18_phase4_roadmap.md Part 3's trimmed list, not all 12) re-run
at a new seed. `train_arm()` (run_epidemic_fit.py) has no seed parameter --
it reads module-level `BASE["seed"]` at call time, so this script sets that
dict entry before each call instead of editing the file (in scope: Track E
owns everything under experiments/sindy_epidemic/).

Output goes under results/phase4/sindy_epidemic_seeds/, NOT into
results/phase3/sindy_epidemic/ or results/benchmarks/noise/ -- keeps Phase 4's
new multi-seed artifacts separate from Phase 2/3's original, already-reviewed
results (docs/18 §2.5's convention, applied uniformly across every track).

SKIP-EXISTING: a job whose metrics.json already exists is skipped -- safe to
re-run this same script across multiple Kaggle sessions.

SETUP:
1. Upload the kinetic-kan repo (at least implementation/) as a Kaggle Dataset,
   attach it to this notebook, and set REPO_ROOT below to its mount path.
2. Set SEED to whichever new seed this notebook covers (one seed per
   notebook is the recommended split -- see docs/19_kaggle_execution_guide.md).
3. Run this as a single notebook cell. No internet access needed.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
OUT_ROOT = "/kaggle/working/implementation/results/phase4/sindy_epidemic_seeds"
SEED = 1337  # <-- one seed per notebook is the recommended split (see docstring)

NOISE_LEVELS = [0.00, 0.01, 0.05, 0.10]  # E1: matches Phase 2's own 4 sigma levels
NOISE_EPOCHS = 10000
NOISE_LR = 2e-3

# E2: (tag, time_scale, vanish_dim, epochs) -- the 3 headline arms, per docs/18 Part 3
EPIDEMIC_ARMS = [
    ("full_plain", 24.0, None, 5000),
    ("full_vanish", 24.0, 0, 5000),
    ("ts24", 24.0, None, 2000),
]

MAX_WORKERS = 4  # further capped to os.cpu_count() below -- matches a Kaggle CPU notebook's 4 cores
SKIP_EXISTING = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
sys.path.insert(0, os.path.join(REPO_ROOT, "implementation", "experiments", "sindy_epidemic"))


def _noise_dir(sigma, seed):
    return os.path.join(OUT_ROOT, f"seed{seed}", "noise", f"sigma{sigma}")


def _arm_dir(tag, seed):
    return os.path.join(OUT_ROOT, f"seed{seed}", "epidemic_arms", tag)


def _run_noise_job(sigma, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from train import train_kan_ode

    cell_dir = _noise_dir(sigma, seed)
    t0 = time.time()
    metrics = train_kan_ode(
        model_type="kan", dataset="lotka_volterra", basis_func="rbf", solver="tsit5",
        lr=NOISE_LR, num_epochs=NOISE_EPOCHS, noise_std=sigma, seed=seed,
        save_dir=cell_dir, print_freq=0, device="cpu",
    )
    elapsed = time.time() - t0
    return f"noise_sigma{sigma}", metrics["best"]["train_mse"], elapsed


def _run_arm_job(tag, time_scale, vanish_dim, epochs, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import run_epidemic_fit as ref

    ref.BASE["seed"] = seed  # train_arm() reads this at call time -- see module docstring
    cell_dir = _arm_dir(tag, seed)
    t0 = time.time()
    metrics = ref.train_arm(
        tag, time_scale=time_scale, vanish_dim=vanish_dim, epochs=epochs,
        out_root=os.path.dirname(cell_dir),
    )
    elapsed = time.time() - t0
    return f"arm_{tag}", metrics["best"]["train_mse"], elapsed


def main():
    n_cores = os.cpu_count() or 1
    max_workers = min(MAX_WORKERS, n_cores)
    print(f"Detected {n_cores} CPU core(s); using {max_workers} concurrent worker(s).")

    jobs = []
    skipped = 0
    for sigma in NOISE_LEVELS:
        cell_dir = _noise_dir(sigma, SEED)
        if SKIP_EXISTING and os.path.isfile(os.path.join(cell_dir, "metrics.json")):
            skipped += 1
            continue
        jobs.append(("noise", (sigma, SEED)))
    for (tag, time_scale, vanish_dim, epochs) in EPIDEMIC_ARMS:
        cell_dir = _arm_dir(tag, SEED)
        if SKIP_EXISTING and os.path.isfile(os.path.join(cell_dir, "metrics.json")):
            skipped += 1
            continue
        jobs.append(("arm", (tag, time_scale, vanish_dim, epochs, SEED)))

    print(f"{len(jobs)} job(s) to run, {skipped} already complete (skipped).")
    if not jobs:
        print("Nothing to do.")
        return

    t_start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for (kind, args) in jobs:
            if kind == "noise":
                fut = executor.submit(_run_noise_job, *args)
            else:
                fut = executor.submit(_run_arm_job, *args)
            futures[fut] = args
        for future in as_completed(futures):
            try:
                name, mse, elapsed = future.result()
                completed += 1
                print(f"[{completed}/{len(jobs)}] {name} -> train_mse={mse:.4e} ({elapsed/60:.1f} min)")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] {futures[future]} -> FAILED: {e}")

    total_elapsed = time.time() - t_start
    print(f"\nAll jobs finished in {total_elapsed/60:.1f} min.")

    zip_base = "/kaggle/working/sindy_epidemic_seeds_results"
    shutil.make_archive(zip_base, "zip", OUT_ROOT)
    print(f"Zipped to {zip_base}.zip -- download it from the notebook's Output tab, then extract "
          f"straight into implementation/results/phase4/sindy_epidemic_seeds/ locally.")


if __name__ == "__main__":
    main()
