"""
Kaggle-notebook script: Phase 4 multi-seed extension of Track C (learnable
hybrid basis). Runs a small list of (dataset, seed, epoch-budget) jobs via
`run_hybrid.run()` (implementation/experiments/hybrid_basis/run_hybrid.py)
called directly, not through a subprocess.

Output goes under results/phase4/hybrid_basis_seeds/, NOT into
results/phase3/hybrid_basis/ -- keeps Phase 4's new multi-seed artifacts
separate from Phase 3's original, already-reviewed N=1 results (docs/18
§2.5's convention, applied uniformly across every track's seed extension).

THE THREE JOBS docs/18_phase4_roadmap.md Part 3 recommends for Track C's N=3
pass (edit JOBS below to add/remove):
  1. Lotka-Volterra, seed A, full 10,000 epochs        (~2.4 h)
  2. Lotka-Volterra, seed B, full 10,000 epochs        (~2.4 h)
  3. Damped pendulum, ONE new seed, 5,000 epochs        (~3.5 h)
     -- enough to see whether the epoch-3,000 overfitting onset
        (docs/15 §4) recurs at a different seed; NOT the full 10k budget.

HOW MANY NOTEBOOKS: these 3 jobs can all run in ONE notebook (set
MAX_WORKERS=3; wall-clock is bounded by the slowest job, ~3.5h, using only
1 of the team's 5 concurrent Kaggle slots) or be split across up to 3
notebooks (trim JOBS to one entry per notebook) if the team would rather
free up cores for other tracks running at the same time. Either is fine --
see docs/19_kaggle_execution_guide.md for the full walkthrough.

SKIP-EXISTING: a job whose metrics.json already exists is skipped -- safe to
re-run this same script across multiple Kaggle sessions.

SETUP:
1. Upload the kinetic-kan repo (at least implementation/) as a Kaggle Dataset,
   attach it to this notebook, and set REPO_ROOT below to its mount path.
2. Edit JOBS to whichever of the 3 jobs above this notebook should cover.
3. Run this as a single notebook cell. No internet access needed.
"""
import os

# ==============================================================================
# CONFIGURE THIS before running
# ==============================================================================
REPO_ROOT = "/kaggle/input/kinetic-kan/kinetic-kan"  # <-- adjust to your dataset's mount path
OUT_ROOT = "/kaggle/working/implementation/results/phase4/hybrid_basis_seeds"

# Each job: (dataset, seed, epochs, blend_lr_mult, tag)
# blend_lr_mult=1.0 for lotka_volterra (no gate-speed fix needed there -- docs/15 §2);
# blend_lr_mult=15.0 for damped_pendulum (the validated fix -- docs/15 §2/§4).
JOBS = [
    ("lotka_volterra", 1337, 10000, 1.0, "lv_seedA"),
    ("lotka_volterra", 2024, 10000, 1.0, "lv_seedB"),
    ("damped_pendulum", 1337, 5000, 15.0, "pendulum_seed1337_5k"),
]
MAX_WORKERS = 3  # further capped to os.cpu_count() below -- see docstring for the tradeoff
SKIP_EXISTING = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(REPO_ROOT, "implementation"))
sys.path.insert(0, os.path.join(REPO_ROOT, "implementation", "experiments", "hybrid_basis"))


def _cell_dir(tag):
    return os.path.join(OUT_ROOT, tag)


def _cell_complete(cell_dir):
    return os.path.isfile(os.path.join(cell_dir, "metrics.json"))


def _run_one_job(dataset, seed, epochs, blend_lr_mult, tag):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import run_hybrid  # imported inside the worker, same reasoning as Track D's kaggle scripts

    cell_dir = _cell_dir(tag)
    t0 = time.time()
    metrics = run_hybrid.run(
        dataset=dataset, epochs=epochs, save_dir=cell_dir, seed=seed,
        log_every=max(500, epochs // 10), blend_lr_mult=blend_lr_mult,
    )
    elapsed = time.time() - t0
    return tag, metrics["best"]["train_mse"], elapsed


def main():
    n_cores = os.cpu_count() or 1
    max_workers = min(MAX_WORKERS, n_cores)
    print(f"Detected {n_cores} CPU core(s); using {max_workers} concurrent worker(s).")

    jobs = []
    skipped = 0
    for (dataset, seed, epochs, blend_lr_mult, tag) in JOBS:
        cell_dir = _cell_dir(tag)
        if SKIP_EXISTING and _cell_complete(cell_dir):
            skipped += 1
            continue
        jobs.append((dataset, seed, epochs, blend_lr_mult, tag))

    print(f"{len(jobs)} job(s) to run, {skipped} already complete (skipped).")
    if not jobs:
        print("Nothing to do.")
        return

    t_start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_job, *j): j[-1] for j in jobs}
        for future in as_completed(futures):
            tag = futures[future]
            try:
                tag, mse, elapsed = future.result()
                completed += 1
                print(f"[{completed}/{len(jobs)}] {tag} -> train_mse={mse:.4e} ({elapsed/60:.1f} min)")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] {tag} -> FAILED: {e}")

    total_elapsed = time.time() - t_start
    print(f"\nAll jobs finished in {total_elapsed/60:.1f} min.")

    zip_base = "/kaggle/working/hybrid_basis_seeds_results"
    shutil.make_archive(zip_base, "zip", OUT_ROOT)
    print(f"Zipped to {zip_base}.zip -- download it from the notebook's Output tab, then extract "
          f"straight into implementation/results/phase4/hybrid_basis_seeds/ locally.")


if __name__ == "__main__":
    main()
