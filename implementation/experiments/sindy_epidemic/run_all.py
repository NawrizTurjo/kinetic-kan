"""
Track E entry point: reproduce every number and figure in
`docs/17_p3_sindy_epidemic_findings.md` from scratch.

    python run_all.py                 # everything (E1 + E2)
    python run_all.py --skip-e2       # E1 only, ~1 min
    python run_all.py --probe-epochs 30 --full-epochs 40   # fast smoke run

What E2 runs, and why each arm exists
-------------------------------------
  Stage 1  time-scale sweep   s in {1, 10, 20, 24, 30, 40}, 2,000 ep
           s=1 is the no-fix baseline; the rest test whether docs/10's
           rescaling transfers. s=24 is the value the init-time diagnostic
           predicts a priori (it puts the rescaled horizon at 5.0).

  Stage 2  structural controls at s=24, 2,000 ep
           `proj`   -- conserve_mode projection, predicted inapplicable
           `vanish` -- vanish_dim 0, also predicted inapplicable (WRONGLY --
                       see the findings doc; this is exactly why the arms are
                       run rather than the predictions simply asserted)

  Stage 3  split controls at s=24, 2,000 ep, train_days in {60, 70}
           The dataset's default split is day 45, which is EXACTLY the
           outbreak peak, so the training window contains 0% of the 75-day
           decay. These arms separate "the method cannot extrapolate" from
           "the window held nothing to extrapolate from".

  Stage 4  full-length runs at s=24, 5,000 ep: `full_vanish` and `full_plain`
           Matched budget and matched time_scale, differing only in the
           vanish_dim prior, so the headline comparison stays OFAT-clean.

Parallelism
-----------
Arms are independent, so stages 1-3 run as parallel processes, each pinned to
one thread: the cost is a sequential Python loop over 44 intervals x 2 substeps
x 6 Tsit5 stages, so intra-op threading buys nothing (measured 0.82 s/epoch on
one thread vs 0.80 on ten) and the cores are better spent on other arms.

**Scaling is sublinear on a power-limited machine** -- 8 concurrent arms
measured only ~1.5x the aggregate throughput of one. `--max-parallel` bounds
concurrency for that reason, and stage 4 runs its long arms SERIALLY, which on
such a machine finishes sooner than running them together.

Uses `.wait()` on the Popen handles rather than polling by PID -- the same
correctness point `docs/07_fix_changelog.md` raises about the old `Wait-Process`
pattern, which could latch onto a recycled PID.

Owner: Monjur Hossain Khan (Shovon), 2105043.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import common
from common import ensure_results_dir

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_SUBDIR = "epidemic_runs"

# Bracket the s ~= 24 the init-time diagnostic predicts. s=1 is the no-fix
# baseline; s=10 is SIR's own value, included to test whether it transfers
# unchanged.
TIME_SCALES = [1.0, 10.0, 20.0, 24.0, 30.0, 40.0]

# The time_scale used for every control and full-length arm. Chosen a priori
# from the diagnostic (horizon 120/24 = 5.0, docs/10's trainable regime) rather
# than from the sweep, whose training loss is a flat plateau across [10, 40]
# and therefore does not discriminate. Holding it fixed keeps the structural
# comparison one-variable-at-a-time.
S_REF = 24.0

SPLIT_DAYS = [60, 70]


def launch(args_list, tag, log_dir):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = open(os.path.join(log_dir, f"{tag}.log"), "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(HERE, "run_epidemic_fit.py")] + args_list,
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE,
    )
    proc._log_handle = log
    return proc


def run_batch(specs, log_dir, max_parallel, label):
    """Run `specs` ({tag: argv}) with bounded concurrency; raise if any fails."""
    print(f"  {label}: {len(specs)} arms, up to {max_parallel} at a time", flush=True)
    pending = list(specs.items())
    running, failed = {}, []

    while pending or running:
        while pending and len(running) < max_parallel:
            tag, argv = pending.pop(0)
            running[tag] = launch(argv, tag, log_dir)
        # Block on one arm; whichever it is, finishing frees a slot.
        tag, proc = next(iter(running.items()))
        rc = proc.wait()
        proc._log_handle.close()
        del running[tag]
        print(f"    {tag:<12} {'ok' if rc == 0 else f'FAILED (rc={rc})'}", flush=True)
        if rc != 0:
            failed.append(tag)

    if failed:
        raise SystemExit(f"Arms failed: {failed}. See {log_dir} for their logs.")


def main():
    ap = argparse.ArgumentParser(description="Run all of Phase 3 Track E")
    ap.add_argument("--skip-e1", action="store_true")
    ap.add_argument("--skip-e2", action="store_true")
    ap.add_argument("--probe-epochs", type=int, default=2000,
                    help="Budget per probe/control arm (roadmap discipline: "
                         "probe first, commit the full budget only after).")
    ap.add_argument("--full-epochs", type=int, default=5000)
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="Concurrent arms. Scaling is sublinear on a "
                         "power-limited CPU; 4 was the useful ceiling here.")
    args = ap.parse_args()

    out_dir = ensure_results_dir()
    log_dir = ensure_results_dir(os.path.join(RUNS_SUBDIR, "_logs"))
    run_root = os.path.join(out_dir, RUNS_SUBDIR)
    t_start = time.time()

    if not args.skip_e1:
        print("=" * 78)
        print("E1: SINDy vs KAN-ODE under noise + L1 edge pruning")
        print("=" * 78, flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "run_sindy_noise.py")],
                       check=True, cwd=HERE)

    if args.skip_e2:
        print(f"\nTrack E (E1 only) finished in {time.time() - t_start:.0f}s.")
        return

    print("\n" + "=" * 78)
    print("E2 stage 0: init-time diagnostic (no training)")
    print("=" * 78, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "run_epidemic_fit.py"),
                    "--diagnose"], check=True, cwd=HERE, stdout=subprocess.DEVNULL)
    with open(os.path.join(out_dir, "init_diagnostic.json")) as f:
        diag = json.load(f)
    print(f"  |f_init| = {diag['init_field_magnitude']:.4g}  vs  true |f| = "
          f"{diag['true_field_magnitude']:.4g}   ratio {diag['init_over_true_ratio']:.2f}x")
    print(f"  sum(y) range {diag['sum_invariant']['sum_range']:.4f} "
          f"-> zero-sum invariant holds: {diag['sum_invariant']['is_invariant']}")
    print(f"  y0[infected] = {diag['vanish_dim_gate']['y0_infected']:.4g}")

    ep = str(args.probe_epochs)
    print("\n" + "=" * 78)
    print(f"E2 stages 1-3: sweep, structural controls, split controls ({ep} epochs)")
    print("=" * 78, flush=True)
    specs = {f"ts{ts:g}": ["--tag", f"ts{ts:g}", "--time_scale", str(ts), "--epochs", ep]
             for ts in TIME_SCALES}
    specs["proj"] = ["--tag", "proj", "--time_scale", str(S_REF),
                     "--conserve_projection", "--epochs", ep]
    specs["vanish"] = ["--tag", "vanish", "--time_scale", str(S_REF),
                       "--vanish_dim", "0", "--epochs", ep]
    for d in SPLIT_DAYS:
        specs[f"split{d}"] = ["--tag", f"split{d}", "--time_scale", str(S_REF),
                              "--train_days", str(d), "--epochs", ep]
    run_batch(specs, log_dir, args.max_parallel, "probe + control arms")

    for ts in TIME_SCALES:
        with open(os.path.join(run_root, f"ts{ts:g}", "metrics.json")) as f:
            m = json.load(f)
        print(f"  s={ts:<5g} train_mse={m['best']['train_mse']:.4e}  "
              f"extrap={m['best']['extrap_mse']:.4e}  "
              f"spike={m['stability']['grad_norm_spike_ratio']:.1f}")

    fe = str(args.full_epochs)
    print("\n" + "=" * 78)
    print(f"E2 stage 4: full-length runs at s={S_REF:g} ({fe} epochs, serial)")
    print("=" * 78, flush=True)
    # Serial (max_parallel=1): on a power-limited CPU two long arms together
    # finish later than one after the other.
    run_batch({
        "full_vanish": ["--tag", "full_vanish", "--time_scale", str(S_REF),
                        "--vanish_dim", "0", "--epochs", fe],
        "full_plain": ["--tag", "full_plain", "--time_scale", str(S_REF),
                       "--epochs", fe],
    }, log_dir, 1, "full-length arms")

    print("\n" + "=" * 78)
    print("E2: collecting")
    print("=" * 78, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "run_epidemic_fit.py"),
                    "--collect"], check=True, cwd=HERE)

    print(f"\nTrack E finished in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
