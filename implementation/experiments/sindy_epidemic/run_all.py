"""
Track E entry point: reproduce every number and figure in
`docs/17_p3_sindy_epidemic_findings.md` from scratch.

    python run_all.py                 # everything (E1 + E2), ~3 h -- the
                                      # 10,000-epoch full run alone is ~2 h 15 min
    python run_all.py --skip-e2       # E1 only, ~1 min
    python run_all.py --probe-epochs 500 --full-epochs 2000   # fast smoke run

E2's arms are independent, so they run as parallel PROCESSES. On this workload
that is a real speedup and not just a hope: the cost is a sequential Python loop
over 44 intervals x 2 substeps x 6 Tsit5 stages, which is latency-bound rather
than BLAS-bound -- measured at 0.82 s/epoch on one thread versus 0.80 s/epoch on
ten, so intra-op threading buys nothing and the cores are free for other arms.
Each child is therefore pinned to a single thread to stop the arms fighting over
them.

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

# Sweep values bracket the s ~= 24 that the init-time diagnostic predicts (it is
# the value putting the rescaled horizon at 5.0, the regime docs/10 identifies as
# where the systems that train actually live). s=1 is the no-fix baseline; s=10
# is SIR's own value, included to test whether it transfers unchanged.
TIME_SCALES = [1.0, 10.0, 20.0, 24.0, 30.0, 40.0]


def launch(args_list, tag, log_dir):
    env = dict(os.environ)
    # One thread per child: intra-op parallelism is worthless here (measured),
    # and oversubscribing the arms across the cores would only add contention.
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = open(os.path.join(log_dir, f"{tag}.log"), "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(HERE, "run_epidemic_fit.py")] + args_list,
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE,
    )
    proc._log_handle = log
    return proc


def wait_all(procs, label, log_dir):
    print(f"  waiting on {len(procs)} {label} ...", flush=True)
    failed = []
    for tag, p in procs.items():
        rc = p.wait()
        p._log_handle.close()
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"    {tag:<10} {status}", flush=True)
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
    ap.add_argument("--full-epochs", type=int, default=10000)
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
    print("E2: real epidemic fit -- init diagnostic")
    print("=" * 78, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "run_epidemic_fit.py"),
                    "--diagnose"], check=True, cwd=HERE, stdout=subprocess.DEVNULL)
    with open(os.path.join(out_dir, "init_diagnostic.json")) as f:
        diag = json.load(f)
    print(f"  |f_init| = {diag['init_field_magnitude']:.4g}  vs  true |f| = "
          f"{diag['true_field_magnitude']:.4g}   ratio {diag['init_over_true_ratio']:.2f}x")
    print(f"  sum(y) range {diag['sum_invariant']['sum_range']:.4f} "
          f"-> zero-sum invariant holds: {diag['sum_invariant']['is_invariant']}")

    print("\n" + "=" * 78)
    print(f"E2: probe arms ({args.probe_epochs} epochs each, in parallel)")
    print("=" * 78, flush=True)
    procs = {}
    for ts in TIME_SCALES:
        tag = f"ts{ts:g}"
        procs[tag] = launch(["--tag", tag, "--time_scale", str(ts),
                             "--epochs", str(args.probe_epochs)], tag, log_dir)
    # Control arms, at the predicted time_scale, for the two SIR fixes whose
    # preconditions this dataset does not satisfy. Run so the write-up rests on
    # measured losses rather than on the algebra alone.
    procs["proj"] = launch(["--tag", "proj", "--time_scale", "24",
                            "--conserve_projection",
                            "--epochs", str(args.probe_epochs)], "proj", log_dir)
    procs["vanish"] = launch(["--tag", "vanish", "--time_scale", "24",
                              "--vanish_dim", "0",
                              "--epochs", str(args.probe_epochs)], "vanish", log_dir)
    wait_all(procs, "probe arms", log_dir)

    # Select the time_scale on TRAINING loss only -- the same rule train.py uses
    # for checkpoint selection. Selecting on extrapolation would tune a
    # hyperparameter on the window the extrapolation claim is made against.
    best_ts, best_loss = None, float("inf")
    for ts in TIME_SCALES:
        with open(os.path.join(run_root, f"ts{ts:g}", "metrics.json")) as f:
            m = json.load(f)
        loss = m["best"]["train_mse"]
        print(f"  s={ts:<5g} train_mse={loss:.4e}  extrap={m['best']['extrap_mse']:.4e}")
        if loss < best_loss:
            best_ts, best_loss = ts, loss
    print(f"  selected time_scale = {best_ts:g} (train MSE {best_loss:.4e})", flush=True)

    print("\n" + "=" * 78)
    print(f"E2: full run at s={best_ts:g} ({args.full_epochs} epochs)")
    print("=" * 78, flush=True)
    procs = {"full": launch(["--tag", "full", "--time_scale", str(best_ts),
                             "--epochs", str(args.full_epochs)], "full", log_dir)}
    wait_all(procs, "full run", log_dir)

    print("\n" + "=" * 78)
    print("E2: collecting")
    print("=" * 78, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "run_epidemic_fit.py"),
                    "--collect"], check=True, cwd=HERE)

    print(f"\nTrack E finished in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
