"""
Collate individual run artifacts into Phase-2 result tables.

Each `train.py` invocation writes its own `metrics.json` into its own `--save_dir`.
When runs are launched separately (e.g. as parallel background processes) there is no
aggregate summary, so this walks a results tree, reads every `metrics.json`, and emits
one CSV plus a printed table per sweep directory.

Runs repeated across seeds are detected by a trailing `_seed<N>` in the directory name
and collapsed into mean +/- std.

Usage:
    python collate_results.py --root results/benchmarks
    python collate_results.py --root results/benchmarks --bucket final
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict

SEED_SUFFIX = re.compile(r"_seed(\d+)$")

# Columns pulled from each run, in table order.
COLUMNS = [
    ("train_mse", "Train MSE"),
    ("extrap_mse", "Extrap MSE"),
    ("full_mse", "Full MSE"),
    ("extrap_r2", "Extrap R2"),
    ("extrap_rel_l2", "Extrap RelL2"),
]


def load_runs(root, bucket):
    """Find every metrics.json under `root` and flatten the fields we tabulate."""
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "metrics.json" not in filenames:
            continue
        with open(os.path.join(dirpath, "metrics.json")) as f:
            m = json.load(f)
        if "config" not in m or bucket not in m:
            print(f"  ! skipping {dirpath}: old flat schema, re-run with current train.py")
            continue

        cfg = m["config"]
        name = os.path.basename(dirpath)
        seed_match = SEED_SUFFIX.search(name)
        group = SEED_SUFFIX.sub("", name)

        row = {
            "sweep": os.path.relpath(os.path.dirname(dirpath), root).replace("\\", "/"),
            "group": group,
            "seed": int(seed_match.group(1)) if seed_match else cfg.get("seed"),
            "model": cfg.get("model_type"),
            "dataset": cfg.get("dataset"),
            "basis": cfg.get("basis_func"),
            "solver": cfg.get("solver"),
            "params": cfg.get("parameters"),
            "lr": cfg.get("lr"),
            "epochs": cfg.get("num_epochs"),
            "dt": cfg.get("dt"),
            "substeps": cfg.get("substeps"),
            "noise_std": cfg.get("noise_std"),
            # [FIX-2026-08 / S3] provenance: two runs with the same lr/epochs but
            # different time_scale are NOT the same experiment, and the CSVs are
            # the only place that distinction survives.
            "time_scale": cfg.get("time_scale", 1.0),
            "best_epoch": m.get("selection", {}).get("best_epoch"),
            "nfe_per_traj": m.get("nfe_per_trajectory"),
            "lipschitz": m.get("estimated_lipschitz_bound"),
            "time_sec": m.get("training_time_seconds"),
            "git_sha": cfg.get("git_sha"),
        }
        for key, _ in COLUMNS:
            row[key] = m[bucket].get(key)
        runs.append(row)
    return runs


def summarise(runs):
    """Group runs by (sweep, group) and compute mean/std across seeds."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["sweep"], r["group"])].append(r)

    out = []
    for (sweep, group), rows in sorted(groups.items()):
        entry = {"sweep": sweep, "config": group, "n_seeds": len(rows)}
        for key in ("model", "dataset", "basis", "solver", "params",
                    "lr", "epochs", "dt", "substeps", "noise_std", "git_sha"):
            entry[key] = rows[0][key]
        for key, _ in COLUMNS + [("time_sec", ""), ("lipschitz", ""), ("nfe_per_traj", "")]:
            vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
            if not vals:
                entry[f"{key}_mean"], entry[f"{key}_std"] = None, None
                continue
            entry[f"{key}_mean"] = statistics.mean(vals)
            entry[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(entry)
    return out


def print_table(rows, bucket):
    """Print one aligned block per sweep directory."""
    by_sweep = defaultdict(list)
    for r in rows:
        by_sweep[r["sweep"]].append(r)

    for sweep, entries in sorted(by_sweep.items()):
        print("\n" + "=" * 118)
        print(f"SWEEP: {sweep or '(root)'}   [reporting '{bucket}' checkpoint]")
        print("=" * 118)
        head = f"{'Configuration':<24} | {'N':<3} | " + " | ".join(
            f"{label:<20}" for _, label in COLUMNS
        ) + f" | {'Time (s)':<10}"
        print(head)
        print("-" * len(head))
        for e in entries:
            cells = []
            for key, _ in COLUMNS:
                mu, sd = e[f"{key}_mean"], e[f"{key}_std"]
                if mu is None:
                    cells.append(f"{'--':<20}")
                elif e["n_seeds"] > 1:
                    cells.append(f"{mu:.3e}+-{sd:.1e}"[:20].ljust(20))
                else:
                    cells.append(f"{mu:.4e}".ljust(20))
            t = e["time_sec_mean"]
            t_cell = f"{t:<10.1f}" if isinstance(t, (int, float)) else f"{'--':<10}"
            print(f"{e['config']:<24} | {e['n_seeds']:<3} | " + " | ".join(cells) + f" | {t_cell}")
        print("=" * 118)


def main():
    parser = argparse.ArgumentParser(description="Collate Phase-2 run artifacts into tables")
    parser.add_argument("--root", type=str, default="results/benchmarks", help="Results tree to walk")
    parser.add_argument("--bucket", type=str, default="best", choices=["best", "final"],
                        help="Report the best-epoch or final-epoch checkpoint")
    parser.add_argument("--out", type=str, default="results/tables", help="CSV output directory")
    args = parser.parse_args()

    runs = load_runs(args.root, args.bucket)
    if not runs:
        print(f"No usable metrics.json found under '{args.root}'.")
        return

    rows = summarise(runs)
    print_table(rows, args.bucket)

    os.makedirs(args.out, exist_ok=True)

    per_run = os.path.join(args.out, f"per_run_{args.bucket}.csv")
    with open(per_run, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)

    summary = os.path.join(args.out, f"summary_{args.bucket}.csv")
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(runs)} runs -> {per_run}")
    print(f"Wrote {len(rows)} configs -> {summary}")

    shas = {r["git_sha"] for r in runs}
    lrs = {r["lr"] for r in runs}
    if len(shas) > 1:
        print(f"\nWARNING: runs span multiple commits {shas} -- they may not be comparable.")
    if len(lrs) > 1:
        print(f"WARNING: runs span multiple learning rates {lrs} -- check before tabulating.")


if __name__ == "__main__":
    main()
