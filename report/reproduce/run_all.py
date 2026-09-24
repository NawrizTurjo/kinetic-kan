"""
Regenerate every report figure, table and derived number that is not a raw training output.

Runs each step in its own Python process, in order, and stops at the first failure.
Nothing is retrained: all inputs are the saved runs under implementation/results/.

Usage (from the repository root):
    python report/reproduce/run_all.py              # everything, then build report.pdf
    python report/reproduce/run_all.py --no-report  # skip the LaTeX build of report.pdf
    python report/reproduce/run_all.py s04 s09      # only the named steps (prefix match)

Takes about 2.5 minutes on a laptop CPU, plus about a minute for the report build.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent

STEPS = [
    ("s00_verify_checkpoints.py", "checkpoints reproduce their stored metrics"),
    ("s01_tables.py", "results tables from metrics.json"),
    ("s02_order_conditions.py", "Runge-Kutta order conditions"),
    ("s03_work_precision.py", "work-precision on the true field"),
    ("s04_stability_regions.py", "stability regions + learned spectrum"),
    ("s05_basis_catalog.py", "basis catalogue, conditioning, knot use"),
    ("s06_equilibria_invariant.py", "equilibria, period, first integral"),
    ("s07_limit_cycle.py", "limit-cycle test"),
    ("s08_cross_solver.py", "cross-solver transfer + Euler modified equation"),
    ("s09_field_error_maps.py", "vector-field error maps"),
    ("s10_edge_functions.py", "learned edge functions"),
    ("s11_training_dynamics.py", "milestones + KAN/MLP crossover"),
    ("s12_pareto.py", "cost-accuracy Pareto front"),
    ("s13_noise_stepsize.py", "noise and step-size sweeps"),
    ("s14_error_vs_time.py", "error against time"),
    ("collect_result_figures.py", "copy the team's original result plots"),
    ("build_tikz.py", "compile the TikZ diagrams"),
]


def run(cmd, cwd, label):
    t0 = time.time()
    print(f"\n=== {label}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    print(f"--- {'done' if r.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f} s", flush=True)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    steps = [s for s in STEPS if not args or any(s[0].startswith(a) for a in args)]
    for script, label in steps:
        run([sys.executable, str(HERE / script)], HERE, f"{script}: {label}")
    if not args and "--no-report" not in sys.argv:
        run([str(REPORT / "tools" / "tectonic.exe"), "-k", "report.tex"], REPORT, "build report.pdf (tectonic)")


if __name__ == "__main__":
    main()
