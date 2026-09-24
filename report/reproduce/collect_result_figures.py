"""
Copy the team's original matplotlib plots from implementation/results/ into report/figures/.

These figures were produced by the training and collation code
(implementation/train.py writes the per-run phase_space.png and loss_curves.png;
implementation/collate_results.py writes results/figures/0*_*.png). The report
uses them unchanged; this script only copies them, then checks byte-for-byte
that each report copy is identical to its source.

Usage:  python report/reproduce/collect_result_figures.py [--check]
        --check  only verify that the copies in report/figures/ match, copy nothing
"""

import filecmp
import shutil
import sys

from common import REPORT, RES, ROOT

ABL = RES / "benchmarks"
P4 = RES / "phase4" / "epoch_budget_check"

# report/figures/<dest>  <-  implementation/results/<source>
MANIFEST = {
    # Section 2: KAN against MLP (collate_results.py)
    "ablation/03_kan_vs_mlp_convergence.png": RES / "figures" / "03_kan_vs_mlp_convergence.png",
    "ablation/07_extrapolation_t28.png": RES / "figures" / "07_extrapolation_t28.png",
    # Section 3: solver and basis phase portraits (train.py, one per run)
    "ablation/solver_euler_phase.png": ABL / "ablation_solvers" / "solver_euler" / "phase_space.png",
    "ablation/solver_midpoint_phase.png": ABL / "ablation_solvers" / "solver_midpoint" / "phase_space.png",
    "ablation/solver_tsit5_phase.png": ABL / "ablation_solvers" / "solver_tsit5" / "phase_space.png",
    "ablation/basis_bspline_phase.png": ABL / "ablation_activations" / "basis_bspline" / "phase_space.png",
    "ablation/basis_rbf_phase.png": ABL / "ablation_activations" / "basis_rbf" / "phase_space.png",
    "ablation/basis_chebyshev_phase.png": ABL / "ablation_activations" / "basis_chebyshev" / "phase_space.png",
    "ablation/basis_newton_phase.png": ABL / "ablation_activations" / "basis_newton" / "phase_space.png",
    # Section 3: step size and training dynamics (collate_results.py)
    "ablation/02_error_vs_stepsize_loglog.png": RES / "figures" / "02_error_vs_stepsize_loglog.png",
    "ablation/05_solver_convergence.png": RES / "figures" / "05_solver_convergence.png",
    "ablation/06_basis_convergence.png": RES / "figures" / "06_basis_convergence.png",
    # Section 9: extended-budget runs (train.py, one per run)
    "phase4/tsit5_rbf_50k_loss_curves.png": P4 / "tsit5_rbf_50k" / "loss_curves.png",
    "phase4/mlp_silu_50k_loss_curves.png": P4 / "mlp_silu_50k" / "loss_curves.png",
    "phase4/euler_50k_loss_curves.png": P4 / "euler_50k" / "loss_curves.png",
    "phase4/bspline_25k_loss_curves.png": P4 / "bspline_25k" / "loss_curves.png",
    "phase4/mlp_paperspec_50k_loss_curves.png": P4 / "mlp_paperspec_50k" / "loss_curves.png",
}


def main():
    check_only = "--check" in sys.argv
    bad = 0
    for dest, src in MANIFEST.items():
        dst = REPORT / "figures" / dest
        if not src.exists():
            print(f"  MISSING source {src.relative_to(ROOT)}")
            bad += 1
            continue
        if not check_only:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        same = dst.exists() and filecmp.cmp(src, dst, shallow=False)
        bad += not same
        print(f"  {'ok ' if same else 'DIFF'} report/figures/{dest}  <-  {src.relative_to(ROOT).as_posix()}")
    print(f"{len(MANIFEST) - bad}/{len(MANIFEST)} figures identical to their source")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
