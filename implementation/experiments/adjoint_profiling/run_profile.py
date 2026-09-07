"""
Track A -- Adjoint vs. Autograd Profiling (Novelty 5 -> Table 4).
Owner: Nawriz Ahmed Turjo. See docs/12_phase3_roadmap.md Part 4 / Track A.

Produces:
    results/phase3/adjoint_profiling/table4.json
    results/phase3/adjoint_profiling/memory_vs_trajectory_length.png
    docs/13_p3_adjoint_profiling_findings.md   (draft, auto-filled from real numbers)

Usage
-----
    python run_profile.py                          # full run, GPU if available
    python run_profile.py --epochs 2000 --device cuda
    python run_profile.py --sweep_lengths 36 101 201 401 801 1601 3201
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

from data import generate_lotka_volterra_data, generate_sir_data   # noqa: E402
from profiling import (                                            # noqa: E402
    build_field,
    train_direct_autograd,
    train_adjoint,
    paired_gradient_relative_error,
    memory_vs_trajectory_length,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "phase3", "adjoint_profiling")
DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "docs", "13_p3_adjoint_profiling_findings.md")


# ==============================================================================
# System recipes -- reused verbatim from the project's own "fixed" configs so
# Track A's numbers describe the systems as they are actually trained elsewhere
# in this project, not a bespoke setup invented for this experiment.
# ==============================================================================
def build_lotka_volterra(device):
    """N=36 train points, the paper's own regime. Recipe: results/benchmarks/kanode_flagship."""
    data = generate_lotka_volterra_data()  # all defaults: alpha,beta,gamma,delta,t_end14,dt0.1,t_train_end3.5
    field = build_field(layers_hidden=[2, 10, 2], grid_len=5, conserve=False, seed=42)
    return {
        "name": "lotka_volterra",
        "state_dim": 2,
        "n_train": len(data.t_train),
        "y0": data.y0,
        "t": data.t_train,
        "y_train": data.y_train,
        "field": field,
        "lr": 2e-3,
        "substeps": 2,
    }


def build_sir(device):
    """
    N=101 train points, using the FIXED --time_scale recipe from
    docs/10_sir_root_cause_and_fix.md (results/_fixed/sir_fixed) so the run
    actually converges instead of exploding like the pre-fix recipe.
    """
    TIME_SCALE = 10.0
    data = generate_sir_data(t_end=80.0, dt=0.5, t_train_end=50.0, seed=42)
    field = build_field(layers_hidden=[3, 16, 3], grid_len=8, conserve=True, seed=42)
    t_scaled = data.t_train / TIME_SCALE
    return {
        "name": "sir",
        "state_dim": 3,
        "n_train": len(data.t_train),
        "y0": data.y0,
        "t": t_scaled,
        "y_train": data.y_train,
        "field": field,
        "lr": 3e-3,
        "substeps": 2,
        "time_scale": TIME_SCALE,
    }


def profile_system(system, num_epochs, device):
    print(f"\n{'='*70}\nProfiling: {system['name']} (N={system['n_train']}, d={system['state_dim']})\n{'='*70}")

    grad_check = paired_gradient_relative_error(
        system["field"], system["y0"], system["t"], system["y_train"],
        substeps=system["substeps"], device=device,
    )
    print(f"  gradient relative error (direct vs adjoint): {grad_check['grad_relative_error']:.3e}")

    direct = train_direct_autograd(
        system["field"], system["y0"], system["t"], system["y_train"],
        substeps=system["substeps"], num_epochs=num_epochs, lr=system["lr"], device=device,
        desc=f"{system['name']}/direct",
    )
    print(f"  direct autograd : {direct['wallclock_s_per_1000_epochs']:.2f}s/1k-epoch, "
          f"peak VRAM {direct['peak_vram_mb']}")

    adjoint = train_adjoint(
        system["field"], system["y0"], system["t"], system["y_train"],
        substeps=system["substeps"], num_epochs=num_epochs, lr=system["lr"], device=device,
        desc=f"{system['name']}/adjoint",
    )
    print(f"  adjoint         : {adjoint['wallclock_s_per_1000_epochs']:.2f}s/1k-epoch, "
          f"peak VRAM {adjoint['peak_vram_mb']}")

    return {
        "state_dim": system["state_dim"],
        "n_train_points": system["n_train"],
        "grad_check": grad_check,
        "direct_autograd": direct,
        "adjoint": adjoint,
    }


def plot_memory_vs_length(sweep, systems_summary, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["n_t"], sweep["direct_autograd_mb"], "o-", label="Direct autograd (this project's NeuralODE)")
    ax.plot(sweep["n_t"], sweep["adjoint_mb"], "s-", label="torchdiffeq.odeint_adjoint")
    for name, info in systems_summary.items():
        ax.axvline(info["n_train_points"], color="gray", linestyle=":", alpha=0.6)
        ax.annotate(f"{name} (N={info['n_train_points']})",
                    xy=(info["n_train_points"], ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1),
                    rotation=90, fontsize=8, va="top", ha="right", color="gray")
    ax.set_xlabel("Trajectory length $N_t$ (integration output points)")
    ax.set_ylabel("Peak VRAM (MB)" if sweep["direct_autograd_mb"][0] is not None else "Peak VRAM (N/A on CPU)")
    ax.set_title("Peak memory vs. trajectory length: direct autograd ($O(N_t)$) vs. adjoint ($O(1)$)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def find_crossover(sweep):
    """First N_t at which adjoint's peak VRAM drops below direct autograd's."""
    for n_t, d_mb, a_mb in zip(sweep["n_t"], sweep["direct_autograd_mb"], sweep["adjoint_mb"]):
        if d_mb is None or a_mb is None:
            return None
        if a_mb < d_mb:
            return n_t
    return None


def write_findings_md(table4, sweep, crossover, save_path):
    lv = table4["systems"]["lotka_volterra"]
    sir = table4["systems"]["sir"]
    device_name = table4["meta"]["device_name"]

    def fmt(v, unit=""):
        return "N/A" if v is None else f"{v:.2f}{unit}"

    lines = []
    lines.append("# Track A -- Adjoint vs. Autograd Profiling: Findings\n")
    lines.append("> **Auto-generated draft from `run_profile.py`'s actual output** "
                 "(`results/phase3/adjoint_profiling/table4.json`). Review the prose "
                 "before treating this as final; the numbers are real, the wording is a first pass.\n")
    lines.append("## Research question\n")
    lines.append(
        "`implementation/README.md` documents direct autograd through the unrolled solver "
        "as a deliberate choice for this project's short trajectories ($N=36$-$101$ points), "
        "on the grounds that adjoint sensitivity's $O(1)$-memory advantage doesn't matter until "
        "trajectories get long. This track measures whether that holds, on real hardware "
        f"({device_name}), and where it stops holding.\n"
    )
    lines.append("## Method\n")
    lines.append(
        "Both paths run classical RK4 at a **matched step size** -- this project's own "
        "`NeuralODE(method='rk4')` for direct autograd, `torchdiffeq.odeint_adjoint(method='rk4', "
        "options={'step_size':...})` for the adjoint path. A forward-trajectory check "
        "(`tests/test_p3_adjoint_profiling.py`) confirms the two RK4 implementations agree to "
        "~1e-7 relative error, so this isolates the differentiation method as the only variable -- "
        "any difference below is attributable to backprop-through-unrolled-solver vs. "
        "continuous-adjoint-ODE, not to comparing two different integrators.\n"
    )
    lines.append("## Table 4 -- peak VRAM, wall-clock, gradient accuracy\n")
    lines.append("| System | $N_t$ | Method | Peak VRAM (MB) | s / 1000 epochs | Grad rel. error |")
    lines.append("| :--- | :---: | :--- | :---: | :---: | :---: |")
    for sys_name, sys_label in [("lotka_volterra", "Lotka-Volterra"), ("sir", "SIR")]:
        s = table4["systems"][sys_name]
        ge = s["grad_check"]["grad_relative_error"]
        lines.append(
            f"| {sys_label} | {s['n_train_points']} | Direct autograd | "
            f"{fmt(s['direct_autograd']['peak_vram_mb'])} | "
            f"{fmt(s['direct_autograd']['wallclock_s_per_1000_epochs'])} | {ge:.2e} |"
        )
        lines.append(
            f"| {sys_label} | {s['n_train_points']} | Adjoint | "
            f"{fmt(s['adjoint']['peak_vram_mb'])} | "
            f"{fmt(s['adjoint']['wallclock_s_per_1000_epochs'])} | (same row) |"
        )
    lines.append("")

    crossover_str = (f"$N_t \\approx {crossover}$" if crossover is not None
                      else "not reached within the swept range "
                           f"(up to $N_t={sweep['n_t'][-1]}$)")
    lines.append("## Memory vs. trajectory length\n")
    lines.append(
        "![memory vs trajectory length](../../implementation/results/phase3/adjoint_profiling/"
        "memory_vs_trajectory_length.png)\n"
    )
    lines.append(
        f"Sweeping $N_t \\in \\{{{', '.join(str(n) for n in sweep['n_t'])}\\}}$ with a single "
        "forward+backward pass per point (isolating the memory claim from training-loop/optimizer "
        f"overhead), the crossover where adjoint's $O(1)$ memory starts winning is: **{crossover_str}**.\n"
    )
    lines.append("## Finding\n")
    both_ok = all(
        table4["systems"][s]["grad_check"]["grad_relative_error"] < 5e-3
        for s in ("lotka_volterra", "sir")
    )
    verdict = (
        "the gradient computed via the continuous adjoint matches direct autograd to within "
        "tolerance ($< 5\\times10^{-3}$ relative error) on both systems, "
        if both_ok else
        "the gradient computed via the continuous adjoint DEVIATES from direct autograd beyond "
        "the $5\\times10^{-3}$ tolerance on at least one system -- see the table above, "
    )
    lines.append(
        f"At this project's actual trajectory lengths ($N=36$ for Lotka-Volterra, $N=101$ for SIR), "
        f"{verdict}so `implementation/README.md`'s claim -- adjoint sensitivity is unnecessary here "
        "-- is evaluated on both memory and wall-clock in the table above. State explicitly, after "
        "reviewing the real numbers: does direct autograd remain cheaper in both VRAM and wall-clock "
        "at these lengths? At what $N_t$ (see crossover above) would that stop being true?\n"
    )
    lines.append(
        "*(TODO: replace this paragraph with one written after reading the actual table above -- "
        "the auto-generated text states the setup and the crossover mechanically; the judgment call "
        "about what it MEANS for the project's short-trajectory regime is not something to leave "
        "auto-generated.)*\n"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Track A: adjoint vs. autograd profiling")
    parser.add_argument("--epochs", type=int, default=2000,
                         help="Epochs per (system, method) profiling run (default: 2000, per roadmap budget)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sweep_lengths", type=int, nargs="+",
                         default=[36, 101, 201, 401, 801, 1601, 3201],
                         help="N_t values for the memory-vs-length sweep")
    parser.add_argument("--skip_training_profile", action="store_true",
                         help="Only run the memory-vs-length sweep + gradient check (fast smoke run)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    systems = {}
    if not args.skip_training_profile:
        systems["lotka_volterra"] = profile_system(build_lotka_volterra(device), args.epochs, device)
        systems["sir"] = profile_system(build_sir(device), args.epochs, device)

    print(f"\n{'='*70}\nMemory-vs-trajectory-length sweep (N_t = {args.sweep_lengths})\n{'='*70}")
    sweep = memory_vs_trajectory_length(
        state_dim=2, layers_hidden=[2, 10, 2], grid_len=5,
        n_t_values=args.sweep_lengths, device=device, substeps=2,
    )
    print(f"  direct autograd peak VRAM (MB): {sweep['direct_autograd_mb']}")
    print(f"  adjoint peak VRAM (MB)        : {sweep['adjoint_mb']}")

    table4 = {
        "meta": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "epochs_per_profiling_run": args.epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "systems": systems,
        "memory_vs_trajectory_length": sweep,
    }

    table4_path = os.path.join(RESULTS_DIR, "table4.json")
    with open(table4_path, "w") as f:
        json.dump(table4, f, indent=2)
    print(f"\nWrote {table4_path}")

    fig_path = os.path.join(RESULTS_DIR, "memory_vs_trajectory_length.png")
    plot_memory_vs_length(sweep, systems, fig_path)
    print(f"Wrote {fig_path}")

    if systems:
        crossover = find_crossover(sweep)
        write_findings_md(table4, sweep, crossover, DOCS_PATH)
        print(f"Wrote draft findings: {DOCS_PATH}")
    else:
        print("Skipped findings draft (--skip_training_profile was set, no per-system table to report)")


if __name__ == "__main__":
    main()
