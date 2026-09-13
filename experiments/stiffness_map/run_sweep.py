"""
Track D -- Stiffness-Solver Stability Phase Map (Phase 3, Novelty 3 -> Table 5).
Owner: Abrar Jahin (2105055). See docs/12_phase3_roadmap.md Part 4 / Track D.

Research question
------------------
Sweep the damped pendulum's damping ratio mu across {euler, midpoint, rk4, tsit5} and
step size(s) dt: which (solver, mu, dt) combinations converge, and which explode or
fail to fit? Produces a 2D stability heatmap (mu x solver) and table5.json.

Why this can't go through train.py's CLI
-----------------------------------------
`generate_damped_pendulum_data(mu=...)` is a plain function, but `train_kan_ode()`
only threads Lotka-Volterra kwargs (alpha, beta, gamma, delta) into the dataset
generator for any dataset -- confirmed by reading train.py directly, not assumed from
the roadmap doc. So a mu-sweep cannot be done via `python train.py --dataset
damped_pendulum`. This script reuses the same primitives train_kan_ode() itself
composes (KAN, NeuralODE, Adam, the X1 non-finite-gradient guard) in a standalone
~150-line loop, per docs/12_phase3_roadmap.md Part 2's "zero-conflict folder"
pattern. It does NOT edit train.py, kan/, ode/, data/, or utils/.

Baseline recipe (every cell starts here; only solver, mu, dt vary)
--------------------------------------------------------------------
This is the `pendulum_control_win5` config from docs/09_stability_fix_results.md,
confirmed against its own results/_fixed/pendulum_control_win5/metrics.json config
block (not retyped from memory):
    KAN([2, 10, 2]), grid_len=8, basis=rbf, normalizer=tanh, base_act=silu,
    substeps=2, lr=0.003, grad_clip=1.0, t_train_end=5.0, t_end=10.0, dt=0.05,
    seed=42, mu=0.5 (default pendulum damping).
Using the pre-fix t_train_end=3.0 recipe here would just re-measure the pendulum's
known optimisation failure at every mu, not stiffness -- see docs/09 and
docs/12_phase3_roadmap.md Track D's warning about this.

Two-phase budget (OFAT discipline established in docs/09_stability_fix_results.md
Methodological lessons)
------------------------------------------------------------------------------------
1. `--stage probe`: every (solver, mu, dt) cell at PROBE_EPOCHS (2,000), cheap.
2. `--stage list-extensions`: read the probe results and print which cells are
   promising/borderline enough to justify the full 10,000-epoch budget -- WITHOUT
   running anything. Use this to decide before spending the extra compute.
3. `--stage full`: run only the cells named via --solvers/--mus/--dts (or
   --from-extensions, which reads the same list `list-extensions` would print) at
   FULL_EPOCHS (10,000).

Nothing in this script runs anything at import time -- `train_cell`, `classify_cell`,
and the sweep drivers are plain functions, safe to import from tests or a notebook
without triggering training. Only `if __name__ == "__main__":` launches runs.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make `implementation/` importable regardless of the cwd this script is invoked
# from -- experiments/stiffness_map/ is two levels below implementation/'s parent.
_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from kan import KAN  # noqa: E402
from ode import NeuralODE  # noqa: E402
from data import generate_damped_pendulum_data  # noqa: E402
from utils import compute_gradient_norm, compute_mse, compute_r2_score  # noqa: E402


# ==============================================================================
# Fixed baseline and sweep axes
# ==============================================================================
SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
MU_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
DEFAULT_DT = 0.05

BASELINE = dict(
    layers_hidden=[2, 10, 2],
    grid_len=8,
    grid_lims=(-1.0, 1.0),
    basis_func="rbf",
    normalizer="tanh",
    base_act="silu",
    substeps=2,
    lr=0.003,
    grad_clip=1.0,
    t_start=0.0,
    t_end=10.0,
    t_train_end=5.0,
    seed=42,
)

PROBE_EPOCHS = 2000
FULL_EPOCHS = 10000
NONFINITE_ABORT_STREAK = 100  # identical to train.py's X1 guard

# Classification thresholds. `converged` uses train MSE < 1e-2, matching the
# threshold suggested in docs/12_phase3_roadmap.md Track D task 4. `unstable`
# additionally catches a recovered-but-noisy run (final >> best, mirroring the
# "REGRESSED" flatness verdict analyze_fixes.py uses elsewhere in this project).
CONVERGED_MSE_THRESHOLD = 1e-2
UNSTABLE_FINAL_OVER_BEST_RATIO = 10.0


# ==============================================================================
# Single-cell training loop
# ==============================================================================
def train_cell(
    solver: str,
    mu: float,
    dt: float = DEFAULT_DT,
    num_epochs: int = PROBE_EPOCHS,
    seed: int = BASELINE["seed"],
    device: str = "cpu",
    print_freq: int = 100,
    save_trajectory_to: Optional[str] = None,
    save_checkpoint_to: Optional[str] = None,
) -> Dict:
    """
    Train one (solver, mu, dt) cell of the stiffness map from scratch.

    Mirrors train_kan_ode()'s loop structure (data -> model -> Adam -> per-epoch
    forward/backward with the X1 non-finite-gradient guard) but is intentionally
    narrower: no checkpoint files, no plots, no CLI -- this is a sweep cell, not a
    standalone experiment. Returns a metrics dict that is the sole record of the run
    by default (nothing is written to disk here unless save_trajectory_to is given).

    save_trajectory_to: if given, save {t_full, y_true, y_pred, t_train_end, mu,
    solver} as an .npz at this path -- the ground-truth and best-checkpoint
    predicted trajectories, for later plotting. NOT saved by any of the 24 cells
    already run in results/phase3/stiffness_map/probe/ (added after that sweep
    completed) -- a cell must be RE-RUN with this set to get its trajectory.

    print_freq: print a one-line progress update every this many epochs (0 disables
    all per-epoch printing). A single cell at PROBE_EPOCHS=2000 can take 20-40+
    minutes wall-clock; with no live output that looks indistinguishable from a
    hang. Plain `print()` line-by-line (not tqdm) so it stays readable when stdout
    is redirected to a log file, not just an interactive terminal.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = generate_damped_pendulum_data(
        mu=mu,
        t_start=BASELINE["t_start"],
        t_end=BASELINE["t_end"],
        dt=dt,
        t_train_end=BASELINE["t_train_end"],
        seed=seed,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_train = data.t_train
    t_full = data.t_full
    y_train = data.y_train
    y_full = data.y_full
    y0_init = data.y0
    n_train = len(t_train)

    model = KAN(
        layers_hidden=BASELINE["layers_hidden"],
        grid_len=BASELINE["grid_len"],
        grid_lims=BASELINE["grid_lims"],
        basis_func=BASELINE["basis_func"],
        normalizer=BASELINE["normalizer"],
        base_act=BASELINE["base_act"],
    ).to(device)

    node = NeuralODE(func=model, method=solver, substeps=BASELINE["substeps"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE["lr"])

    train_losses: List[float] = []
    grad_norms: List[float] = []
    best_train_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    nonfinite_grad_steps = 0
    first_nonfinite_epoch: Optional[int] = None
    nonfinite_streak = 0
    aborted_at_epoch: Optional[int] = None
    epochs_run = num_epochs

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        pred_train = node(y0=y0_init, t=t_train)
        mse_train = F.mse_loss(pred_train, y_train)
        mse_train.backward()

        gnorm = compute_gradient_norm(model)
        grad_norms.append(gnorm)
        train_loss_val = mse_train.item()
        train_losses.append(train_loss_val)

        if train_loss_val < best_train_loss:
            best_train_loss = train_loss_val
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        # [X1 guard, ported from train.py -- see its own comment for the full
        # derivation of why clip_grad_norm_ cannot substitute for this check.]
        step_is_finite = math.isfinite(gnorm) and math.isfinite(train_loss_val)
        if step_is_finite:
            nonfinite_streak = 0
            if BASELINE["grad_clip"] and BASELINE["grad_clip"] > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=BASELINE["grad_clip"])
            optimizer.step()
        else:
            nonfinite_grad_steps += 1
            nonfinite_streak += 1
            if first_nonfinite_epoch is None:
                first_nonfinite_epoch = epoch
                print(f"    [epoch {epoch}] non-finite gradient (gnorm={gnorm}, "
                      f"loss={train_loss_val}) -- step skipped, guard engaged", flush=True)
            optimizer.zero_grad(set_to_none=True)

        if print_freq and (epoch % print_freq == 0 or epoch == 1 or epoch == num_epochs):
            elapsed_so_far = time.time() - start_time
            rate = elapsed_so_far / epoch
            eta = rate * (num_epochs - epoch)
            print(
                f"    [epoch {epoch}/{num_epochs}] loss={train_loss_val:.4e} "
                f"best={best_train_loss:.4e} gnorm={gnorm:.3e} "
                f"({rate:.3f} s/epoch, ETA {eta/60:.1f} min)",
                flush=True,
            )

        if nonfinite_streak >= NONFINITE_ABORT_STREAK:
            aborted_at_epoch = epoch
            epochs_run = epoch
            print(f"    [epoch {epoch}] ABORTING: {nonfinite_streak} consecutive "
                  f"non-finite steps -- parameters unrecoverable, stopping early.", flush=True)
            break

    elapsed = time.time() - start_time

    # Score the best checkpoint (selection-on-training-loss, matching train.py).
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        pred_full = node(y0=y0_init, t=t_full)
    pred_full_np = pred_full.numpy()
    y_full_np = y_full.numpy()
    pred_train_np = pred_full_np[:n_train]
    y_train_np = y_full_np[:n_train]
    pred_extrap_np = pred_full_np[n_train:]
    y_extrap_np = y_full_np[n_train:]

    if save_trajectory_to is not None:
        os.makedirs(os.path.dirname(save_trajectory_to), exist_ok=True)
        np.savez(
            save_trajectory_to,
            t_full=t_full.numpy(),
            y_true=y_full_np,
            y_pred=pred_full_np,
            t_train_end=BASELINE["t_train_end"],
            n_train=n_train,
            mu=mu,
            solver=solver,
        )

    if save_checkpoint_to is not None:
        os.makedirs(os.path.dirname(save_checkpoint_to), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "layers_hidden": BASELINE["layers_hidden"],
            "grid_len": BASELINE["grid_len"],
            "grid_lims": BASELINE["grid_lims"],
            "basis_func": BASELINE["basis_func"],
            "normalizer": BASELINE["normalizer"],
            "base_act": BASELINE["base_act"],
            "solver": solver,
            "mu": mu,
            "dt": dt,
        }, save_checkpoint_to)

    final_train_loss = train_losses[-1] if train_losses else float("nan")

    return {
        "solver": solver,
        "mu": mu,
        "dt": dt,
        "num_epochs_requested": num_epochs,
        "epochs_run": epochs_run,
        "aborted_at_epoch": aborted_at_epoch,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "first_nonfinite_epoch": first_nonfinite_epoch,
        "best_epoch": best_epoch,
        "best_train_mse": best_train_loss if math.isfinite(best_train_loss) else None,
        "final_train_mse": final_train_loss if math.isfinite(final_train_loss) else None,
        "full_mse": compute_mse(y_full_np, pred_full_np) if best_state_dict is not None else None,
        "extrap_mse": compute_mse(y_extrap_np, pred_extrap_np) if best_state_dict is not None else None,
        "extrap_r2": compute_r2_score(y_extrap_np, pred_extrap_np) if best_state_dict is not None else None,
        "grad_norm_max": max((g for g in grad_norms if math.isfinite(g)), default=None),
        "seconds": elapsed,
    }


# ==============================================================================
# Classification
# ==============================================================================
def classify_cell(metrics: Dict) -> str:
    """
    Classify a completed cell as 'converged' / 'unstable' / 'diverged'.

    diverged  -- the run aborted (100 consecutive non-finite steps), or no valid
                 checkpoint was ever produced (best_train_mse is None).
    unstable  -- finite throughout, but either (a) at least one non-finite gradient
                 step was skipped-and-recovered-from, or (b) the final-epoch loss is
                 far worse than the best checkpoint (a late-training regression, the
                 same 'REGRESSED' pattern analyze_fixes.py flags elsewhere in this
                 project).
    converged -- finite throughout, no guard activity, and best_train_mse below
                 CONVERGED_MSE_THRESHOLD.

    A cell that is finite, guard-free, but ABOVE the MSE threshold is also reported
    as 'unstable' rather than a silent 'converged' -- it failed to fit, which is a
    real (solver, mu) outcome worth showing on the heatmap, not a diverged run.
    """
    if metrics["aborted_at_epoch"] is not None or metrics["best_train_mse"] is None:
        return "diverged"

    if metrics["nonfinite_grad_steps"] > 0:
        return "unstable"

    best = metrics["best_train_mse"]
    final = metrics["final_train_mse"]
    if final is not None and best > 0 and (final / best) > UNSTABLE_FINAL_OVER_BEST_RATIO:
        return "unstable"

    if best < CONVERGED_MSE_THRESHOLD:
        return "converged"
    return "unstable"


# ==============================================================================
# Sweep drivers
# ==============================================================================
def _cell_dir(root: str, solver: str, mu: float, dt: float) -> str:
    return os.path.join(root, f"{solver}_mu{mu}_dt{dt}")


def run_stage(
    stage_root: str,
    solvers: List[str],
    mus: List[float],
    dts: List[float],
    num_epochs: int,
    device: str = "cpu",
    skip_existing: bool = False,
    save_trajectories: bool = False,
    save_checkpoints: bool = False,
) -> List[Dict]:
    """Run every (solver, mu, dt) cell in the given grid; write each cell's metrics
    JSON to its own folder under stage_root. Returns the list of result dicts.

    skip_existing: if a cell's metrics.json already exists on disk, load and reuse
    it instead of retraining. This is what makes a stopped-and-restarted run resume
    rather than redo everything from scratch -- e.g. after killing one solver's
    window partway through its mu list (see experiments/stiffness_map/README.md
    "Stopping and resuming"). Off by default so the ordinary case (a fresh sweep,
    or deliberately overwriting a stale/bad result) is unaffected.

    save_trajectories: also write trajectory.npz (ground truth + best-checkpoint
    prediction) into each cell's own folder -- see train_cell's save_trajectory_to.
    Point --out_root at a SEPARATE results root when using this (e.g.
    results/phase3/stiffness_map/probe_traj/), never the original probe/full
    root, so a re-run for plotting purposes can never overwrite or be confused
    with the already-validated sweep results.
    """
    os.makedirs(stage_root, exist_ok=True)
    results = []
    total = len(solvers) * len(mus) * len(dts)
    i = 0
    for dt in dts:
        for mu in mus:
            for solver in solvers:
                i += 1
                cell_dir = _cell_dir(stage_root, solver, mu, dt)
                existing_path = os.path.join(cell_dir, "metrics.json")
                if skip_existing and os.path.isfile(existing_path):
                    with open(existing_path) as f:
                        metrics = json.load(f)
                    results.append(metrics)
                    print(f"[{i}/{total}] solver={solver} mu={mu} dt={dt} "
                          f"-- [skip] metrics.json already present "
                          f"(verdict={metrics.get('verdict')})", flush=True)
                    continue

                print(f"[{i}/{total}] solver={solver} mu={mu} dt={dt} epochs={num_epochs} ...", flush=True)
                traj_path = os.path.join(cell_dir, "trajectory.npz") if save_trajectories else None
                ckpt_path = os.path.join(cell_dir, "checkpoint.pt") if save_checkpoints else None
                metrics = train_cell(solver=solver, mu=mu, dt=dt, num_epochs=num_epochs, device=device,
                                      save_trajectory_to=traj_path, save_checkpoint_to=ckpt_path)
                metrics["verdict"] = classify_cell(metrics)
                os.makedirs(cell_dir, exist_ok=True)
                with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                results.append(metrics)
                print(
                    f"    -> {metrics['verdict']} "
                    f"(best_train_mse={metrics['best_train_mse']}, "
                    f"nonfinite={metrics['nonfinite_grad_steps']}, "
                    f"aborted={metrics['aborted_at_epoch']})",
                    flush=True,
                )
    return results


def load_stage_results(stage_root: str) -> List[Dict]:
    results = []
    if not os.path.isdir(stage_root):
        return results
    for name in sorted(os.listdir(stage_root)):
        path = os.path.join(stage_root, name, "metrics.json")
        if os.path.isfile(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def build_table5(
    probe_root: str,
    full_root: str,
    solvers: List[str],
    mus: List[float],
    dts: List[float],
) -> List[Dict]:
    """
    Combine probe- and full-budget results into the single table the heatmap and
    docs/16_p3_stiffness_map_findings.md are built from: one entry per requested
    (solver, mu, dt) cell, preferring the full-budget run when both exist so the
    reported verdict always reflects the best available evidence for that cell.

    A cell with NEITHER a probe nor a full result yet is included with
    verdict="missing" rather than silently omitted -- the heatmap must show gaps as
    gaps, not as a false 'diverged' or a blank that looks like an oversight.
    """
    probe_by_key = {(r["solver"], r["mu"], r["dt"]): r for r in load_stage_results(probe_root)}
    full_by_key = {(r["solver"], r["mu"], r["dt"]): r for r in load_stage_results(full_root)}

    table = []
    for dt in dts:
        for mu in mus:
            for solver in solvers:
                key = (solver, mu, dt)
                if key in full_by_key:
                    r = full_by_key[key]
                    source = "full"
                elif key in probe_by_key:
                    r = probe_by_key[key]
                    source = "probe"
                else:
                    table.append({
                        "solver": solver, "mu": mu, "dt": dt,
                        "verdict": "missing", "source": None,
                        "best_train_mse": None, "epochs_run": None,
                    })
                    continue
                # NOTE: `r.get("verdict", classify_cell(r))` would look equivalent but
                # is NOT -- dict.get's default argument is evaluated eagerly in Python
                # regardless of whether the key is present, so classify_cell(r) would
                # run (and could KeyError on a dict missing its other fields) even when
                # "verdict" already exists. Branch explicitly instead.
                verdict = r["verdict"] if "verdict" in r else classify_cell(r)
                table.append({
                    "solver": solver, "mu": mu, "dt": dt,
                    "verdict": verdict,
                    "source": source,
                    "best_train_mse": r.get("best_train_mse"),
                    "full_mse": r.get("full_mse"),
                    "extrap_r2": r.get("extrap_r2"),
                    "epochs_run": r.get("epochs_run"),
                    "nonfinite_grad_steps": r.get("nonfinite_grad_steps"),
                })
    return table


def plot_stability_heatmap(table: List[Dict], out_path: str, dt: float, solvers: List[str], mus: List[float]) -> None:
    """
    Render the mu (rows) x solver (columns) grid for a single dt as a standard
    continuous ML-style heatmap: cell COLOR is best_train_mse on a log scale
    (viridis, real colorbar in actual MSE units) -- this is the number that
    actually varies smoothly and is what a reader wants to visually compare at
    a glance, the way a confusion-matrix or correlation heatmap works. The
    earlier version instead flat-colored every cell by its discrete verdict
    (converged/unstable/diverged), which threw away all the within-category
    variation (e.g. two 'unstable' cells at 1.8e-2 and 1.0e-1 looked identical)
    and produced flat, poster-like color BLOCKS rather than something that
    reads as a heatmap.

    Verdict is NOT dropped, just demoted to a secondary channel: a colored
    border around each cell (green=converged, orange=unstable, red=diverged,
    grey=missing) plus the annotation text itself, so "how bad" (color) and
    "which category" (border) are both visible without collapsing into one.

    Kept independent of matplotlib's colormap machinery beyond `table` itself
    -- callable on real OR synthetic data, which is how it gets validated
    before any real sweep data exists.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    verdict_border = {"converged": "#2ca02c", "unstable": "#e67e22", "diverged": "#d62728", "missing": "#999999"}

    by_key = {(r["solver"], r["mu"]): r for r in table if r["dt"] == dt}
    n_rows, n_cols = len(mus), len(solvers)
    log_mse = np.full((n_rows, n_cols), np.nan)
    verdicts = np.empty((n_rows, n_cols), dtype=object)
    mses = np.empty((n_rows, n_cols), dtype=object)
    for i, mu in enumerate(mus):
        for j, solver in enumerate(solvers):
            r = by_key.get((solver, mu))
            verdicts[i, j] = r["verdict"] if r else "missing"
            mse = r.get("best_train_mse") if r else None
            mses[i, j] = mse
            if isinstance(mse, (int, float)) and mse > 0:
                log_mse[i, j] = np.log10(mse)

    finite = log_mse[~np.isnan(log_mse)]
    vmin, vmax = (np.floor(finite.min()), np.ceil(finite.max())) if finite.size else (-5, 0)

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(1.9 * n_cols + 2.2, 1.3 * n_rows + 1.8))
    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("#e8e8e8")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(np.ma.masked_invalid(log_mse), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([s.upper() for s in solvers], fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"μ = {m:g}" for m in mus], fontsize=12, fontweight="bold")
    ax.tick_params(length=0)

    for i in range(n_rows):
        for j in range(n_cols):
            mse = mses[i, j]
            label = f"{mse:.1e}\n{verdicts[i, j]}" if isinstance(mse, (int, float)) else "no data"
            if not np.isnan(log_mse[i, j]):
                r, g, b, _ = cmap(norm(log_mse[i, j]))
                text_color = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"
            else:
                text_color = "#555555"
            ax.text(j, i, label, ha="center", va="center", fontsize=10, color=text_color, linespacing=1.6)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=verdict_border[verdicts[i, j]], linewidth=3.5))

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"Stiffness-Solver Stability Phase Map (dt={dt})", fontsize=14, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks = np.arange(vmin, vmax + 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"1e{int(t)}" for t in ticks])
    cbar.set_label("best_train_mse (log scale) — lower is better", fontsize=10)
    cbar.outline.set_visible(False)

    legend_handles = [Patch(facecolor="none", edgecolor=c, linewidth=2.5, label=v)
                       for v, c in verdict_border.items()]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.32, 1), loc="upper left", fontsize=9,
              title="border = verdict", title_fontsize=9, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.patch.set_facecolor("white")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def list_extensions(probe_root: str) -> List[Tuple[str, float, float]]:
    """
    Read probe-stage results and return the (solver, mu, dt) cells worth extending
    to the full epoch budget: anything that is not clearly 'diverged'. A cell that
    diverges within 2,000 epochs is assumed to diverge at 10,000 too (per the OFAT
    discipline in docs/09_stability_fix_results.md -- don't burn the full budget on
    cells that visibly fail early).
    """
    results = load_stage_results(probe_root)
    if not results:
        raise FileNotFoundError(
            f"No probe results found under '{probe_root}'. Run --stage probe first."
        )
    extend = [
        (r["solver"], r["mu"], r["dt"])
        for r in results
        if classify_cell(r) != "diverged"
    ]
    return extend


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Track D: Stiffness-Solver Stability Phase Map")
    parser.add_argument(
        "--stage", required=True, choices=["probe", "list-extensions", "full", "sanity", "aggregate"],
        help=(
            "probe: run every cell at PROBE_EPOCHS. "
            "list-extensions: print (no training) which probe cells qualify for a full run. "
            "full: run the named cells at FULL_EPOCHS. "
            "sanity: run ONLY (tsit5, mu=0.5, dt=0.05) at FULL_EPOCHS and compare against "
            "the known pendulum_control_win5 result -- the built-in correctness check. "
            "aggregate: build table5.json and the heatmap PNG from whatever probe/full "
            "results already exist on disk -- runs no training, safe to call at any time "
            "(including with zero results, which renders an all-'missing' heatmap)."
        ),
    )
    parser.add_argument("--solvers", nargs="+", default=SOLVERS)
    parser.add_argument("--mus", nargs="+", type=float, default=MU_VALUES)
    parser.add_argument("--dts", nargs="+", type=float, default=[DEFAULT_DT])
    parser.add_argument("--out_root", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                             "results", "phase3", "stiffness_map"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--from-extensions", action="store_true",
        help="For --stage full: ignore --solvers/--mus/--dts and instead run exactly "
             "the cells list-extensions would print, read from the probe results.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override PROBE_EPOCHS/FULL_EPOCHS for --stage probe/full/sanity. For "
             "quick end-to-end pipeline checks (e.g. --epochs 3) only -- the real "
             "sweep should use the stage defaults (2000 / 10000) so results are "
             "directly comparable across cells.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="For --stage probe/full: if a cell's metrics.json already exists on "
             "disk, reuse it instead of retraining. Use this to resume after "
             "stopping a run partway through (e.g. one solver's window was closed "
             "mid-sweep) without redoing already-completed cells.",
    )
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="For --stage probe/full: also save trajectory.npz (ground truth + "
             "best-checkpoint prediction) into each cell's folder, for later "
             "plotting. ALWAYS pair with a separate --out_root (e.g. "
             "results/phase3/stiffness_map/probe_traj) -- never point this at the "
             "original probe/full root, since it re-runs (and re-scores) training "
             "rather than reading the existing metrics.json.",
    )
    parser.add_argument(
        "--save-checkpoints", action="store_true",
        help="For --stage probe/full: also save checkpoint.pt (best-epoch model "
             "state_dict + architecture config) into each cell's folder, for later "
             "direct model inspection (e.g. evaluating f_theta at a specific "
             "point). Same --out_root safety rule as --save-trajectories.",
    )
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    probe_root = os.path.join(out_root, "probe")
    full_root = os.path.join(out_root, "full")
    default_out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                      "results", "phase3", "stiffness_map"))

    if (args.save_trajectories or args.save_checkpoints) and out_root == default_out_root:
        raise SystemExit(
            "--save-trajectories/--save-checkpoints was requested with the DEFAULT "
            "--out_root, which is the same directory as the already-completed sweep "
            "results. This would re-run and overwrite that data's metrics.json for "
            "every cell touched. Pass a separate --out_root (e.g. "
            "--out_root ../../results/phase3/stiffness_map/probe_traj) instead."
        )

    if args.stage == "probe":
        epochs = args.epochs if args.epochs is not None else PROBE_EPOCHS
        run_stage(probe_root, args.solvers, args.mus, args.dts, epochs,
                  device=args.device, skip_existing=args.skip_existing,
                  save_trajectories=args.save_trajectories, save_checkpoints=args.save_checkpoints)

    elif args.stage == "list-extensions":
        cells = list_extensions(probe_root)
        print(f"{len(cells)} of the probe cells qualify for a full ({FULL_EPOCHS}-epoch) run:")
        for solver, mu, dt in cells:
            print(f"  solver={solver} mu={mu} dt={dt}")

    elif args.stage == "full":
        epochs = args.epochs if args.epochs is not None else FULL_EPOCHS
        if args.from_extensions:
            cells = list_extensions(probe_root)
            for solver, mu, dt in cells:
                run_stage(full_root, [solver], [mu], [dt], epochs,
                          device=args.device, skip_existing=args.skip_existing,
                          save_trajectories=args.save_trajectories, save_checkpoints=args.save_checkpoints)
        else:
            run_stage(full_root, args.solvers, args.mus, args.dts, epochs,
                      device=args.device, skip_existing=args.skip_existing,
                      save_trajectories=args.save_trajectories, save_checkpoints=args.save_checkpoints)

    elif args.stage == "sanity":
        epochs = args.epochs if args.epochs is not None else FULL_EPOCHS
        print("Sanity check: (tsit5, mu=0.5, dt=0.05) should reproduce pendulum_control_win5.")
        print("Reference (results/_fixed/pendulum_control_win5/metrics.json):")
        print("  best.train_mse = 9.2797e-05 | best.full_mse = 4.4792e-02 | best.extrap_r2 = 0.6472")
        metrics = train_cell(solver="tsit5", mu=0.5, dt=DEFAULT_DT, num_epochs=epochs, device=args.device)
        print("This run:")
        print(f"  best.train_mse = {metrics['best_train_mse']:.4e} | "
              f"full_mse = {metrics['full_mse']:.4e} | extrap_r2 = {metrics['extrap_r2']:.4f}")
        cell_dir = _cell_dir(os.path.join(out_root, "sanity"), "tsit5", 0.5, DEFAULT_DT)
        os.makedirs(cell_dir, exist_ok=True)
        with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    elif args.stage == "aggregate":
        table = build_table5(probe_root, full_root, args.solvers, args.mus, args.dts)
        table_path = os.path.join(out_root, "table5.json")
        with open(table_path, "w") as f:
            json.dump(table, f, indent=2)
        print(f"Wrote {table_path} ({len(table)} cells)")

        counts = {"converged": 0, "unstable": 0, "diverged": 0, "missing": 0}
        for r in table:
            counts[r["verdict"]] += 1
        print(f"Verdict counts: {counts}")

        for dt in args.dts:
            heatmap_path = os.path.join(out_root, f"stability_heatmap_dt{dt}.png")
            plot_stability_heatmap(table, heatmap_path, dt=dt, solvers=args.solvers, mus=args.mus)
            print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
