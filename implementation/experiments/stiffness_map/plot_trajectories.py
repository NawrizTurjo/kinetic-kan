"""
Track D -- trajectory visualization: 4 panels per mu (theta(t), omega(t), phase
portrait, prediction error), all 4 rendered in 3D (see the _panel_*_3d
docstrings for what each one's third axis is and why), each panel overlaying
every solver that has a saved trajectory.npz for that mu. Then a per-mu
4-across collage, then a 6(mu)x4(panel) grid of all of them together.

Requires trajectory.npz files written by run_sweep.py's --save-trajectories flag
(see README.md "How to run" / "Stopping and resuming") -- the original 24-cell
probe sweep did NOT save these, so a cell must be re-run with that flag before it
can appear here. Missing cells are skipped with a visible placeholder, not a
crash -- this script is meant to be re-run as more trajectories land, showing
partial results.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection
import numpy as np

SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
MU_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
# Color carries SOLVER identity (so lines stay distinguishable even when every
# solver at this mu shares the same verdict, e.g. mu=2.0 -- coloring by verdict
# alone made every line identical there, which is confusing, not illustrative).
# Verdict is instead carried by MARKER shape + line style, both drawn from the
# same VERDICT_* dicts everywhere in this module, so "what does this line mean"
# has one consistent answer across all 4 panels.
# Kept identical to plot_landscape.py's palette (imported there too, for the
# phase_portrait_3d panel) so a solver's color means the same thing across
# every panel of the same figure -- no red/orange, since those collide with
# the landscape terrain's own red/orange peak coloring.
SOLVER_COLORS = {"euler": "#1f77b4", "midpoint": "#2ca02c", "rk4": "#9467bd", "tsit5": "#17becf"}
VERDICT_LINESTYLE = {"converged": "-", "unstable": (0, (5, 2)), "diverged": (0, (1, 1.3))}
VERDICT_MARKER = {"converged": "o", "unstable": "X", "diverged": "v"}
PANEL_NAMES = ["theta_t_3d", "omega_t_3d", "phase_portrait_3d", "error_t_3d"]


def _cell_dir(root, solver, mu, dt):
    return os.path.join(root, "probe", f"{solver}_mu{mu}_dt{dt}")


def load_available_trajectories(root, mu, dt, solvers=SOLVERS):
    """Returns {solver: npz_dict} for every solver that has a trajectory.npz for
    this mu, with that cell's own "verdict" (converged/unstable/diverged, read
    from the metrics.json saved alongside it) merged in as an extra key -- the
    single fact everything else in this module is now drawn to illustrate.
    Silently omits (not errors on) solvers that haven't been run yet."""
    out = {}
    for solver in solvers:
        cell_dir = _cell_dir(root, solver, mu, dt)
        traj_path = os.path.join(cell_dir, "trajectory.npz")
        metrics_path = os.path.join(cell_dir, "metrics.json")
        if os.path.isfile(traj_path):
            d = dict(np.load(traj_path))
            if os.path.isfile(metrics_path):
                with open(metrics_path) as f:
                    d["verdict"] = json.load(f).get("verdict", "unstable")
            else:
                d["verdict"] = "unstable"
            ckpt_path = os.path.join(cell_dir, "checkpoint.pt")
            d["checkpoint_path"] = ckpt_path if os.path.isfile(ckpt_path) else None
            out[solver] = d
    return out


def _style_2d_axes(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)


def _curtain(ax, x, y_const, z, z0, color, alpha=0.18):
    """Fill a translucent vertical ribbon ("curtain") from the baseline z0 up to
    the curve z, at fixed y=y_const -- turns a bare 3D line (which reads as a
    thin wire floating in empty space) into an actual surface patch, the way a
    3D waterfall/curtain chart looks. plot_surface needs a 2-row grid: the
    bottom row sits at z0, the top row traces the real curve."""
    X = np.vstack([x, x])
    Y = np.full_like(X, y_const)
    Z = np.vstack([np.full_like(z, z0), z])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False, zorder=1)


def _zero_plane(ax, x_range, y_range, z=0.0, color="#999999", alpha=0.12):
    """A full grey reference plane at height z, spanning x_range x y_range --
    added because a bare z=0 grid LINE (the previous approach) is too easy to
    lose track of once the view is rotated in 3D; a filled plane stays visually
    anchored so it's obvious whether a curve or endpoint has landed above,
    below, or exactly on it, instead of guessing from tick labels."""
    xs, ys = np.meshgrid(x_range, y_range)
    ax.plot_surface(xs, ys, np.full_like(xs, z, dtype=float), color=color, alpha=alpha,
                     linewidth=0, shade=False, zorder=0)


def _style_3d_axes(ax, title):
    """Shared cosmetics for every 3D panel: white (not default grey) panes so
    lines don't get lost against matplotlib's dim 3D background, a light grid,
    a bold title with room for it, and slightly enlarged tick/axis labels to
    stay readable once shrunk into a collage."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#cccccc")
        axis._axinfo["grid"]["color"] = "#e0e0e0"
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(labelsize=8)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10
    # Steeper elevation + a different azimuth than the first pass: the previous
    # (20, -55) view looked nearly edge-on to the curtains added below, so the
    # filled ribbons collapsed to invisible slivers instead of readable planes.
    ax.view_init(elev=32, azim=-50)


def _legend_label(solver, verdict):
    return f"{solver} ({verdict})"


def _panel_theta_t_3d(ax, trajs, mu, fig=None):
    """
    theta(t) as the same waterfall-plus-curtain treatment as omega_t_3d, for
    consistency -- there was never a good reason for theta_t alone to stay a
    flat 2D line plot while its sibling omega_t got the full 3D/curtain
    treatment; both are "one state variable vs t", so both get the same
    picture. The training/extrapolation boundary that theta_t's old 2D version
    marked with a vertical line + shaded span is redrawn here as a dotted
    vertical grid line across all solver rows, at z=0, so it's still visible
    without needing a filled 2D axvspan (which doesn't have a clean 3D analog).
    """
    ref = next(iter(trajs.values()))
    solvers = list(trajs.keys())
    _zero_plane(ax, ref["t_full"][[0, -1]], [-1, len(solvers) - 1])
    ax.plot(ref["t_full"], np.full_like(ref["t_full"], -1), ref["y_true"][:, 0],
            color="black", ls="--", lw=2.2, label="true", zorder=10)
    for i, (solver, d) in enumerate(trajs.items()):
        color = SOLVER_COLORS[solver]
        _curtain(ax, d["t_full"], i, d["y_pred"][:, 0], z0=0.0, color=color)
        ax.plot(d["t_full"], np.full_like(d["t_full"], i), d["y_pred"][:, 0],
                color=color, ls=VERDICT_LINESTYLE[d["verdict"]], lw=2.2, alpha=0.95,
                label=_legend_label(solver, d["verdict"]))
        ax.scatter(d["t_full"][-1], i, d["y_pred"][-1, 0], color=color, s=45,
                   marker=VERDICT_MARKER[d["verdict"]], edgecolor="black", linewidth=0.6,
                   depthshade=False, zorder=11)
    t_end = ref["t_train_end"]
    ax.plot([t_end, t_end], [-1, len(solvers) - 1], [0, 0], color="#aaaaaa", ls=":", lw=1.2)
    ax.set_yticks([-1] + list(range(len(solvers))))
    ax.set_yticklabels(["true"] + solvers)
    ax.set_xlabel("t"); ax.set_ylabel("solver"); ax.set_zlabel("theta")
    _style_3d_axes(ax, f"theta(t), mu={mu}  (curtain area = deviation from 0)")


def _panel_omega_t_3d(ax, trajs, mu, fig=None):
    """
    omega(t) as a waterfall: t on x, solver on y (each solver's curve drawn at
    its own y-depth instead of overlapping every other solver's line at y=0),
    omega on z. The true reference sits at its own front row (y=-1) so it's
    never occluded by a solver curve behind it. Each curve gets a translucent
    curtain dropped to z=0 -- without it a 3D line plot is just a thin wire in
    empty space; the curtain turns it into an actual surface patch, and its
    filled AREA gives a second, more visceral read of oscillation amplitude
    than the bare line alone.
    """
    ref = next(iter(trajs.values()))
    solvers = list(trajs.keys())
    _zero_plane(ax, ref["t_full"][[0, -1]], [-1, len(solvers) - 1])
    ax.plot(ref["t_full"], np.full_like(ref["t_full"], -1), ref["y_true"][:, 1],
            color="black", ls="--", lw=2.2, label="true", zorder=10)
    for i, (solver, d) in enumerate(trajs.items()):
        color = SOLVER_COLORS[solver]
        _curtain(ax, d["t_full"], i, d["y_pred"][:, 1], z0=0.0, color=color)
        ax.plot(d["t_full"], np.full_like(d["t_full"], i), d["y_pred"][:, 1],
                color=color, ls=VERDICT_LINESTYLE[d["verdict"]], lw=2.2, alpha=0.95,
                label=_legend_label(solver, d["verdict"]))
        ax.scatter(d["t_full"][-1], i, d["y_pred"][-1, 1], color=color, s=45,
                   marker=VERDICT_MARKER[d["verdict"]], edgecolor="black", linewidth=0.6,
                   depthshade=False, zorder=11)
    t_end = ref["t_train_end"]
    ax.plot([t_end, t_end], [-1, len(solvers) - 1], [0, 0], color="#aaaaaa", ls=":", lw=1.2)
    ax.set_yticks([-1] + list(range(len(solvers))))
    ax.set_yticklabels(["true"] + solvers)
    ax.set_xlabel("t"); ax.set_ylabel("solver"); ax.set_zlabel("omega")
    _style_3d_axes(ax, f"omega(t), mu={mu}  (curtain area = oscillation size)")


def _panel_phase_portrait_shadow_fallback(ax, trajs, mu, fig=None):
    """Phase portrait (theta, omega) unfolded along time -- a helix-like curve
    through (t, theta, omega). The curtain-to-omega=0 version of this panel
    (first attempt) was hard to read: dropping a filled sheet under a curve
    that itself winds up and down in omega makes a folded, self-overlapping
    surface, not a clean ribbon -- curtains work for the OTHER two 3D panels
    only because those curves are monotonic in one row per solver.

    Fix, borrowed from how Lorenz-attractor plots are conventionally drawn:
    each curve gets a flattened grey SHADOW projected onto the three back
    walls of the bounding box (the (t, theta), (t, omega) and (theta, omega)
    planes) in addition to the real 3D curve. Shadows give the eye the same
    depth cues real 2D panels would (compare directly against the theta_t
    panel's shape, or an implied omega_t view) while the 3D curve itself still
    shows how the two coordinates evolve jointly. The true equilibrium (0, 0)
    is drawn as a black line down the time axis; each solver's FINAL point is
    marked in 3D *and* on its own shadows, so a trapped run's endpoint visibly
    misses the equilibrium on all three walls at once, not just in one view.
    """
    ref = next(iter(trajs.values()))
    t_full = ref["t_full"]
    all_theta = np.concatenate([ref["y_true"][:, 0]] + [d["y_pred"][:, 0] for d in trajs.values()])
    all_omega = np.concatenate([ref["y_true"][:, 1]] + [d["y_pred"][:, 1] for d in trajs.values()])
    t_lo, t_hi = t_full[0], t_full[-1]
    th_lo, th_hi = all_theta.min(), all_theta.max()
    om_lo, om_hi = all_omega.min(), all_omega.max()
    pad_th = 0.08 * (th_hi - th_lo or 1); pad_om = 0.08 * (om_hi - om_lo or 1)
    th_wall, om_wall = th_hi + pad_th, om_lo - pad_om  # back-right and floor walls
    ax.set_xlim(t_lo, t_hi); ax.set_ylim(th_lo - pad_th, th_wall); ax.set_zlim(om_wall, om_hi + pad_om)

    # Full grey equilibrium PLANE at omega=0 (not just the thin line drawn
    # below) -- a bare line is hard to judge distance from once the view is
    # rotated; a filled plane makes "did this endpoint actually land on the
    # equilibrium" a direct, unambiguous visual read.
    _zero_plane(ax, [t_lo, t_hi], [th_lo - pad_th, th_wall], z=0.0)

    def _shadows(t, theta, omega, color, lw, ls, alpha):
        ax.plot(t, theta, np.full_like(t, om_wall), color=color, lw=lw, ls=ls, alpha=alpha, zorder=1)
        ax.plot(t, np.full_like(t, th_wall), omega, color=color, lw=lw, ls=ls, alpha=alpha, zorder=1)
        ax.plot(np.full_like(t, t_lo), theta, omega, color=color, lw=lw, ls=ls, alpha=alpha, zorder=1)

    # Only the TRUE trajectory gets full line shadows (the depth-cue reference);
    # giving every solver its own full shadow set tripled the clutter without
    # adding information once solvers agree closely (e.g. mu=2.0, where all 4
    # land on nearly the same wrong curve) -- solvers instead get just their
    # final-point projections, which is exactly the fact this panel is trying
    # to show (where did each solver actually end up, relative to the truth).
    _shadows(t_full, ref["y_true"][:, 0], ref["y_true"][:, 1], "#aaaaaa", 1.2, "--", 0.55)
    ax.plot(t_full, ref["y_true"][:, 0], ref["y_true"][:, 1],
            color="black", ls="--", lw=2.2, label="true trajectory", zorder=10)
    ax.plot(t_full, np.zeros_like(t_full), np.zeros_like(t_full),
            color="black", lw=1.4, alpha=0.5, label="true equilibrium (0,0)", zorder=5)

    for solver, d in trajs.items():
        color = SOLVER_COLORS[solver]
        ls = VERDICT_LINESTYLE[d["verdict"]]
        ax.plot(t_full, d["y_pred"][:, 0], d["y_pred"][:, 1],
                color=color, ls=ls, lw=2.2, alpha=0.95, label=_legend_label(solver, d["verdict"]))
        marker = VERDICT_MARKER[d["verdict"]]
        end = (t_full[-1], d["y_pred"][-1, 0], d["y_pred"][-1, 1])
        ax.scatter(*end, color=color, s=55, marker=marker, edgecolor="black",
                   linewidth=0.7, depthshade=False, zorder=11)
        ax.scatter(end[0], end[1], om_wall, color=color, s=25, marker=marker, alpha=0.6, zorder=2)
        ax.scatter(end[0], th_wall, end[2], color=color, s=25, marker=marker, alpha=0.6, zorder=2)
        ax.scatter(t_lo, end[1], end[2], color=color, s=25, marker=marker, alpha=0.6, zorder=2)

    ax.set_xlabel("t"); ax.set_ylabel("theta"); ax.set_zlabel("omega")
    _style_3d_axes(ax, f"phase portrait, mu={mu}  (grey/faint = wall shadows)")


def _panel_phase_portrait_3d(ax, trajs, mu, fig=None):
    """
    FALLBACK ONLY, used when fewer than 2 solvers have a checkpoint (see
    make_panel_images -- when 2+ solvers have their own checkpoint, it calls
    plot_landscape.plot_landscape_grid instead, one terrain PER solver, and
    this function is never reached for that mu). With 0 or 1 checkpoints
    there is nothing to usefully grid, so this either falls back further to
    the wall-shadow phase portrait (0 checkpoints) or renders a single
    solver's real terrain with every solver's trajectory overlaid on it (1
    checkpoint) -- clearly labelled as whose field it is, not silently
    implying it's shared ground truth for solvers that never had their own
    checkpoint trained.
    """
    solver_with_ckpt = next((s for s, d in trajs.items() if d.get("checkpoint_path")), None)
    if solver_with_ckpt is None:
        _panel_phase_portrait_shadow_fallback(ax, trajs, mu, fig=fig)
        return

    from plot_landscape import load_model, render_landscape
    model = load_model(trajs[solver_with_ckpt]["checkpoint_path"])
    surf = render_landscape(ax, model, trajs, mu, solver_with_ckpt, solver_colors=SOLVER_COLORS)
    ax.set_title(f"phase portrait, mu={mu}  (terrain from {solver_with_ckpt}'s learned field)",
                 fontsize=12, fontweight="bold", pad=14)
    if fig is not None:
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.12, aspect=25, label="log(1+|f_theta|)")


def _panel_error_t_3d(ax, trajs, mu, fig=None):
    """Prediction error as a waterfall: t on x, solver on y, log10(error) on z --
    same rationale as omega_t_3d, so different solvers' error curves (which
    otherwise overlap heavily on a shared log-y axis) are visually separated.
    Each curve gets a curtain dropped to a shared floor, plus a translucent
    green reference PLANE at z=log10(1e-3) -- converged runs' curtains end
    below it, unstable/diverged ones visibly poke through."""
    t_full = next(iter(trajs.values()))["t_full"]
    all_log_err = []
    for d in trajs.values():
        err = np.linalg.norm(d["y_pred"] - d["y_true"], axis=-1)
        all_log_err.append(np.log10(np.clip(err, 1e-12, None)))
    z_floor = np.floor(min(a.min() for a in all_log_err)) - 0.5

    xs, ys = np.meshgrid(t_full[[0, -1]], [-0.5, len(trajs) - 0.5])
    ax.plot_surface(xs, ys, np.full_like(xs, -3, dtype=float), color="#2ca02c", alpha=0.10,
                     linewidth=0, shade=False, zorder=0)

    for i, (solver, log_err) in enumerate(zip(trajs.keys(), all_log_err)):
        color = SOLVER_COLORS[solver]
        d = trajs[solver]
        _curtain(ax, d["t_full"], i, log_err, z0=z_floor, color=color)
        ax.plot(d["t_full"], np.full_like(d["t_full"], i), log_err,
                color=color, ls=VERDICT_LINESTYLE[d["verdict"]], lw=2.2, alpha=0.95,
                label=_legend_label(solver, d["verdict"]))
    ax.set_yticks(range(len(trajs)))
    ax.set_yticklabels(list(trajs.keys()))
    ax.set_xlabel("t"); ax.set_ylabel("solver"); ax.set_zlabel("log10|pred - true|")
    _style_3d_axes(ax, f"error(t), mu={mu}  (green plane = 1e-3 'good fit' line)")


_PANEL_FUNCS = {
    "theta_t_3d": _panel_theta_t_3d,
    "omega_t_3d": _panel_omega_t_3d,
    "phase_portrait_3d": _panel_phase_portrait_3d,
    "error_t_3d": _panel_error_t_3d,
}
_PANEL_IS_3D = {name: True for name in PANEL_NAMES}


def make_panel_images(root, mu, dt, out_dir, solvers=SOLVERS):
    """Write the 4 individual panel PNGs for one mu. Returns their paths in
    PANEL_NAMES order, or None for a panel if there was no data at all to plot
    (i.e. zero solvers have run this mu yet)."""
    trajs = load_available_trajectories(root, mu, dt, solvers)
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    n_with_ckpt = sum(1 for d in trajs.values() if d.get("checkpoint_path"))
    for name in PANEL_NAMES:
        if name == "phase_portrait_3d" and n_with_ckpt >= 2:
            # Every solver has its OWN checkpoint here -- render one terrain
            # PER solver (plot_landscape.py's small-multiples grid) instead of
            # picking one solver's field as a stand-in for all of them. The
            # single-shared-terrain dispatcher below is kept only for the case
            # where fewer than 2 solvers have a checkpoint (nothing to
            # meaningfully grid), not as the default once real per-solver data
            # exists -- see the "why are landscapes merged" discussion this
            # replaced.
            import plot_landscape as pl
            out_path = os.path.join(out_dir, f"phase_portrait_3d_mu{mu}.png")
            pl.plot_landscape_grid(root, mu, solvers, out_path, dt=dt)
            paths.append(out_path)
            continue

        out_path = os.path.join(out_dir, f"{name}_mu{mu}.png")
        if _PANEL_IS_3D[name]:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if trajs:
            _PANEL_FUNCS[name](ax, trajs, mu, fig=fig)
            legend_kwargs = dict(fontsize=8, framealpha=0.95, edgecolor="#cccccc")
            if name == "phase_portrait_3d":
                # This panel (when it renders as the landscape) already has a
                # colorbar sitting where the other 3D panels' side legend goes
                # -- putting the legend there too was an unreadable overlap.
                # Below the axes instead, spread across columns.
                ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.02), ncol=3, **legend_kwargs)
            elif _PANEL_IS_3D[name]:
                # Outside the axes entirely (not just anchored inside the corner)
                # -- an inside-corner legend was overlapping the curve itself for
                # panels like phase_portrait_3d where the trajectory sweeps
                # through the upper-left region of the view.
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), **legend_kwargs)
            else:
                ax.legend(**legend_kwargs)
        else:
            ax.text(0.5, 0.5, f"no data yet\n(mu={mu})", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        fig.patch.set_facecolor("white")
        if _PANEL_IS_3D[name]:
            # bbox_inches="tight" (not fig.tight_layout(), which mplot3d doesn't
            # support well) so the legend placed just outside the 3D axes above
            # isn't clipped off the saved image.
            fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
        else:
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, facecolor="white")
        plt.close(fig)
        paths.append(out_path)
    missing = [s for s in solvers if s not in trajs]
    if missing:
        print(f"mu={mu}: {len(trajs)}/{len(solvers)} solvers plotted (missing: {missing})")
    else:
        print(f"mu={mu}: all {len(solvers)} solvers plotted")
    return paths


def make_collage(panel_paths, out_path, mu):
    """Paste 4 already-rendered panel PNGs side by side into one row image."""
    fig, axes = plt.subplots(1, len(panel_paths), figsize=(5.5 * len(panel_paths), 4.6))
    if len(panel_paths) == 1:
        axes = [axes]
    for ax, p in zip(axes, panel_paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"μ = {mu}", fontsize=15, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def make_full_grid(collage_paths_by_mu, out_path, mus=MU_VALUES):
    """Stack the per-mu collage rows into one 6(mu) x 4(panel) grid image."""
    fig, axes = plt.subplots(len(mus), 1, figsize=(20, 4.2 * len(mus)))
    if len(mus) == 1:
        axes = [axes]
    for ax, mu in zip(axes, mus):
        p = collage_paths_by_mu.get(mu)
        ax.axis("off")
        if p and os.path.isfile(p):
            ax.imshow(plt.imread(p))
        else:
            ax.text(0.5, 0.5, f"mu={mu}: no data yet", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Track D trajectory plots (partial-data safe)")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                          "results", "phase3", "stiffness_map", "probe_traj"))
    parser.add_argument("--mus", nargs="+", type=float, default=MU_VALUES)
    parser.add_argument("--solvers", nargs="+", default=SOLVERS)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                             "results", "phase3", "stiffness_map", "figures"))
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir)

    collage_paths = {}
    for mu in args.mus:
        panel_dir = os.path.join(out_dir, f"mu{mu}")
        panel_paths = make_panel_images(root, mu, args.dt, panel_dir, args.solvers)
        collage_path = make_collage(panel_paths, os.path.join(panel_dir, "collage.png"), mu)
        collage_paths[mu] = collage_path
        print(f"  collage: {collage_path}")

    grid_path = make_full_grid(collage_paths, os.path.join(out_dir, "stiffness_trajectories_grid.png"), args.mus)
    print(f"Full grid: {grid_path}")


if __name__ == "__main__":
    main()
