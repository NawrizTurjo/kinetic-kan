"""
Track D -- "hyperspace" landscape plot: the learned vector field's magnitude
|f_theta(theta, omega)| rendered as an actual 3D terrain (blue valleys = near-
zero field = equilibria, red peaks = large field), with the real (theta, omega)
trajectory drawn as a path descending across that terrain.

This replaces the flat grey reference-plane approach in plot_trajectories.py's
phase_portrait_3d panel for one specific reason: a flat plane only tells you
the equilibrium's (theta, omega) coordinates, not the SHAPE of the landscape
around it -- it can't show that mu=2.0's KAN learned a landscape with TWO
basins (the true equilibrium at (0,0) and a second, spurious one nearby) and
that all four solvers' trajectories descend into the wrong one. A real terrain
surface makes that structure directly visible, matching the loss-landscape
style visualizations of an optimizer descending into the wrong basin.

Requires a checkpoint.pt saved by run_sweep.py's --save-checkpoints flag
(seed_test.py / kaggle_seed_test.py now save this too, per the earlier fix) --
without one there is no model to evaluate the field from, so this script (unlike
plot_trajectories.py) is NOT partial-data safe: it errors on a missing
checkpoint rather than drawing a placeholder, since there's nothing meaningful
to draw at all without the model.

Usage: python plot_landscape.py --solver euler --mu 2.0
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from kan import KAN  # noqa: E402


def _cell_dir(root, solver, mu, dt):
    return os.path.join(root, "probe", f"{solver}_mu{mu}_dt{dt}")


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = KAN(
        layers_hidden=ckpt["layers_hidden"], grid_len=ckpt["grid_len"], grid_lims=ckpt["grid_lims"],
        basis_func=ckpt["basis_func"], normalizer=ckpt["normalizer"], base_act=ckpt["base_act"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def field_magnitude_grid(model, theta_range, omega_range, n=80):
    """Evaluate |f_theta(theta, omega)| on an n x n grid -- this IS the terrain
    height, so a converged model (f=0 only at the true equilibrium) produces a
    single clean valley, while a trapped model (f=0 at some OTHER point too)
    produces a second valley exactly where the spurious equilibrium sits."""
    th = np.linspace(*theta_range, n)
    om = np.linspace(*omega_range, n)
    TH, OM = np.meshgrid(th, om)
    pts = torch.tensor(np.stack([TH.ravel(), OM.ravel()], axis=-1), dtype=torch.float32)
    with torch.no_grad():
        f = model(pts).numpy()
    mag = np.linalg.norm(f, axis=-1).reshape(TH.shape)
    return TH, OM, mag


# Fixed across EVERY landscape plot in this module (not auto-scaled per
# panel) -- measured as the true max of log1p(|f_theta|) over all 4 solvers x
# 6 mu's checkpoints on a generous sampling box. Without this, each panel's
# colormap independently stretched to fill red-to-blue over ITS OWN min/max,
# so a mu=8.0 field (magnitudes ~1000x smaller than mu=0.1's at the same
# points -- checked directly against the checkpoints) looked just as "hot"
# as mu=0.1's, hiding the actual difference the whole plot exists to show.
GLOBAL_VMIN = 0.0
GLOBAL_VMAX = 2.9


def render_landscape(ax, terrain_model, trajs_by_solver, mu, terrain_solver,
                      solver_colors=None, pad_frac=0.35, n_grid=90,
                      vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX):
    """
    Draw the terrain from ONE model (terrain_solver's checkpoint) and overlay
    EVERY solver's own (theta, omega) trajectory on top of it, each in its own
    color. One shared terrain (not 4 separate ones) because plot_trajectories.py
    needs a single panel comparable to its other 3 (one image per mu, all
    solvers overlaid) -- and per docs/16 section 3, all 4 solvers converged to
    the SAME wrong trajectory under a shared seed, so one representative
    field's basin structure is a reasonable stand-in for "the landscape at this
    mu", not just an arbitrary pick. Returns the surface artist (for a caller
    that wants to add its own colorbar).
    """
    # mplot3d's DEFAULT depth-sorting (computed_zorder=True) approximates
    # occlusion per-artist using each artist's mean depth -- it does not do
    # real per-pixel z-buffering, so a thin line only slightly above a bumpy
    # surface gets misjudged as "behind" nearby surface faces and vanishes
    # (the "hidden behind the plot" symptom). Turning this off makes mplot3d
    # just use the explicit zorder values set below (surface=0, paths=19-21,
    # star=24-25) directly, which is reliable. Needs matplotlib >= 3.5.
    ax.computed_zorder = False

    if solver_colors is None:
        solver_colors = {}
    all_theta, all_omega = [np.array([0.0])], [np.array([0.0])]
    for d in trajs_by_solver.values():
        all_theta += [d["y_pred"][:, 0], d["y_true"][:, 0]]
        all_omega += [d["y_pred"][:, 1], d["y_true"][:, 1]]
    th_all, om_all = np.concatenate(all_theta), np.concatenate(all_omega)
    th_lo, th_hi = th_all.min(), th_all.max()
    om_lo, om_hi = om_all.min(), om_all.max()
    pad_th = pad_frac * (th_hi - th_lo or 1); pad_om = pad_frac * (om_hi - om_lo or 1)
    theta_range = (th_lo - pad_th, th_hi + pad_th)
    omega_range = (om_lo - pad_om, om_hi + pad_om)

    TH, OM, mag = field_magnitude_grid(terrain_model, theta_range, omega_range, n=n_grid)
    # log1p compresses the large-field region so the near-zero valleys (the
    # actual point of this plot) aren't flattened into invisibility next to a
    # much taller peak elsewhere in the grid.
    Z = np.log1p(mag)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    surf = ax.plot_surface(TH, OM, Z, cmap="RdYlBu_r", norm=norm, alpha=0.85, linewidth=0.15,
                            edgecolor="#00000022", antialiased=True, rcount=n_grid, ccount=n_grid, zorder=0)
    ax.set_zlim(vmin, vmax)

    # mplot3d does NOT have true depth-buffering between a surface and line
    # plots -- it approximates z-order per artist, so a path only ~0.05 above
    # the mesh (the first version's lift) was routinely painted BEHIND nearby
    # surface faces, making it look "hidden behind the plot" even at full
    # opacity/thickness. A lift proportional to the actual z-range (not a tiny
    # fixed constant) keeps the path unambiguously above the surrounding mesh
    # regardless of how tall this particular terrain's peaks are.
    z_lift = 0.06 * (Z.max() - Z.min())

    def _z_at(theta, omega):
        """Height of a path on the surface -- looked up from the SAME field
        evaluation the terrain itself is made of (not re-derived from a smooth
        interpolation), so any path visibly sits ON the terrain."""
        with torch.no_grad():
            f = terrain_model(torch.tensor(np.stack([theta, omega], axis=-1), dtype=torch.float32)).numpy()
        return np.log1p(np.linalg.norm(f, axis=-1))

    # A fixed arrow length relative to the plotted (theta, omega) extent --
    # NOT proportional to each raw step's size -- so arrows stay small, crisp
    # direction markers rather than growing into the big scribbly cones a
    # thick solid line (the previous version) produced.
    arrow_len = 0.035 * max(theta_range[1] - theta_range[0], omega_range[1] - omega_range[0])

    for solver, d in trajs_by_solver.items():
        theta_traj, omega_traj = d["y_pred"][:, 0], d["y_pred"][:, 1]
        color = solver_colors.get(solver, "#1a1a2e")
        z_path = _z_at(theta_traj, omega_traj) + z_lift

        # Black HALO underneath the colored line -- the terrain's RdYlBu_r
        # colormap is red/orange over most of its area (only the valleys near
        # an equilibrium are blue/light), so a solver's own red or orange
        # trajectory line was routinely disappearing into a same-colored patch
        # of terrain. A slightly thicker black line drawn first, then the
        # thinner colored line on top, keeps the path visible against ANY
        # terrain color underneath it.
        ax.plot(theta_traj, omega_traj, z_path, color="black", lw=3.2, alpha=1.0, zorder=19)
        ax.plot(theta_traj, omega_traj, z_path, color=color, lw=1.8, alpha=1.0, zorder=20,
                label=f"{solver} ({d.get('verdict', '?')})")

        n_pts = len(theta_traj)
        n_arrows = min(8, max(1, n_pts // 20))
        arrow_idx = np.linspace(0, n_pts - 2, n_arrows, dtype=int)
        for i in arrow_idx:
            dx, dy, dz = (theta_traj[i + 1] - theta_traj[i], omega_traj[i + 1] - omega_traj[i],
                          z_path[i + 1] - z_path[i])
            norm = np.sqrt(dx * dx + dy * dy) or 1.0
            ax_, ay_, az_ = dx / norm * arrow_len, dy / norm * arrow_len, dz
            # Same halo trick for the arrows: black cone slightly larger,
            # drawn first, colored cone on top and slightly smaller.
            ax.quiver(theta_traj[i], omega_traj[i], z_path[i], ax_, ay_, az_,
                      color="black", arrow_length_ratio=0.55, linewidth=3.6, zorder=20)
            ax.quiver(theta_traj[i], omega_traj[i], z_path[i], ax_, ay_, az_,
                      color=color, arrow_length_ratio=0.5, linewidth=2.0, zorder=21)

        ax.scatter(theta_traj[0], omega_traj[0], z_path[0], color=color, s=55, marker="o",
                   edgecolor="black", linewidth=1.2, zorder=21)

    z_true_eq = _z_at(np.array([0.0]), np.array([0.0]))[0] + z_lift
    # Star drawn as two layers -- a fat white halo underneath the black star --
    # so it stays visible even sitting right on top of a trajectory endpoint
    # or a dark-red terrain peak, both of which swallowed a plain black marker.
    ax.scatter(0, 0, z_true_eq, color="white", s=260, marker="*", zorder=24)
    ax.scatter(0, 0, z_true_eq, color="black", s=140, marker="*", edgecolor="black",
               linewidth=0.8, zorder=25, label="true equilibrium (0,0)")

    ax.set_xlabel("theta"); ax.set_ylabel("omega"); ax.set_zlabel("log(1+|f_theta|)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#cccccc")
    ax.view_init(elev=38, azim=-60)
    return surf


def plot_landscape(checkpoint_path, trajectory_path, out_path, solver, mu, pad_frac=0.35, n_grid=90):
    model = load_model(checkpoint_path)
    traj = dict(np.load(trajectory_path))
    metrics_path = os.path.join(os.path.dirname(trajectory_path), "metrics.json")
    if os.path.isfile(metrics_path):
        import json
        with open(metrics_path) as f:
            traj["verdict"] = json.load(f).get("verdict", "?")

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = render_landscape(ax, model, {solver: traj}, mu, solver,
                             solver_colors={solver: "#1a1a2e"}, pad_frac=pad_frac, n_grid=n_grid)

    ax.set_title(f"Learned vector-field landscape -- {solver}, mu={mu}\n"
                 f"(valleys = equilibria; path shows where training actually ended up)",
                 fontsize=12, fontweight="bold", pad=14)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="log(1+|f_theta(theta,omega)|)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95), fontsize=8, framealpha=0.9)

    fig.patch.set_facecolor("white")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# No red or orange here on purpose -- the terrain's RdYlBu_r colormap uses
# red/orange for its high-field (peak) regions, which is most of the plotted
# area, so a red or orange trajectory line was blending straight into the
# terrain under it even with the black halo behind it. Blue/green/purple/cyan
# don't have that collision.
SOLVER_COLORS = {"euler": "#1f77b4", "midpoint": "#2ca02c", "rk4": "#9467bd", "tsit5": "#17becf"}


def plot_landscape_grid(root, mu, solvers, out_path, dt=0.05, n_grid=70):
    """
    Small-multiples version: ONE subplot per solver, each drawn from that
    solver's OWN checkpoint -- unlike render_landscape's shared-terrain mode
    (used when only some solvers have a saved model), this makes no assumption
    that different solvers' learned fields look alike. A solver missing its own
    checkpoint gets a "no checkpoint" placeholder instead of being silently
    dropped or drawn on a borrowed terrain.
    """
    n = len(solvers)
    ncols = min(n, 2)
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(7.5 * ncols, 6.5 * nrows))

    for i, solver in enumerate(solvers):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        cell_dir = _cell_dir(root, solver, mu, dt)
        ckpt_path = os.path.join(cell_dir, "checkpoint.pt")
        traj_path = os.path.join(cell_dir, "trajectory.npz")
        metrics_path = os.path.join(cell_dir, "metrics.json")
        if not (os.path.isfile(ckpt_path) and os.path.isfile(traj_path)):
            ax.text2D(0.5, 0.5, f"{solver}\nno checkpoint", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(solver, fontsize=12, fontweight="bold")
            continue

        model = load_model(ckpt_path)
        traj = dict(np.load(traj_path))
        if os.path.isfile(metrics_path):
            import json
            with open(metrics_path) as f:
                traj["verdict"] = json.load(f).get("verdict", "?")
        surf = render_landscape(ax, model, {solver: traj}, mu, solver,
                                 solver_colors={solver: SOLVER_COLORS.get(solver, "#1a1a2e")}, n_grid=n_grid)
        ax.set_title(f"{solver} (own field) -- {traj.get('verdict', '?')}", fontsize=12, fontweight="bold", pad=8)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, aspect=20)

    fig.suptitle(f"Learned vector-field landscape per solver, mu={mu} -- EACH panel uses that "
                 f"solver's own trained model (not shared)", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Track D vector-field landscape plot")
    parser.add_argument("--solver", help="Single-solver mode (mutually exclusive with --grid)")
    parser.add_argument("--grid", action="store_true",
                        help="Small-multiples mode: one subplot per solver in SOLVER_COLORS' order, "
                             "each from its OWN checkpoint -- use this instead of --solver whenever "
                             "comparing solvers, since --solver alone only ever shows one model.")
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                          "results", "phase3", "stiffness_map", "probe_traj"))
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                             "results", "phase3", "stiffness_map", "figures"))
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    if args.grid:
        out_path = os.path.join(os.path.abspath(args.out_dir), f"landscape_grid_mu{args.mu}.png")
        plot_landscape_grid(root, args.mu, list(SOLVER_COLORS.keys()), out_path, dt=args.dt)
        return

    if not args.solver:
        raise SystemExit("Pass --solver <name> or --grid.")
    cell_dir = _cell_dir(root, args.solver, args.mu, args.dt)
    checkpoint_path = os.path.join(cell_dir, "checkpoint.pt")
    trajectory_path = os.path.join(cell_dir, "trajectory.npz")
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f"No checkpoint.pt for {args.solver} mu={args.mu} at {checkpoint_path} "
                          f"-- re-run with --save-checkpoints first.")
    out_path = os.path.join(os.path.abspath(args.out_dir), f"landscape_{args.solver}_mu{args.mu}.png")
    plot_landscape(checkpoint_path, trajectory_path, out_path, args.solver, args.mu)


if __name__ == "__main__":
    main()
