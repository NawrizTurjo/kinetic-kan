"""Re-render the damping-study figures with report terminology (repo plotting code untouched).

The original trajectory figures were drawn from trajectory.npz files written on the
cloud machines and never saved locally. This script rebuilds those files for the
seed-42 cells at mu = 1.0 and 2.0 by integrating each saved checkpoint with its own
solver, checks that the rebuilt prediction reproduces the cell's recorded training
MSE, and then calls the project's own plotting functions with plain labels:
"not converged" for the classifier's "unstable", and no working notes in titles.
"""
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
EXP = os.path.join(IMPL, "experiments", "stiffness_map")
PROBE = os.path.join(IMPL, "results", "phase3", "stiffness_map", "probe")
OUT = os.path.join(ROOT, "report", "figures", "track_d")
sys.path.insert(0, EXP)
sys.path.insert(0, IMPL)

from ode import NeuralODE  # noqa: E402
from data import generate_damped_pendulum_data  # noqa: E402

SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
MUS_ALL = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
PRELUDE = (
    '_VERD = {"converged": "converged", "unstable": "not converged", "diverged": "diverged", "missing": "missing"}\n'
    '_NAME = {"euler": "Euler", "midpoint": "Midpoint", "rk4": "RK4", "tsit5": "Tsit5"}\n'
)


def load_patched(name, replacements):
    path = os.path.join(EXP, name + ".py")
    src = open(path, encoding="utf8").read()
    for old, new in replacements:
        assert old in src, (name, old)
        src = src.replace(old, new)
    # insert the label maps right after the module docstring's imports
    src = src.replace("\nimport ", "\n" + PRELUDE + "import ", 1)
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


pl = load_patched("plot_landscape", [
    ("ax.set_title(f\"{solver} (own field) -- {traj.get('verdict', '?')}\", ",
     "ax.set_title(f\"{_NAME.get(solver, solver)}: {_VERD.get(traj.get('verdict', '?'), '?')}\", "),
    ("f\"Learned vector-field landscape per solver, mu={mu} -- EACH panel uses that \"",
     "f\"Learned field magnitude per solver, $\\\\mu$ = {mu}; each panel shows that \""),
    ("f\"solver's own trained model (not shared)\"", "f\"solver's own trained model\""),
    ("label=f\"{solver} ({d.get('verdict', '?')})\")",
     "label=f\"{_NAME.get(solver, solver)} ({_VERD.get(d.get('verdict', '?'), '?')})\")"),
])
sys.modules["plot_landscape"] = pl
pt = load_patched("plot_trajectories", [
    ("return f\"{solver} ({verdict})\"", "return f\"{_NAME.get(solver, solver)} ({_VERD.get(verdict, verdict)})\""),
    ("f\"theta(t), mu={mu}  (curtain area = deviation from 0)\"", "f\"angle theta(t), mu = {mu}\""),
    ("f\"omega(t), mu={mu}  (curtain area = oscillation size)\"", "f\"angular velocity omega(t), mu = {mu}\""),
    ("f\"error(t), mu={mu}  (green plane = 1e-3 'good fit' line)\"",
     "f\"log10 |error|(t), mu = {mu}  (green plane: 1e-3)\""),
])
rs = load_patched("run_sweep", [
    ("ax.set_title(f\"Stiffness-Solver Stability Phase Map (dt={dt})\"",
     "ax.set_title(f\"Damping-solver verdict map, seed 42 (dt = {dt}, two substeps)\""),
    ("label = f\"{mse:.1e}\\n{verdicts[i, j]}\"", "label = f\"{mse:.1e}\\n{_VERD.get(verdicts[i, j], verdicts[i, j])}\""),
    ("cbar.set_label(\"best_train_mse (log scale) — lower is better\"", "cbar.set_label(\"best training MSE (log scale)\""),
    ("Patch(facecolor=\"none\", edgecolor=c, linewidth=2.5, label=v)",
     "Patch(facecolor=\"none\", edgecolor=c, linewidth=2.5, label=_VERD.get(v, v))"),
])
sh = load_patched("plot_seed_heatmap", [
    ("seed_heatmap_dir = os.path.join(OUT_DIR, \"seed_heatmaps\")", "seed_heatmap_dir = _OUT_TRACK_D"),
    ("\"Median best_train_mse across seeds {42, 1, 7}\"", "\"Median best training MSE over seeds 42, 1 and 7\""),
    ("cbar_label=\"median best_train_mse (log scale) — lower is better\"", "cbar_label=\"median best training MSE (log scale)\""),
    ("\"Converged fraction across seeds {42, 1, 7}\"", "\"Converged fraction over seeds 42, 1 and 7\""),
    ("cbar_label=\"fraction of 3 seeds classified 'converged' — higher is better\"",
     "cbar_label=\"fraction of the 3 seeds classified as converged\""),
])
sh._OUT_TRACK_D = OUT

# ---- 1. rebuild trajectory.npz for the seed-42 cells at mu = 1.0, 2.0 -------------
tmp_root = tempfile.mkdtemp(prefix="damping_root_")
for mu in (1.0, 2.0):
    data = generate_damped_pendulum_data(mu=mu, t_end=10.0, dt=0.05, t_train_end=5.0)
    n_train = len(data.t_train)
    for solver in SOLVERS:
        cell = f"{solver}_mu{mu}_dt0.05"
        src_dir, dst_dir = os.path.join(PROBE, cell), os.path.join(tmp_root, "probe", cell)
        os.makedirs(dst_dir, exist_ok=True)
        for f in ("checkpoint.pt", "metrics.json"):
            shutil.copy(os.path.join(src_dir, f), dst_dir)
        model = pl.load_model(os.path.join(dst_dir, "checkpoint.pt"))
        with torch.no_grad():
            pred = NeuralODE(func=model, method=solver, substeps=2)(y0=data.y0, t=data.t_full).numpy()
        y = data.y_full.numpy()
        np.savez(os.path.join(dst_dir, "trajectory.npz"), t_full=data.t_full.numpy(), y_true=y, y_pred=pred,
                 t_train_end=5.0, n_train=n_train, mu=mu, solver=solver)
        rec = json.load(open(os.path.join(dst_dir, "metrics.json")))
        got = float(np.mean((pred[:n_train] - y[:n_train]) ** 2))
        print(f"mu={mu} {solver:8s} recorded best train MSE={rec['best_train_mse']:.4e}  rebuilt={got:.4e}  "
              f"verdict={rec['verdict']}")

# ---- 2. trajectory panels -> 2x2 grid, and the per-solver landscape (Figure 14) ----
from PIL import Image  # noqa: E402

panel_dir = os.path.join(tmp_root, "panels")
for mu in (1.0, 2.0):
    paths = pt.make_panel_images(tmp_root, mu, 0.05, os.path.join(panel_dir, f"mu{mu}"), SOLVERS)
    shutil.copy(paths[2], os.path.join(OUT, f"phase_portrait_3d_mu{mu}.png"))
    ims = [Image.open(p).convert("RGB") for p in paths]
    # the landscape panel is rendered much larger than the other three; scale every
    # panel to one common width so the 2x2 grid has no large empty margins
    w_t = min(i.size[0] for i in ims)
    ims = [im.resize((w_t, round(im.size[1] * w_t / im.size[0])), Image.LANCZOS) for im in ims]
    cw, ch = w_t, max(i.size[1] for i in ims)
    grid = Image.new("RGB", (2 * cw, 2 * ch), "white")
    for k, im in enumerate(ims):
        x0 = (k % 2) * cw + (cw - im.size[0]) // 2
        y0 = (k // 2) * ch + (ch - im.size[1]) // 2
        grid.paste(im, (x0, y0))
    grid.save(os.path.join(OUT, f"mu{mu}_trajectory_grid.png"))
    print("wrote grid and landscape for mu", mu)

# ---- 3. verdict heatmap and seed heatmaps -----------------------------------------
empty = tempfile.mkdtemp(prefix="no_full_")
table = rs.build_table5(PROBE, empty, SOLVERS, MUS_ALL, [0.05])
rs.plot_stability_heatmap(table, os.path.join(OUT, "stability_heatmap_dt0.05.png"), 0.05, SOLVERS, MUS_ALL)
sh.main()
shutil.rmtree(tmp_root, ignore_errors=True)
shutil.rmtree(empty, ignore_errors=True)
print("done")
