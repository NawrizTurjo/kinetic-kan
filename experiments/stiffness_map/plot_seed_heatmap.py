"""
Multi-seed aggregate heatmaps for Track D -- built once all 4 solvers had
3-seed data in results/phase3/stiffness_map/_seed_test/ (euler locally,
midpoint/rk4/tsit5 from the Kaggle run). Per docs/16_p3_stiffness_map_findings.md
section 5, a single seed's verdict is not reported on its own anymore -- median
best_train_mse and converged_fraction across seeds are the two adopted summary
statistics, so this plots exactly those two, solver (columns) x mu (rows).

Reads euler's cells from BOTH naming schemes (legacy "seed<seed>_mu<mu>" with no
solver suffix, and "seed<seed>_mu<mu>_euler" if that ever gets created) --
see seed_test.py's docstring for why the legacy euler folders have no suffix.

Usage: python plot_seed_heatmap.py
Writes results/phase3/stiffness_map/figures/seed_median_mse_heatmap.png and
seed_converged_fraction_heatmap.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOLVERS = ["euler", "midpoint", "rk4", "tsit5"]
MUS = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
SEEDS = [42, 1, 7]

HERE = os.path.dirname(__file__)
STIFFNESS_ROOT = os.path.join(HERE, "..", "..", "results", "phase3", "stiffness_map")
SEED_ROOT = os.path.join(STIFFNESS_ROOT, "_seed_test")
PROBE_ROOT = os.path.join(STIFFNESS_ROOT, "probe")
OUT_DIR = os.path.join(STIFFNESS_ROOT, "figures")


def _fmt_mu(mu):
    return f"{int(mu)}.0" if mu == int(mu) else str(mu)


def _cell_path(solver, mu, seed):
    """
    seed=42 is run_sweep.py's default and was never re-run into _seed_test (only
    the two EXTRA seeds, 1 and 7, live there) -- its data is the original
    single-seed sweep already sitting in probe/, so that's where seed=42 is read
    from. Seeds 1 and 7 come from _seed_test/, with euler's legacy no-suffix
    folder name (see seed_test.py's docstring) checked as a fallback.
    """
    mu_str = _fmt_mu(mu)
    if seed == 42:
        return _existing(os.path.join(PROBE_ROOT, f"{solver}_mu{mu_str}_dt0.05", "metrics.json"))
    candidates = [os.path.join(SEED_ROOT, f"seed{seed}_mu{mu_str}_{solver}", "metrics.json")]
    if solver == "euler":
        candidates.append(os.path.join(SEED_ROOT, f"seed{seed}_mu{mu_str}", "metrics.json"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _existing(p):
    return p if os.path.isfile(p) else None


def load_aggregates():
    """Returns {(solver, mu): {"median_mse": float|None, "converged_fraction": float|None,
    "n_seeds": int}}."""
    out = {}
    for solver in SOLVERS:
        for mu in MUS:
            mses, verdicts = [], []
            for seed in SEEDS:
                p = _cell_path(solver, mu, seed)
                if p is None:
                    continue
                with open(p) as f:
                    m = json.load(f)
                mses.append(m["best_train_mse"])
                verdicts.append(m["verdict"])
            if mses:
                out[(solver, mu)] = {
                    "median_mse": float(np.median(mses)),
                    "converged_fraction": sum(v == "converged" for v in verdicts) / len(verdicts),
                    "n_seeds": len(mses),
                }
            else:
                out[(solver, mu)] = {"median_mse": None, "converged_fraction": None, "n_seeds": 0}
    return out


def _text_color(rgba):
    """White text on dark cells, black text on light cells -- perceptual luminance,
    not a flat threshold, so it holds up across very different colormaps (viridis
    vs. RdYlGn)."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "black"


def _heatmap(agg, field, title, out_path, cmap, fmt, vmin=None, vmax=None, log=False,
             cbar_label=None, cbar_ticks=None, cbar_ticklabels=None):
    grid = np.full((len(MUS), len(SOLVERS)), np.nan)
    annot = np.empty((len(MUS), len(SOLVERS)), dtype=object)
    for i, mu in enumerate(MUS):
        for j, solver in enumerate(SOLVERS):
            v = agg[(solver, mu)][field]
            n = agg[(solver, mu)]["n_seeds"]
            if v is None:
                annot[i, j] = "no data"
                continue
            grid[i, j] = np.log10(v) if log else v
            annot[i, j] = f"{fmt(v)}\nn={n}"

    plt.rcParams["font.family"] = "sans-serif"
    n_rows, n_cols = len(MUS), len(SOLVERS)
    fig, ax = plt.subplots(figsize=(2.1 * n_cols + 1.8, 1.35 * n_rows + 1.5))
    cmap_obj = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap_obj.set_bad("#e8e8e8")
    masked = np.ma.masked_invalid(grid)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) \
        else matplotlib.colors.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([s.upper() for s in SOLVERS], fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"μ = {m:g}" for m in MUS], fontsize=12, fontweight="bold")
    ax.tick_params(length=0)

    # Thin white gridlines between cells (drawn via minor ticks) instead of a
    # bare imshow, which otherwise reads as a flat, seamless block of color.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(grid[i, j]):
                color = "#888888"
            else:
                color = _text_color(cmap_obj(norm(grid[i, j])))
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=11,
                    color=color, linespacing=1.6)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=10)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.patch.set_facecolor("white")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    agg = load_aggregates()

    n_missing = sum(1 for v in agg.values() if v["n_seeds"] == 0)
    n_partial = sum(1 for v in agg.values() if 0 < v["n_seeds"] < len(SEEDS))
    print(f"{len(agg)} (solver, mu) cells: {sum(1 for v in agg.values() if v['n_seeds'] == len(SEEDS))} "
          f"full ({len(SEEDS)}/{len(SEEDS)} seeds), {n_partial} partial, {n_missing} missing.")

    all_mse = [v["median_mse"] for v in agg.values() if v["median_mse"] is not None]
    log_lo, log_hi = np.floor(np.log10(min(all_mse))), np.ceil(np.log10(max(all_mse)))
    tick_exponents = np.arange(log_lo, log_hi + 1)
    _heatmap(
        agg, "median_mse", "Median best_train_mse across seeds {42, 1, 7}",
        os.path.join(OUT_DIR, "seed_median_mse_heatmap.png"),
        cmap="viridis_r", fmt=lambda v: f"{v:.2e}", log=True,
        vmin=log_lo, vmax=log_hi,
        cbar_label="median best_train_mse (log scale) — lower is better",
        cbar_ticks=tick_exponents,
        cbar_ticklabels=[f"1e{int(e)}" for e in tick_exponents],
    )
    _heatmap(
        agg, "converged_fraction", "Converged fraction across seeds {42, 1, 7}",
        os.path.join(OUT_DIR, "seed_converged_fraction_heatmap.png"),
        cmap="RdYlGn", fmt=lambda v: f"{v:.2f}", vmin=0, vmax=1,
        cbar_label="fraction of 3 seeds classified 'converged' — higher is better",
    )


if __name__ == "__main__":
    main()
