"""Re-render Track E figures from saved results with report terminology (repo code untouched)."""
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
EXP = ROOT + r"\implementation\experiments\sindy_epidemic"
OUT = ROOT + r"\report\figures\track_e"
sys.path.insert(0, EXP)
sys.path.insert(0, ROOT + r"\implementation")
os.chdir(EXP)


def load_patched(name, replacements):
    src = open(os.path.join(EXP, name + ".py"), encoding="utf8").read()
    for old, new in replacements:
        assert old in src, (name, old)
        src = src.replace(old, new)
    mod = types.ModuleType(name + "_patched")
    mod.__file__ = os.path.join(EXP, name + ".py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod



ep = load_patched("run_epidemic_fit", [
    ("docs/10 \u201chealthy\u201d: spike ratio in the low tens", "healthy SIR runs: spike ratio in the low tens"),
    ("Time-scale sweep on the empirical outbreak", "Time-scale sweep on the synthetic outbreak curve"),
    ("desc += f\", vanish_dim {cfg['vanish_dim']}\"", "desc += \", vanishing gate on I\""),
    ("\"+ vanish_dim 0\"", "\"+ vanishing gate\""),
    # the off-scale arm is the time-scale-only one (C_FIT), not the projection arm
    ("ha=\"right\", fontsize=8, color=C_ALT)", "ha=\"right\", fontsize=8, color=C_FIT)"),
    ("\"plain arm continues to 6.2 →\"", "\"time-scale-only arm continues to 6.2 ↑\""),
    ("desc += f\", {cfg['num_epochs']:,} ep\"", "desc += f\", {cfg['num_epochs']:,} epochs\""),
])
runs = ep.load_runs(os.path.join(ep.ensure_results_dir(), ep.RUNS_SUBDIR))
ep.plot_time_scale_sweep(runs, os.path.join(OUT, "epidemic_time_scale_sweep.png"))
ep.plot_epidemic_fit(runs, "full", os.path.join(OUT, "real_epidemic_fit.png"))

sn = load_patched("run_sindy_noise", [
    ("on the Phase-2 noise-sweep checkpoints", "on the noise-sweep KAN-ODE models"),
])
results = sn.run_sweep()
sn.plot_pruning(results, os.path.join(OUT, "pruning_degradation.png"))
print("done")
