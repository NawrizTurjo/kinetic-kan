"""Re-render the Phase-2 close-out figures for the report (repo code untouched).

The originals in implementation/results/figures/ label the loss-decay plots
"Extrapolation Test Loss", but they plot the training loss (metric_key =
"train_losses"). This script loads copies of phase2_closeout.py and
utils/plotting.py with the labels corrected and internal task names removed,
writes the figures to report/figures/closeout/, and sends the JSON side outputs to
a temporary folder so the saved results are not overwritten.
"""
import os
import sys
import tempfile
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
OUT = os.path.join(ROOT, "report", "figures", "closeout")
sys.path.insert(0, IMPL)
os.chdir(IMPL)  # phase2_closeout uses paths relative to implementation/


def load_patched(path, name, replacements):
    src = open(path, encoding="utf8").read()
    for old, new in replacements:
        assert old in src, (name, old)
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


plotting = load_patched(os.path.join(IMPL, "utils", "plotting.py"), "plotting_patched", [
    ('ax.set_ylabel("Extrapolation Test Loss (MSE, Log Scale)")', 'ax.set_ylabel("Training loss (MSE, log scale)")'),
])
tmp = tempfile.mkdtemp(prefix="closeout_json_")
pc = load_patched(os.path.join(IMPL, "phase2_closeout.py"), "phase2_closeout_patched", [
    ('FIG_DIR = "results/figures"', f'FIG_DIR = {OUT!r}'),
    ('OUT_DIR = "results/phase2_closeout"', f'OUT_DIR = {tmp!r}'),
    ('"Task 2.3: Extrapolation Horizon Doubled to $t=28$ (8 periods)"',
     '"Extrapolation horizon doubled to $t=28$ (about 8.4 cycles in total)"'),
    ("train | extrapolation | NEW far-extrapolation", "training | near extrapolation | far extrapolation"),
    ("CONFOUNDED: changing", "Note: changing"),
    ("reflects both effects. See docs/05.", "reflects both effects."),
])
pc.plot_model_comparison_curves = plotting.plot_model_comparison_curves
os.makedirs(OUT, exist_ok=True)
pc.task_extrap28()
pc.task_energy()
pc.task_figures()
print("figures written to", OUT)
