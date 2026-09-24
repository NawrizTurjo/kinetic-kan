"""
Compile the TikZ/pgfplots diagrams in report/figures/tikz/ to PDF.

Each .tex there is a standalone document (\\documentclass{standalone}); the
report includes the resulting PDFs. pdflatex (MiKTeX or TeX Live) is used when
it is on PATH, otherwise the bundled report/tools/tectonic.exe.

    kan_ode_system.tex         Figure 1   training loop of a KAN-ODE
    kdense_layer.tex           Figure 2   the KDense dual-branch layer
    data_split.tex             Section 2  training / scored / far-horizon windows
    rk_stages.tex              Section 3  stage structure of the explicit RK methods
    local_vs_global_basis.tex  Section 3  local (B-spline) vs global (Chebyshev) support

Only local_vs_global_basis.tex draws a function of the data; it evaluates the
basis formulas analytically in pgfplots, so no Python output is needed.

Usage:  python report/reproduce/build_tikz.py [name ...]
"""

import shutil
import subprocess
import sys
from pathlib import Path

TIKZ = Path(__file__).resolve().parents[1] / "figures" / "tikz"
TECTONIC = Path(__file__).resolve().parents[1] / "tools" / "tectonic.exe"


def compile_one(tex):
    if shutil.which("pdflatex"):
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name]
    elif TECTONIC.exists():
        cmd = [str(TECTONIC), tex.name]
    else:
        sys.exit("neither pdflatex nor report/tools/tectonic.exe is available")
    r = subprocess.run(cmd, cwd=TIKZ, capture_output=True, text=True)
    for ext in (".aux", ".log"):
        (TIKZ / (tex.stem + ext)).unlink(missing_ok=True)
    ok = r.returncode == 0 and (TIKZ / f"{tex.stem}.pdf").exists()
    print(f"  {'ok  ' if ok else 'FAIL'} figures/tikz/{tex.stem}.pdf")
    if not ok:
        print(r.stdout[-2000:])
    return ok


def main():
    names = sys.argv[1:]
    texs = [TIKZ / f"{n.removesuffix('.tex')}.tex" for n in names] if names else sorted(TIKZ.glob("*.tex"))
    results = [compile_one(t) for t in texs]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
