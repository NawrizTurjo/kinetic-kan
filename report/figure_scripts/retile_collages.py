"""Re-tile the 1x4 trajectory collages into 2x2 grids so each panel prints twice as large."""
from PIL import Image

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMPL = os.path.join(ROOT, "implementation")
D = os.path.join(ROOT, *r"report\figures\track_d".split("\\"))
for mu in ("1.0", "2.0"):
    im = Image.open(rf"{D}\mu{mu}_trajectory_collage.png").convert("RGB")
    w, h = im.size
    top = 70                      # drop the suptitle row; the caption names mu
    q = w // 4
    panels = [im.crop((i * q, top, (i + 1) * q, h)) for i in range(4)]
    pw, ph = panels[0].size
    grid = Image.new("RGB", (2 * pw, 2 * ph), "white")
    for i, p in enumerate(panels):
        grid.paste(p, ((i % 2) * pw, (i // 2) * ph))
    grid.save(rf"{D}\mu{mu}_trajectory_grid.png")
    print(mu, grid.size)
