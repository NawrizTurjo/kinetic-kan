"""
[TRACK C] Plot alpha/beta gate evolution for every completed run.
Run with cwd = implementation/ (same convention as run_hybrid.py's --save_dir),
so paths below are direct children -- no "../" needed.
"""
import json
import os
import matplotlib.pyplot as plt

ROOT = "results/phase3/hybrid_basis"
TAGS = ["probe_lv", "probe_pendulum", "lv_full", "pendulum_full"]

for tag in TAGS:
    hist_path = os.path.join(ROOT, tag, "training_history.json")
    if not os.path.exists(hist_path):
        print(f"  skip {tag}: not run yet ({hist_path} missing)")
        continue
    h = json.load(open(hist_path))
    plt.figure(figsize=(8, 4))
    plt.plot(h["alpha"], label=r"$\alpha$ (B-spline weight)")
    plt.plot(h["beta"], label=r"$\beta$ (RBF weight)")
    plt.xlabel("epoch"); plt.ylabel("softmax weight"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Hybrid basis gate evolution — {tag}")
    plt.tight_layout()
    out = os.path.join(ROOT, f"{tag}_alpha_beta.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  wrote {out}")
