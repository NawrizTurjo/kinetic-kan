"""
Step 0: check that every saved checkpoint still reproduces its own stored metrics.

Every later script recomputes quantities from best_model.pt, so this runs first.
Each noise-free dt=0.1 checkpoint is re-integrated from u(0) and its
extrapolation MSE on (3.5, 14] is compared with the value that training wrote
to metrics.json. The script fails if any relative deviation reaches 1%.

Report: Section 9 ("the rollouts reproduce the stored extrapolation MSE to within 0.6%").
Output: numbers.json -> reproduction_worst_rel_dev, reproduction_rel_dev
"""

import numpy as np

from common import N_14, N_TRAIN, RUNS, Y28, metrics, rollout, save_numbers


def main():
    devs = {}
    for key in RUNS:
        mt = metrics(key)
        if mt["config"]["dt"] != 0.1 or mt["config"]["noise_std"] != 0.0:
            continue
        pr = rollout(key)
        mse = float(np.mean((pr[N_TRAIN:N_14] - Y28[N_TRAIN:N_14]) ** 2))
        ref = mt["best"]["extrap_mse"]
        devs[key] = abs(mse - ref) / ref
        print(f"  {key:16s} stored {ref:.4e}  recomputed {mse:.4e}  rel. dev. {devs[key]:.1e}")
    worst = max(devs.values())
    print(f"  checkpoint reproduction: worst relative deviation {worst:.2e}")
    save_numbers(dict(reproduction_worst_rel_dev=worst, reproduction_rel_dev=devs))
    assert worst < 1e-2, "a checkpoint no longer reproduces its stored metrics"


if __name__ == "__main__":
    main()
