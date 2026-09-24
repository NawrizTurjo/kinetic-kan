"""
Training-dynamics numbers read from each run's training_history.json.

  * milestones: first epoch at which the training MSE falls below 1e-2 ... 1e-5
    (Table "milestones", and the Phase 4 "reaches 1e-4 / 1e-5 at epoch ..." text);
  * paper targets: first epoch at which the extended KAN reaches the paper's
    2.6e-5 and the extended MLPs reach the paper's 3.0e-5 (Section 9);
  * KAN/MLP crossover at 50k epochs: the logged full-horizon test MSE spikes, so
    both curves are smoothed with a trailing rolling geometric median over 50
    logged samples (500 epochs, sampling every 10th epoch) and the last epoch
    at which the MLP is still behind is located.

The loss-curve figures themselves are the team's originals (see
collect_result_figures.py); this script only produces numbers.

Output: numbers.json -> epochs_to_train_mse, paper_target_epochs, phase4_crossover
"""

import numpy as np

from common import RUNS, history, save_numbers


def first_epoch_below(arr, thr):
    idx = np.where(arr <= thr)[0]
    return int(idx[0]) + 1 if len(idx) else None


def roll_gmedian(a, w=50):
    """Rolling geometric median over w logged samples (trailing window)."""
    la = np.log(a)
    return np.exp(np.array([np.median(la[max(0, i - w + 1):i + 1]) for i in range(len(la))]))


def milestones():
    out = {}
    for key in RUNS:
        tr = history(key)["train_losses"]
        out[key] = {f"{thr:.0e}": first_epoch_below(tr, thr) for thr in [1e-2, 1e-3, 1e-4, 1e-5]}
    return out


def paper_targets():
    return {"kan_50k_to_2.6e-5": first_epoch_below(history("kan_50k")["train_losses"], 2.6e-5),
            "mlp_silu_50k_to_3.0e-5": first_epoch_below(history("mlp_silu_50k")["train_losses"], 3.0e-5),
            "mlp_tanh_50k_min_train_loss": float(history("mlp_tanh_50k")["train_losses"].min())}


def phase4_crossover():
    k = roll_gmedian(history("kan_50k")["test_losses"][::10])
    mm = roll_gmedian(history("mlp_silu_50k")["test_losses"][::10])
    ep = np.arange(1, len(history("kan_50k")["test_losses"]) + 1)[::10]
    behind = np.where(mm >= k)[0]
    cross = int(ep[behind[-1] + 1]) if len(behind) and behind[-1] + 1 < len(ep) else None
    at = lambda arr, e: float(arr[np.searchsorted(ep, e)])  # noqa: E731
    marks = [1001, 5001, 10001, 25001, 49991]
    return dict(permanent_crossover_epoch_smoothed=cross,
                kan_full_mse_smoothed={e: at(k, e) for e in marks},
                mlp_full_mse_smoothed={e: at(mm, e) for e in marks})


def main():
    ms = milestones()
    for key in ["solver_euler", "solver_tsit5", "basis_bspline", "mlp_silu", "kan_50k", "mlp_silu_50k"]:
        print(f"  {key:14s} epochs to 1e-2/1e-3/1e-4/1e-5: {list(ms[key].values())}")
    pt = paper_targets()
    print("  paper targets:", pt)
    cr = phase4_crossover()
    print("  crossover:", cr)
    save_numbers(dict(epochs_to_train_mse=ms, paper_target_epochs=pt, phase4_crossover=cr))


if __name__ == "__main__":
    main()
