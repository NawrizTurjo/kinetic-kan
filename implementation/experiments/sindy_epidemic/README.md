# Track E — SINDy Comparison + Real Epidemic Fit

**Owner:** Monjur Hossain Khan ("Shovon") · **2105043**
**Branch:** `feat/p3-sindy-epidemic` · **Plan:** [`docs/12_phase3_roadmap.md`](../../../docs/12_phase3_roadmap.md) §Track E
**Write-up:** [`docs/17_p3_sindy_epidemic_findings.md`](../../../docs/17_p3_sindy_epidemic_findings.md)
**Results:** [`implementation/results/phase3/sindy_epidemic/`](../../results/phase3/sindy_epidemic/)

Two sub-tasks, bundled because both compare a learned model against an
*independent* form of ground truth.

| | Question |
| :-- | :--- |
| **E1** | Sparse symbolic regression (SINDy) and gradient-based KAN-ODE fail for different reasons. Which degrades more gracefully under observational noise — and are KAN's edges sparse enough to compare term-by-term at all? |
| **E2** | `docs/10`'s SIR rescue was derived from properties of the SIR *generator*. Does it transfer to outbreak data that no compartmental ODE produced? |

---

## Running it

```powershell
cd implementation/experiments/sindy_epidemic

python run_all.py                    # everything: E1 + E2  (see Runtime below)
python run_all.py --skip-e2          # E1 only              (~1 min)
python run_all.py --probe-epochs 30 --full-epochs 40   # fast end-to-end smoke test
```

Individual pieces:

```powershell
python run_sindy_noise.py                        # E1
python run_epidemic_fit.py --diagnose            # E2 init-time numbers, no training
python run_epidemic_fit.py --tag ts24 --time_scale 24 --epochs 2000
python run_epidemic_fit.py --collect             # E2 figures + summary JSON
```

Tests: `pytest tests/test_p3_sindy_epidemic.py` (27 tests, no training required).

### Environment

| Package | Version used | Note |
| :--- | :--- | :--- |
| `pysindy` | **2.1.0** | already listed in `requirements.txt`; no edit needed (roadmap §2.4) |
| `torch` | 2.11.0+cpu | |
| `numpy` / `scipy` | 2.4.3 / 1.17.1 | |
| Python | 3.14.3 | |

> `pysindy` 2.x moved `feature_names` from the `SINDy` constructor onto `.fit()`.
> The code here uses the 2.x form, so a 1.7.x environment will raise a
> `TypeError` on `fit`. `requirements.txt` pins `pysindy>=1.7.5`; tightening that
> to `>=2.0` belongs in the single integration PR described in roadmap §2.4, not
> in this branch.

### What E2 runs

| Stage | Arms | Budget | Why |
| :-- | :--- | :-: | :--- |
| 1 | `ts1 ts10 ts20 ts24 ts30 ts40` | 2,000 ep | Does `--time_scale` transfer? `s=1` is the no-fix baseline |
| 2 | `proj`, `vanish` | 2,000 ep | The two structural priors, at matched `s=24` |
| 3 | `split60`, `split70` | 2,000 ep | Move the train/extrapolation split off the outbreak peak |
| 4 | `full_vanish`, `full_plain` | 5,000 ep | Headline runs — matched `s` and budget, differing only in `vanish_dim` |

Stage 3 exists because the dataset's default split (day 45) lands **exactly on
the infected peak**, so the training window holds 0% of the 75-day decay. Those
arms separate *"the method cannot extrapolate"* from *"the window contained
nothing to extrapolate from"*.

### Runtime

Measured at **0.82 s/epoch** single-threaded. Stages 1–3 run in parallel (each
child pinned to one thread — intra-op threading buys nothing here: 0.82 s/epoch
on one thread vs. 0.80 on ten, because the cost is a sequential Python loop over
44 intervals × 2 substeps × 6 Tsit5 stages, not anything BLAS-bound).

**Parallel scaling is sublinear on a power-limited CPU** — 8 concurrent arms
measured only ~1.5× the aggregate throughput of one, and per-arm speed fell from
0.82 to ~3.6 s/epoch. Hence `--max-parallel` (default 4), and hence stage 4 runs
its two long arms *serially*, which on such a machine finishes sooner than
running them together. Budget roughly 1–1.5 h for stages 1–3 and ~1 h per
stage-4 arm; a machine that holds its clocks will be considerably quicker.

---

## Files

| File | What it is |
| :--- | :--- |
| `common.py` | Import-path setup, checkpoint rebuilding, integration, the train/extrap/full error split |
| `pruning.py` | **`prune_edges()`** — the L1 edge-pruning utility the codebase did not have |
| `run_sindy_noise.py` | E1: SINDy at four noise levels, pruning applied to the Phase-2 checkpoints, three figures |
| `run_epidemic_fit.py` | E2: init diagnostic, one training arm per invocation, collection |
| `run_all.py` | Entry point — reproduces every number and figure in the write-up |

Nothing outside this folder, `results/phase3/sindy_epidemic/`,
`tests/test_p3_sindy_epidemic.py` and `docs/17_*.md` is modified.
`results/benchmarks/noise/` is read-only.

---

## Design notes worth knowing before reading the code

### `prune_edges` returns a copy, and prunes whole edges

A KAN edge `i -> o` owns `grid_len` spline coefficients plus one base weight:

```text
phi_{o,i}(x) = sum_g C[o, i*G+g] * basis_g(norm(x))  +  W[o,i] * silu(x)
```

Zeroing scattered entries of `C` would make an edge's curve lumpier, not remove
it, so pruning masks all `G` coefficients of an edge together. Whether the
residual `W * silu(x)` branch also dies is a genuine modelling choice, not an
implementation detail — it is the `prune_base` flag, and **both settings are
measured and reported** rather than one being silently assumed.

`threshold_percentile=0.0` is an exact no-op (`<` not `<=`, so ties at the
minimum survive), and the input model is never mutated. Both are pinned by tests.

### E1's baseline is re-integrated, never retrained

The KAN side of E1 is the existing `results/benchmarks/noise/sigma*` checkpoints,
rebuilt honouring `grid_lims` / `conserve_mode` / `vanish_dim` / `time_scale`
from each saved config — the contract `evaluate.py` and `phase2_closeout.py` both
implement, and whose omissions `docs/10` §Part 5 catalogues. `test_p3_sindy_epidemic.py`
asserts the re-integration reproduces each run's **published** `metrics.json`
numbers to a relative 1e-4, so a silent rebuild bug cannot pass unnoticed.

SINDy is handed data rebuilt from those same configs — same generator, horizon,
split, seed and noise draw — so the comparison is genuinely apples-to-apples.

### E2 writes its own training loop, and does not touch `train.py`

`train.py`'s `DATASETS` registry has no entry for `load_empirical_epidemic_data`,
and its CLI cannot reach the structural field wrappers for a dataset it does not
know. Per roadmap §2.2 that is a reason to compose the same five primitives
(`KAN`, `NeuralODE`, the generator, Adam, `compute_gradient_norm`) in ~120 lines
here — not a reason to edit a shared file. The loop keeps `train.py`'s
non-finite gradient guard (X1), its gradient clipping, and its
select-on-training-loss checkpoint rule.

### E2 predicts, then runs the control arm anyway — and one prediction was wrong

Each of SIR's three fixes was predicted here from its preconditions, then run as
a control arm to check. `--time_scale` transferred as predicted; `conserve_mode
projection` failed as predicted. **`vanish_dim` was predicted to fail and did
not** — it is the best-performing option tested. The prediction had confused a
slow *start* with a frozen one.

That is the argument for running the controls rather than reasoning to a
conclusion and stopping: one arm in three overturned the write-up's expected
result. See the findings doc for the numbers.
