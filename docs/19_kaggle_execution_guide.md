# 🧭 Kaggle Execution Guide — Phase 4 Epoch-Budget Check, Step by Step

> **Who this is for:** anyone running one of the 5 epoch-budget-check notebooks on
> Kaggle, including if you've never used Kaggle notebooks before. Every step is
> spelled out — nothing is assumed.
> **Companion doc:** [`18_phase4_roadmap.md`](./18_phase4_roadmap.md) §2 has the *why*
> (which 5 configs, why those specifically, the real measured cost per one). This doc
> is the *how*.
> **The 5 ready-to-paste scripts** live in
> `implementation/experiments/epoch_budget_check/` — one file per notebook, no code to
> write yourself.

---

## 0. What you need before starting

- A Kaggle account (free) — [kaggle.com](https://www.kaggle.com), sign up if you don't
  have one.
- A local copy of `kinetic-kan`, up to date with `main` (including the 5 new scripts
  this guide points to, under `implementation/experiments/epoch_budget_check/`).
- Nothing else — no GPU needed, everything runs on Kaggle's CPU notebooks.

**One person can run all 5**, or the team can split them across accounts — either
way, all 5 are meant to run **at the same time**, each in its own notebook.

---

## 1. One-time setup: getting the code onto Kaggle, as a "Dataset"

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **"New Dataset"**.
2. Compress your local `kinetic-kan` folder into a single `.zip` and upload it (Kaggle
   unzips it automatically, preserving the folder structure).
3. Title it something simple, e.g. `kinetic-kan`. Set visibility to **Private**.
   Click **"Create"** and wait for upload + processing to finish.
4. Note the dataset's mount path once it's ready — you'll see it as
   `/kaggle/input/<dataset-name>/<folder-name>` (typically
   `/kaggle/input/kinetic-kan/kinetic-kan`) once attached to a notebook (step 2.4 below
   shows exactly where to find it).

**If the code changes later**, go back to this dataset's page and click **"New
Version"** to upload an updated zip — only new notebooks pick up the update
automatically.

---

## 2. Creating a notebook and attaching the dataset (repeat this 5 times, once per config)

1. [kaggle.com/code](https://www.kaggle.com/code) → **"New Notebook"**.
2. Right-hand panel → **"Add Input"** → search for your dataset name → attach it.
3. Hover over the attached dataset in the right panel to see its exact mount path —
   write it down, you'll paste it into the script as `REPO_ROOT`.
4. In the notebook's settings (gear icon):
   - **Internet: Off** (none of these scripts need it — PyTorch is already on Kaggle's
     image).
   - **Accelerator: None / No GPU** (we want a plain 4-core CPU notebook).
5. **Rename the notebook** to something you'll recognize in your notebook list later —
   e.g. `phase4-euler-50k`, `phase4-tsit5-rbf-50k`, etc. (top-left, click the
   auto-generated title). With 5 notebooks running at once, this matters more than
   usual.

---

## 3. The 5 scripts — one per notebook

| # | Notebook name (suggested) | Script to paste | What it trains | Expected time |
| :-: | :--- | :--- | :--- | :---: |
| 1 | `phase4-euler-50k` | `kaggle_50k_1_euler.py` | Euler solver, RBF, LV | $\approx 21$ min |
| 2 | `phase4-tsit5-rbf-50k` | `kaggle_50k_2_tsit5_rbf.py` | Tsit5 solver, RBF, LV (the reference config) | $\approx 2.4$ h |
| 3 | `phase4-bspline-50k` | `kaggle_50k_3_bspline.py` | Tsit5 solver, B-spline, LV | $\approx 9.3$ h (the long pole) |
| 4 | `phase4-mlp-silu-50k` | `kaggle_50k_4_mlp_silu.py` | MLP $[2,14,8,8,2]$+SiLU, LV | $\approx 4.0$ h |
| 5 | `phase4-mlp-paperspec-50k` | `kaggle_50k_5_mlp_paperspec.py` | MLP $[2,50,2]$+tanh, LV (paper's own architecture — currently fails to train) | $\approx 1.0$ h |

All 5 files are in `implementation/experiments/epoch_budget_check/` in your local
repo. For each notebook:

1. Open the matching script locally, select all, copy.
2. Paste into that notebook's single empty code cell.
3. **Edit exactly one line** — `REPO_ROOT`, near the top under
   `# CONFIGURE THIS before running` — to the mount path from step 2.3. Nothing else
   needs to change; every other setting in the script already matches the original
   10k-epoch run's own recorded config, so the only real difference is the epoch count.

---

## 4. Smoke-test first — don't launch a 9-hour run blind

Each script is a **single training run**, so the smoke test is simpler than a
multi-job script: just lower the epoch count temporarily.

1. Find the line `NUM_EPOCHS = 50000` near the top and change it to `NUM_EPOCHS = 5`.
2. Click **"Run All"** (the plain interactive run, not "Save & Run All" yet).
3. You should see the training progress bar print, finish in well under a minute, and
   end with a comparison against the original 10k run's numbers (every script prints
   this automatically) and a line about a zip file being written.
4. **If you see an error:** almost always `REPO_ROOT` doesn't exactly match the real
   mount path — double check step 2.3. §7 (Troubleshooting) has more.
5. Once it prints cleanly, **change `NUM_EPOCHS` back to `50000`** before moving on.

---

## 5. Launching the real run

Click **"Save Version"** → **"Save & Run All (Commit)"**. This runs the notebook in
the background on Kaggle's servers — you can close the tab. Do this for **all 5
notebooks**, one after another, so all 5 start running at roughly the same time.

This is exactly what gives you 5-way parallelism: each "Save & Run All" is its own
independent background job, on its own 4-core sandbox.

---

## 6. Watching progress and getting results back

- Check your notebook list ([kaggle.com/code](https://www.kaggle.com/code)) — each
  shows running/finished status. Open a notebook's **Logs** view to see live
  `print()` output (training progress + the eventual 50k-vs-10k comparison).
- No need to babysit this — the slowest one (`bspline`) takes about 9 hours; check
  back whenever convenient.
- Once a notebook finishes, open its **Output** tab and download the `.zip` file
  listed there (e.g. `euler_50k_result.zip`).
- Unzip it locally into the matching folder under
  `implementation/results/phase4/epoch_budget_check/<config>_50k/` (create the parent
  folder if it doesn't exist yet). Since each script writes to its own uniquely-named
  subfolder, downloading all 5 zips into the same parent folder just adds 5 sibling
  folders — nothing to overwrite.

---

## 7. Reading the result — is the ranking stable?

Each script's own printed output already does the comparison for you — look for the
`50k vs. 10k train_mse improvement` line (or, for notebook 5 specifically, the
explicit "CONVERGED" / "STILL NOT CONVERGED" verdict). To answer
[`18`](./18_phase4_roadmap.md) §2's actual question — **did the relative ranking among
the 5 configs change** — compare all 5 configs' 50k `train_mse` values against each
other, then compare that ordering to the original 10k ordering:

| Config | 10k train_mse (already known) |
| :--- | :---: |
| Euler | $1.377\times10^{-4}$ |
| Tsit5/RBF | $8.833\times10^{-5}$ |
| B-spline | $7.202\times10^{-5}$ |
| MLP-SiLU | $9.505\times10^{-5}$ |
| MLP-paper-spec | $1.083$ (essentially failed) |

Fill in the 50k column from your 5 runs and see whether this same order (B-spline <
Tsit5/RBF < MLP-SiLU < Euler $\ll$ MLP-paper-spec) still holds. If it does,
[`18`](./18_phase4_roadmap.md) §2.5's "ranking holds" outcome applies; if not, that
section's other branch does.

---

## 8. Troubleshooting / FAQ

**"`FileNotFoundError` / can't import `train`."** `REPO_ROOT` doesn't match the real
mount path — re-check step 2.3, copy it rather than retyping from memory.

**"It says a session time limit was reached partway through."** Unlike the earlier
multi-job scripts, these are single runs with no built-in resume — if a session limit
is hit mid-run, you'll need to re-commit the notebook and let it retrain from scratch.
If this becomes a real problem for the `bspline` notebook specifically (the longest at
~9h), it's worth checking Kaggle's current session-length limit before starting (see
the caveat below) rather than discovering it 8 hours in.

**"Can I check progress without waiting for it to finish?"** Yes — the Logs view (§6)
updates live via the training progress bar, even while still running.

**"What if Kaggle's actual concurrent-notebook or session-length limits are different
from what this guide assumes?"** These are platform policies that change over time —
only 5 notebooks are needed for this whole batch, which comfortably fits even the more
conservative limit numbers that have come up in team discussion; double-check current
limits directly in Kaggle's own settings/docs if anything seems off.

---

*Previous: [`18_phase4_roadmap.md`](./18_phase4_roadmap.md) — the reasoning behind
these 5 specific configs and their measured costs.*
