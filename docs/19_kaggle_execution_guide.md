# 🧭 Kaggle Execution Guide — Phase 4 Multi-Seed Runs, Step by Step

> **Who this is for:** anyone on the team running their Phase 4 seed-extension job
> on Kaggle, including if you've never used Kaggle notebooks before. Every step is
> spelled out — nothing is assumed.
> **Companion doc:** [`18_phase4_roadmap.md`](./18_phase4_roadmap.md) has the *why*
> (seed counts, time budgets, per-track scope). This doc is the *how*.
> **The seeds everyone should use:** `1337` and `2024` — the next two seeds in the
> project's original master list (`[42, 1337, 2024, 7, 999]`,
> `04_project_blueprint/PROJECT_IMPLEMENTATION_PLAN.md`). Track D already has its own
> extra seeds (`1`, `7`) from its own earlier investigation and needs nothing new for
> the $N=3$ plan — see §9's quick-reference table.

---

## 0. What you need before starting

- A Kaggle account (free). If you don't have one: go to
  [kaggle.com](https://www.kaggle.com) and sign up — email or Google/GitHub sign-in
  both work.
- A copy of this repo (`kinetic-kan`) on your own computer, up to date with `main`
  (i.e. you've pulled the latest changes, including the new `kaggle_*.py` scripts
  this guide points to).
- Nothing else. No GPU needed, no local Python packages beyond what you already have
  to `git pull` — everything else runs entirely inside Kaggle.

**One Kaggle account per person is fine.** Kaggle's "5 notebooks running at once"
limit is per-account, so if 5 team members each use their own account, the team's
*combined* capacity is much larger than 5 — but even a single person running several
of the jobs below on one account is enough for the $N=3$ plan (§9).

---

## 1. One-time setup: getting the code onto Kaggle, as a "Dataset"

Kaggle notebooks can't see your local hard drive — you have to upload the code once,
as a **Dataset**, then attach that Dataset to any notebook you create. You only do
this step once (or again later if the code changes and you need the newer version).

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click
   **"New Dataset"** (top right).
2. You need to upload the repo as a **folder**, not individual loose files, so the
   folder structure (`implementation/experiments/...`) is preserved. Two ways to do
   this:
   - **Easiest:** in your local `kinetic-kan` folder, select everything and compress
     it into a single `.zip` (right-click → "Compress to ZIP file" on Windows).
     Kaggle will automatically unzip it into the same folder structure once uploaded.
   - Drag that `.zip` file onto the Kaggle upload area (or click to browse and select
     it).
3. Give it a title — e.g. `kinetic-kan` (this exact name matters a little: it
   determines part of the mount path in step 3 below, so keep it simple and remember
   what you typed).
4. Set visibility to **Private** (this is your own working code, no reason to make it
   public).
5. Click **"Create"**. Wait for the upload + processing to finish (a progress bar
   shows this; can take a few minutes depending on the repo's size, mostly the PNG
   figures under `results/`).
6. Once it's done, you'll land on the Dataset's own page. **Note the URL** — it looks
   like `kaggle.com/datasets/<your-username>/kinetic-kan`. You'll need
   `<your-username>` and the dataset name in step 3.

**If the code changes later** (e.g. someone fixes a bug in one of the scripts and you
need the new version): go back to this Dataset's page, click **"New Version"**, upload
the updated zip, and any *new* notebook you create will see the update. Notebooks
you've already created keep the version they were attached to until you explicitly
update them (there's a "Check for updates" option on the notebook's data panel).

---

## 2. Creating a notebook and attaching the dataset

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. You land in an empty notebook editor (looks like Jupyter). On the **right-hand
   side panel**, find the section called **"Add Input"** (sometimes just a `+` icon
   near "Data"). Click it.
3. Search for the dataset name you used in step 1 (e.g. `kinetic-kan`) — if it's
   under your own account, it should show up immediately, possibly under an "Your
   Datasets" tab. Click it to attach.
4. Once attached, you'll see it listed in the right panel under "Input". **Hover over
   it (or click the little copy-path icon)** to see its exact mount path — it will be
   something like:
   ```
   /kaggle/input/kinetic-kan/kinetic-kan
   ```
   (the folder name appears twice — once for the dataset, once for the folder inside
   the zip you uploaded — this is normal). **Write this exact path down** — it's what
   you'll paste into every script's `REPO_ROOT` variable.
5. **Turn off the internet and accelerator** (right panel, under "Notebook options" /
   settings gear icon):
   - **Internet: Off.** None of these scripts need it (they only use PyTorch, which
     is already installed on every Kaggle notebook image), and leaving it off avoids
     any phone-verification requirement some Kaggle features need.
   - **Accelerator: None / No GPU.** We want a plain **CPU** notebook — this is what
     gives the "4 cores" the whole plan is built around. Selecting a GPU accelerator
     here doesn't help these tiny models and would count against a separate, more
     limited GPU quota for no benefit.

You now have a blank, CPU-only notebook with the repo attached and visible at that
mount path. This setup (steps 1–2) only needs to be done once per notebook — from here
on, each new notebook you create for a *different* seed/job just repeats step 2 (data
stays uploaded from step 1).

---

## 3. Picking your script and pasting it in

Each Phase 3 track that needs new seeds has its own ready-to-run script (already in
the repo, under that track's `experiments/<slug>/` folder — no need to write any code
yourself):

| Track | Owner | Script | Location |
| :--- | :--- | :--- | :--- |
| B | Abhishek | `kaggle_gradient_dynamics_seeds.py` | `implementation/experiments/gradient_dynamics/` |
| C | Shams | `kaggle_hybrid_basis_seeds.py` | `implementation/experiments/hybrid_basis/` |
| E | Monjur | `kaggle_sindy_epidemic_seeds.py` | `implementation/experiments/sindy_epidemic/` |
| D | Abrar | `kaggle_full_retrain.py` (already existed, only needed for the optional $N=5$ stretch — §9) | `implementation/experiments/stiffness_map/` |

1. Open the relevant file **locally** (in VS Code, Notepad, whatever you use) and
   select-all + copy its entire contents.
2. Back in your Kaggle notebook, click into the single empty code cell and paste.
3. **Edit exactly two things at the top of the pasted script**, inside the block
   marked `# CONFIGURE THIS before running`:
   - `REPO_ROOT` — replace with the mount path you wrote down in step 2.4.
   - The seed(s)/job list for **this specific notebook** — see §9's table for exactly
     what to put here per track. Each script's own docstring (the big comment at the
     top) also explains this in more detail for that specific track.

**Do not change anything below the `CONFIGURE THIS` block** unless you know what
you're doing — that's the part that was already checked against this project's real
training code.

---

## 4. Smoke-test first — don't launch a multi-hour run blind

Before committing to the real run, prove the script actually works end-to-end, in
under a minute, so a typo or path mistake doesn't waste hours of Kaggle time:

1. Temporarily lower the epoch count. Every script has an epochs variable near the
   top (`NUM_EPOCHS`, `epochs=...` inside `JOBS`, etc.) — temporarily set it to
   something tiny like `10`.
2. Click **"Run All"** (not "Save & Run All / Commit" yet — just the plain interactive
   run, top toolbar or `Shift+Enter` through the cell).
3. Watch the output print below the cell. You should see lines like
   `[1/N] ... -> train_mse=... (0.0 min)` for every job, ending with
   `All jobs finished in ...` and a line about a zip file being written.
4. **If you see an error instead:** almost always either (a) `REPO_ROOT` doesn't
   exactly match the real mount path (double-check step 2.4, including no trailing
   slash mismatch), or (b) the dataset didn't finish attaching — check the right
   panel still shows it under "Input". §10 (Troubleshooting) has more.
5. Once the smoke test prints "All jobs finished" cleanly, **change the epoch count
   back** to the real value (per §9's table, or whatever the script's docstring
   recommends) before moving to the next step.

This costs a couple of minutes and catches the vast majority of mistakes before they
cost hours.

---

## 5. Launching the real run

1. With the epoch count restored to its real value, click the button usually labeled
   **"Save Version"** (top right), then choose **"Save & Run All (Commit)"** in the
   dialog that appears (as opposed to "Quick Save", which does not execute the
   notebook).
2. This starts the notebook running **in the background**, on Kaggle's servers — you
   can close your browser tab, turn off your own computer, whatever. It keeps running
   until it finishes or hits a session time limit.
3. This step is exactly what lets **multiple notebooks run at once**: each
   "Save & Run All" you trigger, on a *different* notebook, is its own independent
   background job. Doing this on 5 different notebooks (your own, or across
   teammates' accounts) is what gives the 5-way parallelism the whole plan is built
   around (§9).

---

## 6. Watching it work

- Go to your **Notebooks list** ([kaggle.com/code](https://www.kaggle.com/code),
  or your profile's "Code" tab). Any notebook you've committed shows a status —
  running (usually a small spinner/progress indicator), or a final "Draft Saved"
  once done.
- Click into a running notebook and look for a **"Logs"** view (sometimes a small
  icon near the run status, or an option in the notebook's menu) to see the live
  `print()` output — the same `[k/N] solver=... -> train_mse=...` lines you saw in
  the smoke test, just from the real run.
- **You do not need to keep watching.** This is a background job — check back
  whenever convenient (an hour later, the next morning, whatever the expected
  duration is per §9).

---

## 7. Getting your results back

1. Once the notebook's status shows finished, open it and find the **"Output"** tab
   (a tab alongside "Notebook"/"Data", usually near the top or in a side panel).
2. You'll see a `.zip` file listed (e.g. `hybrid_basis_seeds_results.zip`) — this is
   exactly what the script's last line built via `shutil.make_archive`. Click the
   download icon next to it.
3. Save it somewhere on your own computer and unzip it. Inside, you'll find the exact
   folder structure the script wrote (e.g.
   `implementation/results/phase4/hybrid_basis_seeds/lv_seedA/metrics.json`, etc.).

---

## 8. Merging results into the shared repo

1. In your **local** copy of the `kinetic-kan` repo, locate the matching folder
   (e.g. `implementation/results/phase4/hybrid_basis_seeds/` — create the folder if
   it doesn't exist yet locally).
2. Copy the unzipped contents in, **merging** rather than overwriting — since every
   script's job list writes to its own uniquely-named subfolder (per seed, per arm,
   per solver), copying two different notebooks' output into the same parent folder
   just adds more subfolders side by side, with nothing to overwrite.
3. That's it — the new seed's results now sit locally exactly where the rest of the
   team can find them. **Committing/pushing this to git is a separate, manual step**
   you do yourself afterward (as with everything else in this project, nobody
   automates git actions on your behalf).

---

## 9. Running this in parallel — the actual per-person plan

Reusing [`18_phase4_roadmap.md`](./18_phase4_roadmap.md) §2.5's numbers. **"Notebook"**
below means: create it following §§1–2 once, then repeat §§3–8 for each row.

### The $N=3$ plan (do this first)

| Who | Script | Notebook(s) to create | Config to set | Expected wall-clock |
| :--- | :--- | :--- | :--- | :---: |
| Abhishek (B) | `kaggle_gradient_dynamics_seeds.py` | 2 — one per seed | Notebook 1: `SEEDS=[1337]`. Notebook 2: `SEEDS=[2024]`. | $\approx 0.4$h each, run together |
| Shams (C) | `kaggle_hybrid_basis_seeds.py` | 1 (all 3 jobs, `MAX_WORKERS=3`) *or* 3 (one job each) | Keep `JOBS` as shipped (LV seed 1337, LV seed 2024, pendulum seed 1337 at 5,000 epochs) for the 1-notebook option; trim to one entry per notebook for the 3-notebook option | $\approx 3.5$h (bounded by the pendulum job either way) |
| Monjur (E) | `kaggle_sindy_epidemic_seeds.py` | 2 — one per seed | Notebook 1: `SEED = 1337`. Notebook 2: `SEED = 2024`. | $\approx 1$h each, run together |
| Abrar (D) | — | **none needed** | Track D is already at $N{=}3$ (seeds 42, 1, 7 from its own earlier work) | — |
| Nawriz (A) | *(no Kaggle script — see below)* | — | — | $\approx 20$ min |

**Nawriz's task (Track A)** is small enough to just run as a plain local Python
snippet or a throwaway Kaggle notebook without any of the machinery above: reuse
`paired_gradient_relative_error()` from `experiments/adjoint_profiling/profiling.py`
at 1–2 more seeds and confirm the relative error stays tiny (see
`docs/18_phase4_roadmap.md` §2.3) — no `kaggle_*.py` script needed for this one.

**Total: 5 notebooks needed at once at most** (2 from B + 2 from E + up to 1–3 from
C) — comfortably inside the 5-concurrent-notebook limit if run as a single wave, or
split into 2 smaller waves if some notebooks are still finishing other work.

### The $N=5$ stretch goal (only after $N=3$ is done and reviewed)

Same scripts, same steps — just more seeds. See
[`18_phase4_roadmap.md`](./18_phase4_roadmap.md) §2.5's "N=5 stretch goal" table for
exactly which extra seeds/notebooks each track needs, and read that section's note on
Track C's pendulum before committing to more than one extra seed there — it's a
judgment call the team should make after seeing the $N=3$ pendulum result, not before.

---

## 10. Troubleshooting / FAQ

**"`FileNotFoundError` / can't find `train.py` or `run_hybrid` / import errors."**
Almost always `REPO_ROOT` doesn't exactly match the real mount path. Re-check step
2.4 — click the dataset in the right panel and copy the path it shows, rather than
retyping it from memory.

**"The notebook says it's still running but I don't see any print output."**
Kaggle can buffer output for a bit, especially right after a commit starts (the
environment is still initializing, e.g. importing torch for the first time across 4
processes at once). Give it a few minutes; if genuinely stuck past the expected
per-job time in §9's table, open the Logs view for more detail, or cancel and re-run
the smoke test (§4) again to isolate whether it's an environment issue.

**"It says a session time limit was reached partway through."** This is expected for
long jobs — every script here is written to be safely re-run: it skips any
already-finished job (checked via `SKIP_EXISTING` / whether `metrics.json` already
exists) and only continues the remaining ones. Just commit the same notebook again.

**"Can I check on progress without waiting for it to fully finish?"** Yes — the Logs
view (§6) updates live as each job completes and prints its own line, even while the
notebook is still running.

**"Do I need to disable internet even though I'm not sure my scripts need it?"**
Every script this guide covers explicitly needs no internet access (PyTorch, and for
Track E, `pysindy`, are both already on Kaggle's standard image) — leaving internet
off is the safe default and simplest account-verification path.

**"What if Kaggle's actual current session-length or weekly-quota limits are
different from what this guide assumes?"** Kaggle's platform policies change over
time and this guide can't guarantee today's exact numbers — check the current limits
in Kaggle's own notebook settings/documentation if anything in §9 seems off; every
individual job in this plan is comfortably under commonly-cited CPU session caps
regardless, so a policy change would need to be fairly drastic to actually block this
plan.

---

*Previous: [`18_phase4_roadmap.md`](./18_phase4_roadmap.md) — the seed-count reasoning
and time budget this guide's steps execute.*
