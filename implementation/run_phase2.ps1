<#
.SYNOPSIS
    Launch the Phase-2 production sweeps as throttled parallel background processes.

.DESCRIPTION
    Each configuration is a separate `train.py` process writing into its own --save_dir,
    reproducing the existing results/benchmarks/ layout exactly:

        results/benchmarks/ablation_solvers/solver_<name>/
        results/benchmarks/ablation_activations/basis_<name>/
        results/benchmarks/kanode_flagship/
        results/benchmarks/mlpode_baseline/
        results/benchmarks/kanode_bspline/

    With -Seeds giving more than one value, each run directory gains a _seed<N> suffix
    (e.g. solver_tsit5_seed1337); collate_results.py groups those back together.

    Thread pinning matters: PyTorch grabs every core by default, so N unpinned parallel
    jobs oversubscribe the CPU and inflate every wall-clock number. ThreadsPerJob is set
    so MaxParallel * ThreadsPerJob stays within the machine's logical processor count.

.PARAMETER Seeds
    Seeds to run per configuration. Default 42. Use 42,1337,2024 for N=3 error bars.

.PARAMETER Only
    Which sweeps to launch:
      solvers, activations, models, bspline, stepsize, noise   (Lotka-Volterra tables)
      mlpfix    converged [2,14,8,8,2]+SiLU MLP baseline for Table 3
      systems   damped pendulum + SIR            (plan Tasks 2.4 / 2.5)
      lorenz    3D Lorenz, coarsened grid        (plan Task 2.5; NOT in "all" -- costly)
      all       everything except lorenz

    [FIX-2026-08] Two targets for the cross-domain stability fixes. Both write to
    their OWN directories so nothing under results/benchmarks/ is ever overwritten:
      probe     8 short one-factor-at-a-time diagnostics  -> results/_probe/
                Isolates WHICH fix matters. Use -Epochs 2000.
      fixes     2 full-length runs with the fixes applied -> results/_fixed/
                Run this only AFTER reading the probe results.
    Neither is included in "all".

.PARAMETER Serial
    Run one job at a time. REQUIRED if you intend to report wall-clock timings.

.EXAMPLE
    .\run_phase2.ps1 -Only solvers,activations,models
    .\run_phase2.ps1 -Only solvers -Seeds 42,1337,2024 -MaxParallel 3
    .\run_phase2.ps1 -Only all -Epochs 20 -DryRun      # verify the plan first

    # [FIX-2026-08] cross-domain stability workflow
    .\run_phase2.ps1 -Only probe -Epochs 2000 -MaxParallel 2   # step 1: diagnose
    .\run_phase2.ps1 -Only fixes -Epochs 10000 -MaxParallel 2  # step 2: full runs
#>

param(
    [int]    $Epochs        = 10000,
    [double] $Lr            = 2e-3,
    [int[]]  $Seeds         = @(42),
    [string] $SaveDir       = "results/benchmarks",
    # [FIX-2026-08] separate output roots so the fix runs can never collide with
    # the committed Phase-2 artifacts under results/benchmarks/.
    [string] $ProbeDir      = "results/_probe",
    [string] $FixedDir      = "results/_fixed",
    [string[]] $Only        = @("solvers", "activations", "models", "bspline"),
    [int]    $MaxParallel   = 0,          # 0 = auto
    [int]    $ThreadsPerJob = 0,          # 0 = auto
    [string] $Device        = "cpu",
    [double[]] $Dts         = @(0.20, 0.10, 0.05),
    [double[]] $Sigmas      = @(0.0, 0.01, 0.05, 0.10),
    [double] $LorenzDt      = 0.02,
    [double] $LorenzTEnd    = 10.0,
    [double] $LorenzTTrain  = 4.0,
    [switch] $Serial,
    [switch] $DryRun,
    [switch] $SkipExisting,   # resume: skip runs that already produced metrics.json
    [switch] $NoCollate       # don't build results/tables/*.csv at the end
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# --- Resolve the interpreter: prefer the repo venv, fall back to PATH ---------
$venv = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
$Python = if (Test-Path $venv) { (Resolve-Path $venv).Path } else { "python" }

# --- Auto-size the thread budget to the actual machine ------------------------
$logical = [Environment]::ProcessorCount
if ($Serial) {
    $MaxParallel = 1
    if ($ThreadsPerJob -le 0) { $ThreadsPerJob = $logical }
} else {
    if ($MaxParallel -le 0)   { $MaxParallel   = [Math]::Max(1, [Math]::Min(4, [Math]::Floor($logical / 2))) }
    if ($ThreadsPerJob -le 0) { $ThreadsPerJob = [Math]::Max(1, [Math]::Floor($logical / $MaxParallel)) }
}

# [FIX-2026-08] Route logs AND collation to the matching root for the probe/fixes
# targets, so results/benchmarks/_logs/ and the committed results/tables/*.csv are
# never touched by a diagnostic run.
$ActiveRoot = $SaveDir
if     ($Only -contains "probe") { $ActiveRoot = $ProbeDir }
elseif ($Only -contains "fixes") { $ActiveRoot = $FixedDir }

$LogDir = Join-Path $ActiveRoot "_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# --- Build the job list -------------------------------------------------------
$jobs = New-Object System.Collections.Generic.List[object]

function Add-Job([string]$Name, [string]$Dir, [string[]]$ExtraArgs) {
    foreach ($seed in $Seeds) {
        $suffix  = if ($Seeds.Count -gt 1) { "_seed$seed" } else { "" }
        $jobs.Add([pscustomobject]@{
            Name = "$Name$suffix"
            Args = @(
                "train.py",
                "--epochs", $Epochs,
                "--lr",     $Lr,
                "--seed",   $seed,
                "--device", $Device,
                "--save_dir", "$Dir$suffix"
            ) + $ExtraArgs
        })
    }
}

if ($Only -contains "solvers" -or $Only -contains "all") {
    foreach ($s in @("tsit5", "rk4", "dopri5", "midpoint", "heun", "euler")) {
        Add-Job "solver_$s" "$SaveDir/ablation_solvers/solver_$s" @("--solver", $s, "--basis", "rbf")
    }
}
if ($Only -contains "activations" -or $Only -contains "all") {
    foreach ($b in @("rbf", "bspline", "chebyshev", "lagrange", "newton", "rswaf", "iqf")) {
        Add-Job "basis_$b" "$SaveDir/ablation_activations/basis_$b" @("--basis", $b, "--solver", "tsit5")
    }
}
if ($Only -contains "models" -or $Only -contains "all") {
    Add-Job "kanode_flagship" "$SaveDir/kanode_flagship" @("--model", "kan", "--basis", "rbf", "--solver", "tsit5")
    Add-Job "mlpode_baseline" "$SaveDir/mlpode_baseline" @("--model", "mlp", "--solver", "tsit5")
}
if ($Only -contains "bspline" -or $Only -contains "all") {
    Add-Job "kanode_bspline" "$SaveDir/kanode_bspline" @("--basis", "bspline", "--solver", "rk4")
}
# Converged parameter-matched MLP baseline (paper's [2,50,2]+tanh does not train;
# see docs/05 section 5). This is the strong baseline Table 3 must be judged against.
if ($Only -contains "mlpfix" -or $Only -contains "all") {
    Add-Job "mlpode_baseline_silu" "$SaveDir/mlpode_baseline_silu" `
        @("--model", "mlp", "--mlp_layers", "2", "14", "8", "8", "2", "--mlp_act", "silu", "--solver", "tsit5")
}

# Plan Tasks 2.4 (pendulum) and 2.5 (Lorenz, SIR) -- required Phase-2 deliverables.
# Lorenz defaults (dt=0.01, t_end=20 -> 801 training points) cost ~22x a Lotka-Volterra
# run, so an explicitly coarsened grid is used for this first pass.
if ($Only -contains "systems" -or $Only -contains "all") {
    Add-Job "pendulum" "$SaveDir/pendulum" @("--dataset", "damped_pendulum", "--basis", "rbf", "--solver", "tsit5", "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0")
    Add-Job "sir"      "$SaveDir/sir"      @("--dataset", "sir",             "--basis", "rbf", "--solver", "tsit5", "--t_train_end", "50.0", "--layers", "3", "16", "3", "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0")
}
if ($Only -contains "lorenz") {
    Add-Job "lorenz" "$SaveDir/lorenz" `
        @("--dataset", "lorenz", "--basis", "rbf", "--solver", "tsit5",
          "--dt", $LorenzDt, "--t_end", $LorenzTEnd, "--t_train_end", $LorenzTTrain)
}

# =============================================================================
# [FIX-2026-08] Cross-domain stability fixes. See docs/06_suggested_fixes.md for
# the diagnosis and docs/08_how_to_run_fixes.md for the workflow.
#
# NOTE: these deliberately write to $ProbeDir / $FixedDir, NOT $SaveDir, so a
# stray invocation can never overwrite results/benchmarks/. Neither target is
# part of "all".
# =============================================================================

# --- probe: one-factor-at-a-time diagnostics ---------------------------------
# Each pendulum job changes exactly ONE thing from the failing baseline, so the
# summary table says which fix is responsible instead of confounding three at
# once (which is what made the last round need forensic log analysis).
if ($Only -contains "probe") {
    # Baseline reproduces the known failure at the short budget -- the control.
    $pendBase = @("--dataset", "damped_pendulum", "--basis", "rbf", "--solver", "tsit5",
                  "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0")
    Add-Job "pend_control"  "$ProbeDir/pend_control"  $pendBase
    Add-Job "pend_identity" "$ProbeDir/pend_identity" ($pendBase + @("--act", "identity"))
    Add-Job "pend_tanhact"  "$ProbeDir/pend_tanhact"  ($pendBase + @("--act", "tanh"))
    Add-Job "pend_lossw"    "$ProbeDir/pend_lossw"    ($pendBase + @("--loss_weighting", "std"))
    Add-Job "pend_gridlims" "$ProbeDir/pend_gridlims" ($pendBase + @("--grid_lims", "-3", "3"))

    $sirBase = @("--dataset", "sir", "--basis", "rbf", "--solver", "tsit5",
                 "--t_train_end", "50.0", "--layers", "3", "16", "3",
                 "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0")
    Add-Job "sir_control"  "$ProbeDir/sir_control"  $sirBase
    Add-Job "sir_conserve" "$ProbeDir/sir_conserve" ($sirBase + @("--conserve_sum", "1.0"))
    Add-Job "sir_lossw"    "$ProbeDir/sir_lossw"    ($sirBase + @("--loss_weighting", "std"))

    # ---- SIR root-cause probes (2,000 epochs) -------------------------------
    # The three above all end STILL FROZEN (I_drift 0.0000 against a true 0.0525);
    # they differ only in how they fail. These isolate the actual causes. Measured
    # extrap RMSE / I_drift / max mass error:
    #   sir_van_only      0.0028  0.0524  1.2e-3   gate alone
    #   sir_van_proj      0.0039  0.0522  0.0e+0   gate + exact conservation  <-- shipped
    #   sir_van_proj_g5   0.0027  0.0527  0.0e+0   same at grid_len 5 (fewer params)
    $sirFixed = $sirBase + @("--time_scale", "10.0")
    Add-Job "sir_van_only"    "$ProbeDir/sir_van_only"    ($sirFixed + @("--vanish_dim", "1"))
    Add-Job "sir_van_proj"    "$ProbeDir/sir_van_proj"    ($sirFixed + @("--conserve_sum", "1.0", "--conserve_mode", "projection", "--vanish_dim", "1"))
    Add-Job "sir_van_proj_g5" "$ProbeDir/sir_van_proj_g5" ($sirFixed + @("--conserve_sum", "1.0", "--conserve_mode", "projection", "--vanish_dim", "1", "--grid_len", "5"))
}

# --- fixes: full-length runs with the fixes combined -------------------------
# Args below are now SET BY THE PROBE RESULTS (results/_probe, 11 runs at 2000
# epochs -- see docs/09_stability_fix_results.md), not by prediction.
#
# Pendulum, ranked by full-horizon MSE over [0,10] (window-independent, so it is
# the only metric comparable across different --t_train_end values):
#     pend_id_win5    0.0432   identity + t_train_end 5.0   <-- CHOSEN, R2 = +0.662
#     pend_tanh_win5  0.1365   tanh     + t_train_end 5.0
#     pend_id_lossw   0.6386   identity + loss_weighting std
#     pend_tanhact    0.7527   tanh
#     pend_control    0.8943   (baseline)
#     pend_identity   1.1100   identity alone -- WORSE than baseline
#     pend_lossw      1.4835   loss_weighting alone -- worst
# --loss_weighting was DROPPED here: it helps the fit but is the worst option on
# extrapolation, and identity+lossw (0.6386) is 15x worse than identity+win5.
if ($Only -contains "fixes") {
    Add-Job "pendulum_fixed" "$FixedDir/pendulum_fixed" `
        @("--dataset", "damped_pendulum", "--basis", "rbf", "--solver", "tsit5",
          "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0",
          "--act", "identity", "--t_train_end", "5.0")

    # SIR. The earlier recipe here was --conserve_sum 1.0 + --loss_weighting std,
    # combining the two best PROBE runs. It diverged at epoch 3055 and was aborted
    # at 3154. Stacking was never the problem; both probes were treating symptoms.
    #
    # Root cause (docs/10_sir_root_cause_and_fix.md): SIR was posed in the wrong
    # units. Over a horizon of 50 with |f| ~ 1e-2, a Glorot-init KAN starts 19x too
    # fast, the trajectory runs to -51.8 by epoch 0, and epoch-0 loss is 3.16e+02.
    # Everything else followed from that -- the "gradient explosion" peaks at epoch
    # ELEVEN, and the "frozen fixed point" is just the optimizer taking the steepest
    # escape from that loss, which is f -> 0.
    #
    # Three independent failures, three separate fixes; measured at 2,000 epochs:
    #   --time_scale 10           train_mse 2.37e-2 -> 2.4e-5   (kills both above)
    #   --conserve_mode projection  mass err 0.35   -> 7e-7     (exact, all horizons)
    #   --vanish_dim 1            extrap RMSE 0.095 -> 0.0039   (I=0 is an equilibrium)
    #
    # CONFIRMED at 10,000 epochs: the first two ALONE pass every acceptance
    # criterion (sir_fixed_noprior: extrap RMSE 0.0140, mass 1.0000, I_drift
    # 0.0837, 0 non-finite steps). --vanish_dim adds a further 25x on extrap RMSE
    # (0.00057) but is NOT required, and it is the only one of the three that
    # assumes anything physical. Both runs finish 10,000/10,000 with a finite
    # final model; the pre-fix run died of NaN at epoch 8,686.
    #
    # --loss_weighting std is deliberately NOT here: with rescaling it makes things
    # worse (extrap RMSE 0.163 vs 0.095), as does lr 1e-3 (0.430).
    Add-Job "sir_fixed" "$FixedDir/sir_fixed" `
        @("--dataset", "sir", "--basis", "rbf", "--solver", "tsit5",
          "--t_train_end", "50.0", "--layers", "3", "16", "3",
          "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0",
          "--time_scale", "10.0",
          "--conserve_sum", "1.0", "--conserve_mode", "projection",
          "--vanish_dim", "1")

    # Ablation, and the honest control for the one fix that is a PHYSICAL claim
    # rather than a numerical one. --vanish_dim asserts that the plane I=0 consists
    # of equilibria (true for compartmental models: every SIR term carries a factor
    # of I). This run is the same recipe WITHOUT that assertion, so its contribution
    # is reported rather than absorbed into the headline number.
    Add-Job "sir_fixed_noprior" "$FixedDir/sir_fixed_noprior" `
        @("--dataset", "sir", "--basis", "rbf", "--solver", "tsit5",
          "--t_train_end", "50.0", "--layers", "3", "16", "3",
          "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0",
          "--time_scale", "10.0",
          "--conserve_sum", "1.0", "--conserve_mode", "projection")

    # Control at full budget: same longer window, but the ORIGINAL SiLU activation.
    # Completes the 2x2 (activation x window) so the claim "both fixes were needed"
    # is backed at 10,000 epochs rather than inferred from the probe.
    Add-Job "pendulum_control_win5" "$FixedDir/pendulum_control_win5" `
        @("--dataset", "damped_pendulum", "--basis", "rbf", "--solver", "tsit5",
          "--grid_len", "8", "--lr", "0.003", "--grad_clip", "1.0",
          "--act", "silu", "--t_train_end", "5.0")
}

if ($Only -contains "stepsize" -or $Only -contains "all") {
    foreach ($d in $Dts) {
        $tag = "dt$($d.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture))"
        Add-Job $tag "$SaveDir/stepsize/$tag" @("--dt", $d, "--basis", "rbf", "--solver", "tsit5")
    }
}
if ($Only -contains "noise" -or $Only -contains "all") {
    foreach ($g in $Sigmas) {
        $tag = "sigma$($g.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture))"
        Add-Job $tag "$SaveDir/noise/$tag" @("--noise_std", $g, "--basis", "rbf", "--solver", "tsit5")
    }
}

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " Phase-2 sweep launcher" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host (" Python        : {0}" -f $Python)
Write-Host (" Jobs          : {0}   (sweeps: {1})" -f $jobs.Count, ($Only -join ", "))
Write-Host (" Seeds         : {0}" -f ($Seeds -join ", "))
Write-Host (" Epochs / LR   : {0} / {1}" -f $Epochs, $Lr)
Write-Host (" Logical CPUs  : {0}" -f $logical)
Write-Host (" Parallelism   : {0} concurrent x {1} threads each" -f $MaxParallel, $ThreadsPerJob)
Write-Host (" Output        : {0}" -f $ActiveRoot)   # [FIX-2026-08] probe/fixes use their own root
Write-Host (" Logs          : {0}" -f $LogDir)
if (-not $Serial) {
    Write-Host " NOTE: wall-clock timings are NOT comparable under parallel execution." -ForegroundColor Yellow
    Write-Host "       Re-run one clean pass with -Serial if you need the timing column." -ForegroundColor Yellow
}
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    foreach ($j in $jobs) { Write-Host ("  {0,-28} {1} {2}" -f $j.Name, $Python, ($j.Args -join " ")) }
    Write-Host "`nDry run only -- nothing launched." -ForegroundColor Yellow
    return
}

# --- Launch with throttling ---------------------------------------------------
$env:OMP_NUM_THREADS      = $ThreadsPerJob
$env:MKL_NUM_THREADS      = $ThreadsPerJob
$env:OPENBLAS_NUM_THREADS = $ThreadsPerJob

$all     = @()
$started = 0
$skipped = 0
$t0 = Get-Date

foreach ($j in $jobs) {
    $dir = $j.Args[$j.Args.IndexOf("--save_dir") + 1]

    if ($SkipExisting -and (Test-Path (Join-Path $dir "metrics.json"))) {
        $skipped++
        Write-Host ("[skip] {0}  (metrics.json already present)" -f $j.Name) -ForegroundColor DarkGray
        continue
    }

    # Throttle: wait until a slot frees up
    while (@($all | Where-Object { -not $_.Proc.HasExited }).Count -ge $MaxParallel) {
        Start-Sleep -Seconds 5
    }

    $out = Join-Path $LogDir "$($j.Name).log"
    $err = Join-Path $LogDir "$($j.Name).err.log"
    $proc = Start-Process -FilePath $Python -ArgumentList $j.Args -NoNewWindow -PassThru `
                          -RedirectStandardOutput $out -RedirectStandardError $err
    $all += [pscustomobject]@{ Name = $j.Name; Dir = $dir; Proc = $proc }
    $started++
    Write-Host ("[{0,3}/{1}] {2,-28} pid {3}  {4:HH:mm:ss}" -f `
                $started, $jobs.Count, $j.Name, $proc.Id, (Get-Date)) -ForegroundColor Green
}

if ($started -gt 0) {
    Write-Host "`n$started launched, $skipped skipped. Waiting for completion..." -ForegroundColor Cyan
    # [FIX-2026-08] Was: `$r.Proc | Wait-Process`, which resolves the target by PID.
    # Windows recycles PIDs, so on a long sweep an early job can exit and have its PID
    # reassigned (observed: an 8-job probe where pid 30400 became an svchost), after
    # which Wait-Process tries to wait on that unrelated process and dies with
    # "Access is denied" -- aborting the script and skipping collation even though
    # every training job was fine.
    #
    # .WaitForExit() is called on the Process object captured by Start-Process, which
    # holds an OS handle rather than a bare PID, so it cannot be confused by reuse.
    foreach ($r in $all) {
        try {
            $r.Proc.WaitForExit()
        } catch {
            Write-Host ("  ! could not wait on {0}: {1}" -f $r.Name, $_.Exception.Message) -ForegroundColor Yellow
        }
    }
}

$elapsed = (Get-Date) - $t0
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host (" Sweep finished in {0:hh\:mm\:ss}   ({1} run, {2} skipped)" -f $elapsed, $started, $skipped) -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

# --- Report any run that failed to produce metrics.json -----------------------
$failed = @()
foreach ($j in $jobs) {
    $dir = $j.Args[$j.Args.IndexOf("--save_dir") + 1]
    if (-not (Test-Path (Join-Path $dir "metrics.json"))) {
        # [FIX-2026-08] .ExitCode throws if the process has not exited (e.g. the wait
        # above was interrupted), which would crash the report instead of printing it.
        $rec  = $all | Where-Object { $_.Name -eq $j.Name } | Select-Object -First 1
        $code = "?"
        if ($rec) { try { if ($rec.Proc.HasExited) { $code = $rec.Proc.ExitCode } else { $code = "still running" } } catch { $code = "unknown" } }
        $failed += [pscustomobject]@{ Name = $j.Name; ExitCode = $code }
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`n$($failed.Count) RUN(S) FAILED -- no metrics.json produced:" -ForegroundColor Red
    foreach ($f in $failed) {
        Write-Host ("  - {0,-28} exit={1}   log: {2}\{0}.err.log" -f $f.Name, $f.ExitCode, $LogDir) -ForegroundColor Red
    }
    Write-Host "`nRe-run with -SkipExisting to retry only the failures." -ForegroundColor Yellow
} else {
    Write-Host "`nAll $($jobs.Count) runs produced metrics.json." -ForegroundColor Green
}

# --- Collate into results/tables/*.csv ----------------------------------------
if (-not $NoCollate) {
    Write-Host "`nCollating results..." -ForegroundColor Cyan
    # [FIX-2026-08] $ActiveRoot / $TableDir keep probe and fixes runs out of the
    # committed results/tables/*.csv. For the normal targets both are unchanged.
    $TableDir = if ($ActiveRoot -eq $SaveDir) { "results/tables" } else { Join-Path $ActiveRoot "tables" }
    & $Python collate_results.py --root $ActiveRoot --bucket best  --out $TableDir
    & $Python collate_results.py --root $ActiveRoot --bucket final --out $TableDir | Out-Null
    Write-Host ("`nTables written to {0}/ (summary_best.csv, per_run_best.csv, *_final.csv)" -f $TableDir) -ForegroundColor Green
}

Write-Host ("`nDone at {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date)) -ForegroundColor Cyan
