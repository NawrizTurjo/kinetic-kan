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

.PARAMETER Serial
    Run one job at a time. REQUIRED if you intend to report wall-clock timings.

.EXAMPLE
    .\run_phase2.ps1 -Only solvers,activations,models
    .\run_phase2.ps1 -Only solvers -Seeds 42,1337,2024 -MaxParallel 3
    .\run_phase2.ps1 -Only all -Epochs 20 -DryRun      # verify the plan first
#>

param(
    [int]    $Epochs        = 10000,
    [double] $Lr            = 2e-3,
    [int[]]  $Seeds         = @(42),
    [string] $SaveDir       = "results/benchmarks",
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

$LogDir = Join-Path $SaveDir "_logs"
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
    Add-Job "pendulum" "$SaveDir/pendulum" @("--dataset", "damped_pendulum", "--basis", "rbf", "--solver", "tsit5")
    Add-Job "sir"      "$SaveDir/sir"      @("--dataset", "sir",             "--basis", "rbf", "--solver", "tsit5")
}
if ($Only -contains "lorenz") {
    Add-Job "lorenz" "$SaveDir/lorenz" `
        @("--dataset", "lorenz", "--basis", "rbf", "--solver", "tsit5",
          "--dt", $LorenzDt, "--t_end", $LorenzTEnd, "--t_train_end", $LorenzTTrain)
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
Write-Host (" Output        : {0}" -f $SaveDir)
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
    foreach ($r in $all) { $r.Proc | Wait-Process }
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
        $code = ($all | Where-Object { $_.Name -eq $j.Name } | Select-Object -First 1).Proc.ExitCode
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
    & $Python collate_results.py --root $SaveDir --bucket best
    & $Python collate_results.py --root $SaveDir --bucket final | Out-Null
    Write-Host "`nTables written to results/tables/ (summary_best.csv, per_run_best.csv, *_final.csv)" -ForegroundColor Green
}

Write-Host ("`nDone at {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date)) -ForegroundColor Cyan
