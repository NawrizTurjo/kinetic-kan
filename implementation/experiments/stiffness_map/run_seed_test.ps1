<#
.SYNOPSIS
    Multi-seed sensitivity check for Track D's stiffness map -- grew out of the
    mu=2.0 local-minimum-trap finding in docs/16_p3_stiffness_map_findings.md.

.DESCRIPTION
    Launches one training run per (solver, mu, seed) combination, throttled to
    at most -MaxParallel concurrent windows (default 5) -- NOT all-at-once like
    the first version of this script, since extending to all 4 solvers means up
    to 72 jobs (4 solvers x 6 mu x 3 seeds), and running that many processes
    simultaneously would badly oversubscribe any normal machine.

    Each job runs in its own separate console window (same reasoning as
    run_parallel.ps1: a shared console means one Ctrl+C kills every job at
    once) but, unlike the first version of this script, does NOT wait on a
    keypress before closing -- with dozens of jobs queued, nobody is going to
    manually dismiss each one, and the throttling loop below needs
    Process.HasExited to become true on its own shortly after training
    finishes, not only after a human notices and presses Enter.

    Jobs are ordered slowest-solver-first (tsit5, then rk4, then midpoint, then
    euler) so the throttler front-loads the longest jobs -- this shortens the
    total makespan versus launching in an arbitrary order, since a slow job
    queued last would otherwise sit waiting behind a slot occupied by a fast
    job that started earlier for no good reason.

    Already-completed (solver, mu, seed) cells are skipped (their
    results/.../seed<seed>_mu<mu>_solver<solver>/metrics.json already exists),
    so re-running this script after a partial run or an interruption is safe
    and resumes rather than redoing everything.

.EXAMPLE
    # Extend the existing euler-only 3-seed check to the other 3 solvers
    .\run_seed_test.ps1 -Solvers midpoint,rk4,tsit5

.EXAMPLE
    # Only 2 windows at once instead of the default 5
    .\run_seed_test.ps1 -Solvers midpoint,rk4,tsit5 -MaxParallel 2
#>
param(
    [double[]] $Mus      = @(0.1, 0.5, 1.0, 2.0, 5.0, 8.0),
    [int[]]    $Seeds    = @(1, 7),
    [string[]] $Solvers  = @("tsit5", "rk4", "midpoint", "euler"),  # slowest first
    [int]      $MaxParallel = 5,
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venv = Join-Path $PSScriptRoot "..\..\venv\Scripts\python.exe"
$Python = if (Test-Path $venv) { (Resolve-Path $venv).Path } else { "python" }

$env:OMP_NUM_THREADS      = 1
$env:MKL_NUM_THREADS      = 1
$env:OPENBLAS_NUM_THREADS = 1

$ResultsRoot = Join-Path $PSScriptRoot "..\..\results\phase3\stiffness_map\_seed_test"

# PowerShell and Python format the SAME double differently: PowerShell's
# default string conversion of 1.0 is "1", Python's str(1.0)/an f-string is
# "1.0" -- and seed_test.py builds its output folder name from the Python-side
# float, so a naive "mu$mu" here would look for "..._mu1_..." while the real
# folder is "..._mu1.0_...". This exact bug was already found and fixed once
# in run_parallel.ps1 (see its Format-PyFloat) -- caught AGAIN here in manual
# validation before this script ever ran for real: the dry run reported only
# 4 of euler's 12 already-completed cells as done (the two non-whole-number mu
# values, 0.1 and 0.5, which happen to format the same way in both languages),
# and queued the other 8 to be uselessly retrained.
function Format-PyFloat($x) {
    if ($x -eq [Math]::Floor($x)) { return "{0:0}.0" -f $x }
    return "$x"
}

# Build the job list, slowest-solver-first (see .DESCRIPTION), and skip cells
# that already have a metrics.json -- resumable, idempotent re-runs.
$jobs = @()
$skipped = 0
foreach ($solver in $Solvers) {
    foreach ($mu in $Mus) {
        $muStr = Format-PyFloat $mu
        foreach ($seed in $Seeds) {
            $cellDir = Join-Path $ResultsRoot "seed${seed}_mu${muStr}_${solver}"
            # euler's existing results used the OLDER "seed<seed>_mu<mu>" naming
            # (no solver suffix, since only euler was tested at first) -- check
            # that path too so euler's already-done work is correctly skipped.
            $legacyCellDir = Join-Path $ResultsRoot "seed${seed}_mu${muStr}"
            if ((Test-Path (Join-Path $cellDir "metrics.json"))) {
                $skipped++
                continue
            }
            if ($solver -eq "euler" -and (Test-Path (Join-Path $legacyCellDir "metrics.json"))) {
                $skipped++
                continue
            }
            $jobs += [pscustomobject]@{ Solver = $solver; Mu = $mu; Seed = $seed; Dir = $cellDir }
        }
    }
}

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " Multi-seed sensitivity check" -ForegroundColor Cyan
Write-Host (" Solvers      : {0}" -f ($Solvers -join ", "))
Write-Host (" Mu values    : {0}" -f ($Mus -join ", "))
Write-Host (" Seeds        : {0}" -f ($Seeds -join ", "))
Write-Host (" Max parallel : {0} windows at a time" -f $MaxParallel)
Write-Host (" Jobs to run  : {0}  (skipped {1} already-done)" -f $jobs.Count, $skipped)
Write-Host "=========================================================================" -ForegroundColor Cyan

if ($jobs.Count -eq 0) {
    Write-Host "Nothing to do -- every requested (solver, mu, seed) cell already has a result." -ForegroundColor Green
    return
}

if ($DryRun) {
    Write-Host "`n--- DRY RUN: would launch these jobs, in this order, throttled to $MaxParallel at a time ---" -ForegroundColor Yellow
    for ($i = 0; $i -lt $jobs.Count; $i++) {
        $j = $jobs[$i]
        Write-Host ("  [{0,3}] solver={1,-8} mu={2,-4} seed={3}  -> {4}" -f ($i+1), $j.Solver, $j.Mu, $j.Seed, $j.Dir)
    }
    return
}

$all = @()
$launched = 0
$t0 = Get-Date

foreach ($j in $jobs) {
    while (@($all | Where-Object { -not $_.Proc.HasExited }).Count -ge $MaxParallel) {
        Start-Sleep -Seconds 10
    }
    # No trailing Read-Host: the window closes on its own a few seconds after
    # training finishes, so Process.HasExited (used by the throttle above)
    # reflects completion promptly instead of waiting indefinitely for a
    # keypress nobody is going to give across dozens of queued jobs.
    $inner = "& '$Python' seed_test.py $($j.Seed) $($j.Mu) $($j.Solver); " +
             "Write-Host ''; Write-Host 'seed=$($j.Seed) mu=$($j.Mu) solver=$($j.Solver) finished' -ForegroundColor Cyan; " +
             "Start-Sleep -Seconds 3"
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-Command", $inner) -PassThru
    $all += [pscustomobject]@{ Job = $j; Proc = $proc }
    $launched++
    Write-Host ("[{0,3}/{1}] solver={2,-8} mu={3,-4} seed={4}  pid {5}  {6:HH:mm:ss}" -f `
                $launched, $jobs.Count, $j.Solver, $j.Mu, $j.Seed, $proc.Id, (Get-Date)) -ForegroundColor Green
}

Write-Host "`nAll $launched job(s) launched (throttled to $MaxParallel at a time.) Waiting for the last ones..." -ForegroundColor Cyan
foreach ($r in $all) {
    try { $r.Proc.WaitForExit() } catch {}
}

$elapsed = (Get-Date) - $t0
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host (" Multi-seed sweep finished in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$missing = @()
foreach ($j in $jobs) {
    if (-not (Test-Path (Join-Path $j.Dir "metrics.json"))) { $missing += $j }
}
if ($missing.Count -gt 0) {
    Write-Host "`n$($missing.Count) job(s) did NOT produce metrics.json:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host ("  - solver={0} mu={1} seed={2}" -f $m.Solver, $m.Mu, $m.Seed) -ForegroundColor Red }
} else {
    Write-Host "`nAll $($jobs.Count) job(s) produced metrics.json." -ForegroundColor Green
}
