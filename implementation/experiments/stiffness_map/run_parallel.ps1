<#
.SYNOPSIS
    Parallel launcher for Track D's stiffness-map sweep (run_sweep.py).

.DESCRIPTION
    Splits the (solver x mu) grid into one subprocess PER SOLVER (4 jobs by
    default -- one per SOLVERS entry) and runs them concurrently, each writing
    to its own disjoint cell folders under the same results root, so there is
    no write collision between jobs.

    Three mutually exclusive launch modes, resolved ONCE up front into
    $Mode before anything else happens:
      - "windows"  (default): each job gets its OWN separate console window
        (Start-Process WITHOUT -NoNewWindow). No wt.exe dependency. This is
        also more robust than the headless mode below: a -NoNewWindow child
        shares the PARENT console's Ctrl+C / close signal, so interrupting
        (or closing) the launcher's own window kills every child job at once
        -- this is what actually happened in production (all 4 jobs died
        simultaneously with KeyboardInterrupt seconds after launch, mid
        `import torch`, when the parent window was interrupted). A real,
        separate console window is immune to that.
      - "grid": one Windows Terminal window, one live pane per job, tiled.
        Requires wt.exe on PATH.
      - "headless": hidden background processes, stdout/stderr redirected to
        per-job log files under results/phase3/stiffness_map/_logs/. Useful
        for CI or a scripted context with no interactive console at all.

    All three reuse the same underlying wait/report pattern already validated
    in implementation/run_phase2.ps1:
      - Start-Process -PassThru, then $proc.WaitForExit() (an OS handle), NOT
        `Wait-Process` (resolves by PID; can be confused by Windows recycling
        a PID after an early job exits -- see docs/07_fix_changelog.md "Bug
        fix: PID-reuse race in the completion wait"). Wrapped in try/catch.
      - .ExitCode is only trusted after checking .HasExited AND is NOT the
        authoritative success signal (it can read back $null even on a
        successful exit, a known .NET redirected-stream race) -- actual
        metrics.json presence on disk is authoritative, same as
        run_phase2.ps1.
      - OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS are set so
        MaxParallel * ThreadsPerJob stays within the machine's logical core
        count.

.PARAMETER Stage
    "probe" (2,000 epochs/cell) or "full" (10,000 epochs/cell).

.PARAMETER Solvers
    Which solvers to split across parallel jobs. Default: all four.

.PARAMETER Mus
    Damping ratios every job sweeps. Default: the full 6-value grid.

.PARAMETER Dt
    Fixed step size. Default: 0.05.

.EXAMPLE
    # Default: 4 separate visible PowerShell windows, no wt.exe needed
    .\run_parallel.ps1 -Stage probe

.EXAMPLE
    # Tiled single-window grid (needs Windows Terminal)
    .\run_parallel.ps1 -Stage probe -Mode grid

.EXAMPLE
    # Hidden background + log files (e.g. no interactive console available)
    .\run_parallel.ps1 -Stage probe -Mode headless
#>
param(
    [ValidateSet("probe", "full")]
    [string]   $Stage        = "probe",
    [string[]] $Solvers      = @("euler", "midpoint", "rk4", "tsit5"),
    [double[]] $Mus          = @(0.1, 0.5, 1.0, 2.0, 5.0, 8.0),
    [double]   $Dt           = 0.05,
    [int]      $MaxParallel  = 0,          # 0 = auto (one slot per job, capped by cores)
    [int]      $ThreadsPerJob = 0,         # 0 = auto
    [string]   $OutRoot      = "",         # "" = run_sweep.py's own default
    [int]      $Epochs       = 0,          # 0 = use run_sweep.py's stage default (2000/10000)
    [switch]   $SkipExisting,              # resume: reuse a cell's metrics.json instead of retraining it
    [switch]   $SaveTrajectories,          # also save trajectory.npz per cell, for plotting
    [ValidateSet("windows", "grid", "headless")]
    [string]   $Mode         = "windows",  # default: separate real console windows, no wt.exe
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venv = Join-Path $PSScriptRoot "..\..\venv\Scripts\python.exe"
$Python = if (Test-Path $venv) { (Resolve-Path $venv).Path } else { "python" }

# --- Auto-size parallelism/threading to the actual machine ---------------
$logical = [Environment]::ProcessorCount
if ($MaxParallel -le 0)   { $MaxParallel   = [Math]::Max(1, [Math]::Min($Solvers.Count, $logical)) }
if ($ThreadsPerJob -le 0) { $ThreadsPerJob = [Math]::Max(1, [Math]::Floor($logical / $MaxParallel)) }

$DefaultStiffnessMapRoot = Join-Path $PSScriptRoot "..\..\results\phase3\stiffness_map"
$LogDir = Join-Path $DefaultStiffnessMapRoot "_logs"
# NOTE: $stageRoot (used by the end-of-run "did every cell produce metrics.json"
# check) is resolved LATER, after -SaveTrajectories' auto-redirect of $OutRoot
# (see below) has already happened -- not here. An earlier version computed it
# from the hardcoded default path at this point in the script, which silently
# looked in the wrong directory whenever -OutRoot (or -SaveTrajectories, which
# sets it) was used, and would have falsely reported every real cell as missing.

# ===========================================================================
# STEP 1: resolve $Mode fully, with any fallback, BEFORE anything else reads
# it. This is deliberate -- an earlier version resolved -Grid's wt.exe
# availability fallback AFTER the dry-run check had already passed for the
# pre-fallback value, so `-Grid -DryRun` on a machine without wt.exe silently
# ran a REAL sweep instead of a dry run. Keeping mode resolution as a single
# unconditional first step, before either the dry-run branch or the launch
# branch exist, makes that class of bug structurally impossible: there is
# only one $Mode value by the time anything downstream looks at it.
# ===========================================================================
if ($Mode -eq "grid") {
    $wt = Get-Command wt.exe -ErrorAction SilentlyContinue
    if (-not $wt) {
        Write-Host "-Mode grid requested but wt.exe (Windows Terminal) was not found on PATH." -ForegroundColor Yellow
        Write-Host "Falling back to -Mode windows (separate console windows)." -ForegroundColor Yellow
        $Mode = "windows"
    } elseif ($Solvers.Count -lt 1 -or $Solvers.Count -gt 4) {
        Write-Host ("-Mode grid only supports 1-4 jobs (got {0}); falling back to -Mode windows." -f $Solvers.Count) -ForegroundColor Yellow
        $Mode = "windows"
    }
}

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host (" Track D parallel {0} sweep  (mode: {1})" -f $Stage, $Mode) -ForegroundColor Cyan
Write-Host (" Logical cores : {0}" -f $logical)
Write-Host (" Parallelism   : {0} concurrent job(s) x {1} thread(s) each" -f $MaxParallel, $ThreadsPerJob)
Write-Host (" Jobs (one per solver): {0}" -f ($Solvers -join ", "))
Write-Host (" Mu grid       : {0}" -f ($Mus -join ", "))
Write-Host (" dt            : {0}" -f $Dt)
Write-Host "=========================================================================" -ForegroundColor Cyan

$env:OMP_NUM_THREADS      = $ThreadsPerJob
$env:MKL_NUM_THREADS      = $ThreadsPerJob
$env:OPENBLAS_NUM_THREADS = $ThreadsPerJob

# -SaveTrajectories re-runs (and re-scores) training rather than reading the
# existing metrics.json, so it must never point at the same root as the
# already-completed sweep -- run_sweep.py itself refuses that combination, but
# defaulting it here means the common case (no -OutRoot given) is safe by
# construction rather than relying on the user remembering to redirect it.
if ($SaveTrajectories -and $OutRoot -eq "") {
    $OutRoot = Join-Path $PSScriptRoot "..\..\results\phase3\stiffness_map\probe_traj"
    Write-Host ("-SaveTrajectories: defaulting -OutRoot to {0} (never the original sweep root)" -f $OutRoot) -ForegroundColor Yellow
}

# Resolved NOW, after $OutRoot's final value (including -SaveTrajectories' own
# override above) is known -- this is the path the end-of-run missing-cell
# check actually looks in, so it must track $OutRoot, not the hardcoded default.
$EffectiveStiffnessMapRoot = if ($OutRoot -ne "") { $OutRoot } else { $DefaultStiffnessMapRoot }
$stageRoot = Join-Path $EffectiveStiffnessMapRoot $Stage

$jobs = @()
foreach ($solver in $Solvers) {
    $args = @("run_sweep.py", "--stage", $Stage, "--solvers", $solver, "--mus") + ($Mus | ForEach-Object { "$_" }) + @("--dts", "$Dt")
    if ($OutRoot -ne "") { $args += @("--out_root", $OutRoot) }
    if ($Epochs -gt 0) { $args += @("--epochs", "$Epochs") }
    if ($SkipExisting) { $args += @("--skip-existing") }
    if ($SaveTrajectories) { $args += @("--save-trajectories") }
    $jobs += [pscustomobject]@{ Name = $solver; Args = $args }
}

# ===========================================================================
# STEP 2: the ONE dry-run checkpoint, covering all three (fully resolved)
# modes. Nothing below this point runs unless -DryRun was NOT passed.
# ===========================================================================
if ($DryRun) {
    Write-Host "`n--- DRY RUN (mode: $Mode) -- nothing will be launched ---" -ForegroundColor Yellow
    switch ($Mode) {
        "windows" {
            foreach ($j in $jobs) {
                $inner = "& '$Python' " + ($j.Args -join ' ') + "; Write-Host ''; Write-Host '[$($j.Name)] finished -- press Enter to close' -ForegroundColor Cyan; Read-Host"
                Write-Host ("  [{0}] new window running: {1}" -f $j.Name, $inner)
            }
        }
        "grid" {
            foreach ($j in $jobs) {
                Write-Host ("  [{0}] pane running: {1} {2}" -f $j.Name, $Python, ($j.Args -join " "))
            }
        }
        "headless" {
            foreach ($j in $jobs) {
                Write-Host ("  [{0}] background, log-redirected: {1} {2}" -f $j.Name, $Python, ($j.Args -join " "))
            }
        }
    }
    return
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ===========================================================================
# STEP 3: launch, by mode. All three populate $all = @({Name, Proc}, ...) so
# the wait/report logic after this block is shared and mode-agnostic.
# ===========================================================================
$all = @()
$t0 = Get-Date

if ($Mode -eq "grid") {
    function New-PaneCommandLine($job) {
        return "& '$Python' " + ($job.Args -join ' ') + "; Write-Host ''; Write-Host '[$($job.Name)] finished -- press Enter to close this pane' -ForegroundColor Cyan; Read-Host"
    }
    $cmds = $jobs | ForEach-Object { New-PaneCommandLine $_ }
    # `;` between wt actions is a literal argv token here (each element of
    # $wtArgs is one argument passed to wt.exe directly), not a PowerShell
    # statement separator -- this is wt's own action-chaining syntax.
    $wtArgs = switch ($cmds.Count) {
        1 { @("new-tab", "powershell", "-NoExit", "-Command", $cmds[0]) }
        2 { @(
                "new-tab", "powershell", "-NoExit", "-Command", $cmds[0], ";",
                "split-pane", "-V", "powershell", "-NoExit", "-Command", $cmds[1]
            ) }
        3 { @(
                "new-tab", "powershell", "-NoExit", "-Command", $cmds[0], ";",
                "split-pane", "-H", "powershell", "-NoExit", "-Command", $cmds[2], ";",
                "move-focus", "up", ";",
                "split-pane", "-V", "powershell", "-NoExit", "-Command", $cmds[1]
            ) }
        4 { @(
                "new-tab", "powershell", "-NoExit", "-Command", $cmds[0], ";",
                "split-pane", "-H", "powershell", "-NoExit", "-Command", $cmds[1], ";",
                "move-focus", "up", ";",
                "split-pane", "-V", "powershell", "-NoExit", "-Command", $cmds[2], ";",
                "move-focus", "down", ";",
                "split-pane", "-V", "powershell", "-NoExit", "-Command", $cmds[3]
            ) }
    }
    Write-Host ("`nOpening Windows Terminal with {0} tiled pane(s), one per solver..." -f $jobs.Count) -ForegroundColor Cyan
    & wt.exe @wtArgs
    Write-Host "Grid window launched. This script does NOT wait for it (wt.exe returns immediately) --"
    Write-Host "each pane runs independently. When all panes report 'finished', run:"
    Write-Host "  python run_sweep.py --stage aggregate"
    return
}

foreach ($j in $jobs) {
    while (@($all | Where-Object { -not $_.Proc.HasExited }).Count -ge $MaxParallel) {
        Start-Sleep -Seconds 5
    }

    if ($Mode -eq "windows") {
        # Genuinely separate console window -- NOT -NoNewWindow. This is the
        # actual fix for the production failure: a -NoNewWindow child shares
        # the parent console's Ctrl+C/close signal, so interrupting the
        # launcher window kills every child simultaneously. A real window has
        # its own console and cannot be killed that way.
        $inner = "& '$Python' " + ($j.Args -join ' ') + "; Write-Host ''; Write-Host '[$($j.Name)] finished -- press Enter to close' -ForegroundColor Cyan; Read-Host"
        $proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $inner) -PassThru
        Write-Host ("[{0,2}/{1}] {2,-10} pid {3}  {4:HH:mm:ss}  (own window)" -f `
                    ($all.Count + 1), $jobs.Count, $j.Name, $proc.Id, (Get-Date)) -ForegroundColor Green
    } else {
        # headless
        $out = Join-Path $LogDir "$($j.Name).log"
        $err = Join-Path $LogDir "$($j.Name).err.log"
        $proc = Start-Process -FilePath $Python -ArgumentList $j.Args -NoNewWindow -PassThru `
                              -RedirectStandardOutput $out -RedirectStandardError $err
        Write-Host ("[{0,2}/{1}] {2,-10} pid {3}  {4:HH:mm:ss}  log: {5}" -f `
                    ($all.Count + 1), $jobs.Count, $j.Name, $proc.Id, (Get-Date), $out) -ForegroundColor Green
    }
    $all += [pscustomobject]@{ Name = $j.Name; Proc = $proc }
}

Write-Host "`n$($all.Count) job(s) launched. Waiting for completion..." -ForegroundColor Cyan
if ($Mode -eq "windows") {
    Write-Host "(Each job has its own window -- watch them directly. Closing THIS window/script"
    Write-Host " does not stop them; they are independent processes.)"
} else {
    Write-Host "(Tail the .log files under $LogDir to watch progress.)"
}

foreach ($r in $all) {
    try {
        $r.Proc.WaitForExit()
    } catch {
        Write-Host ("  ! could not wait on {0}: {1}" -f $r.Name, $_.Exception.Message) -ForegroundColor Yellow
    }
}

$elapsed = (Get-Date) - $t0
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host (" Parallel sweep finished in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$exitCodes = @{}
foreach ($r in $all) {
    # .ExitCode can read back $null immediately after WaitForExit() returns
    # when stdout/stderr are redirected (headless mode) -- a known .NET race
    # between the exit event and the redirected-stream pump finishing. Not
    # itself evidence of failure; the authoritative check is metrics.json
    # presence below, not this.
    $code = "?"
    try {
        if ($r.Proc.HasExited) {
            $ec = $r.Proc.ExitCode
            $code = if ($null -eq $ec) { "unknown (null ExitCode)" } else { $ec }
        } else { $code = "still running" }
    } catch { $code = "unknown (threw)" }
    $exitCodes[$r.Name] = $code
}
Write-Host "`nExit codes: " -NoNewline
Write-Host (($exitCodes.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "  ")

# Authoritative check: did every requested (solver, mu, dt) cell actually
# produce a metrics.json? PowerShell and Python format the SAME double
# differently (PowerShell's "$8.0" -> "8", Python's str(8.0) -> "8.0"), and
# run_sweep.py's _cell_dir() builds folder names from the Python-side float --
# so path-reconstruction here MUST mimic Python's formatting, not PowerShell's
# default. Format-PyFloat does that. (Caught in manual validation: without
# this, 2 genuinely-successful cells were misreported as missing.)
function Format-PyFloat($x) {
    if ($x -eq [Math]::Floor($x)) { return "{0:0}.0" -f $x }
    return "$x"
}
$missing = @()
foreach ($solver in $Solvers) {
    foreach ($mu in $Mus) {
        $cellDir = Join-Path $stageRoot ("{0}_mu{1}_dt{2}" -f $solver, (Format-PyFloat $mu), (Format-PyFloat $Dt))
        if (-not (Test-Path (Join-Path $cellDir "metrics.json"))) {
            $missing += "$solver mu=$mu dt=$Dt"
        }
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`n$($missing.Count) cell(s) did NOT produce metrics.json:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "  - $m" -ForegroundColor Red }
    if ($Mode -eq "headless") { Write-Host "Check the matching .err.log under $LogDir" -ForegroundColor Red }
    else { Write-Host "Check the corresponding window for a traceback." -ForegroundColor Red }
} else {
    Write-Host "`nAll $($Solvers.Count * $Mus.Count) requested cells produced metrics.json." -ForegroundColor Green
}
Write-Host "`nNext: python run_sweep.py --stage aggregate   (builds table5.json + heatmap from whatever landed)"
