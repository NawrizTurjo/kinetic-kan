# Watches A_08.tex, kk-acm.sty, sections/ and references.bib,
# and recompiles A_08.pdf on every save.
# Usage: powershell -ExecutionPolicy Bypass -File watch.ps1

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir

function Build {
    Write-Host "`n[$(Get-Date -Format HH:mm:ss)] rebuilding..." -ForegroundColor Cyan
    & "$dir\tools\tectonic.exe" -k A_08.tex
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] build OK -> A_08.pdf" -ForegroundColor Green
    } else {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] build FAILED" -ForegroundColor Red
    }
}

Build

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $dir
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite
$watcher.EnableRaisingEvents = $true

Write-Host "Watching A_08.tex, kk-acm.sty, sections/, references.bib. Ctrl+C to stop." -ForegroundColor Yellow

while ($true) {
    $result = $watcher.WaitForChanged([System.IO.WatcherChangeTypes]::Changed, 1000)
    if ($result.TimedOut -eq $false) {
        if ($result.Name -match '\.(tex|sty|bib)$') {
            Start-Sleep -Milliseconds 400   # debounce, editors often write twice
            Build
        }
    }
}
