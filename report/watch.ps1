# Watches report.tex, kinetic-kan.sty, sections/, frontmatter/, appendix/ and
# references.bib, and recompiles on every save.
# Usage: powershell -ExecutionPolicy Bypass -File watch.ps1

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir

function Build {
    Write-Host "`n[$(Get-Date -Format HH:mm:ss)] rebuilding..." -ForegroundColor Cyan
    & "$dir\tools\tectonic.exe" -k report.tex
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] build OK -> report.pdf" -ForegroundColor Green
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

Write-Host "Watching report.tex, kinetic-kan.sty, sections/, frontmatter/, appendix/, references.bib. Ctrl+C to stop." -ForegroundColor Yellow

while ($true) {
    $result = $watcher.WaitForChanged([System.IO.WatcherChangeTypes]::Changed, 1000)
    if ($result.TimedOut -eq $false) {
        if ($result.Name -match '\.(tex|sty|bib)$') {
            Start-Sleep -Milliseconds 400   # debounce, editors often write twice
            Build
        }
    }
}
