# Prepare data_signer5/ for cross-signer collection (signeur 5)
# Usage :
#   .\scripts\prepare_signer5.ps1
#   .\scripts\prepare_signer5.ps1 -Fresh

param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$Dir = Join-Path $Root "data_signer5"

if ($Fresh -and (Test-Path $Dir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Write-Host "Archivage de data_signer5 -> data_signer5_archive_$stamp" -ForegroundColor Yellow
    Move-Item $Dir (Join-Path $Root "data_signer5_archive_$stamp")
}

Write-Host ""
Write-Host "Preparation signeur 5" -ForegroundColor Cyan
Write-Host ""

& $Py scripts/prepare_signer5_status.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
