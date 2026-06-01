# Prepare data_signer4/ for cross-signer collection (signeur 4)
# Usage :
#   .\scripts\prepare_signer4.ps1
#   .\scripts\prepare_signer4.ps1 -Fresh

param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$Dir = Join-Path $Root "data_signer4"

if ($Fresh -and (Test-Path $Dir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $archive = Join-Path $Root "data_signer4_archive_$stamp"
    Write-Host "Archivage de data_signer4 -> data_signer4_archive_$stamp" -ForegroundColor Yellow
    Move-Item $Dir $archive
}

Write-Host ""
Write-Host "Preparation signeur 4" -ForegroundColor Cyan
Write-Host ""

& $Py scripts/prepare_signer4_status.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
