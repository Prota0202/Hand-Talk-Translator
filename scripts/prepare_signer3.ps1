# Prepare data_signer3/ for cross-signer collection (signeur 3 — adulte)
# Usage :
#   .\scripts\prepare_signer3.ps1           # cree les sous-dossiers + affiche l etat
#   .\scripts\prepare_signer3.ps1 -Fresh    # archive l ancien dossier puis repart a zero

param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$Dir = Join-Path $Root "data_signer3"

if ($Fresh -and (Test-Path $Dir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $archive = Join-Path $Root "data_signer3_archive_$stamp"
    Write-Host "Archivage de data_signer3 -> data_signer3_archive_$stamp" -ForegroundColor Yellow
    Move-Item $Dir $archive
}

Write-Host ""
Write-Host "Preparation signeur 3" -ForegroundColor Cyan
Write-Host ""

& $Py scripts/prepare_signer3_status.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
