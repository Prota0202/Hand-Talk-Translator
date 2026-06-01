# Collecte signeur 5 — protocole cross-signeur
# Usage : .\scripts\collect_signer5.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COLLECTE SIGNEUR 5" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

& $Py scripts/prepare_signer5_status.py

Write-Host ""
Write-Host "  CONSEILS :"
Write-Host "  - Face a la webcam, mains bien visibles"
Write-Host "  - Maintenir chaque signe ~1 seconde, retirer la main entre deux"
Write-Host "  - Q pour quitter"
Write-Host ""
Read-Host "Appuyez sur Entree quand le signeur est pret"

& $Py collect_data.py `
    --data-dir data_signer5 `
    --signer5 `
    --resume `
    --samples 5 `
    --auto

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Collecte terminee. Lancez :" -ForegroundColor Green
Write-Host "  .\scripts\eval_cross_signers.ps1" -ForegroundColor Green
Write-Host ""
