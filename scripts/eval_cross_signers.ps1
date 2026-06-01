# Evaluation cross-signeur (signeurs 2–5)
# Usage : .\scripts\eval_cross_signers.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

Write-Host ""
Write-Host "Evaluation cross-signeur..." -ForegroundColor Cyan

& $Py evaluate_all_cross_signers.py --skip-missing

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Rapports generes :" -ForegroundColor Green
Write-Host "  models\cross_signer_report.md"
Write-Host "  models\cross_signer_latex.tex  (a coller dans le rapport)"
Write-Host ""
