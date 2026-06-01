# Collecte signeur 3 (adulte) - reprise automatique des signes manquants
# Usage : .\scripts\collect_signer3.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COLLECTE SIGNEUR 3 (ADULTE)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Protocole : 7 glosses (mots + chiffres, sans alphabet)"
Write-Host "  Mode      : reprise - seulement ce qui manque"
Write-Host "  Dossier   : data_signer3\"
Write-Host ""
Write-Host "  Deja fait : MOI, NOM, Bonjour, 2, 4"
Write-Host "  Reste     : ans, ETUDIANT"
Write-Host ""
Write-Host "  CONSEILS :"
Write-Host "  - Face a la webcam, mains bien visibles"
Write-Host "  - Maintenir chaque signe ~1 seconde, retirer la main entre deux"
Write-Host "  - Q pour quitter"
Write-Host ""
Read-Host "Appuyez sur Entree quand l adulte est pret"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "data_signer3") | Out-Null

& $Py collect_data.py `
    --data-dir data_signer3 `
    --signer3 `
    --resume `
    --samples 5 `
    --auto

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Collecte terminee. Lancez :" -ForegroundColor Green
Write-Host "  .\scripts\eval_cross_signers.ps1" -ForegroundColor Green
Write-Host ""
