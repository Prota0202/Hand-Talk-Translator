# Recollecte propre des donnees vision
#
# Usage :
#   .\scripts\recollect_fresh.ps1
#
# Sauvegarde data/ actuel, repart de zero, collecte manuelle puis reentraine.

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$archive = Join-Path $Root "data_archive_$stamp"
if (Test-Path (Join-Path $Root "data")) {
    Write-Host "Archivage data/ -> $archive" -ForegroundColor Yellow
    Move-Item (Join-Path $Root "data") $archive
}
New-Item -ItemType Directory -Path (Join-Path $Root "data") | Out-Null

Write-Host ""
Write-Host "Collecte MANUELLE (ESPACE = enregistrer chaque signe)" -ForegroundColor Cyan
Write-Host "Meme webcam, meme lumiere, meme distance que pour la demo live." -ForegroundColor Cyan
Write-Host ""
& $Py collect_data.py --manual --samples 30

Write-Host ""
Write-Host "Entrainement..." -ForegroundColor Cyan
& $Py train_model.py

Write-Host ""
Write-Host "Termine. Lance : python main.py" -ForegroundColor Green
