# =============================================================================
#  Hand Talk Translator — bootstrap the dedicated "release" venv
#
#  Creates `venv-release/` with **CPU-only PyTorch** + minimal deps so that
#  the resulting .exe weighs ~600 MB instead of ~4.5 GB. The main `venv/`
#  (GPU PyTorch for training) is left untouched.
#
#  Run from project root:
#      .\packaging\setup_release_venv.ps1
#      .\packaging\setup_release_venv.ps1 -Force   # nuke and recreate
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseVenv = Join-Path $ProjectRoot "venv-release"
$Reqs        = Join-Path $PSScriptRoot "requirements-release.txt"

Write-Host ""
Write-Host "=========================================================="
Write-Host " Hand Talk Translator - bootstrap venv-release (CPU-only)"
Write-Host "=========================================================="
Write-Host " Cible    : $ReleaseVenv"
Write-Host " Reqs     : $Reqs"
Write-Host ""

if ($Force -and (Test-Path $ReleaseVenv)) {
    Write-Host "[*] Suppression du venv existant (-Force)..."
    Remove-Item -Recurse -Force $ReleaseVenv
}

if (-not (Test-Path $ReleaseVenv)) {
    Write-Host "[1/3] Creation du venv Python 3.11..."
    & py -3.11 -m venv $ReleaseVenv
    if ($LASTEXITCODE -ne 0) { throw "py -3.11 indisponible" }
} else {
    Write-Host "[1/3] venv-release existe deja."
}

$Python = Join-Path $ReleaseVenv "Scripts\python.exe"

Write-Host "[2/3] Mise a jour de pip..."
$prevPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python -m pip install --upgrade pip 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade a echoue" }

    Write-Host "[3/3] Installation des deps minimales (torch CPU-only, ~10 min)..."
    & $Python -m pip install --no-cache-dir -r $Reqs 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "pip install -r $Reqs a echoue" }
} finally {
    $ErrorActionPreference = $prevPref
}

# Sanity check: confirm torch is the CPU build
& $Python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

Write-Host ""
Write-Host "=========================================================="
Write-Host " venv-release pret"
Write-Host "=========================================================="
Write-Host " Pour builder le .exe optimise :"
Write-Host "    .\packaging\build_release.ps1 -Clean"
Write-Host ""
Write-Host " (Le script detecte automatiquement venv-release et l'utilise)"
Write-Host ""
