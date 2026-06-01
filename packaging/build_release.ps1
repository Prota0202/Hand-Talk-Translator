# =============================================================================
#  Hand Talk Translator — build a standalone .exe with PyInstaller
#
#  Usage (from project root):
#      .\packaging\build_release.ps1
#      .\packaging\build_release.ps1 -Clean   # nuke build/ + dist/ first
#      .\packaging\build_release.ps1 -NoZip   # skip the final .zip
#
#  Output:
#      dist/HandTalkTranslator/HandTalkTranslator.exe   (entry point)
#      dist/HandTalkTranslator/models/*                  (copied next to exe)
#      dist/HandTalkTranslator/sessions/demo.jsonl       (replay demo)
#      dist/HandTalkTranslator-<version>.zip             (one-click distrib)
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$NoZip,
    [string]$Version = (Get-Date -Format "yyyyMMdd")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # silence Invoke-WebRequest progress

# ── Paths ────────────────────────────────────────────────────────────────────
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Spec        = Join-Path $PSScriptRoot "HandTalkTranslator.spec"
$DistDir     = Join-Path $ProjectRoot "dist"
$BuildDir    = Join-Path $ProjectRoot "build"
$AppDir      = Join-Path $DistDir     "HandTalkTranslator"

# Prefer the dedicated CPU-only venv-release if it exists (slim ~600 MB
# bundle); fall back to the main venv (full GPU torch ~4.5 GB bundle).
$ReleaseVenv = Join-Path $ProjectRoot "venv-release"
$MainVenv    = Join-Path $ProjectRoot "venv"
if (Test-Path (Join-Path $ReleaseVenv "Scripts\python.exe")) {
    $Venv     = $ReleaseVenv
    $VenvKind = "venv-release (CPU-only torch, slim build)"
} elseif (Test-Path (Join-Path $MainVenv "Scripts\python.exe")) {
    $Venv     = $MainVenv
    $VenvKind = "venv (GPU torch, full ~4.5 GB build)"
} else {
    Write-Error "Aucun venv trouve. Cree-en un :  py -3.11 -m venv venv  (ou .\packaging\setup_release_venv.ps1)"
    exit 1
}
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host ""
Write-Host "=========================================================="
Write-Host " Hand Talk Translator - build PyInstaller (.exe)"
Write-Host "=========================================================="
Write-Host " Project : $ProjectRoot"
Write-Host " Version : $Version"
Write-Host " Venv    : $VenvKind"
Write-Host " Python  : $Python"
Write-Host ""

# Install/upgrade PyInstaller in the venv if needed
& $Python -m pip show pyinstaller > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[1/5] Installation de pyinstaller dans le venv..."
    & $Python -m pip install --upgrade pyinstaller
} else {
    Write-Host "[1/5] pyinstaller deja installe."
}

# ── Cleanup if asked ─────────────────────────────────────────────────────────
if ($Clean) {
    Write-Host "[2/5] Nettoyage de build/ et dist/ (option -Clean)..."
    Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $DistDir  -ErrorAction SilentlyContinue
} else {
    Write-Host "[2/5] (clean skip - utilise -Clean pour forcer)"
}

# ── Pre-flight: required runtime files ──────────────────────────────────────
Write-Host "[3/5] Verification des modeles..."
$RequiredModels = @(
    "models\gesture_model.pth",
    "models\labels.json"
)
$Missing = @()
foreach ($m in $RequiredModels) {
    if (-not (Test-Path (Join-Path $ProjectRoot $m))) {
        $Missing += $m
    }
}
if ($Missing.Count -gt 0) {
    Write-Warning "Modeles manquants : $($Missing -join ', ')"
    Write-Warning "Le .exe sera construit mais ne pourra pas demarrer sans ces fichiers."
}

# Optional runtime files (copied if present)
$OptionalRuntime = @(
    "models\glove_model.pth",
    "models\glove_labels.json",
    "models\glove_calibration.json",
    "sessions\demo.jsonl"
)

# ── PyInstaller build ────────────────────────────────────────────────────────
Write-Host "[4/5] Lancement de PyInstaller (peut prendre 2-5 min, ~500 MB sortie)..."
Push-Location $ProjectRoot
try {
    # PyInstaller writes its progress to stderr; PowerShell with
    # ErrorActionPreference=Stop would otherwise abort on the first INFO line.
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python -m PyInstaller `
            $Spec `
            --noconfirm `
            --clean `
            --log-level WARN 2>&1 | ForEach-Object { Write-Host $_ }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($code -ne 0) {
        throw "PyInstaller a echoue (code $code)"
    }
} finally {
    Pop-Location
}

# ── Copy runtime assets next to the .exe ─────────────────────────────────────
Write-Host "[5/5] Copie des modeles et sessions a cote du .exe..."
$DestModels   = Join-Path $AppDir "models"
$DestSessions = Join-Path $AppDir "sessions"
New-Item -ItemType Directory -Force -Path $DestModels   | Out-Null
New-Item -ItemType Directory -Force -Path $DestSessions | Out-Null

foreach ($f in ($RequiredModels + $OptionalRuntime)) {
    $src = Join-Path $ProjectRoot $f
    if (Test-Path $src) {
        $dst = Join-Path $AppDir $f
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path $dstDir)) {
            New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        }
        Copy-Item $src $dst -Force
        Write-Host "   + $f"
    }
}

# Ship a tiny README inside the bundle
$ReadmeText = @"
Hand Talk Translator — version $Version
========================================

Double-cliquez sur HandTalkTranslator.exe pour lancer l'application.

Pre-requis :
  - Windows 10/11 64-bit
  - Webcam fonctionnelle (Logitech C920 testee)
  - (optionnel) Connexion internet pour la synthese vocale Edge TTS

Sous-dossiers :
  - models/   : reseaux de neurones (LSTM vision + LSTM gant)
  - sessions/ : JSONL de chaque execution (sign / phrase / voix)
  - data/     : (optionnel) donnees d'entrainement

Mode de demonstration sans webcam :
    HandTalkTranslator.exe --replay sessions/demo.jsonl

Aide complete :
    HandTalkTranslator.exe --help

Genere le $(Get-Date -Format "dd/MM/yyyy HH:mm") par build_release.ps1.
"@
Set-Content -Path (Join-Path $AppDir "README.txt") -Value $ReadmeText -Encoding UTF8

# ── Final size + zip ─────────────────────────────────────────────────────────
$Size = (Get-ChildItem $AppDir -Recurse | Measure-Object Length -Sum).Sum
$SizeMB = [math]::Round($Size / 1MB, 1)

Write-Host ""
Write-Host "=========================================================="
Write-Host " Build OK"
Write-Host "=========================================================="
Write-Host " Dossier : $AppDir"
Write-Host " Taille  : $SizeMB MB"
Write-Host " Lancer  : $AppDir\HandTalkTranslator.exe"
Write-Host ""

if (-not $NoZip) {
    $ZipPath = Join-Path $DistDir "HandTalkTranslator-$Version.zip"
    if (Test-Path $ZipPath) {
        Remove-Item $ZipPath -Force
    }
    Write-Host "Creation du zip de distribution..."
    Compress-Archive -Path "$AppDir\*" -DestinationPath $ZipPath -CompressionLevel Optimal
    $ZipMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
    Write-Host " Zip OK : $ZipPath ($ZipMB MB)"
    Write-Host ""
}

Write-Host "Test rapide (Ctrl+C pour arreter) :"
Write-Host "    & '$AppDir\HandTalkTranslator.exe' --help"
Write-Host ""
