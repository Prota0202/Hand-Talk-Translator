# =============================================================================
#  Hand Talk Translator — PyInstaller spec
#  Build with:   .\packaging\build_release.ps1
#  or directly:  pyinstaller packaging/HandTalkTranslator.spec --clean --noconfirm
#
#  Output       : dist/HandTalkTranslator/HandTalkTranslator.exe   (--onedir)
#  Models / data: copied **next to** the .exe so the user can replace them
#                 (gesture_model.pth, glove_model.pth, labels.json, etc.)
#                 without rebuilding.
# =============================================================================

# pylint: disable=undefined-variable    # PyInstaller injects globals at runtime

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH).resolve().parent

# ── Project Python modules that must be picked up explicitly because
# they are imported via short names from main.py (PyInstaller may already
# trace them; we list them defensively).
HIDDEN_IMPORTS = [
    "config",
    "feature_extractor",
    "gesture_recognizer",
    "hand_detector",
    "latency_tracker",
    "lsf_translator",
    "model",
    "replay_player",
    "sentence_builder",
    "session_logger",
    "speech_engine",
    "speech_listener",
    "splash",
    "ui_renderer",
]

# ── Heavyweight third-party libraries that ship bundled assets.
# MediaPipe in particular keeps tflite models inside `mediapipe/modules/`,
# AND has native C bindings inside `mediapipe.tasks.c.*` (.pyd files).
DATAS = []
DATAS += collect_data_files("mediapipe")
DATAS += collect_data_files("cv2", includes=["**/*.xml"])
DATAS += collect_data_files("edge_tts")
DATAS += collect_data_files("certifi")

# ── Hidden imports that PyInstaller often misses
HIDDEN_IMPORTS += collect_submodules("mediapipe")
HIDDEN_IMPORTS += collect_submodules("edge_tts")
HIDDEN_IMPORTS += collect_submodules("comtypes")     # used by pyttsx3 fallback
HIDDEN_IMPORTS += [
    "pyttsx3.drivers",
    "pyttsx3.drivers.sapi5",
    "pyttsx3.drivers.dummy",
    "speech_recognition",
    "torch",
    "torch.nn",
]

# ── Native binaries (CUDA DLLs, OpenCV libs, .pyd extensions, etc.)
BINARIES = []
BINARIES += collect_dynamic_libs("torch")
BINARIES += collect_dynamic_libs("mediapipe")

# ── Models / data folders sitting next to the .exe -------------------------
# We do **not** put these inside the bundle — they live next to the .exe so
# the user can update them without rebuilding. We just copy them in via
# `additional_files` below (handled by the build script).

# ── Modules we deliberately exclude (training-only / never used at runtime)
EXCLUDES = [
    # NB: matplotlib/scipy/sklearn are training-only for *us*, but mediapipe
    # imports matplotlib transitively (drawing_utils) and scipy is sometimes
    # pulled in too. We keep them rather than fight phantom hidden imports.
    "pytest",
    "tkinter",
    "IPython",
    "notebook",
    "jupyter",
    "jupyter_client",
    "PyQt6",          # main_qt.py is not the entry point of the bundle
    "PyQt5",
    "PySide6",
    "pandas",
    "torchvision",
    "torchaudio",
    "sympy",
    "PIL.ImageQt",
]


# ============================================================================
a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HandTalkTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                         # UPX is fragile with PyTorch DLLs
    console=True,                      # keep stdout for debugging on jury day
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HandTalkTranslator",
)
