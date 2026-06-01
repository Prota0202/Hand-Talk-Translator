"""Capture training samples from the sign-language glove.

Each sample is a fixed-length sequence (``GLOVE.sequence_length`` frames at
``GLOVE.sample_hz`` ≈ 0.6 s) of normalised features and is saved as a
``.npy`` file under ``data_glove/<SIGN>/``.

Usage
-----

    # interactive: cycle through every sign listed in config.RECOMMENDED_SIGNS
    python glove_collect.py --port COM5

    # capture only a subset
    python glove_collect.py --port COM5 --signs MOI BONJOUR MERCI

    # development without HW
    python glove_collect.py --port MOCK --signs DEMO --samples 5

Controls during capture
-----------------------
  Mode AUTO (defaut) :
    compte a rebours puis enregistrement automatique — pas de touche par echantillon
    q = quitter, s = passer au signe suivant (pendant le compte a rebours)

  Mode MANUEL (--manual) :
    ENTER = lancer l'echantillon suivant
    s = passer le signe, q = quitter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

import numpy as np

import config
from glove_reader import (
    FRAMES_PER_SEQUENCE,
    GloveCalibration,
    GloveReader,
)


def _load_calibration(path: Path) -> GloveCalibration:
    if path.exists():
        return GloveCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    print(f"[ATTENTION] aucune calibration trouvee a {path}")
    print("            valeurs par defaut utilisees (lance d'abord glove_calibration.py).")
    return GloveCalibration()


def _poll_key() -> str | None:
    """Return a single key if pressed (Windows console), else None."""
    if not _HAS_MSVCRT or not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        msvcrt.getwch()
        return None
    return ch.lower()


def _countdown(label: str, seconds: int) -> str | None:
    """Print a countdown; return 'q' or 's' if requested, else None."""
    for remaining in range(seconds, 0, -1):
        key = _poll_key()
        if key == "q":
            return "q"
        if key == "s":
            return "s"
        print(f"  {label} dans {remaining}...", end="\r", flush=True)
        time.sleep(1.0)
    print(f"  {label} — GO !          ")
    return None


def _record_sample(reader: GloveReader, calib: GloveCalibration,
                   length: int, sample_hz: int) -> np.ndarray:
    """Capture exactly ``length`` normalised frames at ``sample_hz``."""
    period = 1.0 / sample_hz
    out = np.zeros((length, config.GLOVE["features_per_frame"]), dtype=np.float32)
    last_t_ms = -1
    i = 0
    next_tick = time.perf_counter()
    while i < length:
        next_tick += period
        # Sleep until tick (reasonably accurate)
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        f = reader.read_latest()
        if f is None:
            continue
        # Avoid using the same frame twice if the firmware lags
        if f.t_ms == last_t_ms:
            continue
        last_t_ms = f.t_ms
        out[i] = np.asarray(calib.normalise(f), dtype=np.float32)
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Acquisition de donnees gant")
    ap.add_argument("--port", required=True,
                    help="COM5, /dev/ttyUSB0, ou MOCK")
    ap.add_argument("--baud", type=int, default=config.GLOVE["baud"])
    ap.add_argument("--out", type=Path, default=Path(config.GLOVE_DIR))
    ap.add_argument("--calibration", type=Path, default=Path(config.GLOVE_CALIB_PATH))
    ap.add_argument("--signs", nargs="*",
                    help="Sous-ensemble de signes (par defaut: RECOMMENDED_SIGNS)")
    ap.add_argument("--samples", type=int, default=config.GLOVE["samples_per_sign"])
    ap.add_argument("--length", type=int, default=config.GLOVE["sequence_length"])
    ap.add_argument("--sample-hz", type=int, default=config.GLOVE["sample_hz"])
    ap.add_argument("--manual", action="store_true",
                    help="mode manuel : ENTREE avant chaque echantillon")
    ap.add_argument("--countdown", type=int, default=2,
                    help="secondes avant chaque capture en mode auto (defaut: 2)")
    ap.add_argument("--rest", type=float, default=1.5,
                    help="pause (s) entre deux echantillons en mode auto (defaut: 1.5)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    calib = _load_calibration(args.calibration)
    signs = args.signs if args.signs else config.RECOMMENDED_SIGNS

    reader = GloveReader(port=args.port, baudrate=args.baud)
    if not reader.start():
        print(f"[ERREUR] {reader.error}", file=sys.stderr)
        return 1

    # Wait for first frame
    t0 = time.perf_counter()
    while reader.read_latest() is None and time.perf_counter() - t0 < 5.0:
        time.sleep(0.05)
    if reader.read_latest() is None:
        print("[ERREUR] aucune trame recue, verifie le port et le firmware")
        reader.stop()
        return 2

    print("=" * 64)
    print(" ACQUISITION GANT — Hand Talk Translator")
    print("=" * 64)
    print(f"  signes      : {len(signs)}")
    print(f"  echantillons: {args.samples} par signe")
    print(f"  longueur    : {args.length} frames @ {args.sample_hz} Hz "
          f"(~{args.length/args.sample_hz:.2f} s)")
    auto_mode = not args.manual
    print(f"  mode        : {'AUTO' if auto_mode else 'MANUEL'}")
    if auto_mode:
        print()
        print("  Faites le signe a chaque GO, puis relachez pendant la pause.")
        print("  Touches : q = quitter, s = signe suivant (pendant le compte a rebours)")
    print()

    try:
        for s_idx, sign in enumerate(signs, 1):
            sign_dir = args.out / sign
            sign_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(sign_dir.glob("*.npy"))
            print(f"\n--- [{s_idx:>3}/{len(signs)}] Signe : {sign}  "
                  f"({len(existing)} deja captures) ---")
            for k in range(args.samples):
                idx = len(existing) + k
                if auto_mode:
                    label = f"echantillon {k + 1}/{args.samples}"
                    action = _countdown(label, max(1, args.countdown))
                    if action == "q":
                        print("Arret demande.")
                        return 0
                    if action == "s":
                        print(f"  signe {sign} skippe.")
                        break
                else:
                    cmd = input(
                        f"  echantillon {k+1}/{args.samples}  "
                        f"[ENTREE=GO, s=skip signe, q=quitter] > "
                    ).strip().lower()
                    if cmd == "q":
                        print("Arret demande.")
                        return 0
                    if cmd == "s":
                        print(f"  signe {sign} skippe.")
                        break

                reader.drain()
                seq = _record_sample(reader, calib, args.length, args.sample_hz)
                fname = sign_dir / f"sample_{idx:03d}.npy"
                np.save(fname, seq)
                n_flex = config.GLOVE.get("num_flex", 3)
                print(f"  -> sauve {fname.name}  (mean flex = {seq[:, :n_flex].mean():.3f})")

                if auto_mode and k + 1 < args.samples:
                    time.sleep(max(0.0, args.rest))
    finally:
        reader.stop()

    print("\nTermine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
