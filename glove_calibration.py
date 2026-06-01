"""Interactive calibration tool for the sign-language glove.

Captures the ADC range of every flex sensor between the **open palm** and
**closed fist** poses, then writes ``models/glove_calibration.json``.

Usage
-----

    python glove_calibration.py --port COM5
    python glove_calibration.py --port MOCK     # demo without HW

The resulting calibration file is read by every other glove script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from glove_reader import GloveCalibration, GloveReader


CALIB_PATH = Path("models/glove_calibration.json")
PROMPT_DURATION_S = 4.0


def _capture(reader: GloveReader, duration_s: float, label: str) -> list:
    print(f"\n>> {label}: maintenez la pose pendant {duration_s:.0f} s...")
    for cd in (3, 2, 1):
        print(f"   {cd}...", end="\r", flush=True)
        time.sleep(1.0)
    print("   GO !          ")
    t0 = time.perf_counter()
    samples: list = []
    while time.perf_counter() - t0 < duration_s:
        f = reader.read_latest()
        if f is not None:
            samples.append(f)
        time.sleep(0.02)
    print(f"   ok, {len(samples)} echantillons captures")
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration des capteurs flex du gant")
    ap.add_argument("--port", required=True,
                    help="Port serie de l'ESP32 (ex: COM5, /dev/ttyUSB0) ou 'MOCK'")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", type=Path, default=CALIB_PATH)
    ap.add_argument("--duration", type=float, default=PROMPT_DURATION_S,
                    help="duree (s) de capture pour chaque pose")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    reader = GloveReader(port=args.port, baudrate=args.baud)
    if not reader.start():
        print(f"[ERREUR] {reader.error}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(" CALIBRATION DU GANT — Hand Talk Translator")
    print("=" * 60)
    print("On va capturer la plage ADC de chaque doigt en deux poses :")
    print("  1. main grande ouverte (paume tendue)")
    print("  2. poing serre")
    print("Restez stable pendant chaque capture.")
    time.sleep(1.5)

    try:
        # Wait for the stream to actually produce something
        t0 = time.perf_counter()
        while reader.read_latest() is None:
            if time.perf_counter() - t0 > 5.0:
                print("[ERREUR] aucune trame recue (verifie le cablage / le port)")
                return 2
            time.sleep(0.05)

        open_samples = _capture(reader, args.duration, "Pose OUVERTE")
        fist_samples = _capture(reader, args.duration, "Pose POING FERME")

        all_samples = open_samples + fist_samples
        if len(all_samples) < 50:
            print("[ERREUR] trop peu d'echantillons, recommence")
            return 3

        calib = GloveCalibration.from_samples(all_samples)
    finally:
        reader.stop()

    args.out.write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f" Calibration ecrite dans {args.out}")
    print("=" * 60)
    finger_names = ["pouce", "index", "majeur", "annulaire", "auriculaire"]
    for i in range(len(calib.flex_min)):
        name = finger_names[i] if i < len(finger_names) else f"flex{i+1}"
        lo, hi = calib.flex_min[i], calib.flex_max[i]
        print(f"  {name:12s} : min={lo:4d}  max={hi:4d}  (delta={hi-lo})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
