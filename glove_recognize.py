"""Real-time sign recognition from the glove stream.

Streams 50 Hz frames from the ESP32 (or the MOCK backend), feeds them
into the trained ``GestureLSTM``, prints predictions to the terminal,
and translates them to French via :func:`lsf_translator.translate`.

Usage
-----

    python glove_recognize.py --port COM5
    python glove_recognize.py --port MOCK --print-every 0.2

Controls
--------
  r         effacer la phrase
  u         annuler le dernier signe
  Ctrl+C    quitter
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
import torch

import config
from glove_reader import GloveCalibration, GloveReader
from lsf_translator import translate
from model import GestureLSTM
from sentence_builder import SentenceBuilder


def _poll_key() -> str | None:
    if not _HAS_MSVCRT or not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        msvcrt.getwch()
        return None
    return ch.lower()


# ── helpers ─────────────────────────────────────────────────────────────────


def _load_calibration(path: Path) -> GloveCalibration:
    if path.exists():
        return GloveCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    print(f"[ATTENTION] calibration absente ({path}), valeurs par defaut.")
    return GloveCalibration()


def _load_model(path: Path, labels_path: Path, device: torch.device):
    if not path.exists() or not labels_path.exists():
        raise SystemExit(
            f"Modele introuvable. Lance d'abord python glove_train.py "
            f"({path} / {labels_path})")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    model = GestureLSTM(
        num_features=config.GLOVE["features_per_frame"],
        num_classes=len(labels),
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, labels


# ── main loop ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconnaissance temps reel via gant")
    ap.add_argument("--port", required=True, help="COM5, /dev/ttyUSB0, MOCK")
    ap.add_argument("--baud", type=int, default=config.GLOVE["baud"])
    ap.add_argument("--model",       type=Path, default=Path(config.GLOVE_MODEL_PATH))
    ap.add_argument("--labels",      type=Path, default=Path(config.GLOVE_LABELS))
    ap.add_argument("--calibration", type=Path, default=Path(config.GLOVE_CALIB_PATH))
    ap.add_argument("--threshold",   type=float, default=config.GLOVE["confidence_threshold"],
                    help="confiance softmax min (defaut 0.35 — normal avec 3 classes)")
    ap.add_argument("--margin",      type=float, default=config.GLOVE["margin_threshold"],
                    help="ecart min entre la 1re et 2e classe (defaut 0.06)")
    ap.add_argument("--cooldown",    type=float, default=config.GLOVE["cooldown_seconds"])
    ap.add_argument("--print-every", type=float, default=0.25,
                    help="affichage des predictions hors commit (secondes)")
    ap.add_argument("--repeat", action="store_true",
                    help="autoriser le meme signe plusieurs fois dans la phrase")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    once_each = not args.repeat
    neutral_frames_needed = config.GLOVE.get("neutral_frames", 18)
    rearm_seconds = config.GLOVE.get("rearm_seconds", 1.0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, labels = _load_model(args.model, args.labels, device)
    calib = _load_calibration(args.calibration)

    seq_len = config.GLOVE["sequence_length"]
    period = 1.0 / config.GLOVE["sample_hz"]
    stability_needed = max(config.GLOVE["prediction_buffer"],
                           config.STABILITY_FRAMES_REQUIRED)

    reader = GloveReader(port=args.port, baudrate=args.baud)
    if not reader.start():
        print(f"[ERREUR] {reader.error}", file=sys.stderr)
        return 1

    print("Reconnaissance gant en cours.")
    print(f"Seuils : conf >= {args.threshold*100:.0f}%  |  marge >= {args.margin*100:.0f}%")
    print("(Avec 3 signes, ~40-55% de confiance est normal quand c'est correct.)")
    print("Touches : r = effacer la phrase  |  u = annuler le dernier signe  |  Ctrl+C = quitter")
    if once_each:
        print("Mode : chaque signe une seule fois par phrase (ajoute --repeat pour autoriser les doublons)")
    print("Astuce : enchaine MOI -> BONJOUR -> MERCI (pause ~1 s entre chaque).\n")

    sentence = SentenceBuilder()
    window = np.zeros((seq_len, config.GLOVE["features_per_frame"]), dtype=np.float32)
    filled = 0
    last_t_ms = -1
    last_print = 0.0
    last_commit_t = 0.0
    last_commit_label: str | None = None
    stable_gesture: str | None = None
    stable_count = 0
    committed = False
    armed = True
    neutral_count = 0

    next_tick = time.perf_counter()
    try:
        while True:
            key = _poll_key()
            if key == "r":
                sentence.clear()
                stable_gesture = None
                stable_count = 0
                committed = False
                armed = True
                neutral_count = 0
                print("\n  [phrase effacee]")
                continue
            if key == "u":
                removed = sentence.delete_last()
                if removed:
                    print(f"\n  [retire : {removed}]  gloss: {sentence.gloss or '(vide)'}")
                continue

            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            frame = reader.read_latest()
            if frame is None or frame.t_ms == last_t_ms:
                continue
            last_t_ms = frame.t_ms

            features = np.asarray(calib.normalise(frame), dtype=np.float32)
            window = np.roll(window, -1, axis=0)
            window[-1] = features
            if filled < seq_len:
                filled += 1
                continue

            with torch.no_grad():
                x = torch.from_numpy(window).unsqueeze(0).to(device)
                probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            top = int(np.argmax(probs))
            top_conf = float(probs[top])
            top_label = labels[top]
            order = np.argsort(probs)[::-1]
            second_conf = float(probs[order[1]]) if len(order) > 1 else 0.0
            margin = top_conf - second_conf

            passes = (top_conf >= args.threshold and margin >= args.margin
                      and top_label != config.PAUSE_LABEL)
            commit_gesture = top_label if passes else None

            if not armed:
                pause_elapsed = time.time() - last_commit_t
                new_gesture = (
                    last_commit_label is not None
                    and commit_gesture is not None
                    and commit_gesture != last_commit_label
                    and stable_count >= stability_needed
                )
                if passes:
                    neutral_count = 0
                else:
                    neutral_count += 1

                if (neutral_count >= neutral_frames_needed
                        or pause_elapsed >= rearm_seconds
                        or new_gesture):
                    armed = True
                    neutral_count = 0
                    if not new_gesture:
                        stable_gesture = None
                        stable_count = 0
                    committed = False

            if commit_gesture != stable_gesture:
                stable_gesture = commit_gesture
                stable_count = 1 if commit_gesture else 0
                committed = False
            elif commit_gesture:
                stable_count += 1

            now = time.perf_counter()

            if now - last_print > args.print_every:
                if not armed:
                    pause_left = max(0.0, rearm_seconds - (time.time() - last_commit_t))
                    if pause_left > 0.05:
                        state = f"pause {pause_left:.1f}s"
                    elif not passes:
                        state = "relache..."
                    else:
                        state = top_label
                else:
                    state = stable_gesture or "..."
                print(f"  ~ {state:<12s} {top_conf*100:4.0f}%  "
                      f"(marge {margin*100:4.0f}% vs {labels[order[1]]})",
                      end="\r", flush=True)
                last_print = now

            ready = (armed
                     and commit_gesture is not None
                     and stable_count >= stability_needed
                     and not committed)
            if ready and once_each and commit_gesture in sentence.tokens:
                ready = False
            if ready:
                now_ts = time.time()
                if (commit_gesture == last_commit_label
                        and now_ts - last_commit_t < args.cooldown):
                    ready = False
                else:
                    last_commit_label = commit_gesture
                    last_commit_t = now_ts

            if ready:
                sentence.add(commit_gesture)
                committed = True
                armed = False
                neutral_count = 0
                french = translate(sentence.tokens)
                print(" " * 72, end="\r")
                print(f"  + {commit_gesture}  ({top_conf*100:.1f}%)")
                print(f"     gloss : {sentence.gloss}")
                print(f"     fr    : {french}\n")

    except KeyboardInterrupt:
        print("\n\nArret demande.")
    finally:
        reader.stop()

    if not sentence.is_empty:
        print("\nGloss final :", sentence.gloss)
        print("Phrase finale :", translate(sentence.tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
