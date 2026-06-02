"""Hand Talk Translator — ML Edition.

Real-time sign-language → speech translator (bidirectional).

Usage
─────
1.  python collect_data.py        — record training samples via webcam
2.  python train_model.py         — train the LSTM gesture classifier
3.  python main.py                — launch the real-time translator
    python main.py --debug        — show live probabilities for all classes
    python main.py --no-listen    — disable microphone (offline mode)
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np

import config
from config import (
    CAMERA,
    LABELS_PATH,
    MODEL_PATH,
    MOTION,
    PAUSE_LABEL,
    UNKNOWN_LABEL,
    COMMIT,
    RECOGNITION,
    SPEAK,
    STABILITY_FRAMES_REQUIRED,
)
from latency_tracker import LatencyTracker
from session_logger import SessionLogger


def _draw_debug_panel(frame: np.ndarray, probs: dict[str, float]) -> None:
    """Overlay a probability bar chart for all classes on the right side."""
    if not probs:
        return
    h, w = frame.shape[:2]
    panel_w = 220
    x0 = w - panel_w - 8
    bar_max = panel_w - 90
    labels_sorted = sorted(probs, key=probs.get, reverse=True)[:12]  # top 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, 8), (w - 4, 14 + len(labels_sorted) * 22), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "DEBUG — probas", (x0, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for i, label in enumerate(labels_sorted):
        p = probs[label]
        y = 38 + i * 22
        bar_len = int(p * bar_max)
        color = (0, 220, 120) if i == 0 else (80, 160, 255)
        cv2.rectangle(frame, (x0, y), (x0 + bar_len, y + 14), color, -1)
        cv2.putText(frame, f"{label[:8]:<8s} {p:.0%}", (x0, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


def _draw_conversation(frame: np.ndarray, conversation: list[dict],
                       listen_active: bool) -> None:
    """Overlay the last few conversation turns on the left side of the frame."""
    h, w = frame.shape[:2]
    panel_w = min(340, w // 3)
    max_entries = 6
    entries = conversation[-max_entries:]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 55), (panel_w, h - 110), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Header
    mic_status = "MIC: ON" if listen_active else "MIC: OFF"
    mic_color = (0, 220, 120) if listen_active else (120, 120, 120)
    cv2.putText(frame, "CONVERSATION", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, mic_status, (panel_w - 80, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, mic_color, 1)
    cv2.line(frame, (8, 82), (panel_w - 8, 82), (80, 80, 80), 1)

    y = 102
    row_h = 38
    for entry in entries:
        is_sign = entry["side"] == "sign"
        icon = "[LSF]" if is_sign else "[MIC]"
        icon_col = (0, 180, 255) if is_sign else (0, 220, 120)
        text = entry["text"]

        # Truncate to fit panel
        max_chars = (panel_w - 70) // 7
        if len(text) > max_chars:
            text = text[:max_chars - 1] + "."

        cv2.putText(frame, icon, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, icon_col, 1)
        cv2.putText(frame, text, (62, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        y += row_h

    # Hint at the bottom of the panel
    cv2.putText(frame, "[E] Exporter  [M] Micro", (10, h - 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)


def _export_conversation(conversation: list[dict]) -> None:
    """Save the conversation log to a timestamped .txt file."""
    if not conversation:
        print("  Aucune conversation a exporter.")
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.BASE_DIR, f"session_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(" HAND TALK TRANSLATOR — Session exportee\n")
        f.write(f" {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        for entry in conversation:
            t = datetime.datetime.fromtimestamp(entry["time"]).strftime("%H:%M:%S")
            side = "Sourd  [LSF]" if entry["side"] == "sign" else "Entend [MIC]"
            f.write(f"[{t}] {side} : {entry['text']}\n")
        f.write("\n" + "=" * 50 + "\n")
    print(f"  Session exportee : {path}")


def main():
    parser = argparse.ArgumentParser(description="Hand Talk Translator")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher les probabilites en temps reel pour toutes les classes")
    parser.add_argument("--no-listen", action="store_true",
                        help="Desactiver le microphone (mode hors ligne)")
    parser.add_argument("--no-splash", action="store_true",
                        help="Sauter l'ecran de presentation au demarrage")
    parser.add_argument("--no-latency", action="store_true",
                        help="Cacher le panneau de latence end-to-end")
    parser.add_argument("--no-log", action="store_true",
                        help="Desactiver le logger de session JSONL")
    parser.add_argument("--replay", type=str, default=None, metavar="FICHIER",
                        help="Mode replay : rejoue un sessions/*.jsonl au lieu "
                             "d'utiliser la reconnaissance live (filet de "
                             "secours pour la demo jury)")
    parser.add_argument("--replay-loop", action="store_true",
                        help="En mode replay, recommencer en boucle a la fin")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Vitesse du replay (1.0 = original, 1.5 = 50%% plus rapide)")
    args = parser.parse_args()
    debug_mode = args.debug
    listen_mode = not args.no_listen
    show_latency = not args.no_latency
    enable_log = not args.no_log
    replay_mode = args.replay is not None
    if replay_mode:
        # Replay is silent on the mic side: voice events come from the JSONL
        listen_mode = False

    print("=" * 58)
    print("  HAND TALK TRANSLATOR — ML Edition")
    if debug_mode:
        print("  [MODE DEBUG ACTIF]")
    if replay_mode:
        print(f"  [MODE REPLAY] Source : {args.replay}")
        if args.replay_loop:
            print("  [MODE REPLAY] Boucle activee")
        if args.replay_speed != 1.0:
            print(f"  [MODE REPLAY] Vitesse x{args.replay_speed}")
    print("=" * 58)
    print()

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        print("Aucun modele entraine detecte.\n")
        print("Pour utiliser le traducteur, suivez ces etapes :")
        print("  1. python collect_data.py   (collecter des gestes)")
        print("  2. python train_model.py    (entrainer le modele)")
        print("  3. python main.py           (lancer le traducteur)")
        sys.exit(1)

    from gesture_recognizer import GestureRecognizer
    from hand_detector import HandDetector
    from lsf_translator import translate
    from motion_pause_detector import MotionPauseDetector
    from sentence_builder import SentenceBuilder
    from speech_engine import SpeechEngine
    from speech_listener import SpeechListener
    from ui_renderer import UIRenderer

    print("Chargement du modele...")
    recognizer = GestureRecognizer()
    print(f"  {recognizer.num_signs} signes charges : "
          f"{', '.join(recognizer.labels)}")

    detector = HandDetector()
    speech = SpeechEngine()
    sentence = SentenceBuilder()
    ui = UIRenderer(sign_labels=recognizer.labels)
    pause_motion = MotionPauseDetector()

    def _read_phrase_aloud() -> None:
        """Translate current glosses and send them to TTS."""
        if sentence.is_empty:
            return
        french = translate(sentence.tokens)
        print(f"  LSF:     {sentence.gloss}")
        print(f"  Francais: {french}")
        _speak(french)

    # ── Bidirectional: microphone listener ───────────────────────────────
    listener = SpeechListener(language="fr-FR")
    if listen_mode:
        ok = listener.start()
        if ok:
            print("  Microphone actif (reconnaissance vocale FR)")
        else:
            print("  ATTENTION: speech_recognition non installe.")
            print("  Installez-le: py -3.11 -m pip install SpeechRecognition pyaudio")
            listen_mode = False

    # ── Conversation log (both sides) ────────────────────────────────────
    # Each entry: {"side": "sign"|"voice", "text": str, "time": float}
    conversation: list[dict] = []

    # ── Persistent JSONL session log ─────────────────────────────────────
    session_log = SessionLogger(directory=config.SESSIONS_DIR) if enable_log else None
    if session_log is not None:
        session_log.log_event("session_start_args",
                              debug=debug_mode, listen=listen_mode)

    # ── End-to-end latency tracking ──────────────────────────────────────
    latency = LatencyTracker(window=60)

    def _speak(text: str, force: bool = True) -> None:
        """Speak a French sentence and log it to the conversation."""
        if not text:
            return
        speech.speak(text, force=force)
        conversation.append({"side": "sign", "text": text, "time": time.time()})
        if session_log is not None:
            session_log.log_phrase(lsf=sentence.gloss, french=text)

    print("\nOuverture de la camera...")
    cap = cv2.VideoCapture(CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

    if not cap.isOpened():
        if replay_mode:
            # In replay mode the camera is not strictly required: we can
            # generate synthetic blank frames so the demo runs even on a
            # PC without a working webcam (true safety net for the jury).
            print("  Camera indisponible -- mode replay autonome (frame noir).")
            cap.release()
            cap = None
        else:
            print("ERREUR: Camera non disponible.")
            sys.exit(1)

    window_name = "Hand Talk Translator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, CAMERA["width"], CAMERA["height"])

    # ── Splash screen (TFE branding) ──────────────────────────────────────
    if not args.no_splash:
        try:
            from splash import show_splash
            show_splash(window_name, CAMERA["width"], CAMERA["height"])
        except Exception as exc:  # pragma: no cover
            print(f"  (Splash desactive: {exc})")

    # ── Replay player (jury safety net) ───────────────────────────────────
    replay = None
    if replay_mode:
        from replay_player import ReplayPlayer
        try:
            replay = ReplayPlayer(args.replay,
                                  speed=args.replay_speed,
                                  loop=args.replay_loop)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERREUR replay : {exc}")
            sys.exit(1)
        print(f"  Replay : {replay.total_events} evenements, "
              f"duree {replay.duration:.1f}s, "
              f"{len(replay.signs())} signes")
        if session_log is not None:
            session_log.log_event("replay_mode", source=str(args.replay),
                                  loop=args.replay_loop,
                                  speed=args.replay_speed)
        replay.start()

    print("Systeme pret !\n")
    print("Controles :")
    print("  [H]           Afficher / masquer les signes disponibles")
    print("  [D]           Activer / desactiver mode debug (probas)")
    print("  [M]           Activer / desactiver le microphone")
    print("  [V]           Demarrer / arreter l'enregistrement video")
    print("  [L]           Afficher / masquer le panneau de latence")
    print("  [E]           Exporter la conversation en .txt")
    print("  [Espace]      Traduire et prononcer la phrase")
    print("  [Retour arr.] Supprimer le dernier signe")
    print("  [Entree]      Effacer la phrase")
    if replay is not None:
        print("  >>> MODE REPLAY ACTIF — sortie auto a la fin <<<")
    print("  [R]           Reinitialiser l'historique")
    print("  [Q] / Echap   Quitter")
    print()

    fps_t = time.time()
    fps = 0.0
    stable_gesture = None
    stable_count = 0
    committed = False
    pause_available = PAUSE_LABEL in recognizer.labels
    last_commit_ts = 0.0
    last_commit_label = None
    replay_done_at: float | None = None

    # ── Video recording ───────────────────────────────────────────────────
    recording = False
    video_writer: cv2.VideoWriter | None = None

    def _start_recording(width: int, height: int) -> cv2.VideoWriter:
        os.makedirs(os.path.join(os.path.dirname(__file__), "recordings"), exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(__file__), "recordings", f"session_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 20.0, (width, height))
        print(f"  Enregistrement demarre : {path}")
        return writer

    try:
        while True:
            t_frame = time.perf_counter()

            with latency.measure("camera"):
                if cap is not None:
                    ret, frame = cap.read()
                else:
                    # Replay-only mode without webcam: synthetic dark frame
                    ret = True
                    frame = np.zeros((CAMERA["height"], CAMERA["width"], 3),
                                     dtype=np.uint8)
                    time.sleep(1.0 / 30.0)  # cap to ~30 FPS
            if not ret:
                print("Erreur de lecture camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with latency.measure("mediapipe"):
                results = detector.detect(rgb)
            detector.draw_landmarks(frame, results)

            with latency.measure("lstm"):
                gesture, confidence, motion, accepted = recognizer.process_result(results)

            if replay is None:
                # ── Live recognition path ─────────────────────────────────
                commit_gesture = gesture if confidence >= RECOGNITION["min_commit_confidence"] else None

                if commit_gesture != stable_gesture:
                    stable_gesture = commit_gesture
                    stable_count = 1 if commit_gesture else 0
                    committed = False
                elif commit_gesture:
                    stable_count += 1

                now_ts = time.time()
                if commit_gesture and stable_count >= STABILITY_FRAMES_REQUIRED and not committed:
                    # debounce: avoid rapid repeated commits
                    if (commit_gesture == last_commit_label and
                            now_ts - last_commit_ts < COMMIT["min_interval_seconds"]):
                        commit_gesture = None
                    else:
                        last_commit_label = commit_gesture
                        last_commit_ts = now_ts

                if commit_gesture and stable_count >= STABILITY_FRAMES_REQUIRED and not committed:
                    if gesture == UNKNOWN_LABEL:
                        committed = True
                    elif pause_available and gesture == PAUSE_LABEL:
                        sentence.add_pause()
                        ui.add_history("| PAUSE")
                        committed = True
                    else:
                        if recognizer.motion_active or not MOTION.get("require_active", False):
                            sentence.add(gesture)
                            ui.add_history(gesture)
                            if session_log is not None:
                                session_log.log_sign(gesture)
                        if SPEAK.get("on_commit", False) and not sentence.is_empty:
                            french_now = translate(sentence.tokens)
                            if french_now:
                                _speak(french_now)
                        committed = True
                    if committed:
                        recognizer.reset()
            else:
                # ── Replay path: events drive the state, not the model ────
                # Show a smooth "fake" prediction in the gauge UI so the jury
                # sees the same visual feedback as in live mode.
                upcoming = replay.upcoming_gesture()
                if upcoming is not None:
                    gesture, confidence = upcoming

                for ev in replay.pop_due():
                    etype = ev.get("type")
                    if etype == "sign":
                        label = ev.get("text", "")
                        if not label:
                            continue
                        if pause_available and label == PAUSE_LABEL:
                            sentence.add_pause()
                            ui.add_history("| PAUSE")
                        else:
                            sentence.add(label)
                            ui.add_history(label)
                    elif etype == "phrase":
                        fr = ev.get("fr", "")
                        if fr:
                            speech.speak(fr, force=True)
                            conversation.append({
                                "side": "sign",
                                "text": fr,
                                "time": time.time(),
                            })
                    elif etype == "voice":
                        text = ev.get("text", "")
                        if text:
                            conversation.append({
                                "side": "voice",
                                "text": text,
                                "time": time.time(),
                            })
                            print(f"  [REPLAY MIC] {text}")

            # Motion-based Pause gesture (shake left-right) — skipped in replay
            if replay is None and pause_motion.update(
                results.hand_landmarks[0] if results.hand_landmarks else None, time.time()
            ):
                sentence.add_pause()
                ui.add_history("| PAUSE")

            # ── Microphone transcription (hearing person → deaf person) ──
            if listen_mode:
                heard = listener.get_text()
                if heard:
                    conversation.append({
                        "side": "voice",
                        "text": heard,
                        "time": time.time(),
                    })
                    print(f"  [MIC] {heard}")
                    if session_log is not None:
                        session_log.log_voice(heard)

            # Translate LSF gloss → French
            with latency.measure("translate"):
                french = translate(sentence.tokens) if not sentence.is_empty else ""

            now = time.time()
            fps = 1.0 / max(now - fps_t, 0.001)
            fps_t = now

            # End-to-end frame latency (everything above)
            latency.record("total", (time.perf_counter() - t_frame) * 1000.0)

            ui.draw(
                frame,
                gesture=gesture,
                confidence=confidence,
                gloss=sentence.gloss,
                french=french,
                buf_fill=recognizer.buffer_fill,
                is_speaking=speech.is_speaking,
                fps=fps,
                motion_active=recognizer.motion_active,
                latency=latency.snapshot() if show_latency else None,
            )

            if debug_mode and recognizer.last_probs:
                _draw_debug_panel(frame, recognizer.last_probs)

            # ── Conversation panel ────────────────────────────────────────
            _draw_conversation(frame, conversation, listen_mode)

            # ── Recording indicator ───────────────────────────────────────
            if recording:
                if video_writer:
                    video_writer.write(frame)
                cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 220), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 55, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

            cv2.imshow(window_name, frame)

            # Auto-exit when the replay is fully played and not looping
            if replay is not None and replay.is_done() and not speech.is_speaking:
                # leave a 600 ms tail so the last phrase fully renders + speaks
                if replay_done_at is None:
                    replay_done_at = time.time()
                elif time.time() - replay_done_at > 0.6:
                    print("Replay termine.")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                ui.toggle_help()
            elif key == ord("d"):
                debug_mode = not debug_mode
                print(f"  Mode debug: {'ON' if debug_mode else 'OFF'}")
            elif key == ord("m"):
                if listen_mode:
                    listener.stop()
                    listen_mode = False
                    print("  Microphone desactive.")
                else:
                    listen_mode = listener.start()
                    print(f"  Microphone: {'actif' if listen_mode else 'indisponible'}")
            elif key == ord("e"):
                _export_conversation(conversation)
                if session_log is not None:
                    session_log.log_event("export")
            elif key == ord("l"):
                show_latency = not show_latency
                print(f"  Panneau latence: {'ON' if show_latency else 'OFF'}")
            elif key == ord("v"):
                if not recording:
                    if cap is not None:
                        h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    else:
                        h_cam, w_cam = CAMERA["height"], CAMERA["width"]
                    video_writer = _start_recording(w_cam, h_cam)
                    recording = True
                else:
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print("  Enregistrement arrete.")
            elif key == ord("r"):
                ui.reset()
                sentence.clear()
                recognizer.reset()
                if session_log is not None:
                    session_log.log_event("reset")
                print("Historique reinitialise.")
            elif key == ord(" "):
                if replay is None:
                    _read_phrase_aloud()
            elif key == 8:  # Backspace
                removed = sentence.delete_last()
                if removed:
                    print(f"  Supprime: {removed}")
            elif key == 13:  # Enter
                sentence.clear()
                print("  Phrase effacee.")

    except KeyboardInterrupt:
        print("\nArret par l'utilisateur.")

    finally:
        print("Fermeture...")
        if recording and video_writer:
            video_writer.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        detector.release()
        speech.release()
        listener.stop()
        if session_log is not None:
            session_log.close()
            if session_log.path:
                print(f"  Session enregistree : {session_log.path}")
        print("Au revoir !")


if __name__ == "__main__":
    main()
