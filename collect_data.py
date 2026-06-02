"""Interactive data-collection tool for gesture training samples.

Workflow
────────
1.  Choose which signs to collect (recommended list or custom).
2.  For each sign:
      - Press SPACE to start recording
      - Perform the sign NATURALLY (any duration)
      - Press SPACE to stop recording
3.  The variable-length recording is resampled to SEQUENCE_LENGTH
    and saved as a .npy file under  data/<sign_name>/.
"""

import os
import time

import cv2
import numpy as np

from config import (
    CAMERA,
    COLLECTION,
    CROSS_SIGNER3_SIGNS,
    CROSS_SIGNER4_SIGNS,
    CROSS_SIGNER5_SIGNS,
    CROSS_SIGNER_EVAL_SIGNS,
    DATA_DIR,
    DATA_SIGNER2_DIR,
    DATA_SIGNER3_DIR,
    DATA_SIGNER4_DIR,
    DATA_SIGNER5_DIR,
    EXCLUDED_VISION_LABELS,
    FEATURES_PER_FRAME,
    RECOMMENDED_SIGNS,
    SEQUENCE_LENGTH,
    UI_COLORS,
    canonical_gloss,
)
from feature_extractor import FeatureExtractor
from hand_detector import HandDetector
from sequence_utils import resample_sequence as _resample


def _signs_missing_samples(data_dir: str, signs: list[str], num_samples: int) -> list[str]:
    """Signes absents ou avec moins de *num_samples* fichiers .npy."""
    missing: list[str] = []
    for sign in signs:
        sign_dir = os.path.join(data_dir, sign)
        if not os.path.isdir(sign_dir):
            missing.append(sign)
            continue
        count = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        if count < num_samples:
            missing.append(sign)
    return missing


def _normalize_sign_list(signs: list[str]) -> list[str]:
    """Glosses LSF uniques (ex. m'appelle -> NOM)."""
    seen: set[str] = set()
    out: list[str] = []
    for sign in signs:
        canon = canonical_gloss(sign)
        if canon.lower() in {x.lower() for x in EXCLUDED_VISION_LABELS}:
            print(
                f"  IGNORE {sign!r} — pas un gloss vision "
                f"(l'age = MOI + chiffres, ex. MOI 2 4)."
            )
            continue
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _collect_sign_auto(sign_name, detector, extractor, cap, num_samples,
                       hold_frames: int = 20, rest_frames: int = 15,
                       data_dir: str = DATA_DIR):
    """Auto-record mode: hold the sign for ~20 frames, rest, repeat.

    No key press needed — just perform the sign repeatedly.
    Returns ``(collected_count, quit_requested)``.
    """
    sign_dir = os.path.join(data_dir, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    sample_idx = existing
    target = existing + num_samples

    # State machine: WAITING → RECORDING → COOLDOWN → WAITING …
    state = "WAITING"
    sequence: list[np.ndarray] = []
    hand_absent_count = 0
    cooldown_count = 0
    extractor.reset()

    while sample_idx < target:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = detector.detect(rgb)
        detector.draw_landmarks(frame, results)
        hand_ok = bool(results.hand_landmarks)

        # ── State machine ─────────────────────────────────────────────
        if state == "WAITING":
            if hand_ok:
                state = "RECORDING"
                sequence.clear()
                extractor.reset()
            hand_absent_count = 0

        elif state == "RECORDING":
            if hand_ok:
                features, _ = extractor.extract_from_result(results)
                sequence.append(features)
                hand_absent_count = 0
            else:
                hand_absent_count += 1
                if hand_absent_count > 5:
                    state = "WAITING"
                    sequence.clear()
                    continue

            if len(sequence) >= hold_frames:
                if len(sequence) >= COLLECTION["min_frames"]:
                    resampled = _resample(np.array(sequence), SEQUENCE_LENGTH)
                    np.save(os.path.join(sign_dir, f"{sample_idx}.npy"), resampled)
                    sample_idx += 1
                    print(f"    Echantillon {sample_idx - existing}/{num_samples} sauvegarde")
                sequence.clear()
                state = "COOLDOWN"
                cooldown_count = 0

        elif state == "COOLDOWN":
            cooldown_count += 1
            if cooldown_count >= rest_frames:
                state = "WAITING"

        # ── UI ────────────────────────────────────────────────────────
        progress = (sample_idx - existing) / num_samples
        bar_l, bar_r = 50, w - 50
        cv2.rectangle(frame, (bar_l, h - 50), (bar_r, h - 30), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_l, h - 50),
                      (bar_l + int((bar_r - bar_l) * progress), h - 30),
                      (0, 200, 100), -1)

        state_colors = {"WAITING": (200, 200, 0), "RECORDING": (0, 0, 255), "COOLDOWN": (0, 200, 200)}
        state_labels = {"WAITING": "En attente...", "RECORDING": "ENREGISTREMENT", "COOLDOWN": "Pause..."}
        cv2.putText(frame,
                    f"AUTO — '{sign_name}'  [{sample_idx - existing}/{num_samples}]",
                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 168, 0), 2)
        cv2.putText(frame, state_labels[state],
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors[state], 2)
        if state == "RECORDING":
            cv2.putText(frame, f"Frames: {len(sequence)}/{hold_frames}",
                        (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "[Echap] Signe suivant   [Q] Quitter",
                    (50, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Collecte de donnees", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return sample_idx - existing, False
        if key == ord("q"):
            return sample_idx - existing, True

    return num_samples, False


def _collect_sign(sign_name, detector, extractor, cap, num_samples,
                  data_dir: str = DATA_DIR):
    """Record *num_samples* for a single sign using press-to-record.

    Returns ``(collected_count, quit_requested)``.
    """
    sign_dir = os.path.join(data_dir, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    sample_idx = existing

    while sample_idx < existing + num_samples:
        recording = False
        sequence: list[np.ndarray] = []
        extractor.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = detector.detect(rgb)
            detector.draw_landmarks(frame, results)

            hand_ok = bool(results.hand_landmarks)

            if recording:
                if hand_ok:
                    features, _ = extractor.extract_from_result(results)
                    sequence.append(features)

                # Recording header
                n_frames = len(sequence)
                cv2.circle(frame, (30, 40), 10, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ENREGISTREMENT — '{sign_name}' [{sample_idx + 1}]"
                    f"   {n_frames} frames",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

                # Progress bar
                fill = min(n_frames / COLLECTION["max_frames"], 1.0)
                bar_l, bar_r = 50, w - 50
                cv2.rectangle(frame, (bar_l, h - 60), (bar_r, h - 40), (60, 60, 60), -1)
                cv2.rectangle(
                    frame, (bar_l, h - 60),
                    (bar_l + int((bar_r - bar_l) * fill), h - 40),
                    UI_COLORS["danger"], -1,
                )

                # Hand status
                status = "Main detectee" if hand_ok else "MAIN NON DETECTEE"
                s_col = UI_COLORS["secondary"] if hand_ok else UI_COLORS["danger"]
                cv2.putText(frame, status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_col, 1)

                cv2.putText(
                    frame, "[ESPACE] Arreter l'enregistrement",
                    (50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                )

                # Auto-stop if max frames reached
                if n_frames >= COLLECTION["max_frames"]:
                    break

            else:
                # Waiting mode
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(
                    frame,
                    f"'{sign_name}' — Echantillon {sample_idx + 1}/{existing + num_samples}",
                    (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI_COLORS["primary"], 2,
                )
                cv2.putText(
                    frame,
                    "Preparez votre geste, puis appuyez sur [ESPACE] pour enregistrer",
                    (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_COLORS["text"], 1,
                )

                status = "Main detectee" if hand_ok else "MAIN NON DETECTEE"
                s_col = UI_COLORS["secondary"] if hand_ok else UI_COLORS["danger"]
                cv2.putText(frame, status, (50, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_col, 1)

                cv2.putText(
                    frame,
                    "[ESPACE] Enregistrer   [Echap] Signe suivant   [Q] Quitter",
                    (50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                )

            cv2.imshow("Collecte de donnees", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if recording:
                    break  # stop recording
                else:
                    recording = True
                    sequence.clear()
                    extractor.reset()
            elif key == 27:  # Escape — skip this sign
                return sample_idx - existing, False
            elif key == ord("q"):
                return sample_idx - existing, True

        # Validate & save
        if len(sequence) < COLLECTION["min_frames"]:
            print(f"    Trop court ({len(sequence)} frames), recommencez.")
            continue

        resampled = _resample(np.array(sequence), SEQUENCE_LENGTH)
        np.save(os.path.join(sign_dir, f"{sample_idx}.npy"), resampled)
        sample_idx += 1
        print(f"    Echantillon {sample_idx} sauvegarde ({len(sequence)} frames -> {SEQUENCE_LENGTH})")

    return num_samples, False


def _next_step_hint(data_dir: str) -> str:
    norm = os.path.normpath(data_dir)
    if norm == os.path.normpath(DATA_SIGNER2_DIR):
        return (
            "  Prochaine etape   : python evaluate_cross_signer.py "
            "--data-dir data_signer2 --signer-name \"Signeur 2 (petit frère)\""
        )
    if norm == os.path.normpath(DATA_SIGNER3_DIR):
        return "  Prochaine etape   : python evaluate_all_cross_signers.py"
    if norm == os.path.normpath(DATA_SIGNER4_DIR):
        return "  Prochaine etape   : python evaluate_all_cross_signers.py"
    if norm == os.path.normpath(DATA_SIGNER5_DIR):
        return "  Prochaine etape   : python evaluate_all_cross_signers.py"
    return "  Prochaine etape   : python train_model.py"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collecte de sequences LSF pour l'entrainement vision.",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=(
            "Dossier de sortie des .npy (defaut: data). "
            "Utilisez data_signer2 / data_signer3 / data_signer4 / data_signer5 pour un autre signeur."
        ),
    )
    parser.add_argument(
        "--signs",
        help="Liste de signes separes par des virgules (mode non interactif).",
    )
    parser.add_argument(
        "--cross-signer",
        action="store_true",
        help=(
            f"Raccourci: collecte les {len(CROSS_SIGNER_EVAL_SIGNS)} glosses "
            "du protocole cross-signeur (signeur 2)."
        ),
    )
    parser.add_argument(
        "--signer3",
        action="store_true",
        help=(
            f"Protocole signeur 3 : {len(CROSS_SIGNER3_SIGNS)} glosses "
            "(mots + chiffres, sans alphabet)."
        ),
    )
    parser.add_argument(
        "--signer4",
        action="store_true",
        help=(
            f"Protocole signeur 4 : {len(CROSS_SIGNER4_SIGNS)} glosses "
            "(mots + chiffres, sans alphabet)."
        ),
    )
    parser.add_argument(
        "--signer5",
        action="store_true",
        help=(
            f"Protocole signeur 5 : {len(CROSS_SIGNER5_SIGNS)} glosses "
            "(mots + chiffres, sans alphabet)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ne collecter que les signes manquants ou incomplets.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=f"Echantillons par signe (defaut: {COLLECTION['samples_per_sign']}).",
    )
    parser.add_argument(
        "--auto",
        dest="auto_mode",
        action="store_true",
        help="Mode AUTO sans prompt (enregistrement en boucle).",
    )
    parser.add_argument(
        "--manual",
        dest="auto_mode",
        action="store_false",
        help="Mode MANUEL (ESPACE pour demarrer/arreter).",
    )
    parser.set_defaults(auto_mode=None)
    cli_args, _unknown = parser.parse_known_args()
    data_dir = os.path.abspath(cli_args.data_dir)

    print("=" * 58)
    print("  COLLECTE DE DONNEES — Hand Talk Translator")
    print("=" * 58)
    print(f"  Dossier sortie : {data_dir}")
    print()

    if cli_args.signer5:
        protocol = list(CROSS_SIGNER5_SIGNS)
        num_samples = cli_args.samples if cli_args.samples is not None else 5
        auto_mode = True if cli_args.auto_mode is None else cli_args.auto_mode
        signs = (
            _signs_missing_samples(data_dir, protocol, num_samples)
            if cli_args.resume
            else protocol
        )
    elif cli_args.signer4:
        protocol = list(CROSS_SIGNER4_SIGNS)
        num_samples = cli_args.samples if cli_args.samples is not None else 5
        auto_mode = True if cli_args.auto_mode is None else cli_args.auto_mode
        signs = (
            _signs_missing_samples(data_dir, protocol, num_samples)
            if cli_args.resume
            else protocol
        )
    elif cli_args.signer3:
        protocol = list(CROSS_SIGNER3_SIGNS)
        num_samples = cli_args.samples if cli_args.samples is not None else 5
        auto_mode = True if cli_args.auto_mode is None else cli_args.auto_mode
        signs = (
            _signs_missing_samples(data_dir, protocol, num_samples)
            if cli_args.resume
            else protocol
        )
    elif cli_args.cross_signer:
        protocol = list(CROSS_SIGNER_EVAL_SIGNS)
        num_samples = cli_args.samples if cli_args.samples is not None else 5
        auto_mode = True if cli_args.auto_mode is None else cli_args.auto_mode
        signs = (
            _signs_missing_samples(data_dir, protocol, num_samples)
            if cli_args.resume
            else protocol
        )
    elif cli_args.signs:
        signs = [s.strip() for s in cli_args.signs.split(",") if s.strip()]
        if not signs:
            print("ERREUR: --signs est vide.")
            return
        num_samples = (
            cli_args.samples
            if cli_args.samples is not None
            else COLLECTION["samples_per_sign"]
        )
        auto_mode = True if cli_args.auto_mode is None else cli_args.auto_mode
        if cli_args.resume:
            signs = _signs_missing_samples(data_dir, signs, num_samples)
    else:
        print("Signes recommandes:")
        for i, sign in enumerate(RECOMMENDED_SIGNS):
            end = "\n" if (i + 1) % 10 == 0 else "  "
            print(f"  {sign}", end=end)
        print("\n")

        print("Options signes:")
        print("  1. Collecter les signes recommandes")
        print("  2. Entrer vos propres signes")
        print("  3. Completer des signes existants")
        print()

        choice = input("Choix (1/2/3): ").strip()

        if choice == "1":
            signs = list(RECOMMENDED_SIGNS)
        elif choice == "2":
            raw = input("Noms des signes (separes par des virgules): ")
            signs = [s.strip() for s in raw.split(",") if s.strip()]
        elif choice == "3":
            if not os.path.isdir(data_dir):
                print("Aucune donnee existante.")
                return
            existing = sorted(
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            )
            if not existing:
                print("Aucun signe existant.")
                return
            print(f"Signes existants: {', '.join(existing)}")
            signs = existing
        else:
            print("Choix invalide.")
            return

        default_n = COLLECTION["samples_per_sign"]
        raw_n = input(f"Echantillons par signe [{default_n}]: ").strip().strip("\\/")
        try:
            num_samples = int(raw_n) if raw_n else default_n
        except ValueError:
            print(f"Valeur invalide '{raw_n}', utilisation de {default_n}.")
            num_samples = default_n

        print()
        print("Mode d'enregistrement :")
        print("  A. AUTO  — Faites le signe en boucle, tout est enregistre automatiquement")
        print("             (rapide : ~1 min par signe pour 30 echantillons)")
        print("  M. MANUEL — Appuyez sur ESPACE pour chaque enregistrement")
        print()
        mode_choice = input("Choix (A/M) [A]: ").strip().upper() or "A"
        auto_mode = mode_choice != "M"

    signs = _normalize_sign_list(signs)

    if not signs:
        print("Rien a collecter : protocole deja complet dans ce dossier.")
        return

    print(f"\n{len(signs)} signes x {num_samples} echantillons  |  Mode: {'AUTO' if auto_mode else 'MANUEL'}")
    print(f"  Signes : {', '.join(signs)}")
    print()
    if auto_mode:
        print("INSTRUCTIONS (mode AUTO) :")
        print("  1. Faites votre signe devant la camera")
        print("  2. Maintenez-le ~1 seconde, puis retirez la main brievement")
        print("  3. Repetez jusqu'a avoir tous les echantillons")
        print("  -> Pas besoin d'appuyer sur des touches !")
    else:
        print("INSTRUCTIONS (mode MANUEL) :")
        print("  1. Preparez votre geste devant la camera")
        print("  2. Appuyez sur [ESPACE] pour COMMENCER l'enregistrement")
        print("  3. Faites votre signe NATURELLEMENT (mouvement complet)")
        print("  4. Appuyez sur [ESPACE] pour ARRETER l'enregistrement")
        print("  5. Repetez pour chaque echantillon")
    print()

    os.makedirs(data_dir, exist_ok=True)

    detector = HandDetector()
    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

    if not cap.isOpened():
        print("ERREUR: Camera non disponible.")
        return

    window_name = "Collecte de donnees"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, CAMERA["width"], CAMERA["height"])

    total_collected = 0

    for i, sign in enumerate(signs):
        print(f"\n[{i + 1}/{len(signs)}] Signe: '{sign}'")
        if auto_mode:
            collected, quit_req = _collect_sign_auto(
                sign, detector, extractor, cap, num_samples,
                data_dir=data_dir,
            )
        else:
            collected, quit_req = _collect_sign(
                sign, detector, extractor, cap, num_samples,
                data_dir=data_dir,
            )
        total_collected += collected
        if quit_req:
            print("\nArret demande par l'utilisateur.")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    print(f"\n{'=' * 58}")
    print(f"  Collecte terminee : {total_collected} echantillons")
    print(f"  Donnees dans      : {data_dir}")
    print(_next_step_hint(data_dir))
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
