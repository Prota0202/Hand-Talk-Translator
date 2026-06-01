"""Central configuration for Hand Talk Translator."""

import os
import sys

# When packaged with PyInstaller (--onedir or --onefile) we resolve mutable
# resources (models, data, sessions) **next to the executable** rather than
# inside the bundle, so that the user can update models / read sessions
# without rebuilding the .exe. The Python source layout is untouched in
# normal dev mode (we just use this file's directory).
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_SIGNER2_DIR = os.path.join(BASE_DIR, "data_signer2")
DATA_SIGNER3_DIR = os.path.join(BASE_DIR, "data_signer3")

# Protocole signeur 2 (enfant) : 13 glosses × 5 échantillons.
CROSS_SIGNER_EVAL_SIGNS = [
    "MOI", "NOM", "Bonjour", "2", "4", "ans", "ETUDIANT",
    "A", "B", "D", "E", "L", "I",
]

# Protocole signeur 3 (adulte) : mots + chiffres, sans alphabet.
CROSS_SIGNER3_SIGNS = [
    "MOI", "NOM", "Bonjour", "2", "4", "ans", "ETUDIANT",
]
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ── MediaPipe ────────────────────────────────────────────────────────────────
MEDIAPIPE = {
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.6,
}

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA = {
    "index": 0,
    "width": 1280,
    "height": 720,
}

# ── Features ─────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21
NUM_HANDS = 2
INCLUDE_VELOCITY = True
FEATURES_PER_HAND = NUM_LANDMARKS * 3
FEATURES_PER_FRAME = FEATURES_PER_HAND * NUM_HANDS * (2 if INCLUDE_VELOCITY else 1)
SEQUENCE_LENGTH = 30  # frames the model sees (resampled to this length)

# ── Data collection ──────────────────────────────────────────────────────────
COLLECTION = {
    "samples_per_sign": 20,
    "min_frames": 10,
    "max_frames": 150,
}

# ── Training ─────────────────────────────────────────────────────────────────
TRAINING = {
    "epochs": 300,
    "batch_size": 32,
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "early_stopping_patience": 30,
    "augmentation_factor": 4,
    "noise_std": 0.025,
}

# ── Real-time recognition ────────────────────────────────────────────────────
RECOGNITION = {
    "confidence_threshold": 0.85,
    "smoothed_threshold": 0.80,
    "min_commit_confidence": 0.75,
    "cooldown_seconds": 1.5,
    "prediction_buffer": 5,
    "missing_frames_reset": 5,
}

# ── Phrase building ──────────────────────────────────────────────────────────
PAUSE_LABEL = "Pause"
UNKNOWN_LABEL = "AUTRE"
STABILITY_FRAMES_REQUIRED = 5
COMMIT = {
    "min_interval_seconds":      0.8,   # debounce same gesture repeated
    "spell_letter_cooldown":     1,   # min seconds between ANY two letters in spell mode
}

# ── Motion-based Pause (shake) ───────────────────────────────────────────────
MOTION_PAUSE = {
    "window_size": 12,          # frames used to detect motion
    "min_horizontal": 0.08,     # min x-range in normalized coords
    "min_step": 0.01,           # per-frame delta to count as movement
    "min_direction_changes": 2, # shakes left-right at least twice
    "dominance": 1.8,           # horizontal must dominate vertical
    "cooldown_seconds": 1.0,    # pause gesture debounce
}

# ── Motion gating (dynamic gestures) ─────────────────────────────────────────
MOTION = {
    "activation": 0.015,   # motion energy needed to consider a gesture active
    "release": 0.008,      # below this -> motion inactive
    "frames_required": 3,  # consecutive frames to toggle active/inactive
    "require_active": False,  # True -> only accept gestures when moving
}

# ── LSF Translation ──────────────────────────────────────────────────────────
# Signs use LSF gloss names (uppercase in the translator).
# The lsf_translator module converts LSF grammar → French automatically.

# ── Motion-based spelling toggle (up-down nod) ───────────────────────────────
MOTION_SPELL = {
    "window_size": 12,
    "min_vertical": 0.08,
    "min_step": 0.01,
    "min_direction_changes": 2,
    "dominance": 1.8,
    "cooldown_seconds": 1.0,
}

# ── UI ───────────────────────────────────────────────────────────────────────
UI_COLORS = {
    "primary": (255, 168, 0),
    "secondary": (0, 200, 100),
    "bg": (40, 40, 40),
    "text": (255, 255, 255),
    "accent": (0, 150, 255),
    "danger": (0, 0, 255),
}
HISTORY_MAX = 8

# ── Branding (splash + UI footer) ────────────────────────────────────────────
PRESENTER = {
    "name":     "Abdelbadi",
    "school":   "ECAM",
    "year":     "2025 — 2026",
    "title":    "Hand Talk Translator",
    "subtitle": "Traducteur LSF → Français en temps réel",
    "tagline":  "Travail de Fin d'Études",
}
SPLASH = {
    "enabled":  True,
    "duration": 2.8,   # seconds
}

# ── Glove (sensor-based pipeline) ────────────────────────────────────────────
GLOVE_DIR        = os.path.join(BASE_DIR, "data_glove")
GLOVE_MODEL_PATH = os.path.join(MODEL_DIR, "glove_model.pth")
GLOVE_LABELS     = os.path.join(MODEL_DIR, "glove_labels.json")
GLOVE_CALIB_PATH = os.path.join(MODEL_DIR, "glove_calibration.json")

# Convenience: where session JSONLs and exported transcripts go. Always
# next to the executable (or repo root in dev) so the user can find them.
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

GLOVE = {
    "port":               "MOCK",  # default for headless dev; override per script
    "baud":               115200,
    "sample_hz":          50,
    "sequence_length":    30,      # frames per sample (~0.6 s @ 50 Hz)
    "num_flex":           3,       # thumb / index / middle
    "features_per_frame": 9,       # 3 flex + ax/ay/az + gx/gy/gz
    "samples_per_sign":   25,      # collected per sign
    "epochs":             200,
    "batch_size":         32,
    "lr":                 0.001,
    "augmentation":       4,
    "noise_std":          0.02,
    "early_stop":         25,
    # With few classes softmax peaks stay modest (~40-55% for 3 signs).
    # Use margin (top1 - top2) as the main commit signal, not raw confidence.
    "confidence_threshold": 0.35,
    "margin_threshold":     0.06,
    "smoothed_threshold":   0.35,
    "cooldown_seconds":     2.5,
    "neutral_frames":       18,
    "rearm_seconds":        1.0,    # delai min entre deux signes (meme si le modele reste sur MOI)
    "prediction_buffer":    5,
}

# ── TTS ──────────────────────────────────────────────────────────────────────
TTS = {
    "rate": 150,
    "volume": 1.0,
    "engine": "edge",
    "edge_voice": "fr-BE-CharlineNeural",
    "edge_rate": "+0%",
    "edge_volume": "+0%",
}

SPEAK = {
    "on_commit": False,  # speak after each accepted sign/word
    "on_space": False,   # speak full phrase when Space is pressed
}

# ── Finish-phrase gesture (both open palms) ───────────────────────────────────
FINISH_GESTURE = {
    "enabled": True,
    "required_frames": 3,
    "cooldown_seconds": 2.0,
    "min_spread": 0.12,  # min distance between index and pinky tips
}

# ── Recommended vocabulary for data collection ───────────────────────────────
RECOMMENDED_SIGNS = [
    # ── Phrase demo ──────────────────────────────────────────────────────
    # MOI NOM ABDELBADI | MOI 24 ANS | MOI ETUDIANT ECAM | ICI PROJET FIN ETUDES
    "MOI", "NOM", "ABDELBADI", "24", "ANS",
    "ETUDIANT", "ECAM", "ICI", "PROJET", "FIN", "ETUDES",
    # ── LSF courant ──────────────────────────────────────────────────────
    "BONJOUR", "AU-REVOIR", "MERCI", "OUI", "NON",
    "TOI", "LUI", "NOUS", "BIEN", "MAL",
    "VOULOIR", "AIMER", "MANGER", "BOIRE", "DORMIR",
    "TRAVAILLER", "PARLER", "ECOUTER", "COMPRENDRE",
    "AIDE", "QUOI", "PAS",
    # ── Chiffres ─────────────────────────────────────────────────────────
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    # ── Alphabet (epellation) ────────────────────────────────────────────
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z",
]
