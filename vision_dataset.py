"""Chargement de séquences vision (.npy) et correspondance dossier → classe."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from config import EXCLUDED_VISION_LABELS, FEATURES_PER_FRAME, LABELS_PATH, SEQUENCE_LENGTH


def load_training_labels(labels_path: str = LABELS_PATH) -> list[str]:
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"labels.json introuvable: {labels_path}\n"
            "Lancez d'abord: py -3.11 train_model.py"
        )
    with open(labels_path, encoding="utf-8") as fh:
        labels = json.load(fh)
    if not labels:
        raise ValueError(f"Aucune classe dans {labels_path}")
    return labels


def resolve_label_index(folder_name: str, labels: list[str]) -> Optional[int]:
    """Retourne l'index LSTM d'un dossier ``data/<signe>/`` (ou None)."""
    if folder_name in labels:
        return labels.index(folder_name)

    by_lower = {label.lower(): idx for idx, label in enumerate(labels)}
    return by_lower.get(folder_name.lower())


def load_sequences_from_dir(
    data_dir: str,
    labels: list[str],
    *,
    skip_unknown: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[str]]:
    """Charge les séquences brutes d'un dossier (sans augmentation).

    Returns
    -------
    X, y, per_class_counts, unknown_folders
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dossier introuvable: {data_dir}\n"
            "Collectez d'abord avec:\n"
            "  py -3.11 collect_data.py --data-dir data_signer2"
        )

    candidate_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    X: list[np.ndarray] = []
    y: list[int] = []
    per_class_counts: dict[str, int] = {label: 0 for label in labels}
    unknown_folders: list[str] = []

    for folder_name in candidate_dirs:
        if folder_name.lower() in {x.lower() for x in EXCLUDED_VISION_LABELS}:
            continue
        label_idx = resolve_label_index(folder_name, labels)
        if label_idx is None:
            folder = os.path.join(data_dir, folder_name)
            if any(f.endswith(".npy") for f in os.listdir(folder)):
                unknown_folders.append(folder_name)
            continue

        canonical = labels[label_idx]
        folder = os.path.join(data_dir, folder_name)
        loaded = 0
        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".npy")):
            arr = np.load(os.path.join(folder, fname))
            if arr.ndim != 2 or arr.shape[1] != FEATURES_PER_FRAME:
                continue
            if arr.shape[0] == SEQUENCE_LENGTH:
                X.append(arr)
                y.append(label_idx)
                loaded += 1
            else:
                for start in range(0, len(arr) - SEQUENCE_LENGTH + 1):
                    X.append(arr[start: start + SEQUENCE_LENGTH])
                    y.append(label_idx)
                    loaded += 1
        if loaded:
            per_class_counts[canonical] += loaded

    if skip_unknown and unknown_folders:
        pass  # caller may warn

    if not X:
        raise ValueError(
            f"Aucune sequence valide dans {data_dir}. "
            "Verifiez les noms de dossiers (ex. MOI, Bonjour, A)."
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        per_class_counts,
        unknown_folders,
    )
