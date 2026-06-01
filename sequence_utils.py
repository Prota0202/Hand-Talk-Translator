"""Utilitaires séquences vision (alignés collecte ↔ inférence live)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Rééchantillonne une séquence variable en *target_len* frames (comme collect_data)."""
    if len(sequence) == 0:
        raise ValueError("sequence vide")
    if len(sequence) == target_len:
        return sequence.astype(np.float32, copy=False)
    x_old = np.linspace(0, 1, len(sequence))
    x_new = np.linspace(0, 1, target_len)
    return interp1d(x_old, sequence, axis=0, kind="linear")(x_new).astype(np.float32)
