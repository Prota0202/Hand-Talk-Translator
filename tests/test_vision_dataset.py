"""Tests for cross-signer dataset loading helpers."""

import os
import tempfile

import numpy as np
import pytest

from config import FEATURES_PER_FRAME, SEQUENCE_LENGTH
from vision_dataset import load_sequences_from_dir, resolve_label_index


LABELS = ["MOI", "Bonjour", "A"]


def test_resolve_label_index_exact_and_case_insensitive():
    assert resolve_label_index("MOI", LABELS) == 0
    assert resolve_label_index("bonjour", LABELS) == 1
    assert resolve_label_index("BONJOUR", LABELS) == 1
    assert resolve_label_index("unknown", LABELS) is None


def test_load_sequences_from_dir_sliding_window():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "MOI")
        os.makedirs(sign_dir)
        long_seq = np.random.randn(45, FEATURES_PER_FRAME).astype(np.float32)
        np.save(os.path.join(sign_dir, "sample.npy"), long_seq)

        X, y, counts, unknown = load_sequences_from_dir(tmp, LABELS)

        assert len(X) == 45 - SEQUENCE_LENGTH + 1
        assert np.all(y == 0)
        assert counts["MOI"] == len(X)
        assert unknown == []


def test_load_sequences_from_dir_unknown_folder():
    with tempfile.TemporaryDirectory() as tmp:
        unknown_dir = os.path.join(tmp, "PAS_UN_SIGNE")
        os.makedirs(unknown_dir)
        arr = np.random.randn(SEQUENCE_LENGTH, FEATURES_PER_FRAME).astype(np.float32)
        np.save(os.path.join(unknown_dir, "0.npy"), arr)

        with pytest.raises(ValueError, match="Aucune sequence valide"):
            load_sequences_from_dir(tmp, LABELS)
