"""Real-time gesture recognition using the trained LSTM model."""

import json
import math
import time
from collections import Counter

import numpy as np
import torch

from config import (
    LABELS_PATH,
    MODEL_PATH,
    MOTION,
    RECOGNITION,
    SEQUENCE_LENGTH,
)
from feature_extractor import FeatureExtractor
from model import GestureLSTM


class GestureRecognizer:
    """Buffers landmark features and classifies gestures with the LSTM."""

    def __init__(self):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

        self.model = GestureLSTM(
            num_features=checkpoint["num_features"],
            num_classes=checkpoint["num_classes"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        with open(LABELS_PATH, "r", encoding="utf-8") as fh:
            self.labels: list[str] = json.load(fh)

        self.extractor = FeatureExtractor()
        self.frame_buffer: list[np.ndarray] = []
        self.pred_buffer: list[str] = []
        self.pred_buf_size = RECOGNITION["prediction_buffer"]
        self.threshold = RECOGNITION["confidence_threshold"]
        self._missing_frames = 0
        self.motion_active = False
        self._motion_count = 0

        self._last_gesture: str | None = None
        self._last_time: float = 0.0
        self._cooldown = RECOGNITION["cooldown_seconds"]
        self.last_probs: dict[str, float] = {}

    # ── public API ───────────────────────────────────────────────────────

    @property
    def num_signs(self) -> int:
        return len(self.labels)

    @property
    def buffer_fill(self) -> float:
        return len(self.frame_buffer) / SEQUENCE_LENGTH

    def process_result(self, result):
        """Feed one frame and return ``(gesture_name, confidence, motion, accepted)``."""
        if result is None or not result.hand_landmarks:
            self._missing_frames += 1
            if self._missing_frames >= RECOGNITION["missing_frames_reset"]:
                self.reset()
                self.extractor.reset()
            return None, 0.0, 0.0, False

        self._missing_frames = 0
        features, motion = self.extractor.extract_from_result(result)
        self.frame_buffer.append(features)
        if len(self.frame_buffer) > SEQUENCE_LENGTH:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) < SEQUENCE_LENGTH:
            return None, 0.0, motion, False

        seq = torch.tensor(
            np.array(self.frame_buffer, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(seq)
            probs = torch.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs))
        conf = float(probs[idx])
        name = self.labels[idx]
        self.last_probs = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        # Entropy-based rejection: high entropy = uncertain = not a known sign
        # Max entropy for N classes = log(N). We reject above 40% of max entropy.
        n = len(self.labels)
        entropy = -sum(float(p) * math.log(float(p) + 1e-9) for p in probs)
        max_entropy = math.log(n)
        if entropy > 0.2 * max_entropy:
            return None, 0.0, motion, False

        # temporal smoothing
        self.pred_buffer.append(name)
        if len(self.pred_buffer) > self.pred_buf_size:
            self.pred_buffer.pop(0)

        counter = Counter(self.pred_buffer)
        top, count = counter.most_common(1)[0]
        smoothed = count / len(self.pred_buffer)

        # motion gating (for dynamic gestures)
        if motion >= MOTION["activation"]:
            self._motion_count = min(self._motion_count + 1, MOTION["frames_required"])
        elif motion <= MOTION["release"]:
            self._motion_count = max(self._motion_count - 1, 0)
        if self._motion_count >= MOTION["frames_required"]:
            self.motion_active = True
        if self._motion_count == 0:
            self.motion_active = False

        accepted = smoothed >= RECOGNITION["smoothed_threshold"] and conf >= self.threshold
        return top, conf, motion, accepted

    def is_new_gesture(self, gesture_name: str | None) -> bool:
        """True when the gesture is different from the last one accepted
        or the cooldown has elapsed."""
        if gesture_name is None:
            self._last_gesture = None
            return False
        now = time.time()
        if gesture_name != self._last_gesture:
            self._last_gesture = gesture_name
            self._last_time = now
            return True
        if now - self._last_time > self._cooldown:
            self._last_time = now
            return True
        return False

    def reset(self):
        self.frame_buffer.clear()
        self.pred_buffer.clear()
        self._last_gesture = None
        self._missing_frames = 0
        self.motion_active = False
        self._motion_count = 0
