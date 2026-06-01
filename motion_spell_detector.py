"""Detects a vertical wrist nod used to trigger phrase read-out (TTS)."""

from collections import deque

from config import MOTION_SPELL


class MotionSpellDetector:
    """Detect an up-down nod of the wrist (head-nod proxy)."""

    def __init__(self):
        self._positions = deque(maxlen=MOTION_SPELL["window_size"])
        self._last_trigger_ts = 0.0

    def update(self, hand_landmarks, ts: float) -> bool:
        """Update motion buffer and return True when a nod is detected."""
        if hand_landmarks is None:
            self._positions.clear()
            return False

        wrist = hand_landmarks[0]
        self._positions.append((wrist.x, wrist.y, ts))

        if len(self._positions) < self._positions.maxlen:
            return False

        if ts - self._last_trigger_ts < MOTION_SPELL["cooldown_seconds"]:
            return False

        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        if dy < MOTION_SPELL["min_vertical"] or dy < dx * MOTION_SPELL["dominance"]:
            return False

        changes = 0
        last_dir = 0
        for i in range(1, len(ys)):
            diff = ys[i] - ys[i - 1]
            curr_dir = 1 if diff > MOTION_SPELL["min_step"] else -1 if diff < -MOTION_SPELL["min_step"] else 0
            if curr_dir != 0 and last_dir != 0 and curr_dir != last_dir:
                changes += 1
            if curr_dir != 0:
                last_dir = curr_dir

        if changes >= MOTION_SPELL["min_direction_changes"]:
            self._last_trigger_ts = ts
            return True

        return False
