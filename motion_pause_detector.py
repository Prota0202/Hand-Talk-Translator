"""Detects a 'Pause' gesture based on hand motion (shake)."""

from collections import deque

from config import MOTION_PAUSE


class MotionPauseDetector:
    """Detect a left-right shake of the wrist as a Pause gesture."""

    def __init__(self):
        self._positions = deque(maxlen=MOTION_PAUSE["window_size"])
        self._last_trigger_ts = 0.0

    def update(self, hand_landmarks, ts: float) -> bool:
        """Update motion buffer and return True when a pause is detected."""
        if hand_landmarks is None:
            self._positions.clear()
            return False

        # Wrist landmark is index 0
        wrist = hand_landmarks[0]
        self._positions.append((wrist.x, wrist.y, ts))

        if len(self._positions) < self._positions.maxlen:
            return False

        # Cooldown to avoid multiple triggers
        if ts - self._last_trigger_ts < MOTION_PAUSE["cooldown_seconds"]:
            return False

        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        # Check dominant horizontal motion
        if dx < MOTION_PAUSE["min_horizontal"] or dx < dy * MOTION_PAUSE["dominance"]:
            return False

        # Count direction changes in x (shake)
        changes = 0
        last_dir = 0
        for i in range(1, len(xs)):
            diff = xs[i] - xs[i - 1]
            curr_dir = 1 if diff > MOTION_PAUSE["min_step"] else -1 if diff < -MOTION_PAUSE["min_step"] else 0
            if curr_dir != 0 and last_dir != 0 and curr_dir != last_dir:
                changes += 1
            if curr_dir != 0:
                last_dir = curr_dir

        if changes >= MOTION_PAUSE["min_direction_changes"]:
            self._last_trigger_ts = ts
            return True

        return False
