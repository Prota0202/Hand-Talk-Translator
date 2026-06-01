"""Rolling end-to-end latency tracker for the recognition pipeline.

Records the duration of each pipeline stage (camera read, MediaPipe
hand detection, LSTM inference, LSF→French translation, total frame
time) over a sliding window and exposes their rolling means.

Usage
─────
    tracker = LatencyTracker(window=60)

    with tracker.measure("camera"):
        ret, frame = cap.read()
    with tracker.measure("mediapipe"):
        results = detector.detect(rgb)
    ...
    tracker.record("total", (time.perf_counter() - t_frame) * 1000)

    print(tracker.snapshot())
    # {'camera': 1.7, 'mediapipe': 12.4, 'lstm': 0.5, ...}
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager


class LatencyTracker:
    """Maintains a deque of recent timings (in ms) per stage."""

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._timings: dict[str, deque[float]] = {}

    def record(self, stage: str, ms: float) -> None:
        buf = self._timings.get(stage)
        if buf is None:
            buf = deque(maxlen=self.window)
            self._timings[stage] = buf
        buf.append(ms)

    @contextmanager
    def measure(self, stage: str):
        """Context manager: time the wrapped block and store the result."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def mean(self, stage: str) -> float:
        buf = self._timings.get(stage)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def snapshot(self) -> dict[str, float]:
        """Rolling means for every recorded stage (ms)."""
        return {k: self.mean(k) for k in self._timings}

    def reset(self) -> None:
        self._timings.clear()
