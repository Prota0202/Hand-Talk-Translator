"""Deterministic replay of a session JSONL file.

This module enables a "demo backup mode" for jury defenses:
should the live recognition fail (camera issue, MediaPipe drop,
network glitch on the TTS), the presenter can launch the program with
``--replay sessions/demo.jsonl`` and the application will deliver the
**exact same recorded sequence** of signs, phrases and voice events,
preserving original timing.

The camera and MediaPipe are still active so the user can keep their
hands in front of the lens (visual continuity) — but the recognizer
output is bypassed and replaced with the events stored in the JSONL.

Usage
─────

    from replay_player import ReplayPlayer

    rp = ReplayPlayer("sessions/demo.jsonl", speed=1.0, loop=False)
    rp.start()

    while True:
        for ev in rp.pop_due():
            ...   # apply event to UI / sentence / TTS

        ug = rp.upcoming_gesture()
        if ug:
            label, fake_confidence = ug
            ...   # show in UI as if the LSTM had predicted it

        if rp.is_done():
            break
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


_REPLAYABLE_TYPES = {"sign", "phrase", "voice"}


class ReplayPlayer:
    """Replay a session JSONL file with original (or scaled) timing."""

    def __init__(self, path: str | Path, *, speed: float = 1.0,
                 loop: bool = False) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.path = Path(path)
        self.speed = speed
        self.loop = loop

        self._events = self._load(self.path)
        if not self._events:
            raise ValueError(
                f"Aucun evenement rejouable dans {self.path}. "
                "Le fichier est vide ou ne contient que des events."
            )

        self._idx = 0
        self._t_start: float | None = None
        self._loop_count = 0

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def duration(self) -> float:
        """Original recorded duration in seconds."""
        return self._events[-1]["rel"]

    @property
    def loop_count(self) -> int:
        return self._loop_count

    def signs(self) -> list[str]:
        return [e["text"] for e in self._events if e["type"] == "sign"]

    def phrases(self) -> list[str]:
        return [e["fr"] for e in self._events if e["type"] == "phrase"]

    def start(self) -> None:
        self._t_start = time.perf_counter()
        self._idx = 0

    def restart(self) -> None:
        self.start()
        self._loop_count += 1

    def is_done(self) -> bool:
        if self._idx < len(self._events):
            return False
        if self.loop:
            return False
        return True

    def now_rel(self) -> float:
        """Time since :meth:`start` was called, scaled by ``speed``."""
        if self._t_start is None:
            return 0.0
        return (time.perf_counter() - self._t_start) * self.speed

    def pop_due(self) -> list[dict]:
        """Return (and consume) every event whose time has elapsed."""
        due: list[dict] = []
        if self._t_start is None:
            return due

        # Loop wrap-around if needed
        if self.loop and self._idx >= len(self._events):
            self.restart()

        now = self.now_rel()
        while self._idx < len(self._events):
            e = self._events[self._idx]
            if e["rel"] > now:
                break
            due.append(e)
            self._idx += 1
        return due

    def upcoming_gesture(self, lookahead: float = 1.5
                         ) -> tuple[str, float] | None:
        """Next imminent ``sign`` event within *lookahead* seconds.

        Returns ``(label, fake_confidence)`` so the UI can display a
        smoothly-rising confidence bar that mimics the real recognizer.
        ``None`` if there is no upcoming sign.
        """
        if self._t_start is None or self._idx >= len(self._events):
            return None

        # Find the next sign event
        for j in range(self._idx, len(self._events)):
            e = self._events[j]
            if e["type"] == "sign":
                gap = e["rel"] - self.now_rel()
                if gap < 0 or gap > lookahead:
                    return None
                # Confidence ramps from 0.55 (lookahead away) to 0.97 (firing)
                conf = 0.55 + (lookahead - gap) / lookahead * 0.42
                return e["text"], conf
            # Non-sign events between us and the next sign: ignore
        return None

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"Fichier de replay introuvable : {path}")

        raw: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        events = [r for r in raw if r.get("type") in _REPLAYABLE_TYPES]
        if not events:
            return []

        # Parse timestamps -> seconds relative to the first event
        first_ts = _parse_ts(events[0]["ts"])
        for e in events:
            t = _parse_ts(e["ts"])
            e["rel"] = (t - first_ts).total_seconds()
        return events


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp (handles trailing 'Z' just in case)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def summarise(path: str | Path) -> dict:
    """Quick stats about a session JSONL file (used by inspect_session.py)."""
    rp = ReplayPlayer(path)
    return {
        "path": str(rp.path),
        "duration_s": rp.duration,
        "events": rp.total_events,
        "signs": rp.signs(),
        "phrases": rp.phrases(),
    }
