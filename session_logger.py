"""Append-only JSONL logger for live translation sessions.

Each translator session writes a single ``sessions/<YYYYMMDD_HHMMSS>.jsonl``
file in which **every committed sign**, every transcribed voice
utterance and every spoken French sentence is recorded with a
millisecond-resolution timestamp. The file is flushed after every
write so a crash does not lose history.

The format is intentionally line-delimited JSON to make it trivially
parsable by external scripts (analytics, replay, dataset enrichment).

Example record::

    {"ts": "2026-04-19T22:14:33.812", "type": "sign",   "text": "MOI"}
    {"ts": "2026-04-19T22:14:35.501", "type": "phrase", "lsf": "MOI BIEN", "fr": "Je vais bien."}
    {"ts": "2026-04-19T22:14:40.200", "type": "voice",  "text": "comment ca va"}
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import IO


class SessionLogger:
    """Lightweight JSONL session writer.

    The file is created lazily on the first ``log_*`` call so an
    immediately-aborted run does not litter the ``sessions/`` directory
    with empty files.
    """

    def __init__(self, directory: str = "sessions",
                 prefix: str = "session") -> None:
        self.directory = directory
        self.prefix = prefix
        self._file: IO[str] | None = None
        self._path: str | None = None

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def path(self) -> str | None:
        return self._path

    def log_sign(self, gloss: str) -> None:
        self._write({"type": "sign", "text": gloss})

    def log_phrase(self, lsf: str, french: str) -> None:
        self._write({"type": "phrase", "lsf": lsf, "fr": french})

    def log_voice(self, text: str) -> None:
        self._write({"type": "voice", "text": text})

    def log_event(self, name: str, **payload) -> None:
        """Generic event hook (e.g. ``"reset"``, ``"export"``)."""
        record = {"type": "event", "name": name}
        record.update(payload)
        self._write(record)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._write({"type": "event", "name": "session_end"})
            finally:
                self._file.close()
                self._file = None

    # ── internal ────────────────────────────────────────────────────────────

    def _ensure_open(self) -> IO[str]:
        if self._file is not None:
            return self._file
        os.makedirs(self.directory, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(self.directory, f"{self.prefix}_{ts}.jsonl")
        self._file = open(self._path, "w", encoding="utf-8")
        # Header record makes the file self-describing
        header = {
            "type": "event",
            "name": "session_start",
            "ts": datetime.now().isoformat(timespec="milliseconds"),
        }
        self._file.write(json.dumps(header, ensure_ascii=False) + "\n")
        self._file.flush()
        return self._file

    def _write(self, record: dict) -> None:
        f = self._ensure_open()
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"),
                  **record}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
