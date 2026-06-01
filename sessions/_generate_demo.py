"""Generate a synthetic demo.jsonl file matching the TFE presentation phrase.

Run once to bootstrap a replay file before you've recorded a real one:

    py -3.11 sessions/_generate_demo.py

The output (`sessions/demo.jsonl`) is then usable as:

    py -3.11 main.py --replay sessions/demo.jsonl

Once you've recorded a real session, you can replace it with:

    py -3.11 inspect_session.py --promote-to-demo
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

T0 = datetime(2026, 4, 19, 21, 0, 0)


def _ev(dt_seconds: float, **payload) -> str:
    ts = (T0 + timedelta(seconds=dt_seconds)).isoformat(timespec="milliseconds")
    return json.dumps({"ts": ts, **payload}, ensure_ascii=False)


# Timeline (seconds, payload). Spacing chosen to feel natural on stage.
EVENTS = [
    (0.0,  {"type": "event",  "name": "session_start"}),
    (0.1,  {"type": "event",  "name": "session_start_args",
            "debug": False, "listen": False}),

    # "Bonjour, je m'appelle Abdelbadi."
    (1.5,  {"type": "sign",   "text": "Bonjour"}),
    (2.6,  {"type": "sign",   "text": "MOI"}),
    (3.4,  {"type": "sign",   "text": "NOM"}),
    (4.1,  {"type": "sign",   "text": "A"}),
    (4.6,  {"type": "sign",   "text": "B"}),
    (5.1,  {"type": "sign",   "text": "D"}),
    (5.6,  {"type": "sign",   "text": "E"}),
    (6.1,  {"type": "sign",   "text": "L"}),
    (6.6,  {"type": "sign",   "text": "B"}),
    (7.1,  {"type": "sign",   "text": "A"}),
    (7.6,  {"type": "sign",   "text": "D"}),
    (8.1,  {"type": "sign",   "text": "I"}),
    (9.0,  {"type": "phrase",
            "lsf": "Bonjour MOI NOM A B D E L B A D I",
            "fr":  "Bonjour, je m'appelle Abdelbadi."}),

    # "Je suis étudiant."
    (11.5, {"type": "sign",   "text": "MOI"}),
    (12.3, {"type": "sign",   "text": "ETUDIANT"}),
    (13.5, {"type": "phrase",
            "lsf": "MOI ETUDIANT",
            "fr":  "Je suis étudiant."}),

    # "J'ai 24 ans."
    (15.5, {"type": "sign",   "text": "MOI"}),
    (16.2, {"type": "sign",   "text": "2"}),
    (16.7, {"type": "sign",   "text": "4"}),
    (18.5, {"type": "phrase",
            "lsf": "MOI 2 4",
            "fr":  "J'ai 24 ans."}),

    # "Aujourd'hui je présente."
    (20.5, {"type": "sign",   "text": "Aujourd'hui"}),
    (21.5, {"type": "sign",   "text": "MOI"}),
    (22.2, {"type": "sign",   "text": "PRESENT"}),
    (23.5, {"type": "phrase",
            "lsf": "Aujourd'hui MOI PRESENT",
            "fr":  "Aujourd'hui je présente."}),

    (24.5, {"type": "event",  "name": "session_end"}),
]


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "demo.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for dt, payload in EVENTS:
            fh.write(_ev(dt, **payload) + "\n")
    print(f"Demo written: {out_path}")
    print(f"Try it with: py -3.11 main.py --replay {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
