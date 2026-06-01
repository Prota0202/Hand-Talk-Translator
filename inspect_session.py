"""Inspect / validate a session JSONL file before promoting it to a
replay demo.

Use this **before the jury defense** to make sure ``demo.jsonl``
contains exactly what you want to replay (right signs in the right
order, no rogue voice events, expected duration).

Usage
─────

    py -3.11 inspect_session.py                    # most recent session
    py -3.11 inspect_session.py sessions/foo.jsonl
    py -3.11 inspect_session.py --promote-to-demo  # copy latest as demo.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from replay_player import _parse_ts


def _find_latest(directory: Path) -> Path | None:
    files = sorted(directory.glob("session_*.jsonl"))
    return files[-1] if files else None


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:5.1f} s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}m {s:4.1f}s"


def inspect(path: Path, *, verbose: bool = False) -> int:
    if not path.is_file():
        print(f"[ERREUR] Fichier introuvable : {path}")
        return 1

    print()
    print("=" * 64)
    print(f"  {path}")
    print("=" * 64)

    type_counts: Counter[str] = Counter()
    signs: list[str] = []
    phrases: list[tuple[str, str]] = []
    voices: list[str] = []
    events: list[dict] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(ev)
            t = ev.get("type", "?")
            type_counts[t] += 1
            if t == "sign":
                signs.append(ev.get("text", ""))
            elif t == "phrase":
                phrases.append((ev.get("lsf", ""), ev.get("fr", "")))
            elif t == "voice":
                voices.append(ev.get("text", ""))

    if not events:
        print("\n  [VIDE] Aucun evenement.")
        return 1

    first_ts = _parse_ts(events[0]["ts"])
    last_ts = _parse_ts(events[-1]["ts"])
    duration = (last_ts - first_ts).total_seconds()

    # ── header summary ─────────────────────────────────────────────────────
    print(f"\n  Date         : {first_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duree        : {_format_seconds(duration)}")
    print(f"  Evenements   : {len(events)}")
    for t, n in type_counts.most_common():
        print(f"    - {t:<12s} {n}")

    # ── signs ──────────────────────────────────────────────────────────────
    print(f"\n  SIGNES ({len(signs)}):")
    if signs:
        # 8 columns
        for i in range(0, len(signs), 8):
            row = signs[i:i + 8]
            print("    " + "  ".join(f"{s:<10s}" for s in row))
        most_common = Counter(signs).most_common(5)
        print("\n  Top 5 signes : "
              + ", ".join(f"{s} ({n})" for s, n in most_common))
    else:
        print("    (aucun)")

    # ── phrases ────────────────────────────────────────────────────────────
    print(f"\n  PHRASES PRONONCEES ({len(phrases)}):")
    if phrases:
        for lsf, fr in phrases:
            print(f"    LSF : {lsf}")
            print(f"    FR  : {fr}")
            print()
    else:
        print("    (aucune)")

    # ── voice ──────────────────────────────────────────────────────────────
    if voices:
        print(f"\n  ENTREES VOCALES ({len(voices)}):")
        for v in voices:
            print(f"    > {v}")

    # ── verbose timeline ───────────────────────────────────────────────────
    if verbose:
        print("\n  TIMELINE :")
        for ev in events:
            t = _parse_ts(ev["ts"])
            rel = (t - first_ts).total_seconds()
            t_str = ev.get("type", "?")
            payload = ev.get("text") or ev.get("fr") or ev.get("name") or ""
            print(f"    [{rel:6.2f}s] {t_str:<10s} {payload}")

    # ── verdict ────────────────────────────────────────────────────────────
    print()
    issues: list[str] = []
    if not signs and not phrases:
        issues.append("Aucun signe ni phrase : la session est vide.")
    if duration < 3:
        issues.append(f"Duree tres courte ({duration:.1f}s).")
    if duration > 300:
        issues.append(f"Duree tres longue ({duration:.1f}s) — probablement "
                      "pas une demo propre.")

    if issues:
        print("  [WARN]")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  [OK] Cette session ressemble a une demo viable.")

    print()
    return 0 if not issues else 2


def promote_to_demo(latest: Path, demo_path: Path) -> int:
    if not latest.is_file():
        print(f"[ERREUR] Fichier source introuvable : {latest}")
        return 1
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, demo_path)
    print(f"\n  Demo promue : {demo_path}")
    print(f"  Lance la replay :")
    print(f"    py -3.11 main.py --replay {demo_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?",
                        help="Chemin du JSONL (par defaut: dernier dans sessions/)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Afficher la timeline complete")
    parser.add_argument("--promote-to-demo", action="store_true",
                        help="Copier le fichier vers sessions/demo.jsonl")
    args = parser.parse_args()

    sessions_dir = Path("sessions")
    if args.file:
        path = Path(args.file)
    else:
        latest = _find_latest(sessions_dir)
        if latest is None:
            print(f"[ERREUR] Aucune session dans {sessions_dir}/")
            return 1
        path = latest
        print(f"  (Inspection de la derniere session: {path.name})")

    rc = inspect(path, verbose=args.verbose)

    if args.promote_to_demo:
        rc2 = promote_to_demo(path, sessions_dir / "demo.jsonl")
        rc = max(rc, rc2)

    return rc


if __name__ == "__main__":
    sys.exit(main())
