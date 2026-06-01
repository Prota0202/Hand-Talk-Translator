"""Renomme les dossiers data/ qui utilisent des mots français au lieu de glosses LSF."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DATA_DIR, EXCLUDED_VISION_LABELS, canonical_gloss


def _next_index(dest_dir: str) -> int:
    indices = []
    for fname in os.listdir(dest_dir):
        if not fname.endswith(".npy"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices, default=-1) + 1


def _merge_dir(src_dir: str, dest_dir: str) -> None:
    next_idx = _next_index(dest_dir)
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".npy"):
            continue
        shutil.move(
            os.path.join(src_dir, fname),
            os.path.join(dest_dir, f"{next_idx}.npy"),
        )
        next_idx += 1
    os.rmdir(src_dir)


def _rename_folder(src: str, dest: str) -> None:
    """Rename *src* to *dest*, including case-only changes on Windows."""
    if os.path.normcase(src) == os.path.normcase(dest):
        if os.path.basename(src) == os.path.basename(dest):
            return
        tmp = f"{src}.__gloss_rename__"
        os.rename(src, tmp)
        os.rename(tmp, dest)
        return

    if os.path.isdir(dest):
        _merge_dir(src, dest)
    else:
        os.rename(src, dest)


def _renumber_npy_files(folder: str) -> None:
    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".npy") and os.path.splitext(f)[0].isdigit()
    )
    if not files:
        return
    tmp_dir = os.path.join(folder, ".__renumber__")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, fname in enumerate(files):
        shutil.move(
            os.path.join(folder, fname),
            os.path.join(tmp_dir, f"{i}.npy"),
        )
    for fname in os.listdir(tmp_dir):
        shutil.move(os.path.join(tmp_dir, fname), os.path.join(folder, fname))
    os.rmdir(tmp_dir)


def _remove_excluded(data_dir: str, *, dry_run: bool = False) -> int:
    removed = 0
    for name in sorted(os.listdir(data_dir)):
        if name.lower() not in {x.lower() for x in EXCLUDED_VISION_LABELS}:
            continue
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        print(f"  SUPPRIME {name!r} (gloss vision interdit)")
        removed += 1
        if not dry_run:
            shutil.rmtree(path)
    return removed


def fix_data_dir(data_dir: str, *, dry_run: bool = False) -> int:
    if not os.path.isdir(data_dir):
        print(f"Dossier introuvable : {data_dir}")
        return 1

    removed = _remove_excluded(data_dir, dry_run=dry_run)
    renamed = 0
    for name in sorted(os.listdir(data_dir)):
        src = os.path.join(data_dir, name)
        if not os.path.isdir(src) or name.startswith("."):
            continue
        canon = canonical_gloss(name)
        if canon == name:
            continue

        dest = os.path.join(data_dir, canon)
        print(f"  {name!r} -> {canon!r}")
        renamed += 1
        if dry_run:
            continue

        _rename_folder(src, dest)
        if os.path.isdir(dest):
            _renumber_npy_files(dest)

    if renamed == 0 and removed == 0:
        print("Aucun dossier a renommer ou supprimer.")
    else:
        if removed:
            action = "seraient supprimes" if dry_run else "supprimes"
            print(f"\n{removed} dossier(s) interdit(s) {action}.")
        if renamed:
            action = "seraient renommes" if dry_run else "renommes"
            print(f"{renamed} dossier(s) {action}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Dossier de donnees (defaut: data/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les renommages sans les appliquer",
    )
    args = parser.parse_args()
    return fix_data_dir(os.path.abspath(args.data_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
