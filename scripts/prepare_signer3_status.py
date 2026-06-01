"""Print data_signer3 collection status (used by prepare/collect scripts)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CROSS_SIGNER3_SIGNS, DATA_SIGNER3_DIR

TARGET = 5


def main() -> None:
    os.makedirs(DATA_SIGNER3_DIR, exist_ok=True)
    missing: list[str] = []
    done: list[str] = []

    for sign in CROSS_SIGNER3_SIGNS:
        sign_dir = os.path.join(DATA_SIGNER3_DIR, sign)
        os.makedirs(sign_dir, exist_ok=True)
        n = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        if n >= TARGET:
            done.append(sign)
        else:
            missing.append(f"{sign} ({n}/{TARGET})")

    print(f"  Protocole : {len(CROSS_SIGNER3_SIGNS)} glosses (mots + chiffres, sans alphabet)")
    print(f"  Glosses   : {', '.join(CROSS_SIGNER3_SIGNS)}")
    print(f"  Dossier   : data_signer3\\")
    print(f"  Mode      : reprise (--resume) — seulement ce qui manque")
    if done:
        print(f"  Complets  : {', '.join(done)}")
    if missing:
        print(f"  Reste     : {', '.join(missing)}")
    else:
        print("  Reste     : rien — collecte deja complete")


if __name__ == "__main__":
    main()
