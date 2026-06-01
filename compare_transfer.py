"""Compare training *with* and *without* transfer-learning from ASL.

Runs ``train_model.py`` twice in sequence:

1. ``--no-transfer`` → trains the LSTM from scratch on the LSF data.
2. ``--transfer``    → fine-tunes from the pre-trained ASL weights.

Both runs save their training history (``loss`` / ``accuracy`` per
epoch) and final validation accuracy to JSON. This script then
produces:

* ``models/transfer_comparison.png`` — side-by-side learning curves.
* ``models/transfer_comparison.md``  — Markdown table summarising the
  difference (epochs to convergence, final accuracy, total epochs run).

The original ``models/gesture_model.pth`` is **preserved** — both runs
write to dedicated checkpoints in ``models/_compare/``.

Usage
─────
    py -3.11 compare_transfer.py
    py -3.11 compare_transfer.py --skip-scratch     # only re-run transfer
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MODEL_DIR

OUT_DIR = Path(MODEL_DIR) / "_compare"
HIST_SCRATCH = OUT_DIR / "history_scratch.json"
HIST_TRANSFER = OUT_DIR / "history_transfer.json"
MODEL_SCRATCH = OUT_DIR / "model_scratch.pth"
MODEL_TRANSFER = OUT_DIR / "model_transfer.pth"
PLOT_PATH = Path(MODEL_DIR) / "transfer_comparison.png"
REPORT_PATH = Path(MODEL_DIR) / "transfer_comparison.md"


def _run(cmd: list[str], expected_output: Path | None = None) -> int:
    """Run a subprocess and return its exit code.

    On Windows, PyTorch+CUDA processes routinely exit with the harmless
    code ``0xC0000409`` (STATUS_STACK_BUFFER_OVERRUN) during interpreter
    teardown, even when training has fully succeeded. To be robust, if
    *expected_output* is given and the file exists after the run, the
    exit code is forced to 0 with a warning printed.

    The subprocess is forced to UTF-8 output (PYTHONIOENCODING + PYTHONUTF8)
    so that Unicode print statements (arrows, em-dash, accents) don't crash
    on PowerShell's default cp1252 encoding.
    """
    print("\n" + "=" * 70)
    print("  $ " + " ".join(cmd))
    print("=" * 70)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False, env=env)
    dt = time.perf_counter() - t0
    rc = proc.returncode
    print(f"\n  -> exit {rc} in {dt:.1f} s")

    if rc != 0 and expected_output is not None and expected_output.is_file():
        print(f"  [WARN] exit code {rc} mais {expected_output.name} a ete cree."
              " Considere comme reussi (artefact Windows PyTorch shutdown).")
        return 0
    return rc


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _epochs_to_threshold(accs: list[float], threshold: float) -> int | None:
    for i, a in enumerate(accs, start=1):
        if a >= threshold:
            return i
    return None


def _plot(scratch: dict, transfer: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Transfer learning ASL → LSF — comparaison", fontsize=13)

    # Loss
    h_s, h_t = scratch["history"], transfer["history"]
    ep_s = range(1, len(h_s["val_loss"]) + 1)
    ep_t = range(1, len(h_t["val_loss"]) + 1)

    ax1.plot(ep_s, h_s["val_loss"], label="Scratch — val", color="#ff6b2b")
    ax1.plot(ep_t, h_t["val_loss"], label="Transfer — val",
             color="#4ecdc4", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation loss")
    ax1.set_title("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(ep_s, [a * 100 for a in h_s["val_acc"]],
             label="Scratch — val", color="#ff6b2b")
    ax2.plot(ep_t, [a * 100 for a in h_t["val_acc"]],
             label="Transfer — val", color="#4ecdc4", linestyle="--")
    ax2.axhline(scratch["final_val_acc"] * 100, color="#ff6b2b",
                linestyle=":", alpha=0.5)
    ax2.axhline(transfer["final_val_acc"] * 100, color="#4ecdc4",
                linestyle=":", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot ecrit : {PLOT_PATH}")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = lambda c: "| " + " | ".join(c) + " |"
    sep = "|" + "|".join(":---:" if i > 0 else ":---"
                         for i in range(len(headers))) + "|"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _report(scratch: dict, transfer: dict) -> None:
    final_s = scratch["final_val_acc"]
    final_t = transfer["final_val_acc"]
    ep_s = scratch["epochs_run"]
    ep_t = transfer["epochs_run"]

    target = max(final_s, final_t) * 0.95  # 95 % of the best final score
    e_s_target = _epochs_to_threshold(scratch["history"]["val_acc"], target)
    e_t_target = _epochs_to_threshold(transfer["history"]["val_acc"], target)

    md = []
    md.append("# Comparaison — entraînement *from scratch* vs *transfer learning*")
    md.append("")
    md.append("Pré-entraînement source : **ASL** (`pretrain_asl.py`).  ")
    md.append("Fine-tuning cible : **LSF** (`train_model.py`).")
    md.append("")
    md.append("## Résultats finaux")
    md.append("")
    md.append(_md_table(
        ["Configuration", "Accuracy finale (val)", "Epochs exécutées",
         f"Epochs pour atteindre {target:.1%}"],
        [
            ["From scratch", f"{final_s:.2%}", str(ep_s),
             str(e_s_target) if e_s_target else "non atteint"],
            ["Transfer ASL → LSF", f"{final_t:.2%}", str(ep_t),
             str(e_t_target) if e_t_target else "non atteint"],
        ],
    ))
    md.append("")
    md.append("![Comparaison](transfer_comparison.png)")
    md.append("")
    md.append("## Lecture")
    md.append("")
    delta_acc = (final_t - final_s) * 100
    if delta_acc > 0:
        md.append(f"Le transfer learning apporte **+{delta_acc:.2f} pts**"
                  " d'accuracy finale.")
    elif delta_acc < 0:
        md.append(f"Le transfer learning **dégrade** l'accuracy finale de "
                  f"{abs(delta_acc):.2f} pts (peut-être déjà saturé).")
    else:
        md.append("Le transfer learning n'apporte aucune différence "
                  "d'accuracy finale (les deux configurations saturent).")

    if e_s_target is not None and e_t_target is not None:
        if e_t_target < e_s_target:
            md.append(
                f"Pour atteindre {target:.1%}, le transfer learning a besoin "
                f"de **{e_t_target} epochs** contre **{e_s_target}** "
                f"sans transfer (gain de {e_s_target - e_t_target} epochs)."
            )
        elif e_t_target > e_s_target:
            md.append(
                f"Le from-scratch atteint {target:.1%} en "
                f"**{e_s_target} epochs**, contre **{e_t_target}** avec "
                "transfer (overhead du fine-tuning)."
            )
        else:
            md.append(
                f"Les deux configurations atteignent {target:.1%} en "
                f"{e_s_target} epochs."
            )

    md.append("")
    md.append("---")
    md.append("")
    md.append(
        "_Reproduire :_ `py -3.11 compare_transfer.py`. "
        "Les checkpoints détaillés sont dans `models/_compare/`."
    )
    md.append("")
    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"  Rapport ecrit : {REPORT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-scratch", action="store_true",
                        help="Ne pas reentrainer la version from-scratch")
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Ne pas reentrainer la version transfer")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if not args.skip_scratch:
        rc = _run([py, "train_model.py",
                   "--history-out", str(HIST_SCRATCH),
                   "--model-out",   str(MODEL_SCRATCH)],
                  expected_output=HIST_SCRATCH)
        if rc != 0:
            print("[ERROR] Le run from-scratch a echoue.")
            return rc

    if not args.skip_transfer:
        if not (Path(MODEL_DIR) / "pretrained_asl.pth").is_file():
            print("[WARN] models/pretrained_asl.pth introuvable.")
            print("       Lancez d'abord:  py -3.11 pretrain_asl.py")
            return 1
        rc = _run([py, "train_model.py", "--transfer",
                   "--history-out", str(HIST_TRANSFER),
                   "--model-out",   str(MODEL_TRANSFER)],
                  expected_output=HIST_TRANSFER)
        if rc != 0:
            print("[ERROR] Le run transfer a echoue.")
            return rc

    scratch = _load(HIST_SCRATCH)
    transfer = _load(HIST_TRANSFER)
    if scratch is None or transfer is None:
        print("[ERROR] Historique manquant — impossible de produire le rapport.")
        print(f"        scratch:  {HIST_SCRATCH} {'OK' if scratch else 'MANQUANT'}")
        print(f"        transfer: {HIST_TRANSFER} {'OK' if transfer else 'MANQUANT'}")
        return 1

    _plot(scratch, transfer)
    _report(scratch, transfer)
    print("\n" + "=" * 70)
    print(f"  Plot   : {PLOT_PATH}")
    print(f"  Rapport: {REPORT_PATH}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
