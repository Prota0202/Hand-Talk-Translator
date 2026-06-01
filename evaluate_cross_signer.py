"""Évalue le modèle vision entraîné sur le signeur 1, sur les données d'un autre signeur.

Usage typique (Option B — leave-one-signer-out) :

1. Collecter le signeur 2 sans toucher à ``data/`` ::
       py -3.11 collect_data.py --data-dir data_signer2

2. Garder le modèle actuel (entraîné sur vos signes), puis évaluer ::
       py -3.11 evaluate_cross_signer.py --data-dir data_signer2 --signer-name "Signeur 2"

Le script charge ``models/gesture_model.pth`` + ``models/labels.json`` et
rapporte l'accuracy / F1 sur les séquences du dossier externe.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import DATA_DIR, DATA_SIGNER2_DIR, FEATURES_PER_FRAME, MODEL_DIR, MODEL_PATH, SEQUENCE_LENGTH
from evaluate_model import load_model, predict_all
from vision_dataset import load_sequences_from_dir, load_training_labels


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = lambda cells: "| " + " | ".join(cells) + " |"
    sep = "|" + "|".join(":---:" if i > 0 else ":---" for i in range(len(headers))) + "|"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def render_report(
    *,
    signer_name: str,
    data_dir: str,
    train_data_dir: str,
    labels: list[str],
    accuracy: float,
    f1_macro: float,
    f1_weighted: float,
    n_samples: int,
    per_class_rows: list[list[str]],
    top_confusions: list[tuple[str, str, int]],
    unknown_folders: list[str],
) -> str:
    lines = [
        "# Évaluation cross-signeur (vision)",
        "",
        f"- **Date** : {datetime.now():%Y-%m-%d %H:%M}",
        f"- **Signeur évalué** : {signer_name}",
        f"- **Données test** : `{data_dir}`",
        f"- **Modèle** : `{MODEL_PATH}` (entraîné sur `{train_data_dir}`)",
        f"- **Séquences** : {n_samples}",
        "",
        "## Résultats globaux",
        "",
        _md_table(
            ["Métrique", "Valeur"],
            [
                ["Accuracy", f"{accuracy * 100:.2f} %"],
                ["F1 macro", f"{f1_macro:.4f}"],
                ["F1 weighted", f"{f1_weighted:.4f}"],
            ],
        ),
        "",
        "## Détail par classe",
        "",
        _md_table(
            ["Classe", "Préc.", "Rappel", "F1", "Support"],
            per_class_rows,
        ),
        "",
    ]

    if top_confusions:
        lines.extend([
            "## Principales confusions",
            "",
            _md_table(
                ["Vrai", "Prédit", "Count"],
                [[a, b, str(c)] for a, b, c in top_confusions],
            ),
            "",
        ])

    if unknown_folders:
        lines.extend([
            "## Dossiers ignorés (hors vocabulaire du modèle)",
            "",
            ", ".join(f"`{name}`" for name in unknown_folders),
            "",
        ])

    lines.extend([
        "---",
        "",
        "_Reproduire :_ "
        f"`py -3.11 evaluate_cross_signer.py --data-dir {data_dir}`",
        "",
    ])
    return "\n".join(lines)


def top_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    *,
    k: int = 10,
) -> list[tuple[str, str, int]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=DATA_SIGNER2_DIR,
        help="Dossier des séquences du signeur externe (défaut: data_signer2)",
    )
    parser.add_argument(
        "--train-data-dir",
        default=DATA_DIR,
        help="Dossier sur lequel le modèle a été entraîné (info rapport)",
    )
    parser.add_argument(
        "--signer-name",
        default="Signeur 2",
        help="Nom affiché dans le rapport",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(MODEL_DIR, "cross_signer_report.md"),
        help="Fichier Markdown de sortie",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
    )
    parser.add_argument(
        "--top-confusions", type=int, default=10,
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    train_data_dir = os.path.abspath(args.train_data_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  EVALUATION CROSS-SIGNEUR — Hand Talk Translator")
    print("=" * 60)
    print(f"  Signeur test  : {args.signer_name}")
    print(f"  Donnees test  : {data_dir}")
    print(f"  Modele        : {MODEL_PATH}")
    print(f"  Device        : {device}")
    print()

    labels = load_training_labels()
    print(f"[1/3] Chargement des sequences ({len(labels)} classes du modele)...")
    X, y, per_class_counts, unknown_folders = load_sequences_from_dir(data_dir, labels)
    print(f"      {len(X)} sequences chargees")
    if unknown_folders:
        print(f"      [WARN] Dossiers ignores (pas dans labels.json): "
              f"{', '.join(unknown_folders)}")

    print("[2/3] Chargement du modele...")
    model, _, _ = load_model(len(labels), device)

    print("[3/3] Inference...")
    t0 = time.perf_counter()
    y_pred = predict_all(model, X, device)
    elapsed = time.perf_counter() - t0

    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)

    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=list(range(len(labels))), zero_division=0,
    )
    per_class_rows = []
    for idx, label in enumerate(labels):
        if support[idx] == 0:
            continue
        per_class_rows.append([
            label,
            f"{prec[idx]:.3f}",
            f"{rec[idx]:.3f}",
            f"{f1[idx]:.3f}",
            str(int(support[idx])),
        ])

    confusions = top_confusion_pairs(
        y, y_pred, labels, k=args.top_confusions,
    )

    report = render_report(
        signer_name=args.signer_name,
        data_dir=data_dir,
        train_data_dir=train_data_dir,
        labels=labels,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        n_samples=len(X),
        per_class_rows=per_class_rows,
        top_confusions=confusions,
        unknown_folders=unknown_folders,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report)

    print()
    print(f"  Accuracy        : {accuracy * 100:.2f} %")
    print(f"  F1 macro        : {f1_macro:.4f}")
    print(f"  F1 weighted     : {f1_weighted:.4f}")
    print(f"  Sequences       : {len(X)}")
    print(f"  Temps inference : {elapsed:.2f} s")
    print(f"  Rapport         : {args.output}")
    print()
    print("  (Rappel: le modele n'a PAS ete reentraine sur ce signeur.)")
    print("=" * 60)

    # Résumé lisible pour copier dans le rapport TFE
    print()
    print(classification_report(
        y, y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
