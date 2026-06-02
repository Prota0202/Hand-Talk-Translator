"""Évalue le modèle vision sur plusieurs signeurs externes (leave-one-signer-out).

Génère un rapport Markdown combiné + un extrait LaTeX pour le rapport TFE.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from config import (
    DATA_DIR,
    DATA_SIGNER2_DIR,
    DATA_SIGNER3_DIR,
    DATA_SIGNER4_DIR,
    DATA_SIGNER5_DIR,
    MODEL_DIR,
    MODEL_PATH,
)
from evaluate_cross_signer import _md_table, render_report, top_confusion_pairs
from evaluate_model import load_model, predict_all
from vision_dataset import load_sequences_from_dir, load_training_labels


DEFAULT_SIGNERS: list[tuple[str, str]] = [
    ("Signeur 2 (petit frère)", DATA_SIGNER2_DIR),
    ("Signeur 3 (adulte)", DATA_SIGNER3_DIR),
    ("Signeur 4 (adulte)", DATA_SIGNER4_DIR),
    ("Signeur 5 (ami, LSF partielle)", DATA_SIGNER5_DIR),
]


def evaluate_one(
    *,
    signer_name: str,
    data_dir: str,
    labels: list[str],
    device: torch.device,
    model,
) -> dict:
    X, y, _counts, unknown_folders = load_sequences_from_dir(data_dir, labels)
    y_pred = predict_all(model, X, device)

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

    confusions = top_confusion_pairs(y, y_pred, labels, k=10)

    return {
        "signer_name": signer_name,
        "data_dir": data_dir,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "n_samples": len(X),
        "per_class_rows": per_class_rows,
        "top_confusions": confusions,
        "unknown_folders": unknown_folders,
    }


def render_combined_report(
    *,
    train_data_dir: str,
    results: list[dict],
) -> str:
    lines = [
        "# Évaluation cross-signeur (vision) — synthèse",
        "",
        f"- **Date** : {datetime.now():%Y-%m-%d %H:%M}",
        f"- **Modèle** : `{MODEL_PATH}` (entraîné sur `{train_data_dir}`)",
        f"- **Protocole** : modèle figé (signeur 1), test sur signeurs non vus",
        "",
        "## Résultats globaux",
        "",
        _md_table(
            ["Signeur", "Séquences", "Accuracy", "F1 macro", "F1 weighted"],
            [
                [
                    r["signer_name"],
                    str(r["n_samples"]),
                    f"{r['accuracy'] * 100:.2f} %",
                    f"{r['f1_macro']:.4f}",
                    f"{r['f1_weighted']:.4f}",
                ]
                for r in results
            ],
        ),
        "",
    ]

    for r in results:
        lines.extend([
            f"## {r['signer_name']}",
            "",
            f"Dossier : `{r['data_dir']}` — {r['n_samples']} séquences",
            "",
            _md_table(
                ["Classe", "Préc.", "Rappel", "F1", "Support"],
                r["per_class_rows"],
            ),
            "",
            "### Principales confusions",
            "",
            _md_table(
                ["Vrai", "Prédit", "Count"],
                [[a, b, str(c)] for a, b, c in r["top_confusions"]],
            ),
            "",
        ])
        if r["unknown_folders"]:
            lines.extend([
                f"_Dossiers ignorés :_ {', '.join(r['unknown_folders'])}",
                "",
            ])

    lines.extend([
        "---",
        "",
        "_Reproduire :_ `python evaluate_all_cross_signers.py`",
        "",
    ])
    return "\n".join(lines)


def render_latex_snippet(results: list[dict]) -> str:
    def profile(name: str) -> str:
        lower = name.lower()
        if "petit fr" in lower or "enfant" in lower:
            return "petit frère"
        if "lsf" in lower or "ami" in lower:
            return "ami, pratique partielle LSF"
        if "adulte" in lower:
            return "adulte"
        return "---"

    rows = []
    for r in results:
        short = r["signer_name"].split(" (")[0]
        rows.append(
            f"        {short} & {profile(r['signer_name'])} & {r['n_samples']} "
            f"& {r['accuracy'] * 100:.1f}\\,\\% \\\\"
        )

    return "\n".join([
        "% Coller dans rapport_tfe.tex (section cross-signeur)",
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\small",
        "    \\begin{tabular}{lccc}",
        "        \\toprule",
        "        \\textbf{Signeur} & \\textbf{Profil} & \\textbf{Séq.} & \\textbf{Accuracy} \\\\",
        "        \\midrule",
        *rows,
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\caption[Cross-signeur sans réentraînement.]"
        "{Évaluation du modèle entraîné sur le signeur~1, appliqué tel quel "
        "à d'autres personnes (5 échantillons par gloss, sans réentraînement). "
        "Le signeur~5 est le seul testeur avec une pratique partielle de la LSF "
        "(voir section cross-signeur du rapport).}",
        "    \\label{tab:cross-signer}",
        "\\end{table}",
    ])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data-dir",
        default=DATA_DIR,
        help="Dossier d'entraînement du modèle (info rapport)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(MODEL_DIR, "cross_signer_report.md"),
    )
    parser.add_argument(
        "--latex-out",
        default=os.path.join(MODEL_DIR, "cross_signer_latex.tex"),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Ignore les signeurs dont le dossier est absent ou vide.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  EVALUATION CROSS-SIGNEUR — synthese multi-signeurs")
    print("=" * 60)
    print(f"  Modele : {MODEL_PATH}")
    print(f"  Device : {device}")
    print()

    labels = load_training_labels()
    model, _, _ = load_model(len(labels), device)

    results: list[dict] = []
    for signer_name, data_dir in DEFAULT_SIGNERS:
        data_dir = os.path.abspath(data_dir)
        print(f"--- {signer_name} ({data_dir})")
        if not os.path.isdir(data_dir):
            msg = f"Dossier absent: {data_dir}"
            if args.skip_missing:
                print(f"      [SKIP] {msg}")
                continue
            raise FileNotFoundError(msg)

        try:
            t0 = time.perf_counter()
            r = evaluate_one(
                signer_name=signer_name,
                data_dir=data_dir,
                labels=labels,
                device=device,
                model=model,
            )
            elapsed = time.perf_counter() - t0
            print(f"      {r['n_samples']} seq. | accuracy {r['accuracy'] * 100:.2f} % "
                  f"| {elapsed:.2f} s")
            results.append(r)

            detail_path = os.path.join(
                MODEL_DIR,
                f"cross_signer_{signer_name.split()[1].lower()}.md",
            )
            detail = render_report(
                signer_name=signer_name,
                data_dir=data_dir,
                train_data_dir=os.path.abspath(args.train_data_dir),
                labels=labels,
                accuracy=r["accuracy"],
                f1_macro=r["f1_macro"],
                f1_weighted=r["f1_weighted"],
                n_samples=r["n_samples"],
                per_class_rows=r["per_class_rows"],
                top_confusions=r["top_confusions"],
                unknown_folders=r["unknown_folders"],
            )
            with open(detail_path, "w", encoding="utf-8") as fh:
                fh.write(detail)
            print(f"      Rapport detail : {detail_path}")
        except ValueError as exc:
            if args.skip_missing:
                print(f"      [SKIP] {exc}")
                continue
            raise

    if not results:
        print("Aucun signeur evalue. Collectez d'abord data_signer2/ … data_signer5/.")
        return 1

    combined = render_combined_report(
        train_data_dir=os.path.abspath(args.train_data_dir),
        results=results,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(combined)

    latex = render_latex_snippet(results)
    with open(args.latex_out, "w", encoding="utf-8") as fh:
        fh.write(latex)

    print()
    print(f"  Rapport combine : {args.output}")
    print(f"  Snippet LaTeX   : {args.latex_out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
