"""Evaluation script for the Hand Talk Translator gesture model.

Produces a Markdown report (default: ``models/evaluation_report.md``)
containing :

* Environment / dataset / model meta-data
* Overall accuracy, macro-F1, weighted-F1
* Per-class precision / recall / F1 / support table
* Top confusions (the K most-confused class pairs)
* Inference latency benchmark (per-sample CPU and CUDA, plus batched throughput)

The held-out test set is reconstructed with the exact same
``random_state=42`` / ``test_size=0.2`` stratified split used by
``train_model.py``, so the numbers reported here are directly
comparable to those printed at the end of training.

Usage
─────
    py -3.11 evaluate_model.py
    py -3.11 evaluate_model.py --output docs/evaluation.md --latency-iter 500
    py -3.11 evaluate_model.py --device cpu     # force CPU
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    FEATURES_PER_FRAME,
    LABELS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    TRAINING,
)
from model import GestureLSTM


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (identical to train_model._load_dataset, raw / no augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_raw():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"[ERROR] DATA_DIR introuvable: {DATA_DIR}\n"
            "        Lancez d'abord:  py -3.11 collect_data.py"
        )

    candidate_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    labels: list[str] = []
    for d in candidate_dirs:
        folder = os.path.join(DATA_DIR, d)
        if any(f.endswith(".npy") for f in os.listdir(folder)):
            labels.append(d)

    if not labels:
        raise SystemExit("[ERROR] Aucun fichier .npy trouve dans data/")

    X, y = [], []
    per_class_counts: dict[str, int] = {}
    for idx, label in enumerate(labels):
        folder = os.path.join(DATA_DIR, label)
        loaded = 0
        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".npy")):
            arr = np.load(os.path.join(folder, fname))
            if arr.ndim != 2 or arr.shape[1] != FEATURES_PER_FRAME:
                continue
            if arr.shape[0] == SEQUENCE_LENGTH:
                X.append(arr)
                y.append(idx)
                loaded += 1
            else:
                # sliding-window resample, exactly like train_model
                for start in range(0, len(arr) - SEQUENCE_LENGTH + 1):
                    X.append(arr[start: start + SEQUENCE_LENGTH])
                    y.append(idx)
                    loaded += 1
        per_class_counts[label] = loaded

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        labels,
        per_class_counts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(num_classes: int, device: torch.device):
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(
            f"[ERROR] Modele introuvable: {MODEL_PATH}\n"
            "        Lancez d'abord:  py -3.11 train_model.py"
        )
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    n_features = ckpt.get("num_features", FEATURES_PER_FRAME)
    n_classes_ckpt = ckpt.get("num_classes", num_classes)
    if n_classes_ckpt != num_classes:
        print(
            f"[WARN] Le checkpoint a {n_classes_ckpt} classes, "
            f"mais le dataset en contient {num_classes}. "
            "Les labels ne correspondent peut-etre pas."
        )
    model = GestureLSTM(num_features=n_features, num_classes=n_classes_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, n_features, n_classes_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Inference + latency
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(model, X: np.ndarray, device: torch.device,
                batch_size: int = 64) -> np.ndarray:
    """Return integer predictions for every sample of *X*."""
    preds = []
    tensor = torch.from_numpy(X)
    for i in range(0, len(X), batch_size):
        chunk = tensor[i: i + batch_size].to(device)
        logits = model(chunk)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


@torch.no_grad()
def benchmark_latency(model, X: np.ndarray, device: torch.device,
                      *, n_warmup: int = 20, n_iter: int = 200) -> dict:
    """Per-sample (batch=1) inference latency, in milliseconds."""
    if len(X) == 0:
        return {}
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(X), size=n_iter + n_warmup)
    samples = torch.from_numpy(X[indices]).to(device)

    is_cuda = device.type == "cuda"
    times_ms: list[float] = []
    for i in range(n_warmup + n_iter):
        x = samples[i: i + 1]
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            times_ms.append(dt)

    times_ms.sort()
    return {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "p95": times_ms[int(0.95 * (len(times_ms) - 1))],
        "p99": times_ms[int(0.99 * (len(times_ms) - 1))],
        "min": times_ms[0],
        "max": times_ms[-1],
        "n_iter": n_iter,
    }


@torch.no_grad()
def benchmark_throughput(model, X: np.ndarray, device: torch.device,
                         *, batch_size: int = 64, n_iter: int = 50) -> dict:
    """Batched throughput in samples/second."""
    if len(X) == 0:
        return {}
    is_cuda = device.type == "cuda"
    sample = torch.from_numpy(X[:batch_size]).to(device)

    for _ in range(5):  # warmup
        _ = model(sample)
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(sample)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_samples = n_iter * batch_size
    return {
        "batch_size": batch_size,
        "samples_per_sec": total_samples / elapsed,
        "ms_per_batch": (elapsed / n_iter) * 1000.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = lambda cells: "| " + " | ".join(cells) + " |"
    sep = "|" + "|".join(
        ":---:" if i > 0 else ":---" for i in range(len(headers))
    ) + "|"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def render_report(
    *,
    labels: list[str],
    per_class_counts: dict[str, int],
    n_train: int,
    n_test: int,
    n_total_aug_free: int,
    overall_acc: float,
    macro_f1: float,
    weighted_f1: float,
    per_class: list[dict],
    cm: np.ndarray,
    top_confusions: list[tuple[str, str, int]],
    latency_cpu: dict | None,
    latency_gpu: dict | None,
    throughput_cpu: dict | None,
    throughput_gpu: dict | None,
    n_params: int,
    device_name: str,
    eval_seconds: float,
) -> str:
    lines: list[str] = []
    lines.append("# Hand Talk Translator — Rapport d'évaluation")
    lines.append("")
    lines.append(f"_Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")

    # ── Environment / setup ─────────────────────────────────────────────────
    lines.append("## 1. Environnement")
    lines.append("")
    lines.append(_md_table(
        ["Paramètre", "Valeur"],
        [
            ["Python", platform.python_version()],
            ["PyTorch", torch.__version__],
            ["Plateforme", f"{platform.system()} {platform.release()}"],
            ["Device d'évaluation", device_name],
            ["Nombre de paramètres du modèle", f"{n_params:,}"],
            ["Longueur de séquence", str(SEQUENCE_LENGTH)],
            ["Features par frame", str(FEATURES_PER_FRAME)],
            ["Durée totale de l'évaluation", f"{eval_seconds:.2f} s"],
        ],
    ))
    lines.append("")

    # ── Dataset ─────────────────────────────────────────────────────────────
    lines.append("## 2. Données")
    lines.append("")
    lines.append(_md_table(
        ["Item", "Valeur"],
        [
            ["Nombre de classes", str(len(labels))],
            ["Échantillons réels (sans augmentation)", str(n_total_aug_free)],
            ["Train / Test (split stratifié, seed 42)", f"{n_train} / {n_test}"],
            ["Test split ratio", f"{TRAINING['validation_split']:.0%}"],
        ],
    ))
    lines.append("")
    lines.append("**Échantillons par classe (données brutes) :**")
    lines.append("")
    rows = [[lbl, str(per_class_counts.get(lbl, 0))] for lbl in labels]
    lines.append(_md_table(["Classe", "Échantillons"], rows))
    lines.append("")

    # ── Overall metrics ─────────────────────────────────────────────────────
    lines.append("## 3. Métriques globales (test set)")
    lines.append("")
    lines.append(_md_table(
        ["Métrique", "Valeur"],
        [
            ["Accuracy", f"{overall_acc:.2%}"],
            ["F1 macro", f"{macro_f1:.4f}"],
            ["F1 weighted", f"{weighted_f1:.4f}"],
        ],
    ))
    lines.append("")

    # ── Per-class metrics ───────────────────────────────────────────────────
    lines.append("## 4. Métriques par classe")
    lines.append("")
    rows = []
    for d in per_class:
        rows.append([
            d["label"],
            f"{d['precision']:.3f}",
            f"{d['recall']:.3f}",
            f"{d['f1']:.3f}",
            str(int(d["support"])),
        ])
    lines.append(_md_table(
        ["Classe", "Précision", "Rappel", "F1", "Support"],
        rows,
    ))
    lines.append("")

    # ── Confusions ─────────────────────────────────────────────────────────
    if top_confusions:
        lines.append("## 5. Confusions principales")
        lines.append("")
        lines.append(
            "Paires `(vrai → prédit)` les plus fréquentes parmi les erreurs "
            "du test set."
        )
        lines.append("")
        rows = [[t, p, str(n)] for t, p, n in top_confusions]
        lines.append(_md_table(
            ["Classe réelle", "Classe prédite", "Occurrences"],
            rows,
        ))
        lines.append("")
    else:
        lines.append("## 5. Confusions principales")
        lines.append("")
        lines.append("_Aucune confusion sur le test set._")
        lines.append("")

    # ── Latency ────────────────────────────────────────────────────────────
    lines.append("## 6. Latence d'inférence (batch = 1, temps réel)")
    lines.append("")
    rows = []
    if latency_cpu:
        rows.append([
            "CPU",
            f"{latency_cpu['mean']:.3f}",
            f"{latency_cpu['median']:.3f}",
            f"{latency_cpu['p95']:.3f}",
            f"{latency_cpu['p99']:.3f}",
            f"{latency_cpu['min']:.3f}",
            f"{latency_cpu['max']:.3f}",
        ])
    if latency_gpu:
        rows.append([
            "CUDA",
            f"{latency_gpu['mean']:.3f}",
            f"{latency_gpu['median']:.3f}",
            f"{latency_gpu['p95']:.3f}",
            f"{latency_gpu['p99']:.3f}",
            f"{latency_gpu['min']:.3f}",
            f"{latency_gpu['max']:.3f}",
        ])
    if rows:
        lines.append(_md_table(
            ["Device", "Moy. (ms)", "Médiane (ms)",
             "p95 (ms)", "p99 (ms)", "min (ms)", "max (ms)"],
            rows,
        ))
        lines.append("")
        lines.append(
            "_Mesuré sur "
            f"{(latency_cpu or latency_gpu)['n_iter']} itérations "
            "après warm-up. Une latence < 33 ms est compatible avec une "
            "boucle vidéo à 30 FPS._"
        )
        lines.append("")

    # ── Throughput ─────────────────────────────────────────────────────────
    lines.append("## 7. Débit en mode batch")
    lines.append("")
    rows = []
    if throughput_cpu:
        rows.append([
            "CPU",
            str(throughput_cpu["batch_size"]),
            f"{throughput_cpu['samples_per_sec']:.0f}",
            f"{throughput_cpu['ms_per_batch']:.2f}",
        ])
    if throughput_gpu:
        rows.append([
            "CUDA",
            str(throughput_gpu["batch_size"]),
            f"{throughput_gpu['samples_per_sec']:.0f}",
            f"{throughput_gpu['ms_per_batch']:.2f}",
        ])
    if rows:
        lines.append(_md_table(
            ["Device", "Batch", "Échantillons / s", "Temps / batch (ms)"],
            rows,
        ))
        lines.append("")

    # ── Footer ─────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "_Reproduire :_ `py -3.11 evaluate_model.py`. "
        "Le test set est reconstruit avec le seed 42 du fichier "
        "`train_model.py`, donc les chiffres ci-dessus sont identiques "
        "à ceux affichés à la fin de l'entraînement."
    )
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", "-o", default=os.path.join(MODEL_DIR, "evaluation_report.md"),
        help="Fichier Markdown de sortie",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Device principal pour l'évaluation des métriques",
    )
    parser.add_argument(
        "--latency-iter", type=int, default=200,
        help="Nombre d'itérations pour le benchmark de latence (défaut: 200)",
    )
    parser.add_argument(
        "--top-confusions", type=int, default=10,
        help="Nombre de paires confondues à afficher (défaut: 10)",
    )
    parser.add_argument(
        "--no-gpu-bench", action="store_true",
        help="Ne pas lancer le benchmark CUDA même si disponible",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ── Device sélection ────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA demande mais indisponible -> bascule sur CPU.")
        device = torch.device("cpu")

    if device.type == "cuda":
        device_name = f"CUDA — {torch.cuda.get_device_name(0)}"
    else:
        device_name = f"CPU — {platform.processor() or 'unknown'}"

    print("=" * 60)
    print("  EVALUATION — Hand Talk Translator")
    print("=" * 60)
    print(f"  Device      : {device_name}")
    print(f"  Sortie      : {args.output}")
    print()

    # ── Dataset (raw, no augmentation) ──────────────────────────────────────
    print("[1/5] Chargement du dataset...")
    X, y, labels, per_class_counts = load_dataset_raw()
    print(f"      {len(X)} echantillons, {len(labels)} classes")

    # Reproduce the train/test split used during training (seed 42)
    if len(X) >= 5:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=TRAINING["validation_split"],
            random_state=42,
            stratify=y,
        )
    else:
        X_tr, X_te, y_tr, y_te = X[:0], X, y[:0], y
    print(f"      Train / Test : {len(X_tr)} / {len(X_te)}")

    # ── Modèle ──────────────────────────────────────────────────────────────
    print("[2/5] Chargement du modele...")
    model, _, n_classes_ckpt = load_model(len(labels), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      {n_params:,} parametres")

    if len(labels) != n_classes_ckpt:
        print(
            "[WARN] Le nombre de classes du checkpoint et du dataset diffère "
            "— le rapport sera produit mais les labels peuvent etre decales."
        )

    # ── Métriques ───────────────────────────────────────────────────────────
    print("[3/5] Calcul des metriques sur le test set...")
    y_pred = predict_all(model, X_te, device)

    overall_acc = float(accuracy_score(y_te, y_pred))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(
        y_te, y_pred, average="weighted", zero_division=0,
    ))

    all_idx = list(range(len(labels)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_te, y_pred, labels=all_idx, zero_division=0,
    )
    per_class = [
        {
            "label": labels[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i in range(len(labels))
    ]

    cm = confusion_matrix(y_te, y_pred, labels=all_idx)

    # Top confusions (off-diagonal entries, sorted desc)
    confs: list[tuple[str, str, int]] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confs.append((labels[i], labels[j], int(cm[i, j])))
    confs.sort(key=lambda t: t[2], reverse=True)
    top_confusions = confs[: args.top_confusions]

    print(f"      Accuracy : {overall_acc:.2%}")
    print(f"      F1 macro : {macro_f1:.4f}")

    # ── Latence ─────────────────────────────────────────────────────────────
    print(f"[4/5] Benchmark de latence ({args.latency_iter} iterations)...")
    bench_pool = X_te if len(X_te) >= 32 else X

    latency_cpu = latency_gpu = None
    throughput_cpu = throughput_gpu = None

    cpu_device = torch.device("cpu")
    cpu_model = model.to(cpu_device)
    latency_cpu = benchmark_latency(cpu_model, bench_pool, cpu_device,
                                    n_iter=args.latency_iter)
    throughput_cpu = benchmark_throughput(cpu_model, bench_pool, cpu_device)
    print(f"      CPU  : {latency_cpu['mean']:.3f} ms / sample (mean), "
          f"{throughput_cpu['samples_per_sec']:.0f} samples/s en batch")

    if torch.cuda.is_available() and not args.no_gpu_bench:
        cuda_device = torch.device("cuda")
        cuda_model = model.to(cuda_device)
        latency_gpu = benchmark_latency(cuda_model, bench_pool, cuda_device,
                                        n_iter=args.latency_iter)
        throughput_gpu = benchmark_throughput(cuda_model, bench_pool,
                                              cuda_device)
        print(f"      CUDA : {latency_gpu['mean']:.3f} ms / sample (mean), "
              f"{throughput_gpu['samples_per_sec']:.0f} samples/s en batch")

    # ── Rendu Markdown ──────────────────────────────────────────────────────
    print("[5/5] Ecriture du rapport Markdown...")
    eval_seconds = time.perf_counter() - t_start
    md = render_report(
        labels=labels,
        per_class_counts=per_class_counts,
        n_train=len(X_tr),
        n_test=len(X_te),
        n_total_aug_free=len(X),
        overall_acc=overall_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        cm=cm,
        top_confusions=top_confusions,
        latency_cpu=latency_cpu,
        latency_gpu=latency_gpu,
        throughput_cpu=throughput_cpu,
        throughput_gpu=throughput_gpu,
        n_params=n_params,
        device_name=device_name,
        eval_seconds=eval_seconds,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print()
    print("=" * 60)
    print(f"  Rapport ecrit : {out_path}")
    print(f"  Taille        : {out_path.stat().st_size / 1024:.1f} KB")
    print(f"  Duree         : {eval_seconds:.2f} s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
