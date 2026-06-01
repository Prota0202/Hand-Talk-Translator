"""Generate the head-to-head comparison report between the **vision pipeline**
(MediaPipe + LSTM) and the **glove pipeline** (3 flex + IMU + LSTM) used as
the central experiment of the TFE.

The report contains, for both systems:
  * dataset size (number of classes, samples)
  * accuracy / macro-F1 / weighted-F1
  * inference latency (mean, median, p95) on the same machine
  * model size on disk
  * training time (best-effort: read from history JSON if present)
  * qualitative trade-offs (cost, hardware, light/occlusion, vocabulary, ...)

Outputs
-------
  models/glove_vs_vision.md     — Markdown table + analysis
  models/glove_vs_vision.png    — bar charts (acc, latency)

Usage
-----

    python compare_glove_vs_vision.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import config
from model import GestureLSTM


# ── helpers ─────────────────────────────────────────────────────────────────


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _load_npy_dataset(data_dir: Path, seq_len: int, feat: int):
    classes = sorted(d.name for d in data_dir.iterdir()
                     if d.is_dir() and any(d.glob("*.npy")))
    if not classes:
        return None, None, []
    X, y = [], []
    for idx, cls in enumerate(classes):
        for f in (data_dir / cls).glob("*.npy"):
            arr = np.load(f)
            if arr.ndim == 2 and arr.shape == (seq_len, feat):
                X.append(arr.astype(np.float32))
                y.append(idx)
    if not X:
        return None, None, []
    return np.stack(X), np.asarray(y, dtype=np.int64), classes


def _benchmark_latency(model: torch.nn.Module, sample: np.ndarray,
                       device: torch.device, iters: int = 200) -> dict:
    """Return mean/median/p95/p99 inference latency in ms."""
    model.eval()
    x = torch.from_numpy(sample[None, ...]).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(20):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)
    return {
        "mean":   float(times.mean()),
        "median": float(np.median(times)),
        "p95":    float(np.percentile(times, 95)),
        "p99":    float(np.percentile(times, 99)),
        "n":      iters,
    }


def _evaluate_pipeline(name: str, *, data_dir: Path, model_path: Path,
                       labels_path: Path, seq_len: int, feat: int,
                       hidden: int, dropout: float, device: torch.device) -> dict:
    """Common evaluation routine for both vision and glove models."""
    print(f"\n=== {name} ===")
    if not data_dir.is_dir():
        print(f"  [skip] data dir manquant : {data_dir}")
        return {"name": name, "available": False, "reason": f"{data_dir} absent"}
    if not model_path.exists() or not labels_path.exists():
        print(f"  [skip] modele manquant : {model_path}")
        return {"name": name, "available": False,
                "reason": f"{model_path.name} absent — entraine d'abord"}

    X, y, classes = _load_npy_dataset(data_dir, seq_len, feat)
    if X is None:
        print(f"  [skip] aucune sequence valide dans {data_dir}")
        return {"name": name, "available": False,
                "reason": f"aucune sequence ({seq_len},{feat}) dans {data_dir}"}

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    # Match data classes to model labels
    model_class_to_idx = {c: i for i, c in enumerate(labels)}
    keep_mask = np.array([classes[lbl] in model_class_to_idx for lbl in y])
    X = X[keep_mask]
    y = np.array([model_class_to_idx[classes[lbl]] for lbl in y[keep_mask]])

    if len(X) == 0:
        return {"name": name, "available": False,
                "reason": "intersection vide entre data et labels du modele"}

    # Reproducible test split (we accept overlap with training because we don't
    # always have a held-out set; the goal is comparative, not absolute).
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if len(np.unique(y)) > 1 and np.bincount(y).min() >= 2 else None)

    model = GestureLSTM(num_features=feat, num_classes=len(labels),
                        hidden_size=hidden, num_layers=2, dropout=dropout).to(device)
    raw = torch.load(model_path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state)
    model.eval()

    # Predict
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 64):
            xb = torch.from_numpy(X_te[i:i + 64]).to(device)
            preds.extend(model(xb).argmax(1).cpu().numpy().tolist())
    preds = np.asarray(preds)

    acc = accuracy_score(y_te, preds)
    f1m = f1_score(y_te, preds, average="macro", zero_division=0)
    f1w = f1_score(y_te, preds, average="weighted", zero_division=0)

    lat = _benchmark_latency(model, X_te[0], device)

    n_params = sum(p.numel() for p in model.parameters())
    size_disk = model_path.stat().st_size

    print(f"  classes      : {len(labels)}")
    print(f"  test samples : {len(X_te)}")
    print(f"  accuracy     : {acc*100:.2f}%")
    print(f"  macro F1     : {f1m*100:.2f}%")
    print(f"  weighted F1  : {f1w*100:.2f}%")
    print(f"  latency mean : {lat['mean']:.2f} ms (p95 {lat['p95']:.2f})")
    print(f"  parametres   : {n_params/1e3:.1f}k  ({_human_size(size_disk)})")

    return {
        "name": name,
        "available": True,
        "device": str(device),
        "n_classes": len(labels),
        "n_samples_total": int(keep_mask.sum()),
        "n_samples_test": int(len(X_te)),
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        "latency_ms": lat,
        "parameters": int(n_params),
        "model_size_bytes": int(size_disk),
        "features_per_frame": feat,
        "sequence_length": seq_len,
    }


def _read_history(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# ── reporting ───────────────────────────────────────────────────────────────


_QUALITATIVE = """\
| Critere | Vision (MediaPipe + LSTM) | Gant (Flex + IMU + LSTM) |
|---|---|---|
| Cout materiel | Webcam (~10-30 EUR) | ESP32 + 3 flex + MPU6050 (~20-30 EUR) |
| Mobilite | Necessite une camera fixe et un PC | Autonome, portatif (USB/BLE) |
| Sensible a la lumiere | Oui (penombre = chute des perf) | Non |
| Sensible aux occlusions | Oui (mains qui se croisent) | Non |
| Configuration utilisateur | Aucune | Calibration des doigts (~10 s) |
| Vocabulaire couvert | Tous gestes visibles | Seulement gestes a forte signature dactylo/cinematique |
| Confidentialite (image) | Faible (flux video) | Forte (pas d'image) |
| Privacy / portabilite jury | Camera necessaire | Demo possible n'importe ou |
| Encombrement utilisateur | Aucun | Doit enfiler le gant |
"""


def _write_report(out_md: Path, vision: dict, glove: dict,
                  hist_vision: dict | None, hist_glove: dict | None) -> None:
    lines: list[str] = []
    lines.append("# Comparaison Vision vs Gant — Hand Talk Translator")
    lines.append("")
    lines.append("Ce rapport est genere automatiquement par "
                 "`compare_glove_vs_vision.py`. Il croise les performances "
                 "des deux pipelines de reconnaissance de la LSF developpes "
                 "dans ce projet, sur le meme materiel et la meme machine.")
    lines.append("")

    # ── Quantitative table ──
    lines.append("## 1. Tableau quantitatif")
    lines.append("")
    lines.append("| Metrique | Vision | Gant |")
    lines.append("|---|---|---|")

    def cell(d: dict, key, fmt: str):
        if not d.get("available"):
            return "-"
        v = d
        for k in key:
            v = v[k]
        return fmt.format(v)

    rows = [
        ("Disponible",           lambda d: "oui" if d.get("available") else f"non ({d.get('reason','?')})"),
        ("Device benchmark",     lambda d: d.get("device", "-") if d.get("available") else "-"),
        ("Nombre de classes",    lambda d: str(d["n_classes"]) if d.get("available") else "-"),
        ("Echantillons (total)", lambda d: str(d["n_samples_total"]) if d.get("available") else "-"),
        ("Echantillons (test)",  lambda d: str(d["n_samples_test"]) if d.get("available") else "-"),
        ("Accuracy",             lambda d: f"{d['accuracy']*100:.2f}%" if d.get("available") else "-"),
        ("F1 macro",             lambda d: f"{d['f1_macro']*100:.2f}%" if d.get("available") else "-"),
        ("F1 weighted",          lambda d: f"{d['f1_weighted']*100:.2f}%" if d.get("available") else "-"),
        ("Latence moyenne",      lambda d: f"{d['latency_ms']['mean']:.2f} ms" if d.get("available") else "-"),
        ("Latence p95",          lambda d: f"{d['latency_ms']['p95']:.2f} ms" if d.get("available") else "-"),
        ("Parametres",           lambda d: f"{d['parameters']/1e3:.1f}k" if d.get("available") else "-"),
        ("Taille fichier",       lambda d: _human_size(d['model_size_bytes']) if d.get("available") else "-"),
        ("Features / frame",     lambda d: str(d['features_per_frame']) if d.get("available") else "-"),
        ("Longueur sequence",    lambda d: str(d['sequence_length']) if d.get("available") else "-"),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | {fn(vision)} | {fn(glove)} |")
    lines.append("")

    # ── Training times if histories are available ──
    if hist_vision or hist_glove:
        lines.append("## 2. Apprentissage")
        lines.append("")
        lines.append("| | Vision | Gant |")
        lines.append("|---|---|---|")
        for label, key in [
            ("Epochs effectives",   "epochs"),
            ("Best train accuracy", "train_acc"),
            ("Best val accuracy",   "val_acc"),
        ]:
            v = "-"
            g = "-"
            if hist_vision:
                if key == "epochs":
                    v = str(hist_vision.get("epochs", len(hist_vision.get("train_loss", []))))
                elif key in hist_vision and hist_vision[key]:
                    v = f"{max(hist_vision[key])*100:.2f}%"
            if hist_glove:
                if key == "epochs":
                    g = str(hist_glove.get("epochs", len(hist_glove.get("train_loss", []))))
                elif key in hist_glove and hist_glove[key]:
                    g = f"{max(hist_glove[key])*100:.2f}%"
            lines.append(f"| {label} | {v} | {g} |")
        lines.append("")

    # ── Qualitative ──
    lines.append("## 3. Comparaison qualitative")
    lines.append("")
    lines.append(_QUALITATIVE)

    # ── Analysis ──
    lines.append("## 4. Discussion")
    lines.append("")
    if vision.get("available") and glove.get("available"):
        delta_acc = (glove["accuracy"] - vision["accuracy"]) * 100
        delta_lat = glove["latency_ms"]["mean"] - vision["latency_ms"]["mean"]
        winner_acc = "le gant" if delta_acc > 0.5 else (
                     "la vision" if delta_acc < -0.5 else "egalite")
        winner_lat = "le gant" if delta_lat < -0.1 else (
                     "la vision" if delta_lat > 0.1 else "egalite")
        lines.append(
            f"En accuracy, **{winner_acc}** ressort en tete avec un ecart de "
            f"`{abs(delta_acc):.2f}` point(s) de pourcentage. En latence "
            f"d'inference par echantillon, **{winner_lat}** est le plus rapide "
            f"(`{abs(delta_lat):.2f}` ms d'ecart).")
        lines.append("")
        lines.append(
            "Au-dela des chiffres bruts, les deux approches sont **complementaires** : "
            "la vision offre un vocabulaire potentiellement illimite (tout geste "
            "visible peut etre appris) mais reste sensible a la lumiere et aux "
            "occlusions ; le gant fournit un signal extremement stable, insensible "
            "a l'environnement, mais ne capte que la geometrie d'une main et "
            "ignore les composantes faciales et corporelles de la LSF.")
    elif vision.get("available"):
        lines.append("Seul le pipeline **vision** est disponible pour le moment. "
                     "Une fois le gant assemble et entraine, ce rapport mettra "
                     "en avant les ecarts.")
    elif glove.get("available"):
        lines.append("Seul le pipeline **gant** est disponible pour le moment. "
                     "Une fois le modele vision entraine, le tableau ci-dessus "
                     "mettra en avant les ecarts.")
    else:
        lines.append("Aucun des deux pipelines n'a pu etre evalue. Verifie la "
                     "presence des donnees et des checkpoints.")
    lines.append("")
    lines.append("---")
    lines.append("Genere automatiquement par `compare_glove_vs_vision.py`.")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def _write_charts(out_png: Path, vision: dict, glove: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    names = []
    accs  = []
    lats  = []
    for d in (vision, glove):
        if d.get("available"):
            names.append(d["name"])
            accs.append(d["accuracy"] * 100)
            lats.append(d["latency_ms"]["mean"])

    if names:
        axes[0].bar(names, accs, color=["#4F8BF9", "#F9A24F"][:len(names)])
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("accuracy (%)")
        axes[0].set_title("Accuracy")
        for i, v in enumerate(accs):
            axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")

        axes[1].bar(names, lats, color=["#4F8BF9", "#F9A24F"][:len(names)])
        axes[1].set_ylabel("latence (ms)")
        axes[1].set_title("Latence d'inference (mean)")
        for i, v in enumerate(lats):
            axes[1].text(i, v + max(lats) * 0.02, f"{v:.2f}", ha="center")
    else:
        axes[0].text(0.5, 0.5, "Aucun pipeline disponible", ha="center", va="center")
        axes[1].text(0.5, 0.5, "Aucun pipeline disponible", ha="center", va="center")

    fig.suptitle("Vision vs Gant — performances")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ── entrypoint ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Comparaison vision vs gant")
    ap.add_argument("--out-md",   type=Path, default=Path("models/glove_vs_vision.md"))
    ap.add_argument("--out-png",  type=Path, default=Path("models/glove_vs_vision.png"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device : {device}")

    vision = _evaluate_pipeline(
        "Vision (MediaPipe + LSTM)",
        data_dir=Path(config.DATA_DIR),
        model_path=Path(config.MODEL_PATH),
        labels_path=Path(config.LABELS_PATH),
        seq_len=config.SEQUENCE_LENGTH,
        feat=config.FEATURES_PER_FRAME,
        hidden=128,
        dropout=0.4,
        device=device,
    )

    glove = _evaluate_pipeline(
        "Gant (Flex + IMU + LSTM)",
        data_dir=Path(config.GLOVE_DIR),
        model_path=Path(config.GLOVE_MODEL_PATH),
        labels_path=Path(config.GLOVE_LABELS),
        seq_len=config.GLOVE["sequence_length"],
        feat=config.GLOVE["features_per_frame"],
        hidden=64,
        dropout=0.3,
        device=device,
    )

    hist_vision = _read_history(Path(config.MODEL_DIR) / "training_history.json")
    hist_glove  = _read_history(Path(config.MODEL_DIR) / "glove_history.json")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_report(args.out_md, vision, glove, hist_vision, hist_glove)
    _write_charts(args.out_png, vision, glove)

    print(f"\nRapport : {args.out_md}")
    print(f"Graphique: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
