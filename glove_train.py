"""Train the LSTM classifier on glove samples.

Reads ``data_glove/<SIGN>/*.npy`` produced by ``glove_collect.py``,
augments each sequence with Gaussian noise + small time-warps, and
trains a compact LSTM (same :class:`model.GestureLSTM` architecture
as the vision model — only with 11 input features instead of 252).

Outputs
-------
  models/glove_model.pth          state-dict of the best epoch
  models/glove_labels.json        ordered list of class names
  models/glove_curves.png         loss/accuracy curves
  models/glove_history.json       per-epoch metrics (for the comparison report)

Usage
-----

    python glove_train.py
    python glove_train.py --epochs 100 --no-augment
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
from model import GestureLSTM


# ── Data ────────────────────────────────────────────────────────────────────


def _load_dataset(data_dir: Path):
    if not data_dir.is_dir():
        print(f"Aucun dossier {data_dir} — lance d'abord python glove_collect.py")
        return None, None, None

    classes = sorted(d.name for d in data_dir.iterdir()
                     if d.is_dir() and any(d.glob("*.npy")))
    if not classes:
        print(f"{data_dir} vide.")
        return None, None, None

    seq_len = config.GLOVE["sequence_length"]
    feat = config.GLOVE["features_per_frame"]

    X, y = [], []
    print(f"\nClasses ({len(classes)}) :")
    for label_idx, label in enumerate(classes):
        files = sorted((data_dir / label).glob("*.npy"))
        kept = 0
        for f in files:
            arr = np.load(f)
            if arr.ndim != 2 or arr.shape != (seq_len, feat):
                continue
            X.append(arr.astype(np.float32))
            y.append(label_idx)
            kept += 1
        print(f"  {label:>14s}  {kept:>3d} echantillons")

    if not X:
        print("Aucune sequence valide.")
        return None, None, None

    return np.stack(X), np.asarray(y, dtype=np.int64), classes


def _augment(X: np.ndarray, y: np.ndarray, factor: int, noise_std: float):
    """Multiply the dataset by gaussian noise + tiny time-warp."""
    if factor <= 1:
        return X, y
    seq_len = X.shape[1]
    rng = np.random.default_rng(42)
    aug_X = [X.copy()]
    aug_y = [y.copy()]
    for _ in range(factor - 1):
        noisy = X + rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        # tiny time-warp: drop or duplicate one random frame
        out = np.empty_like(noisy)
        for i in range(noisy.shape[0]):
            shift = rng.integers(-1, 2)  # -1, 0, 1
            if shift == 0:
                out[i] = noisy[i]
            else:
                rolled = np.roll(noisy[i], shift, axis=0)
                if shift > 0:
                    rolled[:shift] = noisy[i, 0]
                else:
                    rolled[shift:] = noisy[i, -1]
                out[i] = rolled
        aug_X.append(out)
        aug_y.append(y.copy())
    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


# ── Training ────────────────────────────────────────────────────────────────


def train(args) -> int:
    data_dir = Path(args.data)
    X, y, labels = _load_dataset(data_dir)
    if X is None:
        return 1

    X_aug, y_aug = _augment(
        X, y,
        factor=1 if args.no_augment else config.GLOVE["augmentation"],
        noise_std=config.GLOVE["noise_std"],
    )
    print(f"\nDonnees finales : {X_aug.shape}  (apres augmentation)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device : {device}")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = GestureLSTM(
        num_features=config.GLOVE["features_per_frame"],
        num_classes=len(labels),
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [],  "val_acc":  [], "epochs": 0}

    best_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        tr_loss = ep_loss / total
        tr_acc = correct / total

        # validation
        model.eval()
        vl_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = crit(out, yb)
                vl_loss += loss.item() * xb.size(0)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += xb.size(0)
        vl_loss /= v_total
        vl_acc = v_correct / v_total

        sched.step(vl_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["epochs"] = epoch

        marker = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = "  <-- best"
        else:
            epochs_without_improvement += 1

        print(f"epoch {epoch:>3d}  "
              f"loss {tr_loss:.3f}/{vl_loss:.3f}  "
              f"acc {tr_acc*100:5.1f}%/{vl_acc*100:5.1f}%{marker}")

        if epochs_without_improvement >= config.GLOVE["early_stop"]:
            print(f"Early stop apres {epoch} epochs (pas d'amelioration depuis "
                  f"{epochs_without_improvement}).")
            break

    # ── Save best checkpoint ──────────────────────────────────────────────
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    torch.save(best_state, args.model_out)
    Path(args.labels_out).write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Final report ──────────────────────────────────────────────────────
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            preds.extend(model(xb).argmax(1).cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())
    print("\n" + classification_report(
        trues, preds, target_names=labels, zero_division=0))

    # ── Curves + history JSON ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle("Glove model — training curves")
    fig.tight_layout()
    fig.savefig(args.curves_out, dpi=130)
    plt.close(fig)

    Path(args.history_out).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nModele sauvegarde dans {args.model_out}")
    print(f"Labels        : {args.labels_out}")
    print(f"Courbes       : {args.curves_out}")
    print(f"Historique    : {args.history_out}")
    print(f"\nBest val acc : {best_acc*100:.2f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrainement du modele gant")
    ap.add_argument("--data",        default=config.GLOVE_DIR)
    ap.add_argument("--model-out",   default=config.GLOVE_MODEL_PATH)
    ap.add_argument("--labels-out",  default=config.GLOVE_LABELS)
    ap.add_argument("--curves-out",  default=os.path.join(config.MODEL_DIR, "glove_curves.png"))
    ap.add_argument("--history-out", default=os.path.join(config.MODEL_DIR, "glove_history.json"))
    ap.add_argument("--epochs",      type=int, default=config.GLOVE["epochs"])
    ap.add_argument("--batch-size",  type=int, default=config.GLOVE["batch_size"])
    ap.add_argument("--lr",          type=float, default=config.GLOVE["lr"])
    ap.add_argument("--no-augment",  action="store_true")
    ap.add_argument("--cpu",         action="store_true", help="forcer CPU")
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
