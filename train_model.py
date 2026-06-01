"""Training script for the gesture-recognition LSTM model.

Pipeline
────────
1. Load collected .npy samples from  data/<sign>/
2. Augment with noise, mirror, time-warp, scale  (×8 by default)
3. Stratified train / test split
4. Build & train the LSTM  (PyTorch)
5. Evaluate — overall accuracy, per-class report, confusion matrix
6. Save  →  models/gesture_model.pth  +  models/labels.json
7. Plot  →  models/training_curves.png  +  models/confusion_matrix.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DATA_DIR,
    EXCLUDED_VISION_LABELS,
    FEATURES_PER_FRAME,
    INCLUDE_VELOCITY,
    LABELS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    TRAINING,
)
from model import GestureLSTM

_HAND_SIZE = 63  # 21 landmarks × 3 coords


# ── data helpers ─────────────────────────────────────────────────────────────

def _load_dataset():
    """Return ``(X, y, labels)`` or ``(None, None, None)``."""
    if not os.path.isdir(DATA_DIR):
        print("Aucune donnee trouvee.  Executez d'abord :  python collect_data.py")
        return None, None, None

    all_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    # Skip empty class directories (e.g. leftover ASL folder)
    labels = []
    for d in all_dirs:
        folder = os.path.join(DATA_DIR, d)
        n_npy = sum(1 for f in os.listdir(folder) if f.endswith(".npy"))
        if d in EXCLUDED_VISION_LABELS:
            print(f"  IGNORE {d!r} — classe exclue (config)")
        elif n_npy > 0:
            labels.append(d)
        else:
            print(f"  IGNORE {d!r} — dossier vide (0 fichiers .npy)")

    if not labels:
        print("Aucune donnee trouvee.  Executez d'abord :  python collect_data.py")
        return None, None, None

    X, y = [], []
    print(f"\nClasses trouvees ({len(labels)}) :\n")
    for idx, label in enumerate(labels):
        label_dir = os.path.join(DATA_DIR, label)
        files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
        loaded = 0
        for fname in files:
            arr = np.load(os.path.join(label_dir, fname))
            if arr.ndim != 2 or arr.shape[1] != FEATURES_PER_FRAME:
                continue
            if arr.shape[0] == SEQUENCE_LENGTH:
                X.append(arr)
                y.append(idx)
                loaded += 1
            else:
                for start in range(0, len(arr) - SEQUENCE_LENGTH + 1):
                    X.append(arr[start : start + SEQUENCE_LENGTH])
                    y.append(idx)
                    loaded += 1
        print(f"  {label:20s}  {len(files):3d} fichiers -> {loaded} echantillons")

    return np.array(X, dtype=np.float32), np.array(y), labels


# ── augmentation helpers ──────────────────────────────────────────────────────

def _noise(seq: np.ndarray, std: float = 0.02) -> np.ndarray:
    return (seq + np.random.normal(0, std, seq.shape)).astype(np.float32)


def _mirror(seq: np.ndarray) -> np.ndarray:
    """Swap left/right hands and negate x-coordinates (horizontal mirror)."""
    seq = seq.copy()
    n_feats = seq.shape[1]
    blocks = 4 if INCLUDE_VELOCITY else 2  # [L_pos, R_pos, L_vel, R_vel] or [L_pos, R_pos]

    # Swap L ↔ R blocks pairwise (pos then vel)
    for pair_start in range(0, blocks, 2):
        left  = seq[:, pair_start * _HAND_SIZE:(pair_start + 1) * _HAND_SIZE].copy()
        right = seq[:, (pair_start + 1) * _HAND_SIZE:(pair_start + 2) * _HAND_SIZE].copy()
        seq[:, pair_start * _HAND_SIZE:(pair_start + 1) * _HAND_SIZE] = right
        seq[:, (pair_start + 1) * _HAND_SIZE:(pair_start + 2) * _HAND_SIZE] = left

    # Negate x-coordinates in every block (index 0, 3, 6, ... within each 63-block)
    for b in range(blocks):
        base = b * _HAND_SIZE
        for lm in range(21):
            seq[:, base + lm * 3] = -seq[:, base + lm * 3]

    return seq


def _time_warp(seq: np.ndarray) -> np.ndarray:
    """Randomly stretch or compress the temporal axis then resample to original length."""
    n = len(seq)
    factor = np.random.uniform(0.8, 1.2)
    warped_len = max(5, int(n * factor))
    x_old = np.linspace(0, 1, n)
    x_warped = np.linspace(0, 1, warped_len)
    warped = interp1d(x_old, seq, axis=0, kind="linear")(x_warped)
    return interp1d(np.linspace(0, 1, warped_len), warped, axis=0, kind="linear")(x_old).astype(np.float32)


def _scale(seq: np.ndarray) -> np.ndarray:
    """Uniform random scale of landmark positions (±15 %)."""
    return (seq * np.random.uniform(0.85, 1.15)).astype(np.float32)


def _augment(X: np.ndarray, y: np.ndarray, factor: int = 3, noise_std: float = 0.02):
    """Return augmented dataset with up to ×15 multiplier.

    Augmentation pool (applied in order until factor-1 variants are generated):
      noise (×3 levels) | mirror | time_warp (×2) | scale (×2) | combos (×5)
    """
    parts_X, parts_y = [X], [y]

    aug_fns = [
        lambda s: _noise(s, noise_std),
        lambda s: _noise(s, noise_std * 1.5),
        lambda s: _noise(s, noise_std * 0.5),
        _mirror,
        _time_warp,
        lambda s: _time_warp(_noise(s, noise_std * 0.5)),
        _scale,
        lambda s: _scale(_noise(s, noise_std)),
        lambda s: _noise(_mirror(s), noise_std),
        lambda s: _noise(_mirror(s), noise_std * 1.5),
        lambda s: _time_warp(_mirror(s)),
        lambda s: _scale(_mirror(s)),
        lambda s: _noise(_scale(_mirror(s)), noise_std),
        lambda s: _time_warp(_scale(s)),
    ]

    used = aug_fns[:max(0, factor - 1)]
    for fn in used:
        augmented = np.array([fn(seq) for seq in X], dtype=np.float32)
        parts_X.append(augmented)
        parts_y.append(y)

    return np.concatenate(parts_X), np.concatenate(parts_y)


# ── training utilities ───────────────────────────────────────────────────────

class _EarlyStopping:
    def __init__(self, patience: int = 25):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0
        self.best_loss = float("inf")
        self.best_state: dict | None = None

    def step(self, acc: float, loss: float, model: nn.Module) -> bool:
        improved = acc > self.best_acc or (acc == self.best_acc and loss < self.best_loss)
        if improved:
            self.best_acc = acc
            self.best_loss = loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        preds = out.argmax(1)
        total_loss += loss.item() * len(yb)
        correct += (preds == yb).sum().item()
        total += len(yb)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_true)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraine le modele LSTM de gestes")
    parser.add_argument(
        "--transfer", action="store_true",
        help="Charger les poids pre-entraines ASL (models/pretrained_asl.pth) "
             "avant de fine-tuner sur les donnees LSF"
    )
    parser.add_argument(
        "--history-out", type=str, default=None,
        help="Si fourni, sauvegarde l'historique d'entrainement (loss/acc par "
             "epoch + accuracy finale) au format JSON dans le fichier indique"
    )
    parser.add_argument(
        "--model-out", type=str, default=None,
        help="Si fourni, ecrit le checkpoint a ce chemin au lieu du chemin par defaut"
    )
    args = parser.parse_args()

    print("=" * 58)
    print("  ENTRAINEMENT - Hand Talk Translator")
    if args.transfer:
        print("  [TRANSFER LEARNING: ASL -> LSF]")
    print("=" * 58)
    print()

    X, y, labels = _load_dataset()
    if X is None:
        return

    print(f"\nDataset brut : {len(X)} echantillons, {len(labels)} classes")

    aug = TRAINING["augmentation_factor"]
    X, y = _augment(X, y, factor=aug, noise_std=TRAINING["noise_std"])
    print(f"Apres augmentation (x{aug}) : {len(X)} echantillons")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=TRAINING["validation_split"],
        random_state=42,
        stratify=y,
    )
    print(f"Train : {len(X_tr)}   Test : {len(X_te)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    bs = TRAINING["batch_size"]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=bs, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long)),
        batch_size=bs,
    )

    model = GestureLSTM(
        num_features=FEATURES_PER_FRAME,
        num_classes=len(labels),
    ).to(device)

    # ── Transfer learning: load ASL pre-trained weights ───────────────
    if args.transfer:
        pretrained_path = os.path.join(MODEL_DIR, "pretrained_asl.pth")
        if not os.path.isfile(pretrained_path):
            print("ATTENTION: models/pretrained_asl.pth introuvable.")
            print("  Lancez d'abord: py -3.11 pretrain_asl.py")
            print("  Entrainement classique sans transfer learning...\n")
            _use_grouped_lr = False
        else:
            state = torch.load(pretrained_path, map_location=device, weights_only=True)
            # Load all weights except the final classifier layer (different n_classes)
            own_state = model.state_dict()
            transferred = {k: v for k, v in state.items()
                           if k in own_state and "fc2" not in k
                           and v.shape == own_state[k].shape}
            own_state.update(transferred)
            model.load_state_dict(own_state)
            n_transferred = len(transferred)
            print(f"  Transfer learning: {n_transferred} couches chargees depuis ASL.")
            print(f"  La couche fc2 (classifieur) est reinitialise pour {len(labels)} classes LSF.\n")

            # Lower LR for pre-trained layers, keep full LR for the new classifier
            pretrained_params = [p for n, p in model.named_parameters() if "fc2" not in n]
            classifier_params  = [p for n, p in model.named_parameters() if "fc2"     in n]
            optimizer_groups = [
                {"params": pretrained_params, "lr": TRAINING["learning_rate"] * 0.1},
                {"params": classifier_params, "lr": TRAINING["learning_rate"]},
            ]
            _use_grouped_lr = True
    else:
        _use_grouped_lr = False

    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParametres : {n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    if _use_grouped_lr:
        optimizer = torch.optim.Adam(optimizer_groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )
    es = _EarlyStopping(patience=TRAINING["early_stopping_patience"])

    print("Entrainement en cours...\n")
    epochs = TRAINING["epochs"]

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = _evaluate(model, test_loader, criterion, device)
        scheduler.step(v_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={t_loss:.4f}  train_acc={t_acc:.1%}  "
            f"val_loss={v_loss:.4f}  val_acc={v_acc:.1%}  lr={lr:.1e}"
        )
        if es.step(v_acc, v_loss, model):
            print(f"\n  Early stopping (patience={es.patience})")
            break

    es.restore(model)

    # final evaluation
    v_loss, v_acc, y_pred, y_true = _evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 58}")
    print(f"  Precision globale : {v_acc:.1%}")
    print(f"  Loss              : {v_loss:.4f}")
    print(f"{'=' * 58}\n")

    print("Rapport par classe :\n")
    all_idx = list(range(len(labels)))
    print(classification_report(y_true, y_pred,
                                labels=all_idx, target_names=labels,
                                zero_division=0))

    print("Matrice de confusion :\n")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l[:5]:>6s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:<{col_w}}{''.join(f'{v:6d}' for v in row)}")

    # ── save model + labels ───────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_out_path = args.model_out or MODEL_PATH
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": FEATURES_PER_FRAME,
            "num_classes": len(labels),
        },
        model_out_path,
    )
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False, indent=2)

    # ── save training history (optional) ──────────────────────────────────
    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)) or ".",
                    exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as fh:
            json.dump({
                "history": history,
                "final_val_acc": float(v_acc),
                "final_val_loss": float(v_loss),
                "transfer": args.transfer,
                "epochs_run": len(history["train_loss"]),
                "labels": labels,
            }, fh, ensure_ascii=False, indent=2)
        print(f"  Historique sauvegarde : {args.history_out}")

    # ── plot training curves ──────────────────────────────────────────────
    curves_path = os.path.join(MODEL_DIR, "training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Courbes d'entraînement — Hand Talk Translator", fontsize=13)

    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], label="Train loss", color="#ff6b2b")
    ax1.plot(ep, history["val_loss"],   label="Val loss",   color="#4ecdc4", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, [a * 100 for a in history["train_acc"]], label="Train acc", color="#ff6b2b")
    ax2.plot(ep, [a * 100 for a in history["val_acc"]],   label="Val acc",   color="#4ecdc4", linestyle="--")
    ax2.axhline(v_acc * 100, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Courbes sauvegardees : {curves_path}")

    # ── plot confusion matrix ─────────────────────────────────────────────
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — précision globale : {v_acc:.1%}")

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matrice sauvegardee  : {cm_path}")

    print(f"\n  Modele sauvegarde  : {MODEL_PATH}")
    print(f"  Labels sauvegardes : {LABELS_PATH}")
    print(f"\n  Prochaine etape : python main.py")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()

