"""Pre-train the GestureLSTM on synthetic ASL landmark data.

Trains the full model on the 20 synthetic ASL letter classes, then saves the
weights so that train_model.py can load them for fine-tuning on LSF data.
This demonstrates transfer learning: common hand-configuration representations
learned on a larger (synthetic) ASL corpus are reused as a starting point for
the smaller, real LSF dataset.

Workflow:
    1. py -3.11 generate_asl_data.py       # generate synthetic ASL data
    2. py -3.11 pretrain_asl.py            # pre-train on ASL
    3. py -3.11 train_model.py --transfer  # fine-tune on LSF  ← uses saved weights

Output: models/pretrained_asl.pth   (full model state_dict on ASL classes)
        models/pretrained_asl_labels.json
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ASL_DIR   = os.path.join(BASE_DIR, "data", "ASL")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_PATH  = os.path.join(MODEL_DIR, "pretrained_asl.pth")
LABELS_OUT = os.path.join(MODEL_DIR, "pretrained_asl_labels.json")

# Training hyper-parameters
EPOCHS     = 80
BATCH_SIZE = 32
LR         = 1e-3
VAL_SPLIT  = 0.15
PATIENCE   = 20


def load_dataset():
    """Load .npy sequences from data/ASL/."""
    if not os.path.isdir(ASL_DIR):
        print("ERREUR: Donnees ASL introuvables.")
        print("  Lancez d'abord: py -3.11 generate_asl_data.py")
        sys.exit(1)

    labels_sorted = sorted(d for d in os.listdir(ASL_DIR)
                           if os.path.isdir(os.path.join(ASL_DIR, d)))
    if not labels_sorted:
        print("ERREUR: Aucun dossier de classe dans data/ASL/.")
        sys.exit(1)

    X, y = [], []
    for idx, lbl in enumerate(labels_sorted):
        folder = os.path.join(ASL_DIR, lbl)
        files  = [f for f in os.listdir(folder) if f.endswith(".npy")]
        for f in files:
            seq = np.load(os.path.join(folder, f))
            X.append(seq)
            y.append(idx)
        print(f"  {lbl:>3} ({idx}): {len(files)} echantillons")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), labels_sorted


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 58)
    print("  PRE-ENTRAINEMENT ASL — Hand Talk Translator")
    print("=" * 58)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    print("\nChargement du dataset ASL...")
    X, y, labels = load_dataset()
    print(f"\n  {len(labels)} classes | {len(X)} echantillons")
    print(f"  Shape sequence : {X.shape[1:]}  (frames × features)")

    # Train / test split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42, stratify=y
    )

    # Tensors
    X_tr_t  = torch.tensor(X_tr,  device=device)
    y_tr_t  = torch.tensor(y_tr,  device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)

    # Model (same architecture as gesture_model, different output size)
    sys.path.insert(0, BASE_DIR)
    from model import GestureLSTM

    _, seq_len, n_feat = X_tr_t.shape
    n_classes = len(labels)

    model = GestureLSTM(
        num_features=n_feat,
        num_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )

    print(f"\nArchitecture : {model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres   : {n_params:,}")

    # Training loop
    print("\nEntrainement en cours...")
    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Mini-batch shuffle
        perm = torch.randperm(len(X_tr_t), device=device)
        train_loss, train_correct = 0.0, 0

        for start in range(0, len(X_tr_t), BATCH_SIZE):
            idx  = perm[start: start + BATCH_SIZE]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(xb)
            train_correct += (logits.argmax(1) == yb).sum().item()

        train_loss /= len(X_tr_t)
        train_acc   = train_correct / len(X_tr_t)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_acc    = (val_logits.argmax(1) == y_val_t).float().mean().item()

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1%}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1%}  "
              f"lr={lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUT_PATH)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                break

    # Save labels
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n  Modele pre-entraine sauvegarde : {OUT_PATH}")
    print(f"  Labels ASL sauvegardes        : {LABELS_OUT}")
    print()
    print("  Etape suivante : fine-tuning LSF")
    print("    py -3.11 train_model.py --transfer")
    print("=" * 58)


if __name__ == "__main__":
    main()
