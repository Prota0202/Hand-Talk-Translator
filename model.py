"""LSTM neural network for gesture sequence classification (PyTorch).

Architecture
────────────
Input  (batch, SEQUENCE_LENGTH, 63)  ← 30 frames × 21 landmarks × 3 coords
  │
  ├─ LSTM 2 layers (hidden 128, dropout 0.3)
  ├─ BatchNorm ─► Dropout
  ├─ Dense 64  ─► ReLU ─► BatchNorm ─► Dropout
  └─ Dense N   (logits)               ← N = number of gesture classes
"""

import torch.nn as nn


class GestureLSTM(nn.Module):
    """Two-layer LSTM followed by a classification head."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout / 2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)              # (batch, seq, hidden)
        last = lstm_out[:, -1, :]               # last time-step
        out = self.drop1(self.bn1(last))
        out = self.drop2(self.relu(self.bn2(self.fc1(out))))
        return self.fc2(out)                    # raw logits
