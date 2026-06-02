"""Generate models/pipeline_architecture.png for the TFE report."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join("models", "pipeline_architecture.png")

BOXES = [
    ("Webcam\n1280×720", "#E8F4FD"),
    ("MediaPipe\nHands", "#D6EAF8"),
    ("Normalisation\n+ vélocités\n(252 feat.)", "#AED6F1"),
    ("Segment\n→ resample 30", "#85C1E9"),
    ("LSTM\n16 classes", "#5DADE2"),
    ("Lissage +\nseuil conf.", "#3498DB"),
    ("lsf_translator\n(grammaire)", "#2E86C1"),
    ("TTS Edge\n(touche Espace)", "#1B4F72"),
]

W, H = 1.35, 0.72
GAP = 0.22
Y = 0.42
XS = [0.04 + i * (W + GAP) for i in range(len(BOXES))]


def main() -> None:
    fig_w = max(14.0, XS[-1] + W + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, 1.35)
    ax.axis("off")
    ax.set_title(
        "Architecture du pipeline vision — Hand Talk Translator",
        fontsize=13,
        pad=12,
        color="#1a1a1a",
    )

    for (label, color), x in zip(BOXES, XS):
        box = FancyBboxPatch(
            (x, Y),
            W,
            H,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.2,
            edgecolor="#2C3E50",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            x + W / 2,
            Y + H / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#1a1a1a",
        )

    for i in range(len(XS) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (XS[i] + W + 0.02, Y + H / 2),
                (XS[i + 1] - 0.02, Y + H / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#566573",
            )
        )

    ax.text(
        5.1,
        0.08,
        "Débit cible : 30 FPS  ·  Goulot : MediaPipe (~12 ms)  ·  LSTM : < 1 ms",
        ha="center",
        fontsize=8.5,
        color="#566573",
        style="italic",
    )

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Écrit : {OUT}")


if __name__ == "__main__":
    main()
