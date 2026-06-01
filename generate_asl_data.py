"""Synthetic ASL landmark data generator for transfer learning pre-training.

Generates anatomically plausible hand landmark sequences for 20 ASL letter
configurations.  Each letter is defined by the extension state of the 5
fingers; the generator then builds 21 MediaPipe-compatible (x, y, z)
landmarks around that configuration, adds Gaussian noise, and writes
the result in the same .npy format used by collect_data.py.

Output: data/ASL/<LETTER>/<sample_XXX.npy>  (same structure as LSF data)

Usage:
    py -3.11 generate_asl_data.py            # 120 samples / letter (default)
    py -3.11 generate_asl_data.py --samples 200
"""

import argparse
import os
import sys

import numpy as np

# ── Finger extension table ────────────────────────────────────────────────────
# [thumb, index, middle, ring, pinky]  values in [0, 1] (0=curled, 1=extended)
ASL_CONFIGS: dict[str, list[float]] = {
    "A": [0.6, 0.0, 0.0, 0.0, 0.0],   # fist, thumb resting on side
    "B": [0.0, 1.0, 1.0, 1.0, 1.0],   # four fingers up, thumb tucked
    "C": [0.5, 0.5, 0.5, 0.5, 0.5],   # curved C shape
    "D": [0.7, 1.0, 0.3, 0.3, 0.3],   # index up, thumb touches middle
    "E": [0.2, 0.2, 0.2, 0.2, 0.2],   # all fingers bent
    "F": [0.8, 0.3, 1.0, 1.0, 1.0],   # thumb-index touching, rest extended
    "I": [0.0, 0.0, 0.0, 0.0, 1.0],   # pinky up (ILY base)
    "K": [0.8, 1.0, 1.0, 0.0, 0.0],   # index + middle + thumb up
    "L": [1.0, 1.0, 0.0, 0.0, 0.0],   # L: thumb out, index up
    "M": [0.2, 0.2, 0.2, 0.2, 0.0],   # three fingers folded over thumb
    "N": [0.2, 0.2, 0.2, 0.0, 0.0],   # two fingers folded over thumb
    "O": [0.4, 0.4, 0.4, 0.4, 0.4],   # all curved to make O
    "R": [0.0, 1.0, 1.0, 0.0, 0.0],   # index + middle crossed
    "S": [0.1, 0.1, 0.1, 0.1, 0.1],   # closed fist, thumb over
    "U": [0.0, 1.0, 1.0, 0.0, 0.0],   # index + middle up together
    "V": [0.0, 1.0, 1.0, 0.0, 0.0],   # peace / V sign
    "W": [0.0, 1.0, 1.0, 1.0, 0.0],   # three fingers spread
    "X": [0.0, 0.6, 0.0, 0.0, 0.0],   # index bent hook
    "Y": [1.0, 0.0, 0.0, 0.0, 1.0],   # thumb + pinky out
    "P": [0.8, 1.0, 1.0, 0.0, 0.0],   # K rotated down
}

# MediaPipe landmark indices per finger
FINGER_TIPS  = [4,  8,  12, 16, 20]  # thumb, index, middle, ring, pinky
FINGER_PIPS  = [3,  6,  10, 14, 18]
FINGER_MCPS  = [2,  5,   9, 13, 17]

SEQ_LEN = 30   # must match config.py SEQUENCE_LENGTH


# ── Landmark generation ───────────────────────────────────────────────────────

def _canonical_hand() -> np.ndarray:
    """Returns a neutral open-palm 21×3 landmark array (normalised [0,1])."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    lm[0] = [0.50, 0.90, 0.00]
    # Thumb: 1-4
    thumb_base = np.array([0.38, 0.80, 0.02])
    for i, tip_frac in enumerate([0.0, 0.33, 0.66, 1.0]):
        tip_pos = np.array([0.20, 0.65, 0.05])
        lm[1 + i] = thumb_base + tip_frac * (tip_pos - thumb_base)
    # Four fingers: 5-20
    x_positions = [0.42, 0.52, 0.62, 0.72]   # MCP x per finger
    for fi, xb in enumerate(x_positions):
        mcp = np.array([xb, 0.80, 0.00])
        tip = np.array([xb, 0.30, 0.05])
        for ji, frac in enumerate([0.0, 0.33, 0.66, 1.0]):
            lm[5 + fi * 4 + ji] = mcp + frac * (tip - mcp)
    return lm


def _apply_extension(lm: np.ndarray, config: list[float]) -> np.ndarray:
    """Curl or extend each finger according to config values."""
    lm = lm.copy()
    fingers = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
               (13, 14, 15, 16), (17, 18, 19, 20)]
    for fi, joints in enumerate(fingers):
        ext = config[fi]
        mcp_idx = joints[0]
        tip_idx = joints[-1]
        mcp = lm[mcp_idx].copy()
        tip_extended = lm[tip_idx].copy()
        # Curled tip: same x as mcp, lower y
        tip_curled = mcp + np.array([0.0, 0.12, 0.06], dtype=np.float32)
        tip_actual = ext * tip_extended + (1 - ext) * tip_curled
        # Interpolate intermediate joints
        for ji, idx in enumerate(joints):
            frac = (ji + 1) / len(joints)
            lm[idx] = mcp + frac * (tip_actual - mcp)
    return lm


def _to_feature_vector(lm: np.ndarray) -> np.ndarray:
    """Flatten 21×3 landmarks (one hand) to 63-d, mirrored to fill 126-d slot."""
    flat = lm.flatten()            # 63
    mirrored = flat.copy()
    mirrored[0::3] = 1.0 - flat[0::3]   # flip x for 2nd hand
    return np.concatenate([flat, mirrored])   # 126


def _make_sequence(config: list[float], noise_std: float = 0.015) -> np.ndarray:
    """Generate a SEQ_LEN × 252 feature sequence for one sample."""
    base = _canonical_hand()
    posed = _apply_extension(base, config)
    frames = []
    for _ in range(SEQ_LEN):
        noisy = posed + np.random.randn(*posed.shape).astype(np.float32) * noise_std
        feat  = _to_feature_vector(noisy)   # 126 positional
        # Velocity: difference to previous frame (zero for frame 0)
        frames.append(feat)

    # Stack frames and compute velocity
    pos = np.stack(frames, axis=0)        # (SEQ_LEN, 126)
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    seq = np.concatenate([pos, vel], axis=1)   # (SEQ_LEN, 252)
    return seq.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(n_samples: int = 120) -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "data", "ASL")
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 56)
    print("  GENERATION ASL SYNTHETIQUE — Hand Talk Translator")
    print("=" * 56)
    print(f"  Lettres : {len(ASL_CONFIGS)}   |   Echantillons/lettre : {n_samples}")
    print()

    total = 0
    for letter, config in ASL_CONFIGS.items():
        out_dir = os.path.join(base_dir, letter)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n_samples):
            seq  = _make_sequence(config)
            path = os.path.join(out_dir, f"sample_{i:04d}.npy")
            np.save(path, seq)
        print(f"  {letter:>2} : {n_samples} fichiers generes -> {out_dir}")
        total += n_samples

    print()
    print(f"  Total : {total} echantillons ASL synthetiques")
    print(f"  Repertoire : {base_dir}")
    print()
    print("  Etape suivante :")
    print("    py -3.11 pretrain_asl.py")
    print("=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=120,
                        help="Nombre d'echantillons par lettre (defaut: 120)")
    args = parser.parse_args()
    generate(n_samples=args.samples)
