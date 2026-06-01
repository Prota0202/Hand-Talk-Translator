"""Feature extraction and normalisation from MediaPipe hand landmarks.

Each hand is represented by 21 3-D landmarks.  We make the features
position- and scale-invariant by centering on the wrist and dividing
by the maximum distance from the wrist.
"""

import numpy as np

from config import INCLUDE_VELOCITY, NUM_HANDS, NUM_LANDMARKS


class FeatureExtractor:
    """Converts MediaPipe landmarks into a flat feature vector.

    Supports two hands and optional velocity features for dynamic gestures.
    """

    def __init__(self):
        self._prev_points = None

    def reset(self):
        self._prev_points = None

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        points = points.copy()
        points -= points[0]
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points /= max_dist
        return points

    @staticmethod
    def _get_handedness_label(result, idx):
        if not hasattr(result, "handedness") or not result.handedness:
            return None
        h = result.handedness[idx]
        # Try multiple schema variants for safety
        if hasattr(h, "classification") and h.classification:
            return h.classification[0].label
        if hasattr(h, "categories") and h.categories:
            return h.categories[0].category_name
        if isinstance(h, (list, tuple)) and h:
            cand = h[0]
            return getattr(cand, "category_name", None) or getattr(cand, "label", None)
        return None

    def extract_from_result(self, result):
        """Return (features, motion_energy).

        features: (FEATURES_PER_FRAME,) float32
        motion_energy: mean absolute velocity (x,y) across hands
        """
        hands = {"Left": None, "Right": None}
        if result and result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                label = self._get_handedness_label(result, i)
                if label in hands and hands[label] is None:
                    hands[label] = hand
                else:
                    # fallback to any free slot
                    for key in hands:
                        if hands[key] is None:
                            hands[key] = hand
                            break

        # Build points array: (NUM_HANDS, 21, 3)
        points_list = []
        for key in ("Left", "Right"):
            hand = hands[key]
            if hand is None:
                points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            else:
                points = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                points = self._normalize(points)
            points_list.append(points)

        points_arr = np.stack(points_list, axis=0)  # (2, 21, 3)

        # Velocity features
        if self._prev_points is None:
            vel = np.zeros_like(points_arr)
        else:
            vel = points_arr - self._prev_points
        self._prev_points = points_arr

        motion_energy = float(np.mean(np.abs(vel[:, :, :2])))

        if INCLUDE_VELOCITY:
            features = np.concatenate([points_arr.flatten(), vel.flatten()]).astype(np.float32)
        else:
            features = points_arr.flatten().astype(np.float32)

        return features, motion_energy

    @staticmethod
    def augment(features: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
        """Return a noisy copy of *features* for data augmentation."""
        noise = np.random.normal(0, noise_std, features.shape).astype(np.float32)
        return features + noise
