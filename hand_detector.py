"""Hand detection using the MediaPipe Tasks HandLandmarker.

The .task model file is downloaded automatically on first run.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from config import MEDIAPIPE, MODEL_DIR

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

# 21-point skeleton used for drawing
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


def _ensure_model() -> str:
    """Download the HandLandmarker .task file if it is not already present."""
    if os.path.isfile(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Telechargement du modele MediaPipe HandLandmarker...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"  -> {_MODEL_PATH}")
    return _MODEL_PATH


class HandDetector:
    """Detects hands and extracts 21 landmarks via MediaPipe Tasks API."""

    def __init__(self):
        model_path = _ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MEDIAPIPE["max_num_hands"],
            min_hand_detection_confidence=MEDIAPIPE["min_detection_confidence"],
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=MEDIAPIPE["min_tracking_confidence"],
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._ts = 0  # monotonic timestamp fed to detect_for_video

    def detect(self, frame_rgb):
        """Run detection on an RGB numpy array.

        Returns a ``HandLandmarkerResult`` whose ``.hand_landmarks`` is a
        list of lists of ``NormalizedLandmark`` (empty list when no hand).
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts += 33  # ≈ 30 fps
        return self.landmarker.detect_for_video(image, self._ts)

    _HAND_COLORS = {
        "Left": ((0, 255, 0), (200, 255, 200)),       # green
        "Right": ((255, 165, 0), (255, 220, 150)),     # orange
    }

    def draw_landmarks(self, frame, result):
        """Draw hand skeletons — green for left, orange for right."""
        if not result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for i, hand_lms in enumerate(result.hand_landmarks):
            label = self._get_handedness(result, i)
            line_col, dot_col = self._HAND_COLORS.get(label, ((255, 255, 255), (0, 255, 0)))

            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in _CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], line_col, 2)
            for pt in pts:
                cv2.circle(frame, pt, 5, dot_col, -1)

            # Label above wrist
            wx, wy = pts[0]
            cv2.putText(frame, label, (wx - 20, wy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_col, 2)
        return frame

    @staticmethod
    def _get_handedness(result, idx) -> str:
        if not hasattr(result, "handedness") or not result.handedness:
            return "Right"
        h = result.handedness[idx]
        if hasattr(h, "categories") and h.categories:
            return h.categories[0].category_name
        if isinstance(h, (list, tuple)) and h:
            cand = h[0]
            return getattr(cand, "category_name", None) or getattr(cand, "label", "Right")
        return "Right"

    def release(self):
        self.landmarker.close()
