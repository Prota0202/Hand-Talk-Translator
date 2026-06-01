"""Animated startup splash screen for the Hand Talk Translator.

Renders an OpenCV-window-sized splash for a few seconds before the
real-time translator takes over. The splash advertises the project
title, the presenter and the school. Designed to look credible in
front of a TFE jury (clean dark gradient, accent line, fade-in/out).

The splash uses Pillow for proper Unicode rendering (accents, em-dash,
right-to-left punctuation) and falls back gracefully if Pillow is
unavailable.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

from config import PRESENTER, SPLASH

_FONT_PATHS_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_PATHS_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int, bold: bool):
    if not _PIL_AVAILABLE:
        return None
    paths = _FONT_PATHS_BOLD if bold else _FONT_PATHS_REGULAR
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _vertical_gradient(width: int, height: int,
                       top: tuple[int, int, int],
                       bottom: tuple[int, int, int]) -> np.ndarray:
    """Return an HxWx3 BGR gradient image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        b = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        r = int(top[2] + (bottom[2] - top[2]) * t)
        img[y, :] = (b, g, r)
    return img


def _build_splash(width: int, height: int) -> np.ndarray:
    """Build the static base splash image (BGR)."""
    img = _vertical_gradient(
        width, height,
        top=(28, 22, 18),       # very dark blue
        bottom=(8, 8, 12),      # near black
    )

    gold = (0, 195, 255)        # BGR
    gold_soft = (0, 145, 220)
    text_main = (240, 240, 245)
    subtext = (180, 180, 185)
    dim = (130, 130, 140)

    cx = width // 2

    # Decorative accent lines
    cv2.line(img, (cx - 220, height // 2 - 90),
             (cx + 220, height // 2 - 90), gold_soft, 2)
    cv2.line(img, (cx - 80, height - 90),
             (cx + 80, height - 90), gold, 2)

    # Brand mark (square + dot)
    cv2.rectangle(img, (cx - 32, height // 2 - 170),
                  (cx + 32, height // 2 - 110), gold, 3)
    cv2.circle(img, (cx, height // 2 - 140), 8, gold, -1)

    if not _PIL_AVAILABLE:
        # Fallback: cv2 putText (no accents)
        cv2.putText(img, PRESENTER["title"].upper(),
                    (cx - 320, height // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, text_main, 3, cv2.LINE_AA)
        cv2.putText(img, "Traducteur LSF -> Francais",
                    (cx - 220, height // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, subtext, 1, cv2.LINE_AA)
        cv2.putText(img, f"{PRESENTER['name']} - {PRESENTER['school']} - "
                    f"{PRESENTER['year']}",
                    (cx - 240, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dim, 1, cv2.LINE_AA)
        return img

    # PIL path with proper Unicode
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    f_title = _load_font(72, bold=True)
    f_sub = _load_font(26, bold=False)
    f_tag = _load_font(20, bold=True)
    f_foot = _load_font(22, bold=False)
    f_school = _load_font(34, bold=True)

    title_y = height // 2 - 70

    draw.text((cx, title_y), PRESENTER["title"],
              font=f_title, fill=(245, 245, 250), anchor="mm")
    draw.text((cx, title_y + 60), PRESENTER["subtitle"],
              font=f_sub, fill=(180, 180, 195), anchor="mm")

    draw.text((cx, height // 2 + 60), PRESENTER["tagline"].upper(),
              font=f_tag, fill=(255, 195, 0), anchor="mm")

    # Bottom block: school + name + year
    draw.text((cx, height - 130), PRESENTER["school"],
              font=f_school, fill=(255, 195, 0), anchor="mm")
    draw.text((cx, height - 60),
              f"{PRESENTER['name']}  •  {PRESENTER['year']}",
              font=f_foot, fill=(170, 170, 180), anchor="mm")

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out


def show_splash(window_name: str, width: int, height: int,
                duration: float | None = None) -> None:
    """Display the splash for *duration* seconds with a quick fade in/out.

    The window must already exist (``cv2.namedWindow``). The function
    keeps the GUI event loop alive via ``cv2.waitKey`` so the window
    actually paints.
    """
    if not SPLASH.get("enabled", True):
        return
    if duration is None:
        duration = float(SPLASH.get("duration", 2.5))

    base = _build_splash(width, height)
    fade_in = 0.4
    fade_out = 0.4
    hold = max(0.1, duration - fade_in - fade_out)

    t0 = time.perf_counter()
    last = 0.0
    while True:
        t = time.perf_counter() - t0
        if t < fade_in:
            alpha = t / fade_in
        elif t < fade_in + hold:
            alpha = 1.0
        elif t < duration:
            alpha = 1.0 - (t - fade_in - hold) / fade_out
        else:
            break

        # blend with black
        frame = (base.astype(np.float32) * alpha).astype(np.uint8)
        cv2.imshow(window_name, frame)

        # ~60 FPS render but let the user skip
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27, ord(" "), 13):
            break
        last = t

    # Final clear (one black frame so transition is clean)
    cv2.imshow(window_name, np.zeros_like(base))
    cv2.waitKey(1)
