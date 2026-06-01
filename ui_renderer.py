"""Cinematic UI overlay for the Hand Talk Translator (Jury Demo Edition).

Premium presentation mode designed for a final-year project defense:
  - dark cinematic vignette
  - glassmorphism-style cards with subtle borders
  - hero text for the detected gesture with neon glow
  - animated confidence ring gauge
  - prominent LSF / French translation panel
  - branded header bar
  - fluid status pills (FPS, time, signs, detections)
"""

import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import HISTORY_MAX, UI_COLORS


# ── Unicode-capable text renderer (PIL backed) ───────────────────────────────

_FONT_CANDIDATES_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_CANDIDATES_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


class _TextEngine:
    """Caches PIL fonts and draws Unicode text onto a BGR OpenCV frame.

    Draw calls are queued during a frame and flushed in a single
    BGR→PIL→BGR round-trip (cheap enough at 30 FPS, and gives us
    proper accents, apostrophes and emoji glyphs).
    """

    def __init__(self):
        self._fonts: dict[tuple[int, bool], ImageFont.ImageFont] = {}
        self._queue: list[tuple] = []

    def _load_font(self, size: int, bold: bool) -> ImageFont.ImageFont:
        candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
        for path in candidates:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def get_font(self, size: int, bold: bool) -> ImageFont.ImageFont:
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            f = self._load_font(size, bold)
            self._fonts[key] = f
        return f

    def measure(self, text: str, size: int, bold: bool) -> tuple[int, int]:
        font = self.get_font(size, bold)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def queue(self, text, pos, size, color_bgr, bold=False, stroke=0,
              stroke_color=(0, 0, 0), anchor="ls"):
        self._queue.append((text, pos, size, color_bgr, bold, stroke,
                            stroke_color, anchor))

    def flush(self, frame_bgr):
        if not self._queue:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for text, pos, size, col_bgr, bold, stroke, stroke_col, anchor in self._queue:
            font = self.get_font(size, bold)
            rgb_col = (int(col_bgr[2]), int(col_bgr[1]), int(col_bgr[0]))
            stroke_rgb = (int(stroke_col[2]), int(stroke_col[1]), int(stroke_col[0]))
            try:
                draw.text(pos, text, font=font, fill=rgb_col,
                          stroke_width=stroke, stroke_fill=stroke_rgb,
                          anchor=anchor)
            except (TypeError, ValueError):
                draw.text((pos[0], pos[1] - size), text,
                          font=font, fill=rgb_col)
        frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self._queue.clear()


class UIRenderer:
    """Composites every overlay element onto the OpenCV frame."""

    # Brand palette (BGR for OpenCV)
    GOLD = (0, 195, 255)
    GOLD_SOFT = (0, 145, 220)
    CYAN = (220, 200, 0)
    GREEN = (60, 220, 130)
    RED = (60, 70, 235)
    INK = (10, 10, 12)
    PANEL = (22, 22, 28)
    PANEL_BORDER = (78, 78, 96)
    TEXT = (240, 240, 245)
    SUBTEXT = (170, 170, 180)
    DIM = (110, 110, 120)

    def __init__(self, sign_labels: list[str] | None = None):
        self.C = UI_COLORS
        self.labels = sign_labels or []
        self.history: list[dict] = []
        self.show_help = False
        self.t0 = time.time()
        self._font = cv2.FONT_HERSHEY_SIMPLEX  # legacy reference
        self._text = _TextEngine()
        self._last_gesture = None
        self._last_change_ts = 0.0

    # ── public ───────────────────────────────────────────────────────────────

    def draw(self, frame, *, gesture, confidence, gloss="", french="",
             buf_fill, is_speaking, fps, motion_active=False,
             latency: dict | None = None):
        """Draw every UI element (mutates *frame* in place).

        ``latency`` is an optional ``{stage: ms}`` mapping (typically
        produced by :class:`latency_tracker.LatencyTracker`). When
        provided, a dedicated card is rendered top-left.
        """
        h, w = frame.shape[:2]

        if gesture and gesture != self._last_gesture:
            self._last_gesture = gesture
            self._last_change_ts = time.time()

        self._vignette(frame, w, h)
        self._header(frame, w)
        self._hero_panel(frame, gesture, confidence, is_speaking, w, h)
        self._buffer_pill(frame, buf_fill, w)
        self._motion_pill(frame, motion_active)
        if latency:
            self._latency_panel(frame, latency)
        self._translation_panel(frame, gloss, french, w, h)
        self._history_panel(frame, h)
        self._footer_status(frame, w, h, fps)
        if self.show_help:
            self._help_panel(frame, w, h)
        # Single Unicode-aware text pass (PIL) for the whole frame
        self._text.flush(frame)
        return frame

    def add_history(self, name):
        if not self.history or self.history[-1]["g"] != name:
            self.history.append({"g": name, "t": time.time()})
            if len(self.history) > HISTORY_MAX:
                self.history.pop(0)

    def toggle_help(self):
        self.show_help = not self.show_help

    def reset(self):
        self.history.clear()

    # ── primitives ───────────────────────────────────────────────────────────

    @staticmethod
    def _scale_to_size(scale: float) -> int:
        """Map legacy cv2 putText scale → PIL pixel font size."""
        return max(8, int(round(scale * 28)))

    def _put(self, f, text, pos, scale=0.55, color=TEXT, thick=1):
        size = self._scale_to_size(scale)
        bold = thick >= 2
        self._text.queue(text, pos, size, color, bold)

    def _put_centered(self, f, text, center, scale=0.55, color=TEXT, thick=1):
        size = self._scale_to_size(scale)
        bold = thick >= 2
        self._text.queue(text, (int(center[0]), int(center[1])),
                         size, color, bold, anchor="mm")

    def _measure(self, text, scale, thick=1):
        size = self._scale_to_size(scale)
        return self._text.measure(text, size, thick >= 2)

    def _rect(self, f, p1, p2, color, alpha=0.7):
        ov = f.copy()
        cv2.rectangle(ov, p1, p2, color, -1)
        cv2.addWeighted(ov, alpha, f, 1 - alpha, 0, f)

    def _glass(self, f, p1, p2, *, alpha=0.78, fill=PANEL,
               border=PANEL_BORDER, accent: tuple | None = None):
        """Glassmorphism-style panel with subtle border + optional accent line."""
        self._rect(f, p1, p2, fill, alpha)
        cv2.rectangle(f, p1, p2, border, 1)
        if accent is not None:
            cv2.line(f, (p1[0], p1[1]), (p1[0], p2[1]), accent, 2)

    def _vignette(self, f, w: int, h: int):
        """Dark cinematic borders (top/bottom + soft sides)."""
        self._rect(f, (0, 0), (w, 70), self.INK, 0.45)
        self._rect(f, (0, h - 56), (w, h), self.INK, 0.55)
        self._rect(f, (0, 0), (12, h), self.INK, 0.35)
        self._rect(f, (w - 12, 0), (w, h), self.INK, 0.35)

    def _truncate(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    @staticmethod
    def _conf_color(conf: float) -> tuple[int, int, int]:
        if conf >= 0.9:
            return 60, 220, 130
        if conf >= 0.75:
            return 0, 195, 255
        return 0, 135, 255

    def _glow_text(self, f, text, pos, scale, color, thick):
        """Neon-style text: dark stroke around + colored fill on top."""
        size = self._scale_to_size(scale)
        bold = thick >= 2
        self._text.queue(text, pos, size, color, bold,
                         stroke=2 if size < 20 else 3,
                         stroke_color=(0, 0, 0))

    # ── elements ─────────────────────────────────────────────────────────────

    def _header(self, f, w):
        self._rect(f, (0, 0), (w, 60), self.INK, 0.92)
        cv2.line(f, (0, 60), (w, 60), self.GOLD_SOFT, 2)

        # Brand mark (square + dot)
        cv2.rectangle(f, (16, 14), (44, 42), self.GOLD, 2)
        cv2.circle(f, (30, 28), 5, self.GOLD, -1)

        self._put(f, "HAND TALK TRANSLATOR",
                  (58, 30), 0.72, self.GOLD, 2)
        self._put(f, "ML EDITION  |  TFE DEMO MODE",
                  (58, 50), 0.42, self.SUBTEXT, 1)

        # Right-side keybinds chip
        chip = "[H] Aide   [D] Debug   [M] Micro   [E] Export   [R] Reset   [Q] Quitter"
        tw, _ = self._measure(chip, 0.42)
        x0 = w - tw - 22
        self._rect(f, (x0 - 12, 14), (w - 14, 42), (28, 28, 32), 0.85)
        cv2.rectangle(f, (x0 - 12, 14), (w - 14, 42), self.PANEL_BORDER, 1)
        self._put(f, chip, (x0 - 4, 32), 0.42, self.TEXT, 1)

    def _hero_panel(self, f, gesture, conf, speaking, w, h):
        """Big card top-right: detected gesture + neon ring gauge."""
        bw, bh = 440, 170
        x, y = w - bw - 18, 78
        self._glass(f, (x, y), (x + bw, y + bh),
                    alpha=0.82, fill=self.PANEL,
                    border=self.PANEL_BORDER, accent=self.GOLD)

        self._put(f, "GESTE DETECTE",
                  (x + 18, y + 26), 0.46, self.SUBTEXT, 1)

        # Pulse on new gesture
        elapsed = time.time() - self._last_change_ts
        pulse = max(0.0, 1.0 - elapsed * 1.6) if gesture else 0.0

        # Confidence ring (right side)
        cx, cy = x + bw - 70, y + bh // 2 + 6
        self._draw_ring_gauge(f, (cx, cy), conf if gesture else 0.0, pulse)

        if gesture:
            color = self._conf_color(conf)
            display = self._truncate(gesture, 14)
            self._glow_text(f, display, (x + 18, y + 92), 1.55, color, 3)

            # Confidence bar
            bl, bt, bwi = x + 18, y + 112, bw - 170
            cv2.rectangle(f, (bl, bt), (bl + bwi, bt + 10), (50, 50, 58), -1)
            fill = int(bwi * max(0.0, min(1.0, conf)))
            cv2.rectangle(f, (bl, bt), (bl + fill, bt + 10), color, -1)
            self._put(f, f"Confiance  {conf:.0%}",
                      (bl, bt + 30), 0.42, self.TEXT, 1)

            if speaking:
                dots = "." * (1 + int(time.time() * 3) % 3)
                self._put(f, f"SYNTHESE VOCALE EN COURS{dots}",
                          (x + 18, y + bh - 14), 0.4, self.GOLD, 1)
        else:
            self._put(f, "EN ATTENTE",
                      (x + 18, y + 92), 1.1, self.DIM, 2)
            self._put(f, "Montrez un signe a la camera",
                      (x + 18, y + 122), 0.45, self.SUBTEXT, 1)

    def _draw_ring_gauge(self, f, center, conf: float, pulse: float = 0.0):
        x, y = center
        conf = max(0.0, min(1.0, conf))
        radius = 46
        # Pulse ring (fades out after a new detection)
        if pulse > 0.0:
            r2 = int(radius + 14 * pulse)
            overlay = f.copy()
            cv2.circle(overlay, (x, y), r2, self.GOLD, 2)
            cv2.addWeighted(overlay, pulse * 0.8, f, 1 - pulse * 0.8, 0, f)

        # Track
        cv2.circle(f, (x, y), radius, (55, 55, 62), 8)
        # Arc
        sweep = int(360 * conf)
        col = self._conf_color(conf)
        cv2.ellipse(f, (x, y), (radius, radius), -90, 0, sweep, col, 8)

        # Center text
        self._put_centered(f, f"{int(conf * 100)}", (x, y - 2), 0.95,
                           self.TEXT, 2)
        self._put_centered(f, "CONF", (x, y + 22), 0.36,
                           self.SUBTEXT, 1)

    def _buffer_pill(self, f, fill, w):
        bw = 440
        x, y = w - bw - 18, 258
        self._glass(f, (x, y), (x + bw, y + 36),
                    fill=self.PANEL, border=self.PANEL_BORDER)
        self._put(f, "BUFFER LSTM", (x + 18, y + 23), 0.4, self.SUBTEXT, 1)
        # Bar
        bl = x + 140
        bt = y + 14
        bwi = bw - 200
        cv2.rectangle(f, (bl, bt), (bl + bwi, bt + 10), (50, 50, 58), -1)
        fw = int(bwi * max(0.0, min(1.0, fill)))
        c = self.GREEN if fill >= 1.0 else self.GOLD_SOFT
        cv2.rectangle(f, (bl, bt), (bl + fw, bt + 10), c, -1)
        self._put(f, f"{int(fill * 100):>3d}%",
                  (x + bw - 50, y + 23), 0.42, self.TEXT, 1)

    def _motion_pill(self, f, active):
        x, y = 18, 78
        bw, bh = 220, 44
        self._glass(f, (x, y), (x + bw, y + bh),
                    fill=self.PANEL, border=self.PANEL_BORDER)
        col = self.GREEN if active else self.DIM
        cv2.circle(f, (x + 16, y + bh // 2), 6, col, -1)
        cv2.circle(f, (x + 16, y + bh // 2), 9, col, 1)
        self._put(f, "MOUVEMENT", (x + 32, y + 19), 0.4, self.SUBTEXT, 1)
        self._put(f, "ACTIF" if active else "STABLE",
                  (x + 32, y + 36), 0.5, col, 2)

    # ── latency card ─────────────────────────────────────────────────────────

    # Order in which stages are rendered if present
    _LATENCY_STAGES = (
        ("camera",    "CAMERA"),
        ("mediapipe", "MEDIAPIPE"),
        ("lstm",      "LSTM"),
        ("translate", "TRADUC."),
        ("total",     "TOTAL"),
    )

    @staticmethod
    def _latency_color(ms: float, budget_ms: float) -> tuple[int, int, int]:
        """Green / gold / red based on the share of the frame budget used."""
        ratio = ms / max(1.0, budget_ms)
        if ratio < 0.5:
            return 60, 220, 130    # green
        if ratio < 0.85:
            return 0, 195, 255     # gold
        return 60, 70, 235         # red

    def _latency_panel(self, f, latency: dict):
        """Small glassmorphism card listing per-stage rolling means.

        Shown directly under the MOUVEMENT pill.
        """
        # Filter to known stages, preserve display order
        rows = [(label, latency[k]) for k, label in self._LATENCY_STAGES
                if k in latency]
        if not rows:
            return

        x, y = 18, 132
        bw = 220
        row_h = 22
        bh = 24 + len(rows) * row_h + 8
        self._glass(f, (x, y), (x + bw, y + bh),
                    fill=self.PANEL, border=self.PANEL_BORDER)
        self._put(f, "LATENCE (ms, moy.)",
                  (x + 12, y + 19), 0.4, self.SUBTEXT, 1)

        budget = 33.3  # one frame at 30 FPS
        for i, (label, ms) in enumerate(rows):
            ry = y + 38 + i * row_h
            col = self._latency_color(ms, budget)
            self._put(f, label, (x + 14, ry), 0.4, self.TEXT, 1)
            self._put(f, f"{ms:5.1f}", (x + bw - 64, ry), 0.45, col, 2)
            # tiny inline bar
            bar_x = x + bw - 22
            bar_w = 10
            bar_h = 12
            cv2.rectangle(f, (bar_x, ry - 12), (bar_x + bar_w, ry - 12 + bar_h),
                          (50, 50, 58), -1)
            fill_h = int(bar_h * min(1.0, ms / budget))
            cv2.rectangle(f, (bar_x, ry - 12 + bar_h - fill_h),
                          (bar_x + bar_w, ry - 12 + bar_h), col, -1)

    def _translation_panel(self, f, gloss, french, w, h):
        bh = 156
        y = h - bh - 56
        self._glass(f, (14, y), (w - 14, y + bh),
                    alpha=0.82, fill=(14, 14, 18),
                    border=self.PANEL_BORDER, accent=self.GOLD)

        self._put(f, "TRADUCTION TEMPS REEL",
                  (28, y + 26), 0.45, self.SUBTEXT, 1)

        # Compute character budgets dynamically based on panel width.
        # PIL renders accents/spaces wider than Hershey, so we cap to roughly
        # what fits on a single line for our font sizes.
        avail_px = (w - 28) - 110
        gloss_chars = max(40, avail_px // 9)
        french_chars = max(30, avail_px // 14)

        # LSF row
        self._tag(f, "LSF", (28, y + 46), self.GOLD)
        g_text = gloss if gloss else "(signes détectés ici)"
        g_col = self.TEXT if gloss else self.DIM
        self._put(f, self._truncate(g_text, gloss_chars),
                  (110, y + 66), 0.5, g_col, 1)

        # FR row (bigger, hero translation)
        self._tag(f, "FR", (28, y + 96), self.GREEN)
        f_text = french if french else "(traduction française)"
        f_col = self.TEXT if french else self.DIM
        self._glow_text(f, self._truncate(f_text, french_chars),
                        (110, y + 124), 0.78, f_col, 2 if french else 1)

    def _tag(self, f, text, top_left, color):
        x, y = top_left
        tw, _ = self._measure(text, 0.45, 2)
        cv2.rectangle(f, (x, y), (x + tw + 16, y + 24), color, -1)
        self._put(f, text, (x + 8, y + 18), 0.45, self.INK, 2)

    def _history_panel(self, f, h):
        if not self.history:
            return
        ph = len(self.history) * 30 + 48
        y0 = h - ph - 210
        self._glass(f, (18, y0), (310, y0 + ph),
                    fill=self.PANEL, border=self.PANEL_BORDER)
        self._put(f, "HISTORIQUE", (30, y0 + 26),
                  0.46, self.GOLD, 1)
        cv2.line(f, (30, y0 + 34), (296, y0 + 34), self.PANEL_BORDER, 1)

        for i, e in enumerate(reversed(self.history)):
            ry = y0 + 58 + i * 30
            fade = max(0.45, 1.0 - i * 0.12)
            col = tuple(int(c * fade) for c in self.TEXT)
            elapsed = time.time() - e["t"]
            t = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed / 60:.0f}m"
            cv2.circle(f, (34, ry - 5), 3, self.GOLD_SOFT, -1)
            self._put(f, self._truncate(e["g"], 14), (46, ry), 0.5, col, 1)
            self._put(f, f"il y a {t}",
                      (210, ry), 0.36, self.DIM, 1)

    def _footer_status(self, f, w, h, fps):
        self._rect(f, (0, h - 40), (w, h), self.INK, 0.9)
        cv2.line(f, (0, h - 40), (w, h - 40), self.GOLD_SOFT, 1)

        el = time.time() - self.t0
        m, s = int(el // 60), int(el % 60)
        n = len(self.labels)
        det = len(self.history)

        x = 22
        x = self._stat_pill(f, "FPS", f"{fps:.0f}", x, h)
        x = self._stat_pill(f, "SESSION", f"{m:02d}:{s:02d}", x, h)
        x = self._stat_pill(f, "SIGNES", f"{n}", x, h)
        x = self._stat_pill(f, "DETECTIONS", f"{det}", x, h)

        # Branding right
        brand = "Hand Talk Translator  •  Demo TFE"
        tw, _ = self._measure(brand, 0.42)
        self._put(f, brand, (w - tw - 22, h - 14), 0.42, self.SUBTEXT, 1)

    def _stat_pill(self, f, label, value, x, h):
        # Compute width based on text
        lw, _ = self._measure(label, 0.36)
        vw, _ = self._measure(value, 0.5, 2)
        pw = max(lw, vw) + 28
        y1 = h - 32
        y2 = h - 8
        cv2.rectangle(f, (x, y1), (x + pw, y2), (28, 28, 34), -1)
        cv2.rectangle(f, (x, y1), (x + pw, y2), self.PANEL_BORDER, 1)
        cv2.rectangle(f, (x, y1), (x + 4, y2), self.GOLD, -1)
        self._put(f, label, (x + 12, y1 + 9), 0.32, self.SUBTEXT, 1)
        self._put(f, value, (x + 12, y2 - 5), 0.5, self.TEXT, 2)
        return x + pw + 10

    def _help_panel(self, f, w, h):
        cols = 3
        per_col = math.ceil(len(self.labels) / cols)
        pw = 720
        ph = max(per_col * 26 + 110, 200)
        x, y = (w - pw) // 2, (h - ph) // 2
        self._glass(f, (x, y), (x + pw, y + ph),
                    alpha=0.96, fill=(20, 20, 26),
                    border=(120, 120, 150), accent=self.GOLD)
        self._put(f, "BIBLIOTHEQUE DES SIGNES",
                  (x + 24, y + 36), 0.7, self.GOLD, 2)
        self._put(f, f"{len(self.labels)} signes appris par le modele LSTM",
                  (x + 24, y + 60), 0.42, self.SUBTEXT, 1)
        cv2.line(f, (x + 20, y + 78), (x + pw - 20, y + 78),
                 self.GOLD_SOFT, 1)

        col_w = (pw - 40) // cols
        for i, label in enumerate(self.labels):
            col_idx = i // per_col
            row_idx = i % per_col
            lx = x + 24 + col_idx * col_w
            ly = y + 108 + row_idx * 26
            cv2.circle(f, (lx, ly - 5), 3, self.GOLD_SOFT, -1)
            self._put(f, f"{i + 1:>2}. {self._truncate(label, 22)}",
                      (lx + 10, ly), 0.46, self.TEXT, 1)

        msg = "Appuyez sur [H] pour fermer"
        tw, _ = self._measure(msg, 0.42)
        self._put(f, msg, (x + (pw - tw) // 2, y + ph - 18),
                  0.42, self.SUBTEXT, 1)
