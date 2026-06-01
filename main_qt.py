"""Hand Talk Translator — Qt Edition (JURY DEMO BUILD).

Premium presentation interface designed to genuinely impress a defense jury:

  • Animated dark gradient background with slow-floating accent orbs
  • Glassmorphism cards with soft drop shadows and subtle borders
  • Hero "current sign" panel: HUGE neon confidence ring + giant gesture name
  • Live probability bars with smooth interpolation and color grading
  • Translation card with a dramatic French sentence (typewriter feel)
  • Chat-bubble conversation panel (LSF ↔ Microphone)
  • Modern pill-shaped action buttons with hover glow
  • Premium footer with floating "stat pills" (FPS, session, signs, commits)

Usage
─────
    py -3.11 main_qt.py
    py -3.11 main_qt.py --no-listen
"""

import argparse
import datetime
import math
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

try:
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QSize, QPointF, QRectF,
    )
    from PyQt6.QtGui import (
        QImage, QPixmap, QColor, QPainter, QFont, QFontDatabase,
        QShortcut, QKeySequence, QPen, QBrush, QLinearGradient,
        QRadialGradient, QPainterPath, QConicalGradient,
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
        QTextEdit, QFrame, QSizePolicy, QStackedLayout,
        QGraphicsDropShadowEffect, QScrollArea,
    )
    _QT_OK = True
except ImportError as _e:
    print(f"Import PyQt6 echoue : {_e}")
    _QT_OK = False


# ── Premium palette (BGR-free, just hex / QColor) ─────────────────────────────
C_BG_TOP    = "#08081a"
C_BG_BOT    = "#02020a"
C_CARD      = "rgba(20, 22, 38, 220)"
C_CARD_HI   = "rgba(28, 32, 50, 220)"
C_BORDER    = "rgba(120, 130, 200, 60)"
C_CYAN      = "#00d4ff"
C_VIOLET    = "#a06bff"
C_PINK      = "#ff5da2"
C_GREEN     = "#34f5a8"
C_AMBER     = "#ffb547"
C_RED       = "#ff5577"
C_WHITE     = "#f5f7ff"
C_TEXT      = "#dde2ff"
C_DIM       = "#6f779e"
C_DIM2      = "#3f4870"


STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background-color: transparent;
    color: {C_TEXT};
    font-family: 'Segoe UI Variable', 'Segoe UI', 'Inter', 'Helvetica Neue', Arial;
}}
QWidget {{ color: {C_TEXT}; font-family: 'Segoe UI Variable', 'Segoe UI', 'Inter', 'Helvetica Neue', Arial; }}

QFrame#card {{
    background-color: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 18px;
}}
QFrame#cardHero {{
    background-color: {C_CARD_HI};
    border: 1px solid {C_BORDER};
    border-radius: 22px;
}}

QLabel#title_lbl {{
    font-size: 22px;
    font-weight: 700;
    color: {C_WHITE};
    letter-spacing: 4px;
}}
QLabel#brand_lbl {{
    font-size: 10px;
    color: {C_DIM};
    letter-spacing: 6px;
}}
QLabel#section_lbl {{
    font-size: 10px;
    color: {C_DIM};
    letter-spacing: 3px;
    font-weight: 600;
}}
QLabel#section_lbl_accent {{
    font-size: 10px;
    color: {C_CYAN};
    letter-spacing: 3px;
    font-weight: 700;
}}
QLabel#stat_value {{ font-size: 18px; font-weight: 700; color: {C_WHITE}; }}
QLabel#stat_label {{ font-size: 9px;  color: {C_DIM}; letter-spacing: 2px; }}

QLabel#fr_lbl {{
    color: {C_WHITE};
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QLabel#gloss_lbl {{ color: {C_DIM}; font-size: 13px; letter-spacing: 1px; }}

QPushButton {{
    background-color: rgba(30, 34, 56, 220);
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 18px;
    padding: 10px 18px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QPushButton:hover   {{ border-color: {C_CYAN}; color: {C_CYAN}; }}
QPushButton:pressed {{ background-color: rgba(20, 24, 40, 240); }}
QPushButton#btn_speak  {{
    border-color: {C_CYAN};   color: {C_CYAN};
    background-color: rgba(0, 212, 255, 30);
}}
QPushButton#btn_clear  {{ border-color: {C_RED};   color: {C_RED};   }}
QPushButton#btn_rec_on {{
    background-color: rgba(255, 85, 119, 50);
    border-color: {C_RED}; color: {C_RED};
}}

QTextEdit#conv {{
    background-color: rgba(10, 12, 24, 220);
    border: 1px solid {C_BORDER};
    border-radius: 14px;
    font-size: 13px;
    color: {C_TEXT};
    padding: 12px;
}}
QLabel#cam_lbl {{
    background-color: #000008;
    border: 1px solid {C_BORDER};
    border-radius: 16px;
}}
QScrollBar:vertical {{
    background: transparent; width: 6px; margin: 4px;
}}
QScrollBar::handle:vertical {{
    background: rgba(120,130,200,80); border-radius: 3px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


# ── Animated gradient background ──────────────────────────────────────────────

class AnimatedBackground(QWidget):
    """Slow-moving gradient orbs over a deep navy background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _tick(self):
        self._t += 0.012
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # Base vertical gradient
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, QColor(C_BG_TOP))
        grad.setColorAt(1.0, QColor(C_BG_BOT))
        p.fillRect(0, 0, W, H, grad)

        # Floating accent orbs
        orbs = [
            (QColor(0, 212, 255, 70),  0.20, 0.25, 360, 1.0),
            (QColor(160, 107, 255, 70), 0.85, 0.30, 420, 0.85),
            (QColor(52, 245, 168, 50), 0.65, 0.85, 380, 0.7),
            (QColor(255, 93, 162, 50), 0.15, 0.85, 320, 0.6),
        ]
        for col, fx, fy, base_r, speed in orbs:
            cx = fx * W + math.cos(self._t * speed) * 80
            cy = fy * H + math.sin(self._t * speed * 0.8) * 60
            r = base_r + math.sin(self._t * speed * 0.6) * 30
            rg = QRadialGradient(QPointF(cx, cy), r)
            rg.setColorAt(0.0, col)
            transparent = QColor(col); transparent.setAlpha(0)
            rg.setColorAt(1.0, transparent)
            p.setBrush(QBrush(rg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # Vignette
        v = QRadialGradient(QPointF(W / 2, H / 2), max(W, H) * 0.8)
        v.setColorAt(0.55, QColor(0, 0, 0, 0))
        v.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.setBrush(QBrush(v))
        p.drawRect(0, 0, W, H)
        p.end()


# ── Confidence ring (HUGE, neon, animated) ────────────────────────────────────

class ConfidenceRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = 0.0
        self._value = 0.0
        self._pulse = 0.0
        self._label = "—"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(25)
        self.setMinimumSize(260, 260)

    def set_value(self, v: float, label: str | None = None):
        self._target = max(0.0, min(1.0, float(v)))
        if label is not None:
            self._label = label

    def pulse(self):
        self._pulse = 1.0

    def _tick(self):
        self._value += (self._target - self._value) * 0.22
        if self._pulse > 0:
            self._pulse = max(0.0, self._pulse - 0.04)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        size = min(W, H)
        cx, cy = W / 2, H / 2
        radius = size / 2 - 18
        thickness = 14

        # Pulse ring
        if self._pulse > 0:
            pulse_pen = QPen(QColor(0, 212, 255, int(160 * self._pulse)))
            pulse_pen.setWidth(int(4 + 12 * self._pulse))
            p.setPen(pulse_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            r2 = radius + 8 + 18 * self._pulse
            p.drawEllipse(QPointF(cx, cy), r2, r2)

        # Track
        track_pen = QPen(QColor(40, 46, 78))
        track_pen.setWidth(thickness)
        track_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(track_pen)
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # Conic gradient arc
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        sweep_angle = 360 * self._value

        cg = QConicalGradient(cx, cy, 90)
        cg.setColorAt(0.0, QColor(C_CYAN))
        cg.setColorAt(0.5, QColor(C_VIOLET))
        cg.setColorAt(1.0, QColor(C_PINK))

        arc_pen = QPen(QBrush(cg), thickness)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        # Qt drawArc uses 1/16 degrees, start at top (90°) going clockwise
        p.drawArc(rect, 90 * 16, int(-sweep_angle * 16))

        # Center glow disc
        cdg = QRadialGradient(QPointF(cx, cy), radius * 0.95)
        cdg.setColorAt(0.0, QColor(0, 212, 255, 50))
        cdg.setColorAt(1.0, QColor(0, 212, 255, 0))
        p.setBrush(QBrush(cdg))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), radius * 0.9, radius * 0.9)

        # Big percentage text
        pct = int(round(self._value * 100))
        f = QFont("Segoe UI Variable", int(size * 0.18), QFont.Weight.Bold)
        p.setFont(f)
        p.setPen(QColor(C_WHITE))
        p.drawText(QRectF(0, cy - radius * 0.4, W, radius * 0.8),
                   Qt.AlignmentFlag.AlignCenter, f"{pct}%")

        # Label below
        f2 = QFont("Segoe UI", 10, QFont.Weight.Bold)
        f2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
        p.setFont(f2)
        p.setPen(QColor(C_DIM))
        p.drawText(QRectF(0, cy + radius * 0.05, W, 30),
                   Qt.AlignmentFlag.AlignCenter, "CONFIANCE")

        # Sign label at bottom
        f3 = QFont("Segoe UI Variable", 14, QFont.Weight.Bold)
        p.setFont(f3)
        p.setPen(QColor(C_CYAN))
        p.drawText(QRectF(0, cy + radius * 0.25, W, 30),
                   Qt.AlignmentFlag.AlignCenter,
                   self._label[:20] if self._label else "—")
        p.end()


# ── Pretty animated probability bars ─────────────────────────────────────────

class ProbBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current: dict[str, float] = {}
        self._target:  dict[str, float] = {}
        self._flash = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(28)
        self.setMinimumHeight(220)

    def set_probs(self, probs: dict[str, float]):
        self._target = dict(probs)
        for k in probs:
            self._current.setdefault(k, 0.0)

    def flash(self):
        self._flash = 1.0

    def _tick(self):
        for k, tgt in self._target.items():
            cur = self._current.get(k, 0.0)
            self._current[k] = cur + 0.18 * (tgt - cur)
        self._flash = max(0.0, self._flash - 0.05)
        self.update()

    def paintEvent(self, _e):
        if not self._current:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        items = sorted(self._current.items(), key=lambda x: -x[1])[:8]
        n = len(items)
        if n == 0:
            return

        row_h = min(28, (H - 12) // n)
        lbl_w = 110
        bar_max = W - lbl_w - 60

        for i, (label, prob) in enumerate(items):
            y = 6 + i * (row_h + 4)
            track = QRectF(lbl_w, y, bar_max, row_h - 4)

            # Track
            path_track = QPainterPath()
            path_track.addRoundedRect(track, (row_h - 4) / 2, (row_h - 4) / 2)
            p.fillPath(path_track, QColor(20, 22, 36))

            # Fill
            bar_len = max(1, bar_max * prob)
            bar_rect = QRectF(lbl_w, y, bar_len, row_h - 4)
            grad = QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
            if prob >= 0.75:
                grad.setColorAt(0.0, QColor(C_GREEN)); grad.setColorAt(1.0, QColor(C_CYAN))
            elif prob >= 0.4:
                grad.setColorAt(0.0, QColor(C_CYAN));  grad.setColorAt(1.0, QColor(C_VIOLET))
            elif prob >= 0.1:
                grad.setColorAt(0.0, QColor(C_AMBER)); grad.setColorAt(1.0, QColor(C_PINK))
            else:
                grad.setColorAt(0.0, QColor(60, 64, 96)); grad.setColorAt(1.0, QColor(40, 44, 70))

            path_fill = QPainterPath()
            path_fill.addRoundedRect(bar_rect, (row_h - 4) / 2, (row_h - 4) / 2)
            p.fillPath(path_fill, QBrush(grad))

            # Flash on top item
            if i == 0 and self._flash > 0:
                fc = QColor(255, 255, 255, int(self._flash * 90))
                p.fillPath(path_fill, fc)

            # Label
            p.setPen(QColor(200, 205, 230))
            f = QFont("Segoe UI Variable", 10, QFont.Weight.DemiBold)
            p.setFont(f)
            p.drawText(QRectF(0, y, lbl_w - 8, row_h),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       label[:14])

            # Percentage
            p.setPen(QColor(C_WHITE if i == 0 else C_TEXT))
            f2 = QFont("Segoe UI Variable", 10, QFont.Weight.Bold)
            p.setFont(f2)
            p.drawText(QRectF(lbl_w + bar_max + 8, y, 50, row_h),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{prob:.0%}")
        p.end()


# ── Buffer bar (gradient pill) ───────────────────────────────────────────────

class BufferBar(QWidget):
    def __init__(self):
        super().__init__()
        self._fill = 0.0
        self.setFixedHeight(14)

    def set_fill(self, v: float):
        self._fill = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        track = QRectF(0, 0, W, H)
        path_t = QPainterPath(); path_t.addRoundedRect(track, H / 2, H / 2)
        p.fillPath(path_t, QColor(20, 22, 36))

        if self._fill > 0:
            bar = QRectF(0, 0, W * self._fill, H)
            grad = QLinearGradient(bar.topLeft(), bar.topRight())
            grad.setColorAt(0.0, QColor(C_CYAN))
            grad.setColorAt(1.0, QColor(C_VIOLET))
            path_b = QPainterPath(); path_b.addRoundedRect(bar, H / 2, H / 2)
            p.fillPath(path_b, QBrush(grad))
        p.end()


# ── Stat pill (small footer card) ────────────────────────────────────────────

class StatPill(QFrame):
    def __init__(self, label: str, value: str = "—"):
        super().__init__()
        self.setObjectName("card")
        self.setMinimumHeight(56)
        self.setMaximumHeight(56)
        l = QVBoxLayout(self)
        l.setContentsMargins(16, 8, 16, 8)
        l.setSpacing(0)
        self.lbl_value = QLabel(value); self.lbl_value.setObjectName("stat_value")
        self.lbl_label = QLabel(label.upper()); self.lbl_label.setObjectName("stat_label")
        l.addWidget(self.lbl_value)
        l.addWidget(self.lbl_label)

    def set_value(self, v: str):
        self.lbl_value.setText(v)


# ── Glowing landmark overlay (kept) ──────────────────────────────────────────

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_FINGER_COLORS = [
    (0, 212, 255), (52, 245, 168), (255, 181, 71),
    (160, 107, 255), (255, 93, 162), (200, 200, 220),
]

def _draw_landmarks(frame: np.ndarray, results) -> np.ndarray:
    if not results or not results.hand_landmarks:
        return frame
    h, w = frame.shape[:2]
    for hand_lm in results.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for a, b in _HAND_CONNECTIONS:
            fi = a // 4 if a > 0 else 5
            fi = min(fi, len(_FINGER_COLORS) - 1)
            col = _FINGER_COLORS[fi]
            # BGR conversion for OpenCV
            col_bgr = (col[2], col[1], col[0])
            cv2.line(frame, pts[a], pts[b], tuple(c // 3 for c in col_bgr), 6, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], col_bgr, 2, cv2.LINE_AA)
        for idx, (px, py) in enumerate(pts):
            col = _FINGER_COLORS[min(idx // 4, len(_FINGER_COLORS) - 1)]
            col_bgr = (col[2], col[1], col[0])
            cv2.circle(frame, (px, py), 7, tuple(c // 3 for c in col_bgr), -1)
            cv2.circle(frame, (px, py), 4, col_bgr, -1)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)
    return frame


# ── Recognition worker (logic unchanged) ─────────────────────────────────────

class RecognitionWorker(QThread):
    frame_ready       = pyqtSignal(object)
    gesture_detected  = pyqtSignal(str, float)
    gesture_committed = pyqtSignal(str)
    speech_committed  = pyqtSignal(str)
    translation_ready = pyqtSignal(str, str)
    speech_heard      = pyqtSignal(str)
    stats_updated     = pyqtSignal(float, int, int)
    buf_fill_updated  = pyqtSignal(float)
    probs_updated     = pyqtSignal(dict)

    def __init__(self, listen: bool = True):
        super().__init__()
        self.listen    = listen
        self._running  = True
        self._do_speak = self._do_clear = self._do_delete = False
        self._do_toggle_mic = False

    def stop(self):              self._running = False
    def request_speak(self):     self._do_speak  = True
    def request_clear(self):     self._do_clear  = True
    def request_delete(self):    self._do_delete = True
    def request_toggle_mic(self):self._do_toggle_mic = True

    def run(self):
        from config import (
            CAMERA, COMMIT, FINISH_GESTURE, MOTION,
            PAUSE_LABEL, RECOGNITION, SPEAK,
            STABILITY_FRAMES_REQUIRED, UNKNOWN_LABEL,
        )
        from gesture_recognizer import GestureRecognizer
        from hand_detector import HandDetector
        from lsf_translator import translate
        from motion_pause_detector import MotionPauseDetector
        from sentence_builder import SentenceBuilder
        from speech_engine import SpeechEngine
        from speech_listener import SpeechListener

        recognizer   = GestureRecognizer()
        detector     = HandDetector()
        speech       = SpeechEngine()
        sentence     = SentenceBuilder()
        pause_motion = MotionPauseDetector()
        listener     = SpeechListener(language="fr-FR")

        if self.listen:
            listener.start()

        cap = cv2.VideoCapture(CAMERA["index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

        fps_t = time.time(); fps = 0.0
        stable_gesture = None; stable_count = 0; committed = False
        last_commit_ts = 0.0;  last_commit_lbl = None
        finish_count = 0;      last_finish_ts = 0.0
        n_commits = 0
        pause_available = PAUSE_LABEL in recognizer.labels

        def _is_open_palm(lm):
            if lm is None or len(lm) < 21: return False
            above = sum(1 for i in [4,8,12,16,20] if lm[i].y < lm[0].y)
            if above < 3: return False
            spread = ((lm[8].x-lm[20].x)**2+(lm[8].y-lm[20].y)**2)**0.5
            return spread >= FINISH_GESTURE["min_spread"]

        def _speak_log(text):
            nonlocal n_commits
            if not text: return
            speech.speak(text, force=True)
            n_commits += 1
            self.speech_committed.emit(text)

        while self._running:
            if self._do_speak:
                self._do_speak = False
                if not sentence.is_empty:
                    _speak_log(translate(sentence.tokens))
            if self._do_clear:
                self._do_clear = False
                sentence.clear(); recognizer.reset()
                self.translation_ready.emit("", "")
            if self._do_delete:
                self._do_delete = False
                sentence.delete_last()
                fr = translate(sentence.tokens) if not sentence.is_empty else ""
                self.translation_ready.emit(sentence.gloss, fr)
            if self._do_toggle_mic:
                self._do_toggle_mic = False
                if self.listen: listener.stop(); self.listen = False
                else: self.listen = listener.start()

            if self.listen:
                heard = listener.get_text()
                if heard: self.speech_heard.emit(heard)

            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now   = time.time()
            fps   = 0.9*fps + 0.1*(1.0/max(now-fps_t, 1e-6))
            fps_t = now

            results = detector.detect(rgb)
            frame = _draw_landmarks(frame, results)

            gesture, confidence, motion, accepted = recognizer.process_result(results)
            commit_gesture = gesture if confidence >= RECOGNITION["min_commit_confidence"] else None

            self.buf_fill_updated.emit(recognizer.buffer_fill)
            if recognizer.last_probs:
                self.probs_updated.emit(dict(recognizer.last_probs))

            if gesture and gesture != UNKNOWN_LABEL:
                self.gesture_detected.emit(gesture, float(confidence))

            if commit_gesture != stable_gesture:
                stable_gesture = commit_gesture
                stable_count   = 1 if commit_gesture else 0
                committed      = False
            elif commit_gesture:
                stable_count += 1

            if commit_gesture and stable_count >= STABILITY_FRAMES_REQUIRED and not committed:
                if (commit_gesture == last_commit_lbl
                        and now - last_commit_ts < COMMIT["min_interval_seconds"]):
                    commit_gesture = None
                else:
                    last_commit_lbl = commit_gesture; last_commit_ts = now

            if commit_gesture and stable_count >= STABILITY_FRAMES_REQUIRED and not committed:
                if gesture == UNKNOWN_LABEL:
                    committed = True
                elif pause_available and gesture == PAUSE_LABEL:
                    sentence.add_pause()
                else:
                    if recognizer.motion_active or not MOTION.get("require_active", False):
                        sentence.add(gesture)
                        self.gesture_committed.emit(gesture)
                    if SPEAK.get("on_commit", False) and not sentence.is_empty:
                        fn = translate(sentence.tokens)
                        if fn: _speak_log(fn)
                committed = True

            if FINISH_GESTURE.get("enabled", False) and results.hand_landmarks:
                hands = results.hand_landmarks
                both_open = (len(hands)>=2 and _is_open_palm(hands[0]) and _is_open_palm(hands[1]))
                finish_count = finish_count+1 if both_open else 0
                if (finish_count >= FINISH_GESTURE["required_frames"]
                        and now - last_finish_ts >= FINISH_GESTURE["cooldown_seconds"]
                        and not sentence.is_empty):
                    last_finish_ts = now; finish_count = 0
                    _speak_log(translate(sentence.tokens))

            if pause_motion.update(
                results.hand_landmarks[0] if results.hand_landmarks else None, now
            ):
                sentence.add_pause()

            fr_disp = translate(sentence.tokens) if not sentence.is_empty else ""
            self.translation_ready.emit(sentence.gloss, fr_disp)
            self.stats_updated.emit(fps, recognizer.num_signs, n_commits)
            self.frame_ready.emit(frame)

        cap.release(); listener.stop(); speech.release(); detector.release()


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, listen: bool = True):
        super().__init__()
        self.listen      = listen
        self.conversation: list[dict] = []
        self.recording   = False
        self.video_writer = None
        self.last_frame  = None
        self._t0         = time.time()
        self._last_gesture = ""

        self.setWindowTitle("Hand Talk Translator — Demo Jury")
        self.setMinimumSize(1480, 880)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._start_worker()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Stack: animated background + foreground UI
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        stack = QStackedLayout(root)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.bg = AnimatedBackground()
        stack.addWidget(self.bg)

        ui = QWidget(); ui.setObjectName("root")
        ui.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        stack.addWidget(ui)
        ui.raise_()

        outer = QVBoxLayout(ui)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(18)

        # ── Header ────────────────────────────────────────────────────
        header = QHBoxLayout(); header.setSpacing(16)

        brand_box = QVBoxLayout(); brand_box.setSpacing(0)
        title = QLabel("HAND TALK TRANSLATOR"); title.setObjectName("title_lbl")
        sub = QLabel("REAL-TIME  ·  LSF → FRENCH  ·  DEEP LEARNING (LSTM)")
        sub.setObjectName("brand_lbl")
        brand_box.addWidget(title); brand_box.addWidget(sub)
        header.addLayout(brand_box)
        header.addStretch()

        self.pill_clock = StatPill("HEURE", "00:00")
        self.pill_session = StatPill("SESSION", "00:00")
        self.pill_fps = StatPill("FPS", "—")
        self.pill_signs = StatPill("SIGNES", "—")
        self.pill_commits = StatPill("COMMITS", "0")
        for w in (self.pill_clock, self.pill_session, self.pill_fps,
                  self.pill_signs, self.pill_commits):
            header.addWidget(w)
        outer.addLayout(header)

        # ── Body: 3 columns ───────────────────────────────────────────
        body = QHBoxLayout(); body.setSpacing(18)
        outer.addLayout(body, 1)

        # COLUMN 1 — Camera + buffer + translation + actions
        col1 = QVBoxLayout(); col1.setSpacing(14)

        self.cam_lbl = QLabel(); self.cam_lbl.setObjectName("cam_lbl")
        self.cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_lbl.setMinimumSize(640, 420)
        self.cam_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        self._add_shadow(self.cam_lbl, 60, 0, 0, QColor(0, 212, 255, 60))
        col1.addWidget(self.cam_lbl, 1)

        # Buffer card
        buf_card = QFrame(); buf_card.setObjectName("card")
        bl = QVBoxLayout(buf_card); bl.setContentsMargins(16, 12, 16, 12); bl.setSpacing(8)
        bh = QLabel("BUFFER LSTM"); bh.setObjectName("section_lbl")
        self.buf_bar = BufferBar()
        bl.addWidget(bh); bl.addWidget(self.buf_bar)
        col1.addWidget(buf_card)

        # Translation card (hero)
        tc = QFrame(); tc.setObjectName("cardHero")
        tl = QVBoxLayout(tc); tl.setContentsMargins(22, 18, 22, 22); tl.setSpacing(8)
        h1 = QLabel("TRADUCTION FRANÇAISE"); h1.setObjectName("section_lbl_accent")
        self.french_lbl = QLabel(""); self.french_lbl.setObjectName("fr_lbl")
        self.french_lbl.setWordWrap(True)
        self.french_lbl.setMinimumHeight(80)
        self._add_shadow(self.french_lbl, 32, 0, 0, QColor(0, 212, 255, 120))
        h2 = QLabel("SIGNES LSF"); h2.setObjectName("section_lbl")
        self.gloss_lbl = QLabel("—"); self.gloss_lbl.setObjectName("gloss_lbl")
        self.gloss_lbl.setWordWrap(True)
        for w in (h1, self.french_lbl, h2, self.gloss_lbl):
            tl.addWidget(w)
        self._add_shadow(tc, 50, 0, 12, QColor(0, 0, 0, 160))
        col1.addWidget(tc)

        # Action buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_speak  = QPushButton("▶  PARLER"); self.btn_speak.setObjectName("btn_speak")
        self.btn_delete = QPushButton("⌫  SUPPR")
        self.btn_clear  = QPushButton("✕  EFFACER"); self.btn_clear.setObjectName("btn_clear")
        self.btn_mic    = QPushButton("🎙  MICRO ON")
        self.btn_rec    = QPushButton("⏺  REC")
        self.btn_export = QPushButton("⬇  EXPORT")
        for b in (self.btn_speak, self.btn_delete, self.btn_clear,
                  self.btn_mic, self.btn_rec, self.btn_export):
            btn_row.addWidget(b)
        col1.addLayout(btn_row)
        body.addLayout(col1, 5)

        # COLUMN 2 — Hero confidence ring + probs
        col2 = QVBoxLayout(); col2.setSpacing(14)

        hero = QFrame(); hero.setObjectName("cardHero")
        hl = QVBoxLayout(hero); hl.setContentsMargins(20, 18, 20, 22); hl.setSpacing(6)
        hh = QLabel("SIGNE DÉTECTÉ"); hh.setObjectName("section_lbl_accent")
        self.ring = ConfidenceRing()
        hl.addWidget(hh)
        hl.addWidget(self.ring, 1, Qt.AlignmentFlag.AlignCenter)
        self._add_shadow(hero, 60, 0, 12, QColor(0, 212, 255, 70))
        col2.addWidget(hero, 1)

        # Probabilities card
        pc = QFrame(); pc.setObjectName("card")
        pl = QVBoxLayout(pc); pl.setContentsMargins(20, 16, 20, 16); pl.setSpacing(10)
        ph = QLabel("PROBABILITÉS  ·  TEMPS RÉEL"); ph.setObjectName("section_lbl")
        self.prob_bars = ProbBarWidget()
        pl.addWidget(ph); pl.addWidget(self.prob_bars, 1)
        col2.addWidget(pc, 1)
        body.addLayout(col2, 4)

        # COLUMN 3 — Conversation
        col3 = QVBoxLayout(); col3.setSpacing(14)
        cc = QFrame(); cc.setObjectName("card")
        cl = QVBoxLayout(cc); cl.setContentsMargins(20, 16, 20, 16); cl.setSpacing(10)
        ch_row = QHBoxLayout()
        ch = QLabel("CONVERSATION"); ch.setObjectName("section_lbl_accent")
        self.mic_dot = QLabel("● MIC ON")
        self.mic_dot.setStyleSheet(f"color:{C_GREEN}; font-size:11px; font-weight:bold; letter-spacing:1px;")
        ch_row.addWidget(ch); ch_row.addStretch(); ch_row.addWidget(self.mic_dot)
        self.conv_text = QTextEdit(); self.conv_text.setObjectName("conv")
        self.conv_text.setReadOnly(True); self.conv_text.setMinimumHeight(180)
        cl.addLayout(ch_row); cl.addWidget(self.conv_text, 1)
        col3.addWidget(cc, 1)

        # Hint card
        hint = QFrame(); hint.setObjectName("card")
        hlt = QVBoxLayout(hint); hlt.setContentsMargins(16, 12, 16, 12); hlt.setSpacing(4)
        ht = QLabel("RACCOURCIS"); ht.setObjectName("section_lbl")
        kt = QLabel(
            "<span style='color:#00d4ff;'>Espace</span> Parler   "
            "<span style='color:#00d4ff;'>⌫</span> Suppr   "
            "<span style='color:#ff5577;'>Entrée</span> Effacer<br>"
            "<span style='color:#34f5a8;'>M</span> Micro   "
            "<span style='color:#a06bff;'>V</span> Rec   "
            "<span style='color:#ffb547;'>E</span> Export   "
            "<span style='color:#ff5577;'>Q</span> Quitter"
        )
        kt.setStyleSheet(f"color:{C_TEXT}; font-size:12px; line-height:18px;")
        hlt.addWidget(ht); hlt.addWidget(kt)
        col3.addWidget(hint)
        body.addLayout(col3, 3)

        # ── Shortcuts ────────────────────────────────────────────────
        QShortcut(QKeySequence("Space"),     self, self._on_speak)
        QShortcut(QKeySequence("Return"),    self, self._on_clear)
        QShortcut(QKeySequence("Backspace"), self, self._on_delete)
        QShortcut(QKeySequence("M"),         self, self._on_toggle_mic)
        QShortcut(QKeySequence("V"),         self, self._on_toggle_rec)
        QShortcut(QKeySequence("E"),         self, self._on_export)
        QShortcut(QKeySequence("Q"),         self, self.close)
        QShortcut(QKeySequence("F"),         self, self._toggle_fullscreen)

        self.btn_speak.clicked.connect(self._on_speak)
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_mic.clicked.connect(self._on_toggle_mic)
        self.btn_rec.clicked.connect(self._on_toggle_rec)
        self.btn_export.clicked.connect(self._on_export)

        self._clock = QTimer(); self._clock.timeout.connect(self._tick_clock)
        self._clock.start(1000); self._tick_clock()

    def _add_shadow(self, widget, blur, dx, dy, color):
        e = QGraphicsDropShadowEffect()
        e.setBlurRadius(blur); e.setOffset(dx, dy); e.setColor(color)
        widget.setGraphicsEffect(e)

    def resizeEvent(self, e):
        self.bg.setGeometry(self.centralWidget().rect())
        return super().resizeEvent(e)

    def _toggle_fullscreen(self):
        if self.isFullScreen(): self.showNormal()
        else: self.showFullScreen()

    # ── Worker ────────────────────────────────────────────────────────

    def _start_worker(self):
        self.worker = RecognitionWorker(listen=self.listen)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.gesture_detected.connect(self._on_gesture)
        self.worker.gesture_committed.connect(self._on_committed)
        self.worker.speech_committed.connect(self._on_sign_spoken)
        self.worker.translation_ready.connect(self._on_translation)
        self.worker.speech_heard.connect(self._on_speech_heard)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.buf_fill_updated.connect(self.buf_bar.set_fill)
        self.worker.probs_updated.connect(self.prob_bars.set_probs)
        self.worker.start()

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        self.last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.cam_lbl.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.cam_lbl.setPixmap(pix)
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def _on_gesture(self, label: str, conf: float):
        self.ring.set_value(conf, label)
        self._last_gesture = label

    def _on_committed(self, _label: str):
        self.ring.pulse()
        self.prob_bars.flash()

    def _on_translation(self, gloss: str, french: str):
        self.gloss_lbl.setText(gloss if gloss else "—")
        self.french_lbl.setText(french if french else " ")

    def _on_sign_spoken(self, text: str):
        self.conversation.append({"side": "sign", "text": text, "time": time.time()})
        self._conv_append(C_CYAN, "LSF", text)

    def _on_speech_heard(self, text: str):
        self.conversation.append({"side": "voice", "text": text, "time": time.time()})
        self._conv_append(C_GREEN, "MIC", text)

    def _on_stats(self, fps: float, n: int, commits: int):
        self.pill_fps.set_value(f"{fps:.0f}")
        self.pill_signs.set_value(f"{n}")
        self.pill_commits.set_value(f"{commits}")

    def _tick_clock(self):
        el = int(time.time() - self._t0)
        self.pill_session.set_value(f"{el // 60:02d}:{el % 60:02d}")
        self.pill_clock.set_value(datetime.datetime.now().strftime("%H:%M"))

    # ── Buttons ──────────────────────────────────────────────────────

    def _on_speak(self):  self.worker.request_speak()
    def _on_delete(self): self.worker.request_delete()
    def _on_clear(self):  self.worker.request_clear()

    def _on_toggle_mic(self):
        self.worker.request_toggle_mic()
        self.worker.listen = not self.worker.listen
        on = self.worker.listen
        self.btn_mic.setText(f"🎙  MICRO {'ON' if on else 'OFF'}")
        self.mic_dot.setText(f"● MIC {'ON' if on else 'OFF'}")
        col = C_GREEN if on else C_DIM
        self.mic_dot.setStyleSheet(
            f"color:{col}; font-size:11px; font-weight:bold; letter-spacing:1px;"
        )

    def _on_toggle_rec(self):
        if not self.recording:
            if self.last_frame is not None:
                h, w = self.last_frame.shape[:2]
                os.makedirs(os.path.join(os.path.dirname(__file__), "recordings"),
                            exist_ok=True)
                ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(os.path.dirname(__file__),
                                    "recordings", f"session_{ts}.mp4")
                self.video_writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                self.recording = True
                self.btn_rec.setObjectName("btn_rec_on")
                self.btn_rec.setText("⏹  STOP REC")
                self.btn_rec.setStyleSheet("")
                self.setStyleSheet(self.styleSheet())
                print(f"  Enregistrement : {path}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release(); self.video_writer = None
            self.btn_rec.setObjectName("")
            self.btn_rec.setText("⏺  REC")
            self.setStyleSheet(self.styleSheet())
            print("  Enregistrement arrete.")

    def _on_export(self):
        if not self.conversation: return
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(__file__), f"session_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n HAND TALK TRANSLATOR — Session\n")
            f.write(f" {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                    + "=" * 50 + "\n\n")
            for e in self.conversation:
                t    = datetime.datetime.fromtimestamp(e["time"]).strftime("%H:%M:%S")
                side = "Sourd  [LSF]" if e["side"] == "sign" else "Entend [MIC]"
                f.write(f"[{t}] {side} : {e['text']}\n")
        print(f"  Export : {path}")

    # ── Helpers ───────────────────────────────────────────────────────

    def _conv_append(self, color: str, tag: str, text: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        align = "left" if tag == "LSF" else "right"
        bubble_bg = "rgba(0, 212, 255, 35)" if tag == "LSF" else "rgba(52, 245, 168, 35)"
        border = color
        html = (
            f"<div style='margin:6px 0; text-align:{align};'>"
            f"<span style='display:inline-block; max-width:75%; "
            f"background:{bubble_bg}; border:1px solid {border}; "
            f"border-radius:14px; padding:6px 12px; color:{C_WHITE};"
            f"font-size:13px;'>"
            f"<span style='color:{color}; font-weight:bold; font-size:10px; "
            f"letter-spacing:1px;'>{tag}</span> &nbsp; {text}"
            f"</span> "
            f"<span style='color:{C_DIM2}; font-size:9px;'>&nbsp;{ts}</span>"
            f"</div>"
        )
        self.conv_text.append(html)
        sb = self.conv_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        self.worker.stop(); self.worker.wait(3000)
        if self.recording and self.video_writer: self.video_writer.release()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not _QT_OK:
        print("ERREUR: PyQt6 non installe.  py -3.11 -m pip install PyQt6")
        sys.exit(1)

    from config import MODEL_PATH, LABELS_PATH
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        print("Aucun modele — lancez d'abord collect_data.py puis train_model.py")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-listen", action="store_true")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Demarrer en plein ecran (jury demo)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Hand Talk Translator")
    win = MainWindow(listen=not args.no_listen)
    if args.fullscreen:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
