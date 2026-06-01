"""Serial-port reader for the ESP32 sign-language glove.

Parses the firmware's CSV stream (see ``firmware/sign_glove/sign_glove.ino``)
into typed ``GloveFrame`` records and exposes a thread-safe rolling buffer
so the rest of the application can poll the latest reading at its own pace.

A drop-in **mock backend** is included so the entire Python pipeline
(collection, training, evaluation, real-time UI) can be developed and
unit-tested without the hardware physically connected.

Usage
─────

    from glove_reader import GloveReader

    # Real hardware on COM5 (Windows) / /dev/ttyUSB0 (Linux)
    reader = GloveReader(port="COM5")

    # Headless mock for CI / dev
    reader = GloveReader(port="MOCK")

    reader.start()
    while True:
        frame = reader.read_latest()
        if frame:
            print(frame.flex, frame.accel, frame.gyro)

The CSV record format (firmware-side) is:

    t_ms,f1,f2,f3,ax,ay,az,gx,gy,gz

where ``f1=thumb``, ``f2=index``, ``f3=middle`` are the three flex sensors
mounted on the glove.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# ── Public types ─────────────────────────────────────────────────────────────


NUM_FLEX = 3                # thumb / index / middle
NUM_CSV_FIELDS = 1 + NUM_FLEX + 6   # t_ms + flex + accel(3) + gyro(3) = 10


@dataclass(frozen=True)
class GloveFrame:
    """A single 50 Hz sample from the glove."""

    t_ms: int
    flex: tuple[int, int, int]            # raw 12-bit ADC, 0..4095
    accel: tuple[float, float, float]     # g, ±2g range
    gyro:  tuple[float, float, float]     # dps


# ── Parser ───────────────────────────────────────────────────────────────────


def parse_csv_line(line: str) -> GloveFrame | None:
    """Parse a single firmware CSV line into a :class:`GloveFrame`.

    Returns ``None`` for comments (``# ...``), blank lines or malformed
    records — never raises. This makes it safe to feed every byte of the
    raw serial stream through this function.
    """
    if not line:
        return None
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split(",")
    if len(parts) != NUM_CSV_FIELDS:
        return None

    try:
        t_ms = int(parts[0])
        flex = tuple(int(x) for x in parts[1:1 + NUM_FLEX])
        accel = tuple(float(x) for x in parts[1 + NUM_FLEX:4 + NUM_FLEX])
        gyro  = tuple(float(x) for x in parts[4 + NUM_FLEX:7 + NUM_FLEX])
    except ValueError:
        return None

    return GloveFrame(t_ms=t_ms, flex=flex, accel=accel, gyro=gyro)  # type: ignore[arg-type]


# ── Reader (background thread) ───────────────────────────────────────────────


class GloveReader:
    """Background thread that drains the serial port and exposes the latest frame.

    Pass ``port="MOCK"`` to use the synthetic backend (no hardware needed).
    """

    def __init__(self, port: str, baudrate: int = 115200,
                 buffer_size: int = 256) -> None:
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size

        self._buffer: deque[GloveFrame] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial = None
        self._mock = port.upper() == "MOCK"
        self._frames_seen = 0
        self._error: str | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self, timeout_open_s: float = 3.0) -> bool:
        """Open the serial port and start the background reader thread.

        Returns ``True`` on success, ``False`` otherwise (in which case
        :attr:`error` describes the problem).
        """
        if self._thread is not None and self._thread.is_alive():
            return True

        if self._mock:
            self._thread = threading.Thread(
                target=self._mock_loop, name="GloveReader[MOCK]", daemon=True)
            self._stop_evt.clear()
            self._thread.start()
            return True

        if not _SERIAL_AVAILABLE:
            self._error = ("pyserial non installe — "
                           "pip install pyserial, ou utilise port='MOCK'")
            return False

        try:
            self._serial = serial.Serial(
                self.port, self.baudrate, timeout=0.1)
        except (serial.SerialException, OSError) as exc:
            self._error = f"impossible d'ouvrir {self.port}: {exc}"
            return False

        # Some boards reset on open: give them a moment to come back
        t0 = time.time()
        while time.time() - t0 < timeout_open_s:
            try:
                self._serial.reset_input_buffer()
                break
            except Exception:
                time.sleep(0.05)

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._serial_loop, name="GloveReader", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    # ── public read API ─────────────────────────────────────────────────────

    def read_latest(self) -> GloveFrame | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def drain(self, max_frames: int | None = None) -> list[GloveFrame]:
        """Pop up to *max_frames* oldest frames (or all)."""
        with self._lock:
            if max_frames is None or max_frames >= len(self._buffer):
                out = list(self._buffer)
                self._buffer.clear()
            else:
                out = [self._buffer.popleft() for _ in range(max_frames)]
        return out

    def push(self, frame: GloveFrame) -> None:
        """Inject a frame externally (used by tests and the mock backend)."""
        with self._lock:
            self._buffer.append(frame)
            self._frames_seen += 1

    @property
    def frames_seen(self) -> int:
        with self._lock:
            return self._frames_seen

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_mock(self) -> bool:
        return self._mock

    def __enter__(self) -> "GloveReader":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ── internal loops ──────────────────────────────────────────────────────

    def _serial_loop(self) -> None:
        assert self._serial is not None
        buf = b""
        while not self._stop_evt.is_set():
            try:
                chunk = self._serial.read(256)
            except Exception as exc:
                self._error = f"lecture serie: {exc}"
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                try:
                    text = line.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                frame = parse_csv_line(text)
                if frame is not None:
                    self.push(frame)

    def _mock_loop(self) -> None:
        """Synthetic 50 Hz stream: a slow open/close fist plus a slow wave."""
        period_s = 1.0 / 50.0
        t0 = time.perf_counter()
        while not self._stop_evt.is_set():
            elapsed = time.perf_counter() - t0
            phase = elapsed * 1.2  # rad/s
            base = 2200 + 700 * math.sin(phase)
            flex = tuple(int(base + 60 * math.sin(phase + i))
                         for i in range(NUM_FLEX))
            accel = (
                0.05 * math.sin(phase * 0.5),
                0.05 * math.cos(phase * 0.5),
                0.98,
            )
            gyro = (
                15.0 * math.sin(phase * 1.3),
                10.0 * math.cos(phase * 1.7),
                3.0  * math.sin(phase * 0.9),
            )
            frame = GloveFrame(
                t_ms=int(elapsed * 1000),
                flex=flex,            # type: ignore[arg-type]
                accel=accel,
                gyro=gyro,
            )
            self.push(frame)
            self._stop_evt.wait(period_s)


# ── Calibration + feature normalisation ──────────────────────────────────────


_DEFAULT_FLEX_MIN: tuple[int, ...] = (1500,) * NUM_FLEX
_DEFAULT_FLEX_MAX: tuple[int, ...] = (3000,) * NUM_FLEX


@dataclass
class GloveCalibration:
    """Per-finger ADC range collected during the calibration step.

    ``flex_min`` corresponds to fully extended (open palm) and ``flex_max``
    to fully flexed (closed fist). Values outside the range are clamped.
    """

    flex_min: tuple[int, ...] = _DEFAULT_FLEX_MIN
    flex_max: tuple[int, ...] = _DEFAULT_FLEX_MAX

    def normalise(self, frame: GloveFrame) -> tuple[float, ...]:
        """Return the 9-D feature vector ``(f1..f3, ax, ay, az, gx, gy, gz)``.

        Flex values are mapped to ``[0, 1]`` using the calibration range.
        Accelerometer is left in g (already ~[-2, 2]) and gyro is divided by
        250 to bring it into ``~[-1, 1]``.
        """
        flex_norm = []
        for i in range(NUM_FLEX):
            lo, hi = self.flex_min[i], self.flex_max[i]
            span = max(1, hi - lo)
            v = (frame.flex[i] - lo) / span
            flex_norm.append(min(1.0, max(0.0, v)))
        ax, ay, az = frame.accel
        gx, gy, gz = frame.gyro
        return (*flex_norm,
                ax, ay, az,
                gx / 250.0, gy / 250.0, gz / 250.0)

    def to_dict(self) -> dict:
        return {
            "flex_min": list(self.flex_min),
            "flex_max": list(self.flex_max),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GloveCalibration":
        # Tolerate legacy 5-flex calibrations: take only the first NUM_FLEX entries.
        flex_min = tuple(data["flex_min"])[:NUM_FLEX]
        flex_max = tuple(data["flex_max"])[:NUM_FLEX]
        if len(flex_min) < NUM_FLEX or len(flex_max) < NUM_FLEX:
            raise ValueError(
                f"calibration has {len(flex_min)} flex entries, "
                f"expected at least {NUM_FLEX}")
        return cls(flex_min=flex_min, flex_max=flex_max)

    @classmethod
    def from_samples(cls, frames: Iterable[GloveFrame]) -> "GloveCalibration":
        """Build a calibration from a captured set of frames spanning open
        palm AND closed fist (in any order)."""
        mins = [4095] * NUM_FLEX
        maxs = [0] * NUM_FLEX
        for f in frames:
            for i in range(NUM_FLEX):
                v = f.flex[i]
                if v < mins[i]:
                    mins[i] = v
                if v > maxs[i]:
                    maxs[i] = v
        # Avoid degenerate (min == max) ranges
        for i in range(NUM_FLEX):
            if maxs[i] - mins[i] < 100:
                maxs[i] = mins[i] + 100
        return cls(flex_min=tuple(mins),  # type: ignore[arg-type]
                   flex_max=tuple(maxs))  # type: ignore[arg-type]


FEATURES_PER_FRAME = NUM_FLEX + 6   # 9
FRAMES_PER_SEQUENCE = 30
GLOVE_HZ = 50  # nominal firmware rate
