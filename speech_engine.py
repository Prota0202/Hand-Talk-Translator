"""Text-to-speech engine using pyttsx3 (offline, no internet required).

A dedicated worker thread owns the pyttsx3 engine and processes speech
requests from a queue, ensuring the main loop is never blocked.
"""

import asyncio
import ctypes
import os
import queue
import tempfile
import threading
import time
import traceback

import edge_tts

from config import TTS, RECOGNITION


class SpeechEngine:
    """Non-blocking TTS that processes speech requests from a queue."""

    def __init__(self):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._speaking = False
        self._last_spoken: dict[str, float] = {}
        self._cooldown = RECOGNITION["cooldown_seconds"]
        self._last_error_ts = 0.0

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── public API ───────────────────────────────────────────────────────────

    def speak(self, text: str, force: bool = False) -> bool:
        """Enqueue *text* for speech.  Set *force* to bypass the cooldown."""
        if not force:
            now = time.time()
            if text in self._last_spoken:
                if now - self._last_spoken[text] < self._cooldown:
                    return False
            self._last_spoken[text] = now
        self._queue.put(text)
        return True

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def release(self):
        """Signal the worker thread to stop."""
        self._queue.put(None)

    # ── internals ────────────────────────────────────────────────────────────

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._speaking = True
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp_path = tmp.name
                voice = TTS.get("edge_voice", "fr-BE-CharlineNeural")
                rate = TTS.get("edge_rate", "+0%")
                volume = TTS.get("edge_volume", "+0%")
                communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
                asyncio.run(communicate.save(tmp_path))
                self._play_mp3(tmp_path)
            except Exception:
                now = time.time()
                if now - self._last_error_ts > 5:
                    self._last_error_ts = now
                    print("[TTS] Erreur Edge TTS:")
                    traceback.print_exc(limit=2)
            finally:
                try:
                    if "tmp_path" in locals() and os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                self._speaking = False

    @staticmethod
    def _play_mp3(path: str):
        """Play an MP3 file using Windows MCI (no extra deps)."""
        cmd = f'open "{path}" type mpegvideo alias ttsmp3'
        ctypes.windll.winmm.mciSendStringW(cmd, None, 0, None)
        ctypes.windll.winmm.mciSendStringW("play ttsmp3 wait", None, 0, None)
        ctypes.windll.winmm.mciSendStringW("close ttsmp3", None, 0, None)
