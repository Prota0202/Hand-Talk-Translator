"""Background speech listener for bidirectional communication.

Listens to the microphone in a separate daemon thread and transcribes
speech using Google Speech Recognition (requires internet).
The transcribed text is pushed to a thread-safe queue.
"""

import queue
import threading

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


class SpeechListener:
    """Continuously listens to the microphone and transcribes speech to text.

    Usage:
        listener = SpeechListener()
        listener.start()
        text = listener.get_text()   # None if nothing new
        listener.stop()
    """

    def __init__(self, language: str = "fr-FR"):
        self.language = language
        self.available = _SR_AVAILABLE
        self._q: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

        if self.available:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.8

    def start(self) -> bool:
        """Start listening in background. Returns False if unavailable."""
        if not self.available or self._running:
            return self.available
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False

    def get_text(self) -> str | None:
        """Return the latest transcribed text, or None if nothing new."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def _listen_loop(self):
        if not self.available:
            return
        try:
            with sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                while self._running:
                    try:
                        audio = self._recognizer.listen(
                            source, timeout=2, phrase_time_limit=10
                        )
                        text = self._recognizer.recognize_google(
                            audio, language=self.language
                        )
                        if text.strip():
                            self._q.put(text.strip())
                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        pass
                    except Exception:
                        pass
        except Exception:
            self._running = False
