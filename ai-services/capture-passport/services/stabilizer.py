"""
services/stabilizer.py
Stabilise la validation sur un buffer de frames consécutives (5 frames).
"""
from collections import deque

STABILITY_BUFFER_SIZE = 5


class Stabilizer:
    def __init__(self):
        self._buffer: deque[bool] = deque(maxlen=STABILITY_BUFFER_SIZE)
        self._streak: int = 0
        self._capture_sent_for_current_streak: bool = False

    def reset(self) -> None:
        self._buffer.clear()
        self._streak = 0
        self._capture_sent_for_current_streak = False

    def update(self, is_valid: bool) -> tuple[str, int, bool]:
        self._buffer.append(bool(is_valid))

        if is_valid:
            self._streak += 1
        else:
            self._streak = 0
            self._capture_sent_for_current_streak = False

        progress = min(self._streak, STABILITY_BUFFER_SIZE)
        confirmed = self._streak >= STABILITY_BUFFER_SIZE
        should_capture = False

        if confirmed and not self._capture_sent_for_current_streak:
            should_capture = True
            self._capture_sent_for_current_streak = True

        return ("CONFIRMED" if confirmed else "DETECTED"), progress, should_capture
