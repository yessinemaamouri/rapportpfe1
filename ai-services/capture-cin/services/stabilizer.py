"""
services/stabilizer.py
Stabilise la validation sur un buffer de frames consécutives.
Objectif: CONFIRMED uniquement si les 5 dernières frames consécutives valident:
  - conf >= 0.50
  - bbox contenue dans le guide (tolérance gérée dans detector)
"""

from collections import deque

STABILITY_BUFFER_SIZE = 5

class Stabilizer:
    """
    Instancié pour chaque connexion WebSocket (chaque client ou scan).
    Conserve un buffer circulaire (5 frames) et un compteur de streak consécutif.
    """
    def __init__(self):
        self._buffer: deque[bool] = deque(maxlen=STABILITY_BUFFER_SIZE)
        self._streak: int = 0
        self._capture_sent_for_current_streak: bool = False

    def reset(self) -> None:
        self._buffer.clear()
        self._streak = 0
        self._capture_sent_for_current_streak = False

    def update(self, is_valid: bool) -> tuple[str, int, bool]:
        """
        Ajoute une frame au buffer:
        - True si conditions 1&2 OK
        - False sinon

        Retourne (status, confirm_progress, should_capture)
        - CONFIRMED si les 5 dernières frames consécutives sont True
        - confirm_progress = 0..5 (streak courant)
        - should_capture = True une seule fois lors du passage en CONFIRMED
        """
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
