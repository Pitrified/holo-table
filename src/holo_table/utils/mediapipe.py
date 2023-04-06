"""Misc utils for landmark detection."""

from typing import cast

import mediapipe.python.solutions.hands as mp_hands


def get_default_hand_connections() -> list[tuple[int, int]]:
    """Get the default hand connections.

    Cast the connections to a list of tuples for the sake of type checking.
    """
    hand_connections = cast(list[tuple[int, int]], mp_hands.HAND_CONNECTIONS)
    return hand_connections
