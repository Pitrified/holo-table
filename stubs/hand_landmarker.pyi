"""Stubs for hand_landmarker.py."""

from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.landmark import Landmark

class HandLandmarkerResult:
    handedness: list[list[Category]]
    hand_landmarks: list[list[NormalizedLandmark]]
    hand_world_landmarks: list[list[Landmark]]
