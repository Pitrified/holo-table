"""Compute distances between landmarks."""

from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
import numpy as np

from holo_table.utils.mediapipe import HAND_LANDMARK_MAP


def compute_landmark_dist(
    one_hand_world_landmarks: list[Landmark],
    landmark_name1: str,
    landmark_name2: str,
) -> float:
    """Compute the distance between two landmarks."""
    landmark1 = one_hand_world_landmarks[HAND_LANDMARK_MAP[landmark_name1]]
    landmark2 = one_hand_world_landmarks[HAND_LANDMARK_MAP[landmark_name2]]
    return np.linalg.norm(
        np.array([landmark1.x, landmark1.y, landmark1.z])
        - np.array([landmark2.x, landmark2.y, landmark2.z])
    ).astype(float)


def compute_pinch_level(
    one_hand_world_landmarks: list[Landmark],
) -> float:
    """Compute the pinch size, normalized."""
    dist_thumb_index = compute_landmark_dist(
        one_hand_world_landmarks, "THUMB_TIP", "INDEX_FINGER_TIP"
    )
    dist_wrist_index = compute_landmark_dist(
        one_hand_world_landmarks, "WRIST", "INDEX_FINGER_MCP"
    )
    return dist_thumb_index / dist_wrist_index
