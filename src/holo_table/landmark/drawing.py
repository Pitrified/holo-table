"""Drawing module for landmark detection."""

from pathlib import Path
from typing import cast

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing_utils
import mediapipe.python.solutions.hands as mp_hands
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
import numpy as np

from holo_table.landmark.compute import HandLandmarkerFrame
from holo_table.utils.cv import cv_imshow
from holo_table.utils.data import get_resource
from holo_table.utils.mediapipe import (
    HAND_LANDMARK_MAP,
    HAND_LANDMARK_NAMES,
    get_default_hand_connections,
    get_landmarks_from_result,
    list_land_to_landlist,
)
from holo_table.utils.plt import show_frame
from holo_table.video.frame import Frame
from holo_table.video.load import iterate_video_frames, list_video_frames


def draw_landmarks(
    frame: Frame,
    detection_result: HandLandmarkerResult,
) -> np.ndarray:
    """Draw the landmarks on the image."""
    rgb_image = frame.image.numpy_view()
    annotated_image = np.copy(rgb_image)

    hand_landmarks_list = detection_result.hand_landmarks
    # handedness_list = detection_result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        # handedness = handedness_list[idx]

        # Draw the hand landmarks.
        landmark_list = list_land_to_landlist(hand_landmarks)
        mp_drawing_utils.draw_landmarks(
            annotated_image,
            landmark_list,
            get_default_hand_connections(),
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image