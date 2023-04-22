"""Detect hand landmarks."""

from pathlib import Path
from typing import Literal, overload

from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from typing import Callable

from holo_table.video.frame import Frame


def create_hand_landmarker(
    hand_landmark_model_path: Path,
    **kwargs,
) -> HandLandmarker:
    """Create the landmarker object.

    Default HandLandmarkerOptions kwargs are:
        running_mode: _RunningMode = _RunningMode.IMAGE,
        num_hands: int | None = 1,
        min_hand_detection_confidence: float | None = 0.5,
        min_hand_presence_confidence: float | None = 0.5,
        min_tracking_confidence: float | None = 0.5,
        result_callback: Callable[[HandLandmarkerResult, mediapipe.python.Image, int], None] | None = None
    """
    base_options = BaseOptions(model_asset_path=str(hand_landmark_model_path))
    options = HandLandmarkerOptions(base_options=base_options, **kwargs)
    detector = HandLandmarker.create_from_options(options)
    return detector


class HandLandmarkerFrame:
    """HandLandmarker that can accept Frame objects as input."""

    def __init__(
        self,
        hand_landmark_model_path: Path,
        hand_landmarker_kwargs: dict = {},
    ) -> None:
        """Initialize the HandLandmarkerFrame."""
        self.hand_landmarker = create_hand_landmarker(
            hand_landmark_model_path,
            **hand_landmarker_kwargs,
        )

    def detect(self, frame: Frame) -> HandLandmarkerResult:
        """Process a frame."""
        # return self.hand_landmarker.detect(frame.image)
        return self.hand_landmarker.detect_for_video(frame.image, int(frame.msec))
