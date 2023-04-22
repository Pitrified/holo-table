"""Misc utils for landmark detection."""


from enum import IntEnum
from typing import Literal, Mapping, TypeVar, cast, get_args

import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import numpy as np
from typing import Literal, overload
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmarkList,
    LandmarkList,
)
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers.category import Category

T = TypeVar("T")

HAND_LANDMARK_NAMES = HandLandmark._member_names_
HAND_LANDMARK_MAP = cast(dict[str, IntEnum], HandLandmark._member_map_)


def get_default_hand_connections() -> list[tuple[int, int]]:
    """Get the default hand connections.

    Cast the connections to a list of tuples for the sake of type checking.
    """
    hand_connections = cast(list[tuple[int, int]], mp_hands.HAND_CONNECTIONS)
    return hand_connections


def get_spec_from_map(
    drawing_spec: mp_drawing.DrawingSpec | Mapping[T, mp_drawing.DrawingSpec],
    key: T,
) -> mp_drawing.DrawingSpec:
    """Extract a DrawingSpec from a Mapping or return the DrawingSpec itself."""
    if isinstance(drawing_spec, Mapping):
        return drawing_spec[key]
    return drawing_spec


@overload
def get_landmarks_from_result(
    result: HandLandmarkerResult,
    which_info: Literal["world"],
    hand_idx: int = 0,
) -> list[Landmark] | None:
    ...


@overload
def get_landmarks_from_result(
    result: HandLandmarkerResult,
    which_info: Literal["normalized"],
    hand_idx: int = 0,
) -> list[NormalizedLandmark] | None:
    ...


@overload
def get_landmarks_from_result(
    result: HandLandmarkerResult,
    which_info: Literal["handedness"],
    hand_idx: int = 0,
) -> list[Category] | None:
    ...


def get_landmarks_from_result(
    result: HandLandmarkerResult,
    which_info: Literal["world", "normalized", "handedness"],
    hand_idx: int = 0,
) -> list[Landmark] | list[NormalizedLandmark] | list[Category] | None:
    """Get the info from the result, for a specific hand."""
    if which_info == "world":
        ll = result.hand_world_landmarks
    elif which_info == "normalized":
        ll = result.hand_landmarks
    elif which_info == "handedness":
        ll = result.handedness
    if hand_idx >= len(ll):
        return None
    return ll[hand_idx]


@overload
def list_land_to_landlist(
    hand_landmarks: list[NormalizedLandmark],
) -> NormalizedLandmarkList:
    ...


@overload
def list_land_to_landlist(
    hand_landmarks: list[Landmark],
) -> LandmarkList:
    ...


def list_land_to_landlist(
    hand_landmarks: list[NormalizedLandmark] | list[Landmark],
) -> NormalizedLandmarkList | LandmarkList:
    """Convert a list of [Normalized]Landmark to a [Normalized]LandmarkList."""
    # decide which type of landmark list to use
    if isinstance(hand_landmarks[0], NormalizedLandmark):
        land_type = landmark_pb2.NormalizedLandmark
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    elif isinstance(hand_landmarks[0], Landmark):
        land_type = landmark_pb2.Landmark
        hand_landmarks_proto = landmark_pb2.LandmarkList()
    else:
        raise TypeError(
            "hand_landmarks must be a list of NormalizedLandmark or Landmark"
        )

    # add the landmarks to the list
    hand_landmarks_proto.landmark.extend(  # type: ignore # Member "landmark" is unknown
        [
            land_type(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ]
    )
    return hand_landmarks_proto
