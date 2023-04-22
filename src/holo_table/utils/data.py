"""Utility functions for the whole package."""

from pathlib import Path
from typing import Literal

from loguru import logger as lg


def get_resource(
    which_res: Literal[
        "hand_landmarker.task",
        "root_fol",
        "sample_fol",
        "hand_fol",
    ]
) -> Path:
    """Get the path of the requested resource."""
    # resources
    if which_res == "hand_landmarker.task":
        mp_model_fol = Path("~/.mediapipe/models").expanduser()
        hand_landmark_model_path = mp_model_fol / "hand_landmarker.task"
        return hand_landmark_model_path

    # folders that are not in the package
    if which_res == "hand_fol":
        return Path.home() / "data" / "hand"

    # folders that are in the package
    if which_res == "root_fol":
        return Path(__file__).absolute().parents[3]
    elif which_res == "sample_fol":
        return get_resource("root_fol") / "data" / "sample"
