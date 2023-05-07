"""Compute the landmarks and send the distance into the ether.

Streamlit frontend.
"""

import json
from typing import Self

import av
import click
import cv2 as cv
from loguru import logger as lg
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from holo_table.landmark.compute import HandLandmarkerFrame
from holo_table.landmark.dist import compute_landmark_dist, compute_pinch_level
from holo_table.landmark.drawing import draw_landmarks
from holo_table.socket.socket import UdpSocketSender
from holo_table.utils.cv import cv_imshow_rgb
from holo_table.utils.data import get_resource
from holo_table.utils.mediapipe import (
    HAND_LANDMARK_MAP,
    HAND_LANDMARK_NAMES,
    get_default_hand_connections,
    get_landmarks_from_result,
)
from holo_table.utils.utils import get_current_msec
from holo_table.video.frame import Frame


class Sender:
    """Compute the landmarks and send the distance into the ether."""

    def __init__(self, ip: str, port: int) -> None:
        """Initialize the sender."""
        # setup the landmark detector
        hand_landmark_model_path = get_resource("hand_landmarker.task")
        self.hlf = HandLandmarkerFrame(
            hand_landmark_model_path=hand_landmark_model_path,
            hand_landmarker_kwargs={
                "running_mode": VisionRunningMode.VIDEO,
                "num_hands": 1,
            },
        )

        # setup the socket sender
        self.ip = ip
        self.port = port
        self.uss = UdpSocketSender(ip=self.ip, port=self.port)

        # setup the text settings
        self.text_kwargs = {
            "fontFace": cv.FONT_HERSHEY_SIMPLEX,
            "fontScale": 1,
            "color": (0, 0, 255),
            "thickness": 2,
        }

        self.img_format = "rgb24"
        self.start_pos_msec = get_current_msec()
        self.frame_num = 0

    def img_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format=self.img_format)
        # img = cv.flip(img, 1)
        pos_msec = get_current_msec()
        frame = Frame.from_np_array(img, pos_msec - self.start_pos_msec, self.frame_num)

        # compute the landmarks
        detection_result = self.hlf.detect(frame)

        # draw the landmarks
        annotated_image = draw_landmarks(frame, detection_result)

        # get the landmarks for the hand and compute the pinch level
        one_hand_world_landmarks = get_landmarks_from_result(detection_result, "world")
        if one_hand_world_landmarks is not None:
            pinch_level = compute_pinch_level(one_hand_world_landmarks)
            cv.putText(
                img=annotated_image,
                text=f"pinch: {pinch_level:.2f}",
                org=(10, 30),
                **self.text_kwargs,
            )
            dist_thumb_index = compute_landmark_dist(
                one_hand_world_landmarks, "THUMB_TIP", "INDEX_FINGER_TIP"
            )
            cv.putText(
                img=annotated_image,
                text=f"dist_thumb_index: {dist_thumb_index:.2f}",
                org=(10, 60),
                **self.text_kwargs,
            )
        else:
            pinch_level = None
            dist_thumb_index = None

        # send the pinch level
        payload = self.build_payload(
            pinch_level=pinch_level,
            pos_msec=frame.msec,
            dist_thumb_index=dist_thumb_index,
        )
        self.uss.send(payload)

        # show the frame
        self.frame_num += 1
        return av.VideoFrame.from_ndarray(annotated_image, format=self.img_format)

    def build_payload(self, **kwargs) -> str:
        """Build the payload to send.

        Additional security could be added here.
        """
        return json.dumps(kwargs)


@st.cache_resource
def get_sender(**kwargs) -> Sender:
    return Sender(**kwargs)


@click.command()
@click.option(
    "--ip",
    default="127.0.0.1",
    help="The IP address of the server.",
)
@click.option(
    "--port",
    default=5005,
    help="The port of the server.",
)
def main(ip: str, port: int) -> None:
    """Run the sender."""
    st.title("Holo Table Sender")

    sender = get_sender(ip=ip, port=port)

    webrtc_streamer(
        key="sender",
        video_frame_callback=sender.img_callback,
        # async_transform=True, # ?? suggested by copilot lol
        # client_settings={"video_quality": 3},
    )


if __name__ == "__main__":
    main()
