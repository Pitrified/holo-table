"""Compute the landmarks and send the distance into the ether."""

import json
from typing import Self
import click
import cv2 as cv
from loguru import logger as lg
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
import numpy as np

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

        # setup the webcam
        self.cap = cv.VideoCapture(0)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        lg.info(f"fps: {self.fps}")
        # read one frame to get the start position
        self.cap.read()
        self.start_pos_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)

    def run(self) -> None:
        """Run the sender."""
        frame_num = 0
        while True:
            # get the frame
            ret, frame_opencv = self.cap.read()
            frame_opencv = cv.flip(frame_opencv, 1)
            pos_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)
            frame = Frame.from_opencv(
                frame_opencv, pos_msec - self.start_pos_msec, frame_num
            )

            # compute the landmarks
            detection_result = self.hlf.detect(frame)

            # draw the landmarks
            annotated_image = draw_landmarks(frame, detection_result)

            # get the landmarks for the hand and compute the pinch level
            one_hand_world_landmarks = get_landmarks_from_result(
                detection_result, "world"
            )
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

            # send the pinch level
            payload = self.build_payload(pinch_level=pinch_level, pos_msec=frame.msec)
            self.uss.send(payload)

            # show the frame
            cv_imshow_rgb("Pincher", annotated_image)
            frame_num += 1

            # check for quit
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    def build_payload(self, **kwargs) -> str:
        """Build the payload to send.

        Additional security could be added here.
        """
        return json.dumps(kwargs)

    def exit(self) -> None:
        """Clean up resources."""
        self.cap.release()
        self.uss.quit()
        cv.destroyAllWindows()

    def __enter__(self) -> Self:
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context."""
        self.exit()


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
    with Sender(ip, port) as sender:
        sender.run()


if __name__ == "__main__":
    main()
