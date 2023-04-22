"""Load a video from a file or a camera.

https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
"""

from pathlib import Path
from typing import Generator

import cv2 as cv

from holo_table.video.frame import Frame


def iterate_video_frames(
    in_vid_path: Path,
    keep_every_nth_frame: int = 1,
) -> Generator[Frame, None, None]:
    """Extract frames from video.

    Args:
        in_vid_path: Input video file.
        keep_every_nth_frame: Keep every nth frame in the video.

    Yields:
        Frame.
    """
    # start capturing the feed
    cap = cv.VideoCapture(str(in_vid_path))
    # moving the cap definition outside the try block lets it be used in the
    # finally block, but if something goes wrong before we even get to the try
    # block, then we never get to the finally block ? mmm

    # if you really want a msec_interval,
    # we could compute the keep_every_nth_frame from that and the fps
    # and if we need to skip many frames we could use the set(CAP_PROP_POS_MSEC)
    # (at that point a small change in frame interval does not matter)

    try:
        count = 0
        success = True
        while success:
            # extract the frame
            success, frame = cap.read()
            if not success:
                break

            # get the timestamp
            pos_msec = cap.get(cv.CAP_PROP_POS_MSEC)
            # pos_usec = int(pos_msec * 1000)

            # skip frames
            if count % keep_every_nth_frame == 0:
                # lg.debug(f"Yielding frame {count}")
                yield Frame.from_opencv(frame, pos_msec, count)

            count = count + 1

    finally:
        # close the feed
        # lg.info(f"Closing video feed")
        cap.release()


def list_video_frames(
    in_vid_path: Path,
    keep_every_nth_frame: int = 1,
    max_frame_count: int = 0,
) -> list[Frame]:
    """Extract frames from video, return them as a list.

    Args:
        in_vid_path: Input video file.
        keep_every_nth_frame: Keep every nth frame in the video.
        max_frame_count: Maximum number of frames to extract.
            Set to 0 to extract all frames.
    """
    frames: list[Frame] = []

    frame_num = 0
    for frame in iterate_video_frames(
        in_vid_path,
        keep_every_nth_frame=keep_every_nth_frame,
    ):
        frames.append(frame)
        frame_num += 1
        if max_frame_count > 0 and frame_num >= max_frame_count:
            break

    return frames
