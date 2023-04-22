"""Utility functions for opencv."""

import math
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def cv_imshow(
    img: np.ndarray,
    ax: plt.Axes | None = None,
) -> None:
    """Show a BGR img in a matplotlib figure.

    If ax is not None, then the image is plotted on the given axis,
    and the function returns without showing the image with plt.show().
    """
    # TODO auto detect image type (if it's grayscale)
    # img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if ax is not None:
        ax.imshow(img1)
        return
    plt.imshow(img1)
    plt.show()


def resize(
    image,
    desired_width=640,
    desired_height=480,
) -> np.ndarray:
    """Resize the image to the desired width and height.

    https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv.resize(
            image,
            (desired_width, math.floor(h / (w / desired_width))),
        )
    else:
        img = cv.resize(
            image,
            (math.floor(w / (h / desired_height)), desired_height),
        )
    return img


def cv_imshow_rgb(winname: str, image_rgb: np.ndarray) -> None:
    """Show a RGB image in an opencv window."""
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    cv.imshow(winname, image_bgr)
