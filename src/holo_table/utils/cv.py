"""Utility functions for opencv."""

from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def cv_imshow(
    img: np.ndarray,
    ax: plt.Axes | None = None,
) -> None:
    """Show a BGR img properly.

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
