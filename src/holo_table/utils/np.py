"""Utility functions for numpy."""

from typing import Any
import numpy as np


def diff_pad(x: np.ndarray) -> np.ndarray:
    """Compute the difference between adjacent elements, padding the first element with itself."""
    return np.diff(x, prepend=x[0])


def create_left_triangle_filter(
    window_size: int,
) -> np.ndarray:
    """Create a left triangle filter."""
    triangle = np.arange(1, window_size + 1)
    triangle = triangle / triangle.sum()
    return triangle


def roll_append(x: np.ndarray, val: Any) -> np.ndarray:
    """Roll the array to the left, and assign val at the end of arr."""
    x = np.roll(x, -1)
    x[-1] = val
    return x


def roll_append_smooth(
    hist: np.ndarray,
    hist_smooth: np.ndarray,
    value: Any,
    filt: np.ndarray,
):
    """Roll the data, smooth using the filter and roll the smooth."""
    # update the history of the original data
    hist = roll_append(hist, value)
    # compute the moving average using the filter
    value_smooth = np.dot(hist, filt)
    # update the history of the smoothed data
    hist_smooth = roll_append(hist_smooth, value_smooth)
    return hist, hist_smooth
