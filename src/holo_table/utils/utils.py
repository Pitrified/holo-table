"""General utility functions for the holo_table package."""

from time import time


def get_current_msec() -> int:
    """Get the current time in milliseconds."""
    return int(round(time() * 1000))
