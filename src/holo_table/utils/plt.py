"""Utils for plotting."""

import matplotlib.pyplot as plt

from holo_table.utils.cv import resize
from holo_table.video.frame import Frame


def show_frame(
    frame: Frame,
    ax: plt.Axes | None = None,
    title_suffix: str | None = None,
    do_show: bool = True,
    do_resize: bool = True,
) -> None:
    """Show a Frame."""
    if ax is None:
        _, ax = plt.subplots()

    if do_resize:
        img = resize(frame.image.numpy_view())
    else:
        img = frame.image.numpy_view()
    ax.imshow(img)

    ax.set_axis_off()

    title = f"{frame.idx} @ {frame.msec:.0f}"
    if title_suffix is not None:
        title += f" {title_suffix}"
    ax.set_title(title)

    if do_show:
        plt.show()
