"""Receive the pinch level from the sender and update the plot."""

from time import sleep

from PIL import Image
import numpy as np
import plotly.express as px
import streamlit as st

from holo_table.utils.data import get_resource


def get_square_pattern(size: int) -> np.ndarray:
    """Generate a square pattern."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # img[::2, ::2] = 255
    # img[1::2, 1::2] = 255
    pattern_size = int(size / 10)
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                img[
                    i * pattern_size : (i + 1) * pattern_size,
                    j * pattern_size : (j + 1) * pattern_size,
                ] = 255
    return img


class Fractal:
    """Receive the zoom input and update the fractal image."""

    def __init__(self) -> None:
        """Initialize the template."""
        self.zoom_level = 0
        self.zoom_steps = 10

        # in the template we assume that the image is square
        # and that if we get the center of the image, we can get the whole image
        # self.template_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # self.template_img = get_square_pattern(100)
        # load the template image
        img_path = get_resource("box_fractal")
        # load the png image as numpy array
        img_pil = Image.open(img_path)
        self.template_img = np.array(img_pil)
        print(self.template_img.shape)
        self.template_size = self.template_img.shape[0]

        # compute how much we need to chop off each side, per zoom step
        # /2 to get the free space
        # /zoom_steps to get each individual step
        # /2 to get each side
        self.zoom_chop_size = self.template_size / 2 / self.zoom_steps / 2
        print(self.zoom_chop_size)

        self.update_image()

    def receive_zoom(self, step: int) -> None:
        """Update the zoom level."""
        self.zoom_level += step
        self.update_image()

    def update_image(self) -> None:
        """Update the fractal image.

        We create a view of the template, and resize it to a standard dimension.
        We can just get the mod of the zoom level to fake computing the fractal.
        As the zoom mod goes up, we get a smaller and smaller view of the template.
        """
        zoom_mod = self.zoom_level % self.zoom_steps
        zoom_pad = self.zoom_chop_size * zoom_mod
        zoom_pad_int = int(zoom_pad)
        # extract the view
        self.view = self.template_img[
            zoom_pad_int : self.template_size - zoom_pad_int,
            zoom_pad_int : self.template_size - zoom_pad_int,
        ]
        print(zoom_mod, zoom_pad, self.view.shape)

        # create the figure with plotly
        self.fig = px.imshow(self.view)


def main() -> None:
    """Run the app."""
    print("\nrestarting")
    st.title("Holo Table fractal sample")

    # creating a single-element container
    placeholder = st.empty()

    fractal = Fractal()

    for i in range(19):
        with placeholder.container():
            st.metric(label="Pinch Level", value=f"{i:.3f}")
            fractal.receive_zoom(1)
            st.write(fractal.fig)
            sleep(0.5)

    for i in range(19):
        with placeholder.container():
            st.metric(label="Pinch Level", value=f"{i:.3f}")
            fractal.receive_zoom(-1)
            st.write(fractal.fig)
            sleep(0.5)


if __name__ == "__main__":
    main()
