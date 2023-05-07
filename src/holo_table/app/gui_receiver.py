"""Update a plot using FuncAnimation and a class.

https://matplotlib.org/stable/api/animation_api.html
"""


import json
import time
import click
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from loguru import logger as lg

from holo_table.socket.socket import UdpSocketReceiver


class Receiver:
    """Receive data from UDP socket and update a plot."""

    def __init__(
        self,
        ip: str,
        port: int,
        buffer_size: int = 1024,
    ) -> None:
        """Initialize the plot and the UDP socket."""
        # plot
        self.xdata, self.ydata = [], []
        self.fig, self.ax = plt.subplots()
        (self.ln,) = self.ax.plot([], [], color="C1", marker="o", linestyle="None")
        self.max_data_points = 100

        # socket
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.usr = UdpSocketReceiver(self.ip, self.port)

    def data_update(self, pos_msec: float, pinch_level: float) -> None:
        """Update the data."""
        # self.xdata.append(pos_msec)
        if len(self.xdata) < self.max_data_points:
            self.xdata.append(len(self.xdata))
        self.ydata.append(pinch_level)

    def plot_init(self) -> tuple[Line2D]:
        """Initialize the plot."""
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 2)
        self.ax.grid()
        return (self.ln,)

    def plot_update(self, msg) -> tuple[Line2D]:
        """Update the plot."""
        # unpack the message
        data, addr = msg
        lg.debug(f"Received {data} from {addr}")
        if data == "quit":
            # this message is lost ?
            lg.debug(f"Quitting...")
        info = json.loads(data)
        self.data_update(**info)
        # update the plot
        # self.ln.set_data(self.xdata, self.ydata)
        self.ln.set_data(self.xdata, self.ydata[-100:])
        # return the artists to be updated
        return (self.ln,)

    def run(self) -> None:
        """Run the animation."""
        ani = FuncAnimation(
            self.fig,
            self.plot_update,
            frames=self.usr.receive(self.buffer_size),
            init_func=self.plot_init,
            blit=True,
            interval=1,
            cache_frame_data=False,
            repeat=False,
        )
        plt.show()


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
    """Run the app."""
    app = Receiver(ip, port)
    app.run()


if __name__ == "__main__":
    main()
