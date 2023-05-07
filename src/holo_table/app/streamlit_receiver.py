"""Receive the pinch level from the sender and update the plot."""

import json

import click
from loguru import logger as lg
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

from holo_table.pinch.tracker import PinchTracker
from holo_table.socket.socket import UdpSocketReceiver


class Receiver:
    """Receive the pinch level from the sender and update the plot."""

    def __init__(
        self,
        ip: str,
        port: int,
        buffer_size: int = 1024,
    ) -> None:
        """Initialize the plot and the UDP socket."""
        # socket
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.usr = UdpSocketReceiver(self.ip, self.port)

        # ranges for the derivative
        # TODO slider to change these
        self.sd_max = 0.04
        self.sd_min = 0.006
        self.sdsd_max = 0.005

        # pinch tracker
        self.tracker = PinchTracker(self.sd_max, self.sd_min, self.sdsd_max)

        # tracker also has a max len and a whole bunch of history
        self.pinch_level = 0
        self.dist = 0
        self.max_data_points = 100
        self.xdata, self.ydata = [], []

    def receive(self):
        """Parse the pinch level message and update the plot."""
        for data, addr in self.usr.receive(self.buffer_size):
            lg.debug(f"Received {data} from {addr}")
            info = json.loads(data)
            self.data_update(**info)
            self.plot_update()
            yield info

    def data_update(
        self,
        pos_msec: float,
        pinch_level: float | None,
        dist_thumb_index: float | None,
    ) -> None:
        """Update the data."""
        # sanitize the data
        if pinch_level is None:
            pinch_level = 0
        self.pinch_level = pinch_level
        if dist_thumb_index is None:
            dist_thumb_index = 0
        self.dist = dist_thumb_index

        # update the tracker
        self.tracker.update(pinch_level, pos_msec)

        # # extract relevant data to plot
        # if len(self.xdata) < self.max_data_points:
        #     self.xdata.append(len(self.xdata))
        # self.ydata.append(pinch_level)
        # y = self.ydata[-100:]
        # self.fig = px.scatter(
        #     x=self.xdata,
        #     y=y,
        #     range_y=[min(y), max(y)],
        # )

    def plot_update(self) -> None:
        """Update the plot."""
        # convert to numpy arrays
        plot_len_max = 100
        all_msec = np.array(self.tracker.all_msec_ls[-plot_len_max:])
        all_dist = np.array(self.tracker.all_dist_ls[-plot_len_max:])
        all_dist_s = np.array(self.tracker.all_dist_s_ls[-plot_len_max:])
        all_dist_sd = np.array(self.tracker.all_dist_sd_ls[-plot_len_max:])
        all_dist_sds = np.array(self.tracker.all_dist_sds_ls[-plot_len_max:])
        all_dist_sdss = np.array(self.tracker.all_dist_sdss_ls[-plot_len_max:])
        all_dist_sdsd = np.array(self.tracker.all_dist_sdsd_ls[-plot_len_max:])
        all_dist_sdsds = np.array(self.tracker.all_dist_sdsds_ls[-plot_len_max:])
        all_ispinch = np.array(self.tracker.all_ispinch_ls[-plot_len_max:])
        all_ispinch_sds = np.array(self.tracker.all_ispinch_sds_ls[-plot_len_max:])
        all_ispinch_sdsds = np.array(self.tracker.all_ispinch_sdsds_ls[-plot_len_max:])

        # Create subplot grid
        fig = sp.make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Pinch Data",
                "Pinch Data SD",
                "Pinch Data SDSD",
                "Pinch Data to send",
            ),
        )

        ################
        # pinch data
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist,
                mode="markers",
                marker=dict(size=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_s,
                mode="lines",
                line=dict(color="coral"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(range=[all_dist.min(), all_dist.max()], row=1, col=1)
        fig.add_shape(
            type="rect",
            x0=all_msec[0],
            y0=all_dist.min(),
            x1=all_msec[-1],
            y1=all_dist.max(),
            fillcolor="chartreuse",
            opacity=0.2,
            line=dict(width=0),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_ispinch.astype(int),
                fill="tozeroy",
                mode="none",
                # fillcolor="coral", # name="Pinch Detection",
                opacity=0.2,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # ################
        # first derivative
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_sd,
                mode="markers",
                marker=dict(size=1),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_sds,
                mode="lines",
                line=dict(color="coral"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_ispinch_sds.astype(int),
                fill="tozeroy",
                mode="none",
                # fillcolor="coral", # name="Pinch Detection",
                opacity=0.2,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[all_dist_sd.min(), all_dist_sd.max()], row=2, col=1)

        # ################
        # second derivative
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_sdsd,
                mode="markers",
                marker=dict(size=1),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_sdsds,
                mode="lines",
                line=dict(color="coral"),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_ispinch_sdsds.astype(int),
                fill="tozeroy",
                mode="none",
                # fillcolor="coral", # name="Pinch Detection",
                opacity=0.2,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.update_yaxes(range=[all_dist_sdsd.min(), all_dist_sdsd.max()], row=3, col=1)

        # ################
        # pinch data to send
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_sds,
                mode="lines",
                line=dict(color="coral"),
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=all_msec,
                y=all_dist_sdss,
                mode="lines",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.update_yaxes(range=[all_dist_sds.min(), all_dist_sds.max()], row=4, col=1)

        #################
        # layout
        fig.update_layout(
            height=900,
            width=800,
            # title="Pinch Data title",
            xaxis=dict(title="msec"),
            yaxis=dict(title="dist"),
            xaxis2=dict(title="msec"),
            yaxis2=dict(title="dist sds"),
            xaxis3=dict(title="msec"),
            yaxis3=dict(title="dist sdsds"),
            xaxis4=dict(title="msec"),
            yaxis4=dict(title="dist sdss"),
        )

        # save the fig
        self.fig = fig


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
    st.title("Holo Table Receiver")

    # creating a single-element container
    placeholder = st.empty()

    receiver = Receiver(ip, port)
    for a in receiver.receive():
        # lg.debug(f"Yielding {a}")

        # if the plot update lags, we can skip some updates
        # and just update the receiver

        with placeholder.container():
            st.metric(
                label="Pinch Level",
                value=f"{receiver.pinch_level:.3f}",
            )

            st.write(receiver.fig)


if __name__ == "__main__":
    main()
