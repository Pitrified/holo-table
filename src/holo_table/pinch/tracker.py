"""Track a pinch distance and decide if it is a pinch or not."""

import numpy as np

from holo_table.utils.np import create_left_triangle_filter, roll_append, roll_append_smooth


class PinchTracker:
    """Track a pinch distance and decide if it is a pinch or not."""

    def __init__(
        self,
        sd_max: float,
        sd_min: float,
        sdsd_max: float,
        filter_size: int = 5,
    ) -> None:
        """Initialize the pinch tracker."""
        # ranges for the derivative
        self.sd_max = sd_max
        self.sd_min = sd_min
        self.sdsd_max = sdsd_max

        # filter
        self.filter_size = filter_size
        self.filter = create_left_triangle_filter(self.filter_size)

        # msec data
        self.hmsec = np.zeros(self.filter_size, dtype=float)

        # pinch data
        self.hdist = np.zeros(self.filter_size, dtype=float)
        # pinch data smoothed
        self.hdist_s = np.zeros(self.filter_size, dtype=float)

        # first derivative of the pinch data smoothed
        self.hdist_sd = np.zeros(self.filter_size, dtype=float)
        # first derivative of the pinch data smoothed, smoothed
        self.hdist_sds = np.zeros(self.filter_size, dtype=float)

        # second derivative of the pinch data
        # (computed on the first derivative smoothed)
        self.hdist_sdsd = np.zeros(self.filter_size, dtype=float)
        # second derivative of the pinch data, smoothed
        self.hdist_sdsds = np.zeros(self.filter_size, dtype=float)

        # track them all to plot later
        self.all_msec_ls = []
        self.all_dist_ls = []
        self.all_dist_s_ls = []
        self.all_dist_sd_ls = []
        self.all_dist_sds_ls = []
        self.all_dist_sdss_ls = []
        self.all_dist_sdsd_ls = []
        self.all_dist_sdsds_ls = []
        self.all_ispinch_ls = []
        self.all_ispinch_sds_ls = []
        self.all_ispinch_sdsds_ls = []

        # is it a pinch? if so, how much is it changing?
        self.ispinch = 0
        self.ispinch_sds = 0
        self.ispinch_sdsds = 0
        self.dist_sdss = 0

    def update(self, dist: float, msec: float):
        """Update the pinch tracker with a new distance."""
        # update the history of the msec data
        self.hmsec = roll_append(self.hmsec, msec)

        # update the history of the raw data
        self.hdist, self.hdist_s = roll_append_smooth(
            self.hdist, self.hdist_s, dist, self.filter
        )

        # compute the first derivative of the smoothed data
        dist_sd = self.hdist_s[-1] - self.hdist_s[-2]
        # update the history of the first derivative (and its smoothed version)
        self.hdist_sd, self.hdist_sds = roll_append_smooth(
            self.hdist_sd, self.hdist_sds, dist_sd, self.filter
        )

        # compute the second derivative of the smoothed data
        dist_sdsd = self.hdist_sds[-1] - self.hdist_sds[-2]
        # update the history of the second derivative (and its smoothed version)
        self.hdist_sdsd, self.hdist_sdsds = roll_append_smooth(
            self.hdist_sdsd, self.hdist_sdsds, dist_sdsd, self.filter
        )

        # absolute values
        adist_sds = np.abs(self.hdist_sds)
        adist_sdsds = np.abs(self.hdist_sdsds)

        # check if the first derivative is in a pinching range
        self.ispinch_sds = np.all(adist_sds > self.sd_min) and np.all(
            adist_sds < self.sd_max
        )
        # check if the second derivative is in a pinching range
        self.ispinch_sdsds = np.all(adist_sdsds < self.sdsd_max)
        # if both are in a pinching range
        self.ispinch = self.ispinch_sds and self.ispinch_sdsds

        # smooth the first derivative again
        self.dist_sdss = np.dot(self.hdist_sds, self.filter)
        if self.ispinch:
            self.all_dist_sdss_ls.append(self.dist_sdss)
        else:
            self.all_dist_sdss_ls.append(0)

        # track them all to plot later
        self.all_msec_ls.append(msec)
        self.all_dist_ls.append(dist)
        self.all_dist_s_ls.append(self.hdist_s[-1])
        self.all_dist_sd_ls.append(dist_sd)
        self.all_dist_sds_ls.append(self.hdist_sds[-1])
        self.all_dist_sdsd_ls.append(dist_sdsd)
        self.all_dist_sdsds_ls.append(self.hdist_sdsds[-1])
        self.all_ispinch_ls.append(self.ispinch)
        self.all_ispinch_sds_ls.append(self.ispinch_sds)
        self.all_ispinch_sdsds_ls.append(self.ispinch_sdsds)

        # if hist is too long, remove the oldest elements
        self.check_hist_len()

        # return the change in distance if it is a pinch
        return self.dist_sdss

    def check_hist_len(self):
        """Check that the history is not too long."""
        self.max_all_len = 10000
        new_len = -self.max_all_len // 2
        # or we could pop one ?
        if len(self.all_dist_ls) > self.max_all_len:
            self.all_dist_ls = self.all_dist_ls[new_len:]
            self.all_dist_s_ls = self.all_dist_s_ls[new_len:]
            self.all_dist_sd_ls = self.all_dist_sd_ls[new_len:]
            self.all_dist_sds_ls = self.all_dist_sds_ls[new_len:]
            self.all_dist_sdss_ls = self.all_dist_sdss_ls[new_len:]
            self.all_dist_sdsd_ls = self.all_dist_sdsd_ls[new_len:]
            self.all_dist_sdsds_ls = self.all_dist_sdsds_ls[new_len:]
            self.all_ispinch_ls = self.all_ispinch_ls[new_len:]
            self.all_ispinch_sds_ls = self.all_ispinch_sds_ls[new_len:]
            self.all_ispinch_sdsds_ls = self.all_ispinch_sdsds_ls[new_len:]
